"""APScheduler wiring (SPEC §14).

`build_scheduler()` returns a configured BlockingScheduler. Its cadence:
  - safety.health_check      every 5 min
  - data.bankroll_snapshot   every hour
  - strategy.calibration     every 15 min
  - (content + other strategies: add as they land)

The scheduler is timezone-aware (settings.timezone); "daily 10am local"
jobs (content publish) fire correctly across DST transitions.
"""

from __future__ import annotations

import logging
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from moneybutton.core.config import Settings, get_settings
from moneybutton.scheduler.jobs import (
    bankroll_snapshot_job,
    calibration_scan_job,
    health_check_job,
)

log = logging.getLogger("moneybutton.scheduler")


def build_scheduler(
    *,
    settings: Settings | None = None,
    calibration_feed_fn: Callable | None = None,
    calibration_scanner_fn: Callable | None = None,
) -> BlockingScheduler:
    settings = settings or get_settings()
    scheduler = BlockingScheduler(timezone=str(settings.tzinfo))

    scheduler.add_job(
        _wrap_safe(health_check_job),
        trigger=IntervalTrigger(minutes=5),
        id="safety.health_check",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        _wrap_safe(bankroll_snapshot_job),
        trigger=IntervalTrigger(hours=1),
        id="data.bankroll_snapshot",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    if calibration_feed_fn is not None and calibration_scanner_fn is not None:
        def _calib_run() -> None:
            calibration_scan_job(
                feed=calibration_feed_fn,
                scanner_fn=calibration_scanner_fn,
                settings=settings,
            )

        scheduler.add_job(
            _wrap_safe(_calib_run),
            trigger=IntervalTrigger(minutes=15),
            id="strategy.calibration.scan",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

    return scheduler


def _wrap_safe(fn: Callable) -> Callable:
    """Any uncaught exception in a job should log + notify but never bring
    the whole scheduler down."""

    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("job %s failed: %s", getattr(fn, "__name__", "?"), exc)
            try:
                from moneybutton.core.notifier import notify

                notify(
                    f"scheduled job {getattr(fn, '__name__', '?')} failed: {exc}",
                    level="critical",
                )
            except Exception:  # noqa: BLE001
                pass

    _inner.__name__ = getattr(fn, "__name__", "_wrapped")
    return _inner
