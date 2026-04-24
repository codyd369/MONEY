"""Parquet read/write helpers with partition management (SPEC §6.2).

Layout mirrors SPEC §6.2:
    data/parquet/
        markets/category={cat}/year_month={ym}/markets.parquet
        prices/category={cat}/year_month={ym}/prices.parquet
        orderbook/category={cat}/year_month={ym}/orderbook.parquet
        news/source={src}/year_month={ym}/news.parquet
        features/strategy={strat}/year_month={ym}/features.parquet

The core concern of this module: writes must be idempotent and append-safe
across crashes. We achieve this by:
  1. Writing to a .tmp sibling and os.replace()-ing on success (atomic rename).
  2. Keying each row by a deterministic id (ticker+ts for prices, id for news,
     ticker for markets) so re-writes overwrite duplicates cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from moneybutton.core.config import get_settings


Dataset = Literal["markets", "prices", "orderbook", "news", "features", "trades"]


@dataclass(frozen=True)
class PartitionKey:
    dataset: Dataset
    category_or_source: str
    year_month: str  # "YYYY-MM"


def _base_dir() -> Path:
    return Path(get_settings().data_dir) / "parquet"


def _partition_folder(key: PartitionKey) -> Path:
    base = _base_dir() / key.dataset
    if key.dataset == "news":
        return base / f"source={key.category_or_source}" / f"year_month={key.year_month}"
    if key.dataset == "features":
        return base / f"strategy={key.category_or_source}" / f"year_month={key.year_month}"
    return base / f"category={key.category_or_source}" / f"year_month={key.year_month}"


def _partition_file(key: PartitionKey) -> Path:
    return _partition_folder(key) / f"{key.dataset}.parquet"


_DEDUPE_KEYS: dict[str, list[str]] = {
    "markets": ["ticker"],
    "prices": ["ticker", "ts"],
    "orderbook": ["ticker", "ts", "side", "level"],
    "news": ["id"],
    "features": ["ticker", "ts"],
    "trades": ["trade_id"],
}


def write_partition(
    key: PartitionKey,
    rows: pd.DataFrame | Iterable[dict],
    *,
    dedupe: bool = True,
) -> Path:
    """Write `rows` into the partition, merging with any existing file.

    When `dedupe=True`, rows are deduplicated by the dataset-specific key
    (see _DEDUPE_KEYS) with the NEW rows winning on conflicts. The final
    DataFrame is written atomically via write-tmp-then-rename.
    """
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(list(rows))
    if rows.empty:
        # Still create the folder so subsequent readers see an empty partition,
        # but don't create a zero-row parquet.
        _partition_folder(key).mkdir(parents=True, exist_ok=True)
        return _partition_file(key)

    target = _partition_file(key)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = pd.read_parquet(target)
        merged = pd.concat([existing, rows], ignore_index=True)
    else:
        merged = rows

    if dedupe:
        keys = _DEDUPE_KEYS.get(key.dataset, [])
        if keys:
            # Later rows (i.e., the just-written `rows`) win.
            merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    tmp = target.with_suffix(".tmp")
    table = pa.Table.from_pandas(merged, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)
    return target


def read_partition(key: PartitionKey) -> pd.DataFrame:
    target = _partition_file(key)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_parquet(target)


def read_dataset(
    dataset: Dataset,
    *,
    category_or_source: str | None = None,
    year_month_from: str | None = None,
    year_month_to: str | None = None,
) -> pd.DataFrame:
    """Read all partitions for a dataset, optionally filtered.

    Returns an empty DataFrame if no partitions match; callers that need a
    fixed schema should coerce columns after.
    """
    base = _base_dir() / dataset
    if not base.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    # Directory layout is two levels deep; glob at any depth.
    for pqfile in base.rglob("*.parquet"):
        parts = pqfile.relative_to(base).parts
        if len(parts) < 3:
            continue
        cat_or_src = parts[0].split("=", 1)[-1]
        ym = parts[1].split("=", 1)[-1]
        if category_or_source and cat_or_src != category_or_source:
            continue
        if year_month_from and ym < year_month_from:
            continue
        if year_month_to and ym > year_month_to:
            continue
        frames.append(pd.read_parquet(pqfile))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def list_partitions(dataset: Dataset) -> list[PartitionKey]:
    base = _base_dir() / dataset
    if not base.exists():
        return []
    out: list[PartitionKey] = []
    for pqfile in base.rglob("*.parquet"):
        parts = pqfile.relative_to(base).parts
        if len(parts) < 3:
            continue
        cat_or_src = parts[0].split("=", 1)[-1]
        ym = parts[1].split("=", 1)[-1]
        out.append(PartitionKey(dataset=dataset, category_or_source=cat_or_src, year_month=ym))
    return sorted(out, key=lambda k: (k.category_or_source, k.year_month))


def partition_size_bytes(key: PartitionKey) -> int:
    p = _partition_file(key)
    return p.stat().st_size if p.exists() else 0


def year_month(ts_iso_or_epoch: str | int | float) -> str:
    """Normalize either ISO string or unix seconds into 'YYYY-MM'.

    RSS feeds emit a zoo of date formats ('Thu, 23 Apr 2026 16:16:17 EST',
    missing tzinfo, milliseconds-only, etc.). We try pandas first, then
    dateutil with a named-timezone hint, and fall back to the current UTC
    month so a single unparseable timestamp never breaks a batch write.
    """
    try:
        if isinstance(ts_iso_or_epoch, (int, float)):
            ts = pd.to_datetime(ts_iso_or_epoch, unit="s", utc=True)
        else:
            ts = pd.to_datetime(ts_iso_or_epoch, utc=True)
        return ts.strftime("%Y-%m")
    except (ValueError, TypeError):
        pass

    try:
        from dateutil import parser as _dp

        _TZINFOS = {
            "EST": -5 * 3600, "EDT": -4 * 3600,
            "CST": -6 * 3600, "CDT": -5 * 3600,
            "MST": -7 * 3600, "MDT": -6 * 3600,
            "PST": -8 * 3600, "PDT": -7 * 3600,
            "UTC": 0, "GMT": 0, "UT": 0, "Z": 0,
            "BST": +1 * 3600, "CET": +1 * 3600, "CEST": +2 * 3600,
        }
        parsed = _dp.parse(str(ts_iso_or_epoch), tzinfos=_TZINFOS)
        return parsed.strftime("%Y-%m")
    except Exception:  # noqa: BLE001
        import datetime as _dt

        return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m")
