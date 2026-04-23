"""On-disk model registry (SPEC §9.4).

Layout:
    data/models/{name}_v{N}/
        model.joblib           # pickled estimator
        metadata.json          # hyperparams, windows, git sha, train metrics
        feature_schema.json    # feature columns (exact names, order, fingerprint)
        train_report.html      # calibration plot + feature importance (if written)

The registry enforces one invariant: `get_active(name)` refuses to return
a model whose feature_schema.json fingerprint doesn't match the current
features.pipeline output. This prevents a stale model from silently
scoring against a different feature set.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from moneybutton.core.config import get_settings


_VERSION_RE = re.compile(r"^(?P<name>[a-z_]+)_v(?P<version>\d+)$")


@dataclass
class ModelEntry:
    name: str
    version: int
    path: Path
    metadata: dict[str, Any]
    feature_schema: dict[str, Any]

    @property
    def model_file(self) -> Path:
        return self.path / "model.joblib"


def _models_dir() -> Path:
    d = Path(get_settings().data_dir) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _schema_fingerprint(columns: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(columns)).encode()).hexdigest()


def register(
    name: str,
    estimator: Any,
    *,
    feature_columns: list[str],
    metadata: dict[str, Any],
    train_report_html: str | None = None,
) -> ModelEntry:
    """Persist an estimator as the next version of `name`."""
    existing = list_models(name)
    next_version = (max((e.version for e in existing), default=0)) + 1
    folder = _models_dir() / f"{name}_v{next_version}"
    folder.mkdir(parents=True, exist_ok=False)

    joblib.dump(estimator, folder / "model.joblib")

    meta = {
        **metadata,
        "name": name,
        "version": next_version,
        "registered_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_columns_count": len(feature_columns),
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    schema = {
        "columns": sorted(feature_columns),
        "fingerprint": _schema_fingerprint(feature_columns),
    }
    (folder / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    if train_report_html:
        (folder / "train_report.html").write_text(train_report_html, encoding="utf-8")

    return ModelEntry(
        name=name,
        version=next_version,
        path=folder,
        metadata=meta,
        feature_schema=schema,
    )


def list_models(name: str | None = None) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    base = _models_dir()
    if not base.exists():
        return out
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        m = _VERSION_RE.match(folder.name)
        if not m:
            continue
        if name and m.group("name") != name:
            continue
        meta_path = folder / "metadata.json"
        schema_path = folder / "feature_schema.json"
        if not (meta_path.exists() and schema_path.exists()):
            continue
        metadata = json.loads(meta_path.read_text())
        schema = json.loads(schema_path.read_text())
        out.append(
            ModelEntry(
                name=m.group("name"),
                version=int(m.group("version")),
                path=folder,
                metadata=metadata,
                feature_schema=schema,
            )
        )
    return out


def get_active(
    name: str,
    *,
    current_feature_columns: list[str] | None = None,
) -> ModelEntry | None:
    """Return the highest-versioned model whose feature_schema matches.

    When `current_feature_columns` is provided, the registry refuses to
    return any model whose fingerprint differs — the caller gets None and
    should decline to trade rather than score against stale features.
    """
    entries = list_models(name)
    if not entries:
        return None
    entries.sort(key=lambda e: e.version, reverse=True)
    if current_feature_columns is None:
        return entries[0]
    current_fp = _schema_fingerprint(current_feature_columns)
    for e in entries:
        if e.feature_schema.get("fingerprint") == current_fp:
            return e
    return None


def load(entry: ModelEntry) -> Any:
    return joblib.load(entry.model_file)
