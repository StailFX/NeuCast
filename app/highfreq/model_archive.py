"""Model versioning + rollback (T.17.d, 2026-05-03).

Why
===

The trainer overwrites ``weights/highfreq/<sym>_1m.cbm`` on every
daily cron run. If a run produces a regressing model and pushes it
to production, paper-traders pick it up via mtime-watch within 60 s
and we lose the previous (good) model. With ~1 h to detect a bad
deploy via realized accuracy, we'd be flying blind.

This module:

1. Snapshots ``<sym>_1m.cbm`` + ``<sym>_1m_metrics.json`` +
   ``<sym>_1m_calibrator.pkl`` into ``weights/highfreq/archive/``
   under a timestamped name BEFORE the trainer overwrites.
2. Keeps the last ``N`` snapshots per symbol (default 7 = ~1 week
   of daily training cadence). Older ones get deleted.
3. Exposes ``rollback_to(symbol, snapshot_iso)`` that copies a
   chosen archived version back over the live weights — the
   paper-trader's mtime-watcher then picks it up automatically.

Pure functions only — no DB, no logging side effects beyond INFO.

Used at the trainer's save site + via the
``tools.rollback_model`` CLI for manual operator action.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _archive_dir(weights_dir: Path) -> Path:
    """Subdir of the weights directory where snapshots live."""
    return weights_dir / "archive"


def _stem_for(weights_path: Path) -> str:
    """Symbol+horizon prefix from a weights path.
    ``btcusdt_1m.cbm`` → ``btcusdt_1m``."""
    return weights_path.stem


def archive_existing(
    weights_path: Path,
    *,
    keep_last_n: int = 7,
    now: datetime | None = None,
) -> Path | None:
    """Copy the CURRENT weights file (and its sidecar metrics +
    calibrator if present) to a timestamped archive subdir.

    Called BEFORE the trainer overwrites the production .cbm. If the
    file doesn't yet exist (first-ever training), this is a no-op
    (returns None). Otherwise returns the archive path of the .cbm
    snapshot.

    After the copy, prunes the archive to keep only the last
    ``keep_last_n`` snapshots per symbol+horizon (oldest first).
    """
    if not weights_path.exists():
        return None
    archive_dir = _archive_dir(weights_path.parent)
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem_for(weights_path)
    ts = (now or datetime.now(tz=timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    archive_cbm = archive_dir / f"{stem}_{ts}.cbm"
    shutil.copy2(weights_path, archive_cbm)
    logger.info("archived %s → %s", weights_path.name, archive_cbm.name)

    # Sidecars: metrics.json + calibrator.pkl. If present, copy with
    # matching timestamp suffix so rollback can restore them as a
    # set.
    metrics_src = weights_path.with_name(f"{stem}_metrics.json")
    if metrics_src.exists():
        metrics_dst = archive_dir / f"{stem}_{ts}_metrics.json"
        shutil.copy2(metrics_src, metrics_dst)
    cal_src = weights_path.with_name(f"{stem}_calibrator.pkl")
    if cal_src.exists():
        cal_dst = archive_dir / f"{stem}_{ts}_calibrator.pkl"
        shutil.copy2(cal_src, cal_dst)

    _prune_archive(archive_dir, stem, keep_last_n=keep_last_n)
    return archive_cbm


def _prune_archive(
    archive_dir: Path, stem: str, *, keep_last_n: int,
) -> list[Path]:
    """Delete oldest archived snapshots beyond ``keep_last_n``.

    Returns the list of paths that were deleted (mostly for tests).
    """
    if keep_last_n <= 0:
        return []
    # Find all archived .cbm for this stem. Naming pattern:
    # ``{stem}_{ISO_TS}.cbm``. Sort by mtime so ties are resolved
    # consistently.
    snapshots = sorted(
        archive_dir.glob(f"{stem}_*.cbm"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(snapshots) <= keep_last_n:
        return []
    to_delete = snapshots[:-keep_last_n]
    deleted: list[Path] = []
    for old_cbm in to_delete:
        try:
            old_cbm.unlink()
            deleted.append(old_cbm)
            # Delete sidecars matching the same timestamp.
            ts_part = old_cbm.stem[len(stem) + 1:]  # e.g. 20260503T...
            for ext in ("_metrics.json", "_calibrator.pkl"):
                sidecar = archive_dir / f"{stem}_{ts_part}{ext}"
                if sidecar.exists():
                    sidecar.unlink()
        except OSError as exc:
            logger.warning("failed to prune %s: %s", old_cbm, exc)
    if deleted:
        logger.info(
            "pruned %d old archive snapshot(s) (kept last %d)",
            len(deleted), keep_last_n,
        )
    return deleted


def list_snapshots(weights_path: Path) -> list[dict]:
    """Return all archived snapshots for the given live weights file,
    newest first. Each entry: ``{ts_iso, cbm_path, metrics_path,
    calibrator_path}`` (sidecars are None if absent).

    Useful for the ``tools.rollback_model`` CLI to print the menu of
    rollback targets to the operator.
    """
    archive_dir = _archive_dir(weights_path.parent)
    if not archive_dir.exists():
        return []
    stem = _stem_for(weights_path)
    out: list[dict] = []
    for cbm in sorted(
        archive_dir.glob(f"{stem}_*.cbm"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        ts_part = cbm.stem[len(stem) + 1:]
        metrics = archive_dir / f"{stem}_{ts_part}_metrics.json"
        cal = archive_dir / f"{stem}_{ts_part}_calibrator.pkl"
        out.append({
            "ts_iso": ts_part,
            "cbm_path": str(cbm),
            "metrics_path": str(metrics) if metrics.exists() else None,
            "calibrator_path": str(cal) if cal.exists() else None,
            "size_bytes": cbm.stat().st_size,
        })
    return out


def rollback_to(weights_path: Path, ts_iso: str) -> dict:
    """Copy the archived snapshot identified by ``ts_iso`` back over
    the live weights (+ sidecars). Paper-trader's mtime-watcher picks
    it up automatically within ~60 s.

    ``ts_iso`` format: ``YYYYMMDDTHHMMSSZ`` (matches ``archive_existing``
    output). Returns the restored {cbm, metrics, calibrator} paths.

    Raises ``FileNotFoundError`` if the snapshot doesn't exist.
    """
    archive_dir = _archive_dir(weights_path.parent)
    stem = _stem_for(weights_path)
    src_cbm = archive_dir / f"{stem}_{ts_iso}.cbm"
    if not src_cbm.exists():
        raise FileNotFoundError(
            f"no archived snapshot at {src_cbm} (ts_iso={ts_iso!r})"
        )

    # Before clobbering live, archive THE CURRENT live state first
    # — operator may want to roll forward back to it later if the
    # rollback was a mistake. archive_existing handles the no-op case
    # if live doesn't exist yet.
    archive_existing(weights_path, keep_last_n=14)

    shutil.copy2(src_cbm, weights_path)
    logger.info("rolled back %s ← %s", weights_path.name, src_cbm.name)

    metrics_dst = weights_path.with_name(f"{stem}_metrics.json")
    cal_dst = weights_path.with_name(f"{stem}_calibrator.pkl")
    metrics_src = archive_dir / f"{stem}_{ts_iso}_metrics.json"
    cal_src = archive_dir / f"{stem}_{ts_iso}_calibrator.pkl"
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dst)
    if cal_src.exists():
        shutil.copy2(cal_src, cal_dst)
    return {
        "cbm": str(weights_path),
        "metrics": str(metrics_dst) if metrics_dst.exists() else None,
        "calibrator": str(cal_dst) if cal_dst.exists() else None,
    }
