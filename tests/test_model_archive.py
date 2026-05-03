"""Tests for ``app.highfreq.model_archive`` (T.17.d).

Production safety: trainer overwrites ``<sym>_1m.cbm`` daily; if a
bad model lands we'd lose the previous good one in ~60 s as
paper-traders pick it up. ``archive_existing`` snapshots the file
BEFORE overwrite; ``rollback_to`` restores from the archive.

Tests pin:
1. archive_existing copies cbm + metrics + calibrator under a
   timestamped name, returns the new path.
2. archive_existing on missing file is a no-op (returns None).
3. _prune_archive keeps the last N snapshots only.
4. list_snapshots returns newest-first.
5. rollback_to clobbers live + archives current first (undoable
   misclick).
6. rollback_to raises on missing snapshot.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.highfreq.model_archive import (
    archive_existing,
    list_snapshots,
    rollback_to,
    _prune_archive,
)


def _make_fake_weights(weights_dir: Path, stem: str = "btcusdt_1m") -> Path:
    """Create a placeholder .cbm + metrics + calibrator triplet."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    cbm = weights_dir / f"{stem}.cbm"
    cbm.write_bytes(b"placeholder cbm v1")
    (weights_dir / f"{stem}_metrics.json").write_text('{"v": 1}')
    (weights_dir / f"{stem}_calibrator.pkl").write_bytes(b"placeholder pkl v1")
    return cbm


def test_archive_existing_copies_all_three_sidecars(tmp_path):
    weights = _make_fake_weights(tmp_path)
    fixed_now = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    archived = archive_existing(weights, now=fixed_now)
    assert archived is not None
    assert archived.exists()
    archive_dir = tmp_path / "archive"
    # cbm + metrics + calibrator all present with matching ts suffix.
    assert (archive_dir / "btcusdt_1m_20260503T100000Z.cbm").exists()
    assert (archive_dir / "btcusdt_1m_20260503T100000Z_metrics.json").exists()
    assert (archive_dir / "btcusdt_1m_20260503T100000Z_calibrator.pkl").exists()


def test_archive_existing_no_live_file_returns_none(tmp_path):
    """First-ever training: no live .cbm yet → archive is a no-op."""
    weights = tmp_path / "btcusdt_1m.cbm"
    # File doesn't exist.
    assert archive_existing(weights) is None


def test_archive_skips_missing_sidecars(tmp_path):
    """Calibrator.pkl is sometimes absent (e.g. when calibration fit
    failed). archive_existing must still copy the .cbm + whatever
    sidecars exist, not crash."""
    weights = tmp_path / "btcusdt_1m.cbm"
    weights.write_bytes(b"placeholder")
    # Only .cbm — no metrics, no calibrator.
    archived = archive_existing(weights)
    assert archived is not None
    archive_dir = tmp_path / "archive"
    cbms = list(archive_dir.glob("btcusdt_1m_*.cbm"))
    assert len(cbms) == 1


def test_prune_archive_keeps_last_n(tmp_path):
    """Create 10 snapshots, prune to keep last 3 → 7 deleted."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    stem = "btcusdt_1m"
    # Make 10 cbm files with monotone mtimes.
    for i in range(10):
        f = archive_dir / f"{stem}_2026050{i}T000000Z.cbm"
        f.write_bytes(b"x")
        meta = archive_dir / f"{stem}_2026050{i}T000000Z_metrics.json"
        meta.write_text("{}")
        # Stagger mtime so sorting is deterministic.
        ts = time.time() + i
        import os as _os
        _os.utime(f, (ts, ts))
        _os.utime(meta, (ts, ts))
    deleted = _prune_archive(archive_dir, stem, keep_last_n=3)
    assert len(deleted) == 7
    # 3 cbm + 3 metrics remain.
    remaining_cbm = list(archive_dir.glob(f"{stem}_*.cbm"))
    remaining_meta = list(archive_dir.glob(f"{stem}_*_metrics.json"))
    assert len(remaining_cbm) == 3
    assert len(remaining_meta) == 3


def test_archive_then_prune_keeps_last_n(tmp_path):
    """archive_existing called 8 times with keep_last_n=3 → only 3
    archived snapshots remain (the 5 oldest auto-pruned)."""
    weights = _make_fake_weights(tmp_path)
    for hour in range(8):
        archive_existing(
            weights, keep_last_n=3,
            now=datetime(2026, 5, 3, hour, 0, 0, tzinfo=timezone.utc),
        )
        # Bump live file mtime so each iteration archives a "new" copy.
        weights.write_bytes(f"live v{hour}".encode())
    snapshots = list_snapshots(weights)
    assert len(snapshots) == 3


def test_list_snapshots_newest_first(tmp_path):
    weights = _make_fake_weights(tmp_path)
    for hour in (10, 11, 12):
        archive_existing(
            weights, now=datetime(2026, 5, 3, hour, 0, 0, tzinfo=timezone.utc),
        )
        weights.write_bytes(f"live v{hour}".encode())
        time.sleep(0.01)  # ensure mtime ordering on fast filesystems
    snapshots = list_snapshots(weights)
    # Newest first.
    timestamps = [s["ts_iso"] for s in snapshots]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_snapshots_empty_when_no_archive(tmp_path):
    weights = _make_fake_weights(tmp_path)
    # No archive_existing was called → archive dir doesn't exist.
    assert list_snapshots(weights) == []


def test_rollback_to_restores_live_from_archive(tmp_path):
    weights = _make_fake_weights(tmp_path)
    fixed_now = datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc)
    archive_existing(weights, now=fixed_now)

    # Now overwrite live with a "bad" model.
    weights.write_bytes(b"bad v2")
    weights.with_name("btcusdt_1m_metrics.json").write_text('{"v": 2}')

    result = rollback_to(weights, "20260503T100000Z")
    assert weights.read_bytes() == b"placeholder cbm v1"
    assert result["cbm"] == str(weights)
    assert result["metrics"] is not None


def test_rollback_archives_current_before_clobber(tmp_path):
    """A misclick is undoable: rollback_to first archives the
    CURRENT live state, so the operator can roll forward back to
    it if needed."""
    weights = _make_fake_weights(tmp_path)
    archive_existing(
        weights, now=datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc),
    )
    # Live file is now still v1 in archive AND on disk.
    # Overwrite live to a "bad" model.
    weights.write_bytes(b"bad v2")
    weights.with_name("btcusdt_1m_metrics.json").write_text('{"v": 2}')
    snapshots_before = list_snapshots(weights)
    assert len(snapshots_before) == 1

    rollback_to(weights, "20260503T100000Z")
    # Now there should be 2 snapshots: the original v1 AND the v2
    # that we just clobbered.
    snapshots_after = list_snapshots(weights)
    assert len(snapshots_after) == 2


def test_rollback_to_missing_snapshot_raises(tmp_path):
    weights = _make_fake_weights(tmp_path)
    with pytest.raises(FileNotFoundError, match="no archived snapshot"):
        rollback_to(weights, "99999999T000000Z")
