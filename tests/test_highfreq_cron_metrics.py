"""Tests for ``app.highfreq.cron_metrics`` — the textfile-collector helper
backing the 3 cron-stale alerts.

Why these tests matter
======================

The cron-failure alerting strategy depends on a tight contract between
the *writer* (this module) and the *reader* (node_exporter's
``--collector.textfile.directory``). If the writer:

* writes a malformed body → node_exporter logs a warning and the metric
  *silently disappears* — alert never fires;
* writes non-atomically → node_exporter may pick up a half-written file
  and skip the metric;
* swallows the error too aggressively → the cron itself silently
  succeeds even when its heartbeat couldn't be recorded.

These tests pin the contract so a refactor can't degrade alerting.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.highfreq.cron_metrics import (
    DEFAULT_TEXTFILE_DIR,
    _escape_label_value,
    _format_labels,
    textfile_dir,
    write_cron_success,
)


# ───── label rendering ─────


def test_format_labels_empty_returns_empty_string():
    """No labels → bare metric line, not ``{}`` (which Prometheus rejects)."""
    assert _format_labels(None) == ""
    assert _format_labels({}) == ""


def test_format_labels_single():
    assert _format_labels({"symbol": "BTCUSDT"}) == '{symbol="BTCUSDT"}'


def test_format_labels_multiple_sorted():
    """Order must be deterministic (sorted) so the same metric write
    produces a byte-identical file across reruns — useful for diffing
    or asserting in tests."""
    out = _format_labels({"job": "trainer", "symbol": "BTCUSDT"})
    assert out == '{job="trainer",symbol="BTCUSDT"}'


def test_escape_label_value_quotes_backslash_newline():
    """The exposition spec requires escaping these three characters in
    label values — anything else is allowed verbatim. Wrong escapes
    cause node_exporter to reject the line."""
    assert _escape_label_value('a"b') == 'a\\"b'
    assert _escape_label_value("a\\b") == "a\\\\b"
    assert _escape_label_value("a\nb") == "a\\nb"
    # Other chars passthrough — equals signs, spaces, unicode all OK.
    assert _escape_label_value("a=b ё") == "a=b ё"


# ───── directory resolution ─────


def test_textfile_dir_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("TEXTFILE_COLLECTOR_DIR", raising=False)
    assert str(textfile_dir()) == DEFAULT_TEXTFILE_DIR


def test_textfile_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXTFILE_COLLECTOR_DIR", str(tmp_path))
    assert textfile_dir() == tmp_path


# ───── write_cron_success — happy path ─────


def test_write_cron_success_creates_well_formed_file(tmp_path):
    ok = write_cron_success(
        "neucast_hf_l2_archive_last_success_timestamp_seconds",
        file_stem="neucast_hf_l2_archive",
        now=1714435200,  # 2024-04-29 00:00:00 UTC, deterministic
        directory=tmp_path,
    )
    assert ok is True
    out = (tmp_path / "neucast_hf_l2_archive.prom").read_text()
    assert "# HELP neucast_hf_l2_archive_last_success_timestamp_seconds" in out
    assert "# TYPE neucast_hf_l2_archive_last_success_timestamp_seconds gauge" in out
    assert "neucast_hf_l2_archive_last_success_timestamp_seconds 1714435200" in out


def test_write_cron_success_with_labels(tmp_path):
    ok = write_cron_success(
        "neucast_hf_trainer_last_success_timestamp_seconds",
        file_stem="neucast_hf_trainer_btcusdt",
        labels={"symbol": "BTCUSDT"},
        now=1714435200,
        directory=tmp_path,
    )
    assert ok is True
    body = (tmp_path / "neucast_hf_trainer_btcusdt.prom").read_text()
    expected = (
        'neucast_hf_trainer_last_success_timestamp_seconds'
        '{symbol="BTCUSDT"} 1714435200'
    )
    assert expected in body


def test_write_cron_success_creates_dir_if_missing(tmp_path):
    """Fresh deploy: the textfile dir may not exist yet. The helper
    must create it (mode 0o755 implicit) so the first cron run isn't a
    no-op that silently logs a warning."""
    target = tmp_path / "deeply" / "nested" / "collector"
    assert not target.exists()
    ok = write_cron_success(
        "neucast_hf_test_metric",
        file_stem="test_stem",
        directory=target,
        now=42,
    )
    assert ok is True
    assert target.is_dir()
    assert (target / "test_stem.prom").exists()


def test_write_cron_success_overwrites_previous(tmp_path):
    """Idempotent: a re-run replaces the timestamp in place. node_exporter
    reads the current file each scrape, so a stale value would silently
    break the alert (it'd show "fresh!" forever)."""
    write_cron_success("m", file_stem="s", now=100, directory=tmp_path)
    write_cron_success("m", file_stem="s", now=200, directory=tmp_path)
    body = (tmp_path / "s.prom").read_text()
    assert "m 200" in body
    assert "m 100" not in body


def test_write_cron_success_atomic_no_partial_files(tmp_path):
    """After a successful write, the only file with our stem must be
    the final ``.prom`` — no leftover ``.tmp``s. node_exporter scans
    the dir and would pick up half-written files otherwise."""
    write_cron_success("m", file_stem="s", now=42, directory=tmp_path)
    files = sorted(tmp_path.iterdir())
    assert [f.name for f in files] == ["s.prom"]


def test_write_cron_success_uses_time_time_when_now_omitted(tmp_path, monkeypatch):
    """When ``now`` is not provided, the helper falls through to
    ``time.time()``. Pin via monkeypatch so a fast CI machine doesn't
    race."""
    monkeypatch.setattr("app.highfreq.cron_metrics.time.time", lambda: 1234567890.5)
    write_cron_success("m", file_stem="s", directory=tmp_path)
    # time.time returns float; we format as %.0f → 1234567891 (banker's
    # round; here .5 rounds up by Python's default in printf format).
    body = (tmp_path / "s.prom").read_text()
    # %.0f of 1234567890.5 may round to either 1234567890 or 1234567891
    # depending on Python build's printf — accept either.
    assert "m 123456789" in body


# ───── failure modes (must be fail-soft) ─────


def test_write_cron_success_returns_false_on_unwritable_dir(tmp_path):
    """If the directory can't be written (permissions, full FS, etc.)
    the helper returns False but does NOT raise. The cron's success
    return value must be unaffected by a metric-write failure."""
    if os.geteuid() == 0:  # root bypasses permissions
        pytest.skip("test cannot run as root")
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)  # read-only
    try:
        ok = write_cron_success(
            "m",
            file_stem="s",
            directory=locked,
            now=42,
        )
    finally:
        # Restore so pytest can clean up.
        locked.chmod(0o700)
    assert ok is False
    # Nothing got written.
    assert list(locked.iterdir()) == []


def test_write_cron_success_target_is_a_file_returns_false(tmp_path):
    """If the resolved path is a regular file (operator typo'd
    TEXTFILE_COLLECTOR_DIR pointing at a file), mkdir errors → False."""
    blocker = tmp_path / "is-a-file"
    blocker.write_text("oops")
    ok = write_cron_success("m", file_stem="s", directory=blocker, now=42)
    assert ok is False


# ───── exposition format integration ─────


def test_full_body_matches_prometheus_textfile_spec(tmp_path):
    """End-to-end: lines must be:
        1. ``# HELP <name> <free text>``
        2. ``# TYPE <name> gauge``
        3. ``<name>{<labels>} <value>``
    each terminated with ``\\n``. node_exporter parses this strictly."""
    write_cron_success(
        "neucast_hf_x", file_stem="x", labels={"a": "b"},
        now=42, directory=tmp_path,
    )
    body = (tmp_path / "x.prom").read_text()
    lines = body.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("# HELP neucast_hf_x ")
    assert lines[1] == "# TYPE neucast_hf_x gauge"
    assert lines[2] == 'neucast_hf_x{a="b"} 42'
    # Trailing newline so the file is POSIX-clean (no missing-newline-at-eof).
    assert body.endswith("\n")
