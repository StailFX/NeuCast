"""Tiny helper for emitting **one-shot cron success metrics** to the
node_exporter ``textfile_collector``.

Why this exists
===============

The 5 alert rules in ``docs/highfreq/deploy/grafana/alerting/alerts.yaml``
all watch **long-running services** — ingest, paper-trader — via
counters/gauges from the in-process ``prometheus_client`` registry on
the ``hf_ingest`` / ``hf_paper_trader`` jobs. None of those rules can
catch a **silently failing one-shot cron**:

* ``tools/archive_l2_to_s3.py`` runs daily — if it dies after the S3
  upload but before the Postgres ``DELETE``, the next run sees the
  S3 object already exists and might skip cleanup. **No alert fires.**
* ``tools/backup_prometheus_to_s3.py`` runs daily — if it crashes,
  yesterday's TSDB snapshot is missing in S3 and we won't know until
  we try a DR drill.
* The systemd-templated trainer (``neucast-highfreq-trainer@*.timer``)
  fires daily per symbol. If it crashes (e.g. lowercase-symbol bug from
  earlier this week), the UI shows stale metrics for ~24 h before
  anyone notices. **No alert fires.**

The fix is the standard Prometheus pattern for cron-style jobs: the
job writes its "last success" unix timestamp to a file in the
node_exporter ``textfile_collector`` directory; node_exporter
periodically scrapes the directory and exposes those metrics under
its ``node_exporter`` job. Grafana alert rules then watch::

    time() - max(neucast_hf_<job>_last_success_timestamp_seconds) > 90000

where 90000 ≈ 25 hours, allowing one daily fire to legitimately miss
its window without false-positive paging.

Why not push to a Pushgateway
-----------------------------

Pushgateway requires running yet another service and has subtle
gotchas around stale metric eviction. node_exporter's
``textfile_collector`` is the documented, zero-config alternative for
exactly this use case (one-shot job heartbeats) — see Prometheus docs
"When to use the textfile collector".

Atomic-write contract
---------------------

Files are written via ``tempfile.NamedTemporaryFile`` + ``os.replace``
so node_exporter never reads a half-written file. The helper is
**fail-soft**: any error (permission denied, dir doesn't exist) is
logged at WARNING level but never re-raised — losing a metric write is
much less bad than crashing the actual cron whose success we just
celebrated.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

#: Default location of node_exporter's ``--collector.textfile.directory``.
#:
#: The ``prometheus-node-exporter`` Debian/Ubuntu package (which is what
#: we run on Tokyo) ships a default of ``/var/lib/prometheus/node-exporter``
#: — note the ``prometheus`` parent dir, not ``node_exporter``. The
#: upstream Prometheus docs sample uses ``/var/lib/node_exporter/...`` —
#: confused exactly that path on first deployment (alerts fired with
#: noDataState until 2026-04-27 when this default was corrected).
#:
#: If the deployment puts it elsewhere, set ``$TEXTFILE_COLLECTOR_DIR``.
DEFAULT_TEXTFILE_DIR = "/var/lib/prometheus/node-exporter"


def textfile_dir() -> Path:
    """Resolve the textfile-collector directory from env, with default."""
    return Path(os.getenv("TEXTFILE_COLLECTOR_DIR", DEFAULT_TEXTFILE_DIR))


def _format_labels(labels: Mapping[str, str] | None) -> str:
    """Render ``{job="x", symbol="BTCUSDT"}`` for the Prometheus exposition
    format. Returns ``""`` for empty/None — caller writes a bare metric."""
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


def _escape_label_value(v: str) -> str:
    """Per Prometheus exposition spec: escape ``\\``, ``"``, and newline."""
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_cron_success(
    metric_name: str,
    *,
    file_stem: str,
    labels: Mapping[str, str] | None = None,
    now: float | None = None,
    directory: Path | None = None,
) -> bool:
    """Record a successful cron run.

    Parameters
    ----------
    metric_name
        Prometheus-style metric name (e.g.
        ``"neucast_hf_l2_archive_last_success_timestamp_seconds"``).
    file_stem
        Filename without extension; the final file will be
        ``<directory>/<file_stem>.prom``. Use distinct stems for jobs
        that have multiple instances (e.g. ``trainer_btcusdt``,
        ``trainer_ethusdt``) so concurrent writers don't clobber each
        other.
    labels
        Optional Prometheus labels — applied as ``{k="v",...}``.
    now
        Override of ``time.time()`` for tests.
    directory
        Override of the textfile-collector dir for tests.

    Returns
    -------
    bool
        ``True`` when the write completed; ``False`` on any IO error
        (already logged). Callers should NOT branch on this return —
        it's purely for the unit tests. **A cron that succeeded but
        failed to record its success is still successful.**

    Notes
    -----
    The output file format is the standard Prometheus textfile
    exposition::

        # HELP <name> Last successful run (unix seconds since epoch)
        # TYPE <name> gauge
        <name>{labels} <unix_ts>
    """
    ts = time.time() if now is None else now
    target_dir = directory or textfile_dir()
    rendered_labels = _format_labels(labels)
    body = (
        f"# HELP {metric_name} Last successful run (unix seconds since epoch)\n"
        f"# TYPE {metric_name} gauge\n"
        f"{metric_name}{rendered_labels} {ts:.0f}\n"
    )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: tempfile in same dir → os.replace.
        # node_exporter scans every 15s; if it picks the file up
        # mid-write the metric value is malformed and node_exporter
        # logs a warning — we avoid that entirely.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target_dir,
            prefix=f".{file_stem}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(body)
            tmp_path = Path(f.name)
        final_path = target_dir / f"{file_stem}.prom"
        os.replace(tmp_path, final_path)
        logger.debug(
            "cron_metrics: wrote %s -> %s (ts=%d)",
            metric_name, final_path, int(ts),
        )
        return True
    except OSError as exc:
        logger.warning(
            "cron_metrics: could not write %s to %s: %s",
            metric_name, target_dir, exc,
        )
        return False
