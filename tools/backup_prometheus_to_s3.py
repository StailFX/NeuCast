"""Daily snapshot of Prometheus TSDB → tar.gz → upload to Yandex S3.

Closes the «hybrid» promise: local Prometheus stores 30 days of metrics
fast, daily backup ships a compressed snapshot to s3 so historical
metrics survive a Tokyo crash. Recovery: download last snapshot, untar
into /var/lib/prometheus/metrics2/, restart Prometheus.

Run via systemd timer (deploy/neucast-prom-backup.{service,timer}).
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as _urlreq

logger = logging.getLogger(__name__)

PROM_API = "http://127.0.0.1:9099/api/v1/admin/tsdb/snapshot"
PROM_SNAPSHOTS_DIR = Path("/var/lib/prometheus/metrics2/snapshots")


def trigger_snapshot() -> str:
    """Call Prometheus admin API to create a TSDB snapshot. Returns the
    snapshot directory name (relative to PROM_SNAPSHOTS_DIR)."""
    req = _urlreq.Request(PROM_API, method="POST")
    with _urlreq.urlopen(req, timeout=30) as resp:
        body = resp.read()
    import json
    payload = json.loads(body)
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus snapshot API: {payload}")
    return payload["data"]["name"]


def tar_gz_directory(directory: Path) -> bytes:
    """Stream the snapshot directory into a gzipped tar in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    return buf.getvalue()


def upload_to_s3(*, body: bytes, bucket: str, key: str,
                 endpoint: str, region: str, ak: str, sk: str) -> None:
    import boto3  # noqa: WPS433 — CLI-only dep
    s3 = boto3.client(
        "s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=ak, aws_secret_access_key=sk,
    )
    s3.put_object(
        Bucket=bucket, Key=key, Body=body,
        ContentType="application/gzip",
        Metadata={
            "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": "tokyo-prometheus-snapshot",
            "size_bytes": str(len(body)),
        },
    )


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bucket = os.environ["YANDEX_S3_BUCKET"]
    endpoint = os.environ["YANDEX_S3_ENDPOINT"]
    region = os.environ.get("YANDEX_S3_REGION", "ru-central1")
    ak = os.environ["YANDEX_S3_ACCESS_KEY_ID"]
    sk = os.environ["YANDEX_S3_SECRET_ACCESS_KEY"]

    logger.info("triggering Prometheus TSDB snapshot via admin API")
    t0 = time.time()
    snap_name = trigger_snapshot()
    snap_dir = PROM_SNAPSHOTS_DIR / snap_name
    if not snap_dir.exists():
        logger.error("snapshot directory not found: %s", snap_dir)
        return 2
    logger.info("snapshot %s created in %.1fs", snap_name, time.time() - t0)

    logger.info("creating tar.gz of %s", snap_dir)
    t1 = time.time()
    body = tar_gz_directory(snap_dir)
    logger.info("tar.gz built: %.1f MB in %.1fs", len(body) / 1024 / 1024, time.time() - t1)

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    key = f"prometheus_snapshots/{today}_{snap_name}.tar.gz"
    logger.info("uploading to s3://%s/%s", bucket, key)
    t2 = time.time()
    upload_to_s3(
        body=body, bucket=bucket, key=key,
        endpoint=endpoint, region=region, ak=ak, sk=sk,
    )
    logger.info(
        "uploaded in %.1fs (total elapsed %.1fs)",
        time.time() - t2, time.time() - t0,
    )

    # Local cleanup: remove the snapshot dir (Prometheus keeps them
    # otherwise — they accumulate disk).
    import shutil
    shutil.rmtree(snap_dir, ignore_errors=True)
    logger.info("local snapshot dir removed: %s", snap_dir)

    # Heartbeat: write success timestamp to node_exporter
    # textfile_collector so a stuck cron triggers the
    # "Prom backup stale" alert (see alerts.yaml). Fail-soft —
    # if the textfile dir is missing on a fresh deploy, the cron
    # still counts as having succeeded.
    from app.highfreq.cron_metrics import write_cron_success
    write_cron_success(
        "neucast_hf_prom_backup_last_success_timestamp_seconds",
        file_stem="neucast_hf_prom_backup",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
