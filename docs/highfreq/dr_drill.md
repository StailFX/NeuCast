# Disaster Recovery Drill — HF Archive Read-Path

Purpose: prove the read path of the three HF S3 archives actually
works **before** we need it to.

| Source table | Archive script | S3 prefix | Retention in hot |
|---|---|---|---|
| `highfreq_l2_snapshots` | `tools/archive_l2_to_s3.py` | `highfreq_l2/<sym>/<day>.parquet` | **2 days** (post-2026-04-27) |
| `highfreq_ofi_1s` | `tools/archive_ofi_1s_to_s3.py` | `highfreq_ofi_1s/<sym>/<day>.parquet` | 7 days |
| `paper_trades` | `tools/archive_paper_trades_to_s3.py` | `paper_trades_backup/<sym>/<day>.parquet` | ∞ (backup-only, no delete) |

The original DR drill tooling (`tools/dr_drill_l2.py`) only covers
the **L2** archive. The other two follow identical patterns —
DR-drilling them is a copy of `dr_drill_l2.py` swapping the prefix
and column-set; tracked as a follow-up item below.

## Why this exists

> *A backup that has never been restored is not a backup.*

Hot-cold storage on the HF stack is split into:

* **Hot** — `highfreq_l2_snapshots` in Postgres on Tokyo. 7-day
  retention. Used by `app/highfreq/web.py` for the live orderbook
  heatmap.
* **Cold** — `s3://<bucket>/highfreq_l2/<symbol>/<YYYY-MM-DD>.parquet`
  on Yandex Object Storage. Forever. Written by the daily cron
  `neucast-l2-archive.timer`, then deleted from hot.

Once a row leaves hot storage, S3 is the **only** copy. If a Parquet
encoder regression, a schema migration, or a credential rotation
silently breaks restoration, we won't find out until an incident — at
which point the data is already gone from Postgres.

The DR drill exercises the read-back path end-to-end against real S3
without touching production data. It's read-only by default; with
`--restore-into-table` it also INSERTs into a separate test table to
prove the DB schema is compatible.

## What it validates

`tools/dr_drill_l2.py` performs, per (symbol, day):

1. **S3 GET** — downloads `highfreq_l2/<symbol>/<day>.parquet`. A 404
   here means the archive cron didn't actually upload that day; check
   `journalctl -u neucast-l2-archive.service`.
2. **Parquet decode** — pyarrow round-trip into a pandas DataFrame.
   Catches Parquet codec mismatches (e.g. snappy missing on the
   recovery host).
3. **Schema validation** — column set must equal the writer's
   `SELECT`-list:

   ```
   ts · symbol · bids_price · bids_qty · asks_price · asks_qty · written_at
   ```

   Missing columns → fatal (`overall_ok=false`). Extra columns →
   warning only (deliberate widening is OK).
4. **Array-column shape check** — `bids_price` etc. must be list-like
   on every row. The most insidious silent corruption mode for L2
   data is array columns getting collapsed to their first element by
   a buggy serialiser; this check catches that.
5. **Time-range sanity** — earliest / latest `ts` from the file must
   land inside the requested day in UTC.
6. **Optional restore** (`--restore-into-table`) — INSERTs every row
   into `highfreq_l2_snapshots_dr_test` (auto-created mirror schema),
   reports rows inserted vs rows in Parquet. **MISMATCH** in the
   `notes` field is a defect.

## How to run

On Tokyo, with the same env file as the archive cron
(`/etc/default/neucast-l2-archive`):

```bash
cd /opt/neucast
sudo -u neucast bash -lc '
  set -a
  source /etc/default/neucast-l2-archive
  set +a
  python -m tools.dr_drill_l2 --day "$(date -u -d yesterday +%F)"
'
```

Read-only validation completes in <30 s for one day × 3 symbols
(typical archive size: ~3 MB Parquet per symbol-day).

For the **full** drill (also writes to a test table — adds ~5 s):

```bash
python -m tools.dr_drill_l2 --day 2026-04-20 --restore-into-table
```

Single-symbol drill (faster iteration during debugging):

```bash
python -m tools.dr_drill_l2 --symbol BTCUSDT --day 2026-04-20
```

## Reading the output

The script emits a single JSON object on stdout. Pipe through `jq` for
human-readable form, archive the raw output as evidence:

```bash
python -m tools.dr_drill_l2 --day 2026-04-20 \
  | tee docs/highfreq/dr_drill_runs/$(date -u +%F).json \
  | jq '.overall_ok, (.results[] | {symbol, rows_in_parquet, schema_ok})'
```

A clean run looks like:

```json
{
  "drill_run_at": "2026-04-27T08:00:00+00:00",
  "drill_day": "2026-04-20",
  "bucket": "neucast-hf-cold",
  "endpoint": "https://storage.yandexcloud.net",
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "restore_into_table": false,
  "overall_ok": true,
  "results": [
    {
      "symbol": "BTCUSDT",
      "day": "2026-04-20",
      "s3_key": "highfreq_l2/btcusdt/2026-04-20.parquet",
      "s3_object_size_bytes": 2_945_117,
      "rows_in_parquet": 8640,
      "columns": ["asks_price","asks_qty","bids_price","bids_qty","symbol","ts","written_at"],
      "earliest_ts": "2026-04-20T00:00:00+00:00",
      "latest_ts": "2026-04-20T23:59:59+00:00",
      "schema_ok": true,
      "schema_diff": [],
      "rows_restored": null,
      "notes": []
    }
  ]
}
```

`overall_ok=true` + `schema_ok=true` for every result + `rows_in_parquet`
matching the archive log's `rows_uploaded` for that day = drill PASS.

## Frequency

| Trigger | Action |
|---|---|
| Before any production milestone | Full drill (`--restore-into-table`) on yesterday + last week's day |
| Quarterly | Full drill on a randomly-chosen day from the past 30 days |
| After any change to `archive_l2_to_s3.py` | Read-only drill on yesterday |
| After any change to `highfreq_l2_snapshots` schema | Full drill (`--restore-into-table`) |
| After a credential rotation on Yandex S3 | Read-only drill (proves new keys work) |

## Recording drill runs

Per drill, archive the output JSON in `docs/highfreq/dr_drill_runs/`:

```
docs/highfreq/dr_drill_runs/
    2026-04-27.json   ← read-only, yesterday
    2026-04-28.json   ← full restore, last week's day
    ...
```

The DR-drill output is **the** evidence shown if a reviewer asks
"have you actually tested your backups?". Keep these committed.

## What a failure looks like

```json
"results": [
  {
    "symbol": "BTCUSDT",
    "day": "2026-04-20",
    "schema_ok": false,
    "schema_diff": ["missing columns: ['written_at']"],
    "notes": []
  }
]
```

Triage steps:

1. **schema_ok=false with missing columns** — recent schema change
   that wasn't applied to the archive cron's SELECT. Update
   `fetch_day_dataframe` in `archive_l2_to_s3.py`. Old archives are
   recoverable via column-defaulting (write a migration that adds the
   missing column with `NULL` default, then re-archive any failing
   days from a hot replica if available).
2. **HTTP 404 from S3** — archive cron didn't fire that day, or the
   bucket / endpoint env vars drifted. Check
   `neucast_hf_l2_archive_last_success_timestamp_seconds` in Grafana
   — the new cron-stale alert (added 2026-04-27) catches this within
   25 hours.
3. **Parquet decode error** — boto3 / pyarrow / snappy version skew
   on the host. Reproduce on a clean Python env; `pip install
   pyarrow==<version-from-requirements>`.
4. **rows_restored != rows_in_parquet** — DB INSERT silently dropped
   rows. Most likely a `NOT NULL` violation that psycopg2 caught and
   continued past in a future code-path; check Postgres logs.

## Cost guard

Yandex S3 standard pricing: ~₽1.30 per 10k GET requests. A read-only
drill of 3 symbols × 1 day = 3 GETs (₽0.0004). A full quarterly schedule
of 4 days × 3 symbols × 4 quarters/year = 48 GETs/year ≈ free. Safe to
schedule generously.

## Coverage gaps + follow-ups

* `tools/dr_drill_ofi_1s.py` — TODO. Same shape as `dr_drill_l2.py`
  but for the OFI archive. Schema is simpler (no array columns), so
  the validation step is just "all expected scalar columns present".
* `tools/dr_drill_paper_trades.py` — TODO. Backup-only data; the
  drill should additionally verify the row count in S3 equals the
  row count in Postgres for that day (bidirectional consistency,
  unique to this archive type).
* Both follow-ups are routine; until they exist, the **manual** form
  of the drill works:

  ```bash
  # Manual OFI drill — read one day from S3 and sanity-check.
  aws --endpoint-url "$YANDEX_S3_ENDPOINT" s3 cp \
      "s3://$YANDEX_S3_BUCKET/highfreq_ofi_1s/btcusdt/2026-04-20.parquet" /tmp/
  python3 -c "import pyarrow.parquet as pq; t = pq.read_table('/tmp/2026-04-20.parquet'); print(t.num_rows, t.column_names)"
  ```
