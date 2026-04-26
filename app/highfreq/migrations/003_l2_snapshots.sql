-- NeuCast High-Frequency module — Phase D: L2 snapshots for orderbook heatmap
--
-- Stores top-N price levels of the order book at sampled cadence
-- (1 Hz typical, configurable via HIGHFREQ_L2_SAMPLE_EVERY_N in
-- runner.py). Powers the heatmap UI on /highfreq and feeds the
-- daily Yandex S3 archival cron (see tools/archive_l2_to_s3.py).
--
-- Storage budget at defaults (1 Hz × 3 symbols × top-10):
--   ~7 KB/sec total → ~600 MB/day raw
--   Postgres arrays compressed via TOAST → ~90 MB/day actual on disk
--   With 7-day retention (cron DELETE) → steady-state ~630 MB
--
-- Apply with:
--   docker exec -i neucast-postgres psql -U neucast -d neucast \
--     < app/highfreq/migrations/003_l2_snapshots.sql

BEGIN;

CREATE TABLE IF NOT EXISTS highfreq_l2_snapshots (
    ts          TIMESTAMPTZ        NOT NULL,
    symbol      TEXT               NOT NULL,

    -- Top-N levels on each side, sorted best→worst.
    -- bids: descending price (best bid first); asks: ascending price.
    -- Both arrays have the same length per row (top_n).
    bids_price  DOUBLE PRECISION[] NOT NULL,
    bids_qty    DOUBLE PRECISION[] NOT NULL,
    asks_price  DOUBLE PRECISION[] NOT NULL,
    asks_qty    DOUBLE PRECISION[] NOT NULL,

    -- DB-side audit time (when the writer flushed). Distinct from ``ts``
    -- which is the exchange event time — useful to debug runner gaps.
    written_at  TIMESTAMPTZ        NOT NULL DEFAULT now(),

    PRIMARY KEY (ts, symbol),

    -- Sanity: bids and asks arrays must be the same length on each row.
    CHECK (array_length(bids_price, 1) = array_length(bids_qty, 1)),
    CHECK (array_length(asks_price, 1) = array_length(asks_qty, 1)),
    CHECK (array_length(bids_price, 1) = array_length(asks_price, 1))
);

-- Recent-snapshots-by-symbol query (drives the heatmap endpoint).
CREATE INDEX IF NOT EXISTS idx_highfreq_l2_snapshots_symbol_ts
    ON highfreq_l2_snapshots (symbol, ts DESC);

-- Retention cleanup query (DELETE WHERE ts < cutoff). The plain
-- (ts) index is used by the planner; declared explicitly so EXPLAIN
-- shows it during the cron's DELETE.
CREATE INDEX IF NOT EXISTS idx_highfreq_l2_snapshots_ts
    ON highfreq_l2_snapshots (ts);

COMMIT;
