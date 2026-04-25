-- NeuCast High-Frequency module — Phase A initial schema
--
-- Tables:
--   highfreq_ofi_1s        — 1-second OFI / microprice / depth features
--   highfreq_features_1m   — 1-minute aggregates + label (sign(return_1m))
--
-- Schema rationale: see docs/highfreq/architecture.md §6.
-- Apply with:
--   docker exec -i <postgres-container> psql -U neucast -d neucast \
--     < app/highfreq/migrations/001_initial_schema.sql

BEGIN;

CREATE TABLE IF NOT EXISTS highfreq_ofi_1s (
    ts            TIMESTAMPTZ      NOT NULL,
    symbol        TEXT             NOT NULL,
    ofi           DOUBLE PRECISION,
    microprice    DOUBLE PRECISION,
    depth_imb     DOUBLE PRECISION,
    spread_bps    DOUBLE PRECISION,
    trade_imb     DOUBLE PRECISION,
    vpin          DOUBLE PRECISION,
    n_updates     INTEGER,
    local_recv_ms INTEGER,
    PRIMARY KEY (ts, symbol)
);

CREATE INDEX IF NOT EXISTS idx_highfreq_ofi_1s_symbol_ts
    ON highfreq_ofi_1s (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS highfreq_features_1m (
    ts                TIMESTAMPTZ      NOT NULL,
    symbol            TEXT             NOT NULL,
    ofi_mean          DOUBLE PRECISION,
    ofi_sum           DOUBLE PRECISION,
    ofi_std           DOUBLE PRECISION,
    microprice_open   DOUBLE PRECISION,
    microprice_close  DOUBLE PRECISION,
    microprice_mean   DOUBLE PRECISION,
    depth_imb_mean    DOUBLE PRECISION,
    spread_bps_mean   DOUBLE PRECISION,
    trade_imb_sum     DOUBLE PRECISION,
    vpin_mean         DOUBLE PRECISION,
    -- target columns, populated after the next minute completes:
    return_1m         DOUBLE PRECISION,
    direction         SMALLINT,           -- sign(return_1m): -1, 0, +1
    PRIMARY KEY (ts, symbol)
);

CREATE INDEX IF NOT EXISTS idx_highfreq_features_1m_symbol_ts
    ON highfreq_features_1m (symbol, ts DESC);

COMMIT;
