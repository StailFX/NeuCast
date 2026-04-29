-- NeuCast High-Frequency module — release S: USDM Futures venue
--
-- Adds a parallel data plane for Binance USDM Perpetual Futures
-- (BTCUSDT.P / ETHUSDT.P / BNBUSDT.P).  The spot path stays
-- untouched; this is a true parallel ingest, not a venue column on
-- the existing table.
--
-- Why a separate table
-- --------------------
-- 1. Futures has additional fields the spot table doesn't have
--    (funding_rate, mark_price). Adding nullable columns to the
--    primary table widens the row footprint for spot too — and keeps
--    schema reasoning awkward ("when is funding_rate NULL?").
-- 2. Cleaner blast radius: dropping / altering futures while iterating
--    can't affect the spot ingest that's been running for weeks.
-- 3. Different fee structure (4 bp roundtrip futures-maker vs 15 bp
--    spot-maker) means the trainer / paper-trader outputs will be in
--    parallel weights paths anyway — keeping the source data parallel
--    aligns with that.
--
-- Why USDM (USDT-margined) and not COIN-M
-- ---------------------------------------
-- USDM is collateralised in USDT; PNL is in USDT terms, matching the
-- spot side's reporting currency. COIN-M (BTC-margined) has different
-- mark-price mechanics and would complicate cross-venue P&L
-- comparison. We can add COIN-M later if research suggests it.
--
-- Apply with:
--   docker exec -i <postgres-container> psql -U neucast -d neucast \
--     < app/highfreq/migrations/006_futures_ofi.sql

BEGIN;

CREATE TABLE IF NOT EXISTS highfreq_futures_ofi_1s (
    ts             TIMESTAMPTZ      NOT NULL,
    symbol         TEXT             NOT NULL,
    -- Same OFI / microprice fields as the spot table.
    ofi            DOUBLE PRECISION,
    microprice     DOUBLE PRECISION,
    depth_imb      DOUBLE PRECISION,
    spread_bps     DOUBLE PRECISION,
    trade_imb      DOUBLE PRECISION,
    vpin           DOUBLE PRECISION,
    n_updates      INTEGER,
    local_recv_ms  INTEGER,
    -- USDM-specific: mark price (for liquidation calc, also feature
    -- input — diff between mid and mark indicates premium/discount).
    -- Sourced from the @markPrice@1s stream.
    mark_price     DOUBLE PRECISION,
    -- USDM-specific: predicted next funding rate (in fraction-of-1,
    -- e.g. 0.0001 = 0.01% / 8h). Updates every 1 s on USDM. Used as
    -- a feature ("longs paying shorts → bearish lean") and as a cost
    -- input when the paper trader holds a position over funding.
    funding_rate   DOUBLE PRECISION,
    -- Index hint of next funding settlement (UTC ms). Lets the
    -- trader pre-compute funding-cost expectation when an open
    -- position will straddle the timestamp.
    next_funding_ms BIGINT,
    PRIMARY KEY (ts, symbol)
);

-- Same indexing convention as spot — symbol+ts DESC for "give me last
-- N hours of BTCUSDT.P" queries that the trainer fires.
CREATE INDEX IF NOT EXISTS idx_highfreq_futures_ofi_1s_symbol_ts
    ON highfreq_futures_ofi_1s (symbol, ts DESC);

-- Document the symbol convention. We keep `BTCUSDT` (not `BTCUSDT.P`)
-- in the column for parity with spot — venue is implicit by table.
COMMENT ON TABLE highfreq_futures_ofi_1s IS
    'USDM Perpetual Futures L2 / OFI features at 1-second granularity.
     Symbol values match the spot table (BTCUSDT/ETHUSDT/BNBUSDT) —
     the venue is implicit by table name.  Funding rate stored as
     fraction-of-1 (0.0001 = 1 bp per 8h).';

COMMIT;
