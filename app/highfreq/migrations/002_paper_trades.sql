-- NeuCast High-Frequency module — Phase C: paper_trades persistence
--
-- Schema mirrors the PaperTrade dataclass in app/highfreq/paper_trader.py.
-- One row per closed paper trade (the trader emits nothing on entry; only
-- on close does it return a PaperTrade for the runner to persist here).
--
-- ADR-005: sim-only by design. There is intentionally no `is_live` column
-- — if we ever add live execution it gets a separate table to keep
-- audit trails clean.
--
-- Apply with:
--   docker exec -i neucast-postgres psql -U neucast -d neucast \
--     < app/highfreq/migrations/002_paper_trades.sql

BEGIN;

CREATE TABLE IF NOT EXISTS paper_trades (
    -- Primary key: BIGSERIAL (not (entry_ts, symbol) like highfreq_ofi_1s)
    -- because two trades can in principle open in the same minute on
    -- different symbols, and exit_ts is also a candidate but less stable
    -- under runner restarts. id is the safe choice.
    id                  BIGSERIAL PRIMARY KEY,

    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL
                          CHECK (side IN ('long', 'short')),
    qty                 DOUBLE PRECISION NOT NULL CHECK (qty > 0),

    -- Entry leg
    entry_ts            TIMESTAMPTZ      NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL CHECK (entry_price > 0),
    entry_prob_up       DOUBLE PRECISION NOT NULL
                          CHECK (entry_prob_up >= 0.0 AND entry_prob_up <= 1.0),

    -- Exit leg
    exit_ts             TIMESTAMPTZ      NOT NULL,
    exit_price          DOUBLE PRECISION NOT NULL,
    exit_reason         TEXT             NOT NULL
                          CHECK (exit_reason IN ('time_stop', 'halt_close')),

    -- P&L (denormalised so we can SELECT without re-deriving on every read)
    fee_paid_total_usd  DOUBLE PRECISION NOT NULL CHECK (fee_paid_total_usd >= 0),
    pnl_usd             DOUBLE PRECISION NOT NULL,
    pnl_bps             DOUBLE PRECISION NOT NULL,

    -- Provenance: which model (mtime ISO of the .cbm) opened this trade.
    -- Lets us slice "P&L since trainer rolled v3" without timestamp games.
    model_version       TEXT             NOT NULL,

    -- DB-side audit timestamp (distinct from entry_ts/exit_ts, which are
    -- exchange/wall time of the simulated trade). Used for debugging
    -- "when did the runner persist this row" vs "when was the trade".
    written_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),

    -- Sanity: entry must precede exit. A bug where the runner inverted
    -- them would otherwise ship subtly wrong P&L.
    CHECK (exit_ts >= entry_ts)
);

-- Recent trades by symbol (drives /api/highfreq/paper_trades + UI block)
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol_exit_ts
    ON paper_trades (symbol, exit_ts DESC);

-- Cohort analysis by model version (e.g. "P&L since model v3 went live")
CREATE INDEX IF NOT EXISTS idx_paper_trades_model_version
    ON paper_trades (model_version);

-- Daily roll-up (used by the UI's "today's P&L" widget and the post-Tier-2
-- gate decision). date_trunc on TIMESTAMPTZ is immutable enough for an
-- expression index in Postgres ≥ 14.
CREATE INDEX IF NOT EXISTS idx_paper_trades_day_utc
    ON paper_trades ((date_trunc('day', exit_ts AT TIME ZONE 'UTC')), symbol);

COMMIT;
