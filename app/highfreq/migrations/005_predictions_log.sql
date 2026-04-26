-- NeuCast High-Frequency module — Phase D: predictions_log
--
-- Append-only log of every minute's model output. Until now the
-- predictor scored each closed minute and forwarded the result
-- straight to the paper-trader; if the trader didn't open (calibration
-- gate, halt, neutral signal) the prediction was forgotten. With this
-- table we keep ALL of them — including ones the trader skipped — so:
--
--   * the UI can render a "live signal tape" showing the last 30-60
--     minutes of forecasts even when no paper trade fired;
--   * the Telegram signal-flip bot can replay the most recent N
--     predictions on subscriber start;
--   * realized accuracy (after the next minute lands) can be backfilled
--     against EVERY prediction, not just the ones the trader acted on
--     — separates "model skill" from "trader skill".
--
-- Apply with:
--   docker exec -i neucast-postgres psql -U neucast -d neucast \
--     < app/highfreq/migrations/005_predictions_log.sql

BEGIN;

CREATE TABLE IF NOT EXISTS predictions_log (
    id              BIGSERIAL PRIMARY KEY,

    -- Bar-close timestamp (NOT wall-clock at INSERT — those differ by
    -- ~50 ms because the runner reads the bar then logs). Aligned to
    -- minute boundary by the runner.
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,

    -- Predictor output: P(up) ∈ [0, 1]. The discretised signal
    -- (`up` / `down` / `neutral`) is derived from prob_up via the
    -- runner's threshold logic; we store BOTH so the UI can render
    -- the arrow without re-deriving and so a future threshold change
    -- (e.g. tighter neutral band) doesn't require migration of old rows.
    prob_up         DOUBLE PRECISION NOT NULL
                      CHECK (prob_up >= 0.0 AND prob_up <= 1.0),
    signal          TEXT NOT NULL CHECK (signal IN ('up', 'down', 'neutral')),

    -- The minute's microprice_close — what the predictor "saw" as
    -- the latest price. Used by the UI tape (price next to signal)
    -- and by realized-accuracy backfill (compare against next bar's
    -- microprice).
    microprice      DOUBLE PRECISION NOT NULL CHECK (microprice > 0),

    -- Tags the prediction with the model that produced it. Same
    -- convention as paper_trades.model_version. Predictions with
    -- 'pre-calibration-demo' are still logged but their realized
    -- accuracy must be reported separately.
    model_version   TEXT NOT NULL,

    -- Backfilled by a small follow-up job (NOT yet implemented):
    --   ts+1m's microprice_close, and `realized_correct` =
    --   (signal == 'up' AND realized > microprice) OR
    --   (signal == 'down' AND realized < microprice).
    -- NULL until backfilled. The UI shows ✓/✗/? based on this.
    realized_microprice_1m DOUBLE PRECISION,
    realized_correct       BOOLEAN,

    written_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One prediction per (symbol, minute). If the runner restarts
    -- and re-processes the same bar, the second INSERT must be
    -- a no-op rather than a duplicate row.
    UNIQUE (ts, symbol)
);

-- The dominant query: "show me the last N predictions for this symbol".
CREATE INDEX IF NOT EXISTS idx_predictions_log_symbol_ts
    ON predictions_log (symbol, ts DESC);

-- For the realized-backfill job, find rows missing the realized info.
CREATE INDEX IF NOT EXISTS idx_predictions_log_unbackfilled
    ON predictions_log (ts)
    WHERE realized_microprice_1m IS NULL;

COMMIT;
