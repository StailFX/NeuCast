-- NeuCast High-Frequency module — Phase D: trainer-run history
--
-- Append-only log of every training run. The on-disk metrics.json is
-- overwritten per run so we lose ALL prior context: how did dir_acc
-- evolve as the dataset grew? When did the first walk-forward fold
-- land? Which feature came up in the first run that had calibration?
-- This table answers all of that without ever overwriting.
--
-- Defence-grade utility:
--   "the model improved from dir_acc 0.51 (day 1) to 0.55 (day 14)
--    monotonically as data accumulated — here's the time-series chart."
-- That story is impossible if we only have the latest snapshot.
--
-- Apply with:
--   docker exec -i neucast-postgres psql -U neucast -d neucast \
--     < app/highfreq/migrations/004_training_runs.sql

BEGIN;

CREATE TABLE IF NOT EXISTS training_runs (
    -- Primary key: surrogate ID. (symbol, run_started_at) would also
    -- work but BIGSERIAL keeps inserts cheap and a UI's "last 100
    -- runs" query maps cleanly to ORDER BY id DESC.
    id                              BIGSERIAL PRIMARY KEY,

    symbol                          TEXT NOT NULL,

    -- When the trainer's run_training() call started — wall clock at
    -- the start of training, NOT when this row was inserted (which is
    -- ~10 s later for BTC, ~5 s for ETH/BNB). Uniformly comparable
    -- across symbols and across cadence changes.
    run_started_at                  TIMESTAMPTZ NOT NULL,
    elapsed_seconds                 DOUBLE PRECISION NOT NULL CHECK (elapsed_seconds >= 0),

    -- Data-side counts (mirror TrainingReport.n_*).
    n_seconds_loaded                INT NOT NULL CHECK (n_seconds_loaded >= 0),
    n_minutes_after_aggregation     INT NOT NULL CHECK (n_minutes_after_aggregation >= 0),
    n_minutes_after_neutral_drop    INT NOT NULL CHECK (n_minutes_after_neutral_drop >= 0),

    -- Model-evaluation outputs. Nullable because they're NaN until the
    -- walk-forward CV produces ≥ 1 fold (currently the case for all 3
    -- symbols at the time of this migration — no folds yet).
    n_folds                         INT NOT NULL CHECK (n_folds >= 0),
    dir_acc_mean                    DOUBLE PRECISION,
    dir_acc_ci_low                  DOUBLE PRECISION,
    dir_acc_ci_high                 DOUBLE PRECISION,
    dir_acc_p_value                 DOUBLE PRECISION,
    log_loss_mean                   DOUBLE PRECISION,
    base_rate                       DOUBLE PRECISION,

    -- Frozen-holdout state (introduced in release A). 0 means
    -- frozen_holdout_days was disabled; n_minutes_in_holdout NULL
    -- means the trainer didn't apply the partition (legacy run).
    frozen_holdout_days             INT NOT NULL DEFAULT 0,
    n_minutes_in_holdout            INT,

    -- Convenience: cohort flag — true when this run produced a
    -- model file that the predictor will hot-reload. Helps the UI
    -- distinguish "fitted a new model" from "no-op run for stats only".
    weights_path                    TEXT,

    -- The full TrainingReport JSON. Lets us recover any field we
    -- forgot to migrate as a column without backfilling — a
    -- semi-defensive duplication. JSONB so we can index into it
    -- if needed; cost is one extra ~2 KB per row.
    full_report                     JSONB NOT NULL,

    -- Audit timestamp (distinct from run_started_at — see comment).
    written_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Sanity: a run that loaded zero seconds shouldn't claim folds.
    CHECK (n_folds = 0 OR n_seconds_loaded > 0)
);

-- The dominant query: "show me the last N runs for this symbol".
-- Used by the /api/highfreq/training_history endpoint.
CREATE INDEX IF NOT EXISTS idx_training_runs_symbol_run_started_at
    ON training_runs (symbol, run_started_at DESC);

-- Cohort: filter to runs that actually wrote a new .cbm.
CREATE INDEX IF NOT EXISTS idx_training_runs_weights_not_null
    ON training_runs (symbol, run_started_at DESC)
    WHERE weights_path IS NOT NULL;

COMMIT;
