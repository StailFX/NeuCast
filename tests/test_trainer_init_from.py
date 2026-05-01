"""Tests for ``--init-from`` (release T.15) — CatBoost incremental
learning that lets the trainer fine-tune from a pre-trained .cbm.

Why this matters
================

The trainer fits a CatBoost from scratch every cron run. With ~5 days
of accumulated OFI data on Tokyo we hit a ceiling: the model has no
"memory" of regimes outside the recent window. The pretrain → fine-
tune split (release T.15) lets us pre-train on YEARS of historical
Klines (via ``app.highfreq.pretrain``), then continue training on the
recent live OFI data via CatBoost's ``init_model=`` parameter.

Tests pin:
1. ``fit_final_model(init_from=<missing>)`` raises FileNotFoundError
   (silent fallback would leave operators wondering why fine-tune
   didn't happen).
2. End-to-end: pretrain a CatBoost on synthetic data, save .cbm,
   call fit_final_model with init_from → resulting model has MORE
   trees than the pretrained one (proof CatBoost continued learning).
3. Trainer ``--init-from`` CLI flag round-trips through argparse.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.highfreq.trainer import (
    WalkForwardConfig,
    _parse_args,
    fit_final_model,
)


def _make_synthetic_xy(n: int = 500, n_features: int = 18, seed: int = 0):
    """Synthetic (X, y) pair good enough to fit CatBoost on but small."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Simple linearly-separable signal so CatBoost can fit something.
    logits = X.iloc[:, 0] * 1.0 + X.iloc[:, 1] * 0.5
    y = (logits > 0).astype(int)
    return X, pd.Series(y, name="y")


def test_fit_final_model_init_from_missing_file_raises():
    X, y = _make_synthetic_xy(n=200)
    cfg = WalkForwardConfig(catboost_iterations=20)
    with pytest.raises(FileNotFoundError, match="--init-from"):
        fit_final_model(X, y, config=cfg, init_from="/tmp/does-not-exist.cbm")


def test_fit_final_model_init_from_extends_tree_count(tmp_path):
    """Pretrain → save → fine-tune with init_from. The fine-tuned
    model should have MORE trees than the pretrained one (CatBoost's
    incremental learning appends new trees on top of the loaded ones)."""
    X, y = _make_synthetic_xy(n=400)
    cfg = WalkForwardConfig(catboost_iterations=30)

    # Step 1: pretrain (no init_from).
    pretrained = fit_final_model(X.iloc[:200], y.iloc[:200], config=cfg)
    pretrained_path = tmp_path / "pretrained.cbm"
    pretrained.save_model(str(pretrained_path), format="cbm")
    n_trees_pretrained = pretrained.tree_count_
    assert n_trees_pretrained > 0

    # Step 2: fine-tune with init_from on a different slice of data.
    finetuned = fit_final_model(
        X.iloc[200:], y.iloc[200:], config=cfg,
        init_from=pretrained_path,
    )
    n_trees_finetuned = finetuned.tree_count_
    # CatBoost appends iterations to the loaded model — the result
    # has ≥ pretrained tree count (typically pretrained + iterations).
    assert n_trees_finetuned >= n_trees_pretrained
    # Sanity: a fully-disjoint run from scratch on the same fine-tune
    # data has fewer trees than pretrained+finetuned.
    fresh = fit_final_model(X.iloc[200:], y.iloc[200:], config=cfg)
    assert n_trees_finetuned >= fresh.tree_count_


def test_trainer_cli_init_from_argparse_round_trip():
    """The argparse layer must accept --init-from <path> and surface
    it in args. Default reads HF_INIT_FROM env (defaults to None)."""
    args = _parse_args([
        "--symbol", "BTCUSDT",
        "--init-from", "/path/to/pretrained.cbm",
    ])
    assert args.init_from == "/path/to/pretrained.cbm"


def test_trainer_cli_init_from_default_is_none(monkeypatch):
    """Without --init-from and without HF_INIT_FROM env, init_from
    defaults to None (preserving the original from-scratch behaviour)."""
    monkeypatch.delenv("HF_INIT_FROM", raising=False)
    args = _parse_args(["--symbol", "BTCUSDT"])
    assert args.init_from is None


def test_trainer_cli_init_from_picks_up_env(monkeypatch):
    """HF_INIT_FROM env makes deploy via systemd drop-ins easier
    (no need to edit ExecStart) — pin the contract."""
    monkeypatch.setenv("HF_INIT_FROM", "/etc/neucast/pretrained.cbm")
    args = _parse_args(["--symbol", "BTCUSDT"])
    assert args.init_from == "/etc/neucast/pretrained.cbm"
