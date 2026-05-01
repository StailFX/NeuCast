"""Tests for ``tools.compare_candidate_models``.

Decision rule under test
========================

A candidate fine-tuned model is greenlit for production deploy ONLY
when it strictly improves dir_acc_mean AND its 95% CI lower bound
doesn't regress beyond ``tolerance``. A "tied" verdict (no
improvement, no regression) is permitted but doesn't trigger deploy.
"keep_production" fires when EITHER metric regresses.

Pin this conservative bias loudly: silently shipping a regression
because the script said "deploy" would be worse than the script
existing at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compare_candidate_models import compare_one


def _write_metrics(path: Path, *, dir_acc: float, ci_low: float,
                   feature_set: str = "microstructure",
                   n_minutes: int = 3000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dir_acc_mean": dir_acc,
        "dir_acc_ci_low": ci_low,
        "feature_set": feature_set,
        "n_minutes_after_neutral_drop": n_minutes,
    }))


def test_compare_deploy_when_candidate_strictly_better(tmp_path):
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "btcusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    _write_metrics(cand / "btcusdt_1m_metrics.json", dir_acc=0.58, ci_low=0.56)
    row = compare_one("BTCUSDT", production_dir=prod, candidate_dir=cand)
    assert row.verdict == "deploy"
    assert row.delta == pytest.approx(0.03, abs=1e-9)


def test_compare_keep_production_when_dir_acc_regresses(tmp_path):
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "ethusdt_1m_metrics.json", dir_acc=0.56, ci_low=0.54)
    # Candidate regresses -1pp on dir_acc → must NOT deploy.
    _write_metrics(cand / "ethusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    row = compare_one(
        "ETHUSDT", production_dir=prod, candidate_dir=cand, tolerance=0.005,
    )
    assert row.verdict == "keep_production"
    assert "regressed" in " ".join(row.notes)


def test_compare_keep_production_when_ci_lower_regresses(tmp_path):
    """Even if mean dir_acc nudges up, a meaningful CI-lower regression
    means the candidate is more uncertain — refuse deploy."""
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "bnbusdt_1m_metrics.json", dir_acc=0.56, ci_low=0.55)
    # Candidate: dir_acc +0.001, but CI lower drops 2pp.
    _write_metrics(cand / "bnbusdt_1m_metrics.json", dir_acc=0.561, ci_low=0.53)
    row = compare_one(
        "BNBUSDT", production_dir=prod, candidate_dir=cand, tolerance=0.005,
    )
    assert row.verdict == "keep_production"
    assert any("CI lower regressed" in n for n in row.notes)


def test_compare_tied_within_tolerance(tmp_path):
    """Within ε on both metrics → 'tied' (no deploy, no regression)."""
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "btcusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    _write_metrics(cand / "btcusdt_1m_metrics.json", dir_acc=0.5495, ci_low=0.5298)
    row = compare_one(
        "BTCUSDT", production_dir=prod, candidate_dir=cand, tolerance=0.005,
    )
    assert row.verdict == "tied"


def test_compare_missing_metrics_returns_missing(tmp_path):
    """When one side's metrics.json doesn't exist, verdict='missing'
    so the operator sees there's nothing to compare yet, not a
    misleading verdict."""
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "btcusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    # cand intentionally not written
    row = compare_one("BTCUSDT", production_dir=prod, candidate_dir=cand)
    assert row.verdict == "missing"


def test_compare_records_feature_set_change(tmp_path):
    """If the candidate uses a different feature_set than production,
    the row carries both so the comparison output makes the swap
    explicit (e.g. microstructure → long_horizon under pretrain+ft)."""
    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(
        prod / "btcusdt_1m_metrics.json", dir_acc=0.56, ci_low=0.54,
        feature_set="microstructure",
    )
    _write_metrics(
        cand / "btcusdt_1m_metrics.json", dir_acc=0.59, ci_low=0.57,
        feature_set="long_horizon",
    )
    row = compare_one("BTCUSDT", production_dir=prod, candidate_dir=cand)
    assert row.feature_set_prod == "microstructure"
    assert row.feature_set_cand == "long_horizon"
    assert row.verdict == "deploy"


# ──────────────────── exit-code semantics (release T.15.e) ───────────────────


def test_main_exit_code_all_keep_returns_1(tmp_path, capsys, monkeypatch):
    """When EVERY symbol regresses (keep_production verdict), main()
    must exit 1 — the pipeline runner uses this to drive the
    "⚠️ keep production" Telegram branch.  Earlier T.15.b regression
    where this case returned 2 ("missing") got the runner to send a
    misleading "missing data" message even though comparator clearly
    decided to keep production."""
    from tools.compare_candidate_models import main

    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    for sym in ("btcusdt", "ethusdt", "bnbusdt"):
        _write_metrics(
            prod / f"{sym}_1m_metrics.json", dir_acc=0.56, ci_low=0.54,
        )
        # Each candidate regresses by 2pp on dir_acc.
        _write_metrics(
            cand / f"{sym}_1m_metrics.json", dir_acc=0.54, ci_low=0.52,
        )

    rc = main([
        "--production-dir", str(prod),
        "--candidate-dir", str(cand),
        "--tolerance", "0.005",
    ])
    assert rc == 1, "all-keep should map to exit 1, not 2"


def test_main_exit_code_at_least_one_deploy_returns_0(tmp_path):
    """Exit 0 when at least one symbol genuinely improves AND no
    other symbol regresses — pipeline runner uses this to drive
    the "✅ deploy" branch."""
    from tools.compare_candidate_models import main

    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    # BTC improves, ETH ties.
    _write_metrics(prod / "btcusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    _write_metrics(cand / "btcusdt_1m_metrics.json", dir_acc=0.58, ci_low=0.56)
    _write_metrics(prod / "ethusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    _write_metrics(cand / "ethusdt_1m_metrics.json", dir_acc=0.5495, ci_low=0.5298)

    rc = main([
        "--production-dir", str(prod),
        "--candidate-dir", str(cand),
        "--symbol", "BTCUSDT", "--symbol", "ETHUSDT",
        "--tolerance", "0.005",
    ])
    assert rc == 0


def test_main_exit_code_all_missing_returns_2(tmp_path):
    """Exit 2 only when nothing to compare (operator forgot to write
    metrics or pointed at wrong dir) — pipeline runner uses this
    to send "❓ missing data" Telegram branch."""
    from tools.compare_candidate_models import main

    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    # No metric files written anywhere.
    rc = main([
        "--production-dir", str(prod),
        "--candidate-dir", str(cand),
        "--symbol", "BTCUSDT",
    ])
    assert rc == 2


def test_main_exit_code_one_keep_one_deploy_prefers_keep(tmp_path):
    """Mixed: BTC deploy, ETH keep_production. Conservative rule:
    if ANY symbol regresses, the whole batch goes to keep_production
    (you don't want to ship a partial deploy that saves BTC and tanks
    ETH on the same day). Exit 1."""
    from tools.compare_candidate_models import main

    prod = tmp_path / "prod"
    cand = tmp_path / "cand"
    _write_metrics(prod / "btcusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)
    _write_metrics(cand / "btcusdt_1m_metrics.json", dir_acc=0.58, ci_low=0.56)  # deploy
    _write_metrics(prod / "ethusdt_1m_metrics.json", dir_acc=0.56, ci_low=0.54)
    _write_metrics(cand / "ethusdt_1m_metrics.json", dir_acc=0.55, ci_low=0.53)  # keep
    rc = main([
        "--production-dir", str(prod),
        "--candidate-dir", str(cand),
        "--symbol", "BTCUSDT", "--symbol", "ETHUSDT",
        "--tolerance", "0.005",
    ])
    assert rc == 1, "any keep_production must dominate exit code"
