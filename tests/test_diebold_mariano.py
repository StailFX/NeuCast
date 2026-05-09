"""Тесты для tools.diebold_mariano.

Проверяем:
* идентичные модели → не отвергаем H₀ (p > 0.05);
* модель A систематически точнее → отвергаем H₀ в пользу A;
* HAC-оценка дисперсии корректно учитывает автокорреляцию;
* HLN-коррекция даёт более консервативный (больший) p-value на
  малых выборках.
"""
import numpy as np
import pytest

from tools.diebold_mariano import (
    _newey_west_long_run_variance,
    diebold_mariano,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_identical_models_fails_to_reject_h0(rng):
    """Две модели с идентичным распределением ошибок → DM-тест не
    должен отвергнуть H₀ (модели равно точны)."""
    n = 500
    errors = rng.normal(0, 1, n)
    # Используем те же ошибки для двух моделей — статистика DM = 0,
    # p-value должен быть близок к 1.
    out = diebold_mariano(errors, errors, horizon=1)
    assert out["winner"] == "tie"
    assert out["p_value"] > 0.95


def test_uniformly_better_model_a_rejects_h0(rng):
    """Модель A с систематически меньшей дисперсией ошибок → DM-тест
    отвергает H₀ в пользу A (winner='A', p < 0.01)."""
    n = 300
    # A — ошибки в 2 раза меньше по дисперсии чем B
    errors_a = rng.normal(0, 0.5, n)
    errors_b = rng.normal(0, 1.0, n)
    out = diebold_mariano(errors_a, errors_b, horizon=1, loss="squared")
    assert out["winner"] == "A"
    assert out["p_value"] < 0.01
    assert out["mean_diff"] < 0  # L_a < L_b на среднем


def test_uniformly_worse_model_b_loses(rng):
    """Симметричный сценарий: A явно хуже → winner='B'."""
    n = 300
    errors_a = rng.normal(0, 1.0, n)
    errors_b = rng.normal(0, 0.5, n)
    out = diebold_mariano(errors_a, errors_b, horizon=1)
    assert out["winner"] == "B"
    assert out["p_value"] < 0.01


def test_marginally_different_models_well_calibrated(rng):
    """Модели с очень близкими дисперсиями (1.00 vs 1.05) → DM-тест
    должен давать p-value не близкий к нулю на типичном случае.
    Слабое утверждение: на N независимых прогонах с разными seed
    тест НЕ должен отвергать H₀ слишком часто (false-positive rate
    при истинном equal-accuracy не превышает 0.05 — это и есть
    смысл α-уровня).
    """
    rejections = 0
    n_trials = 50
    for seed in range(n_trials):
        local_rng = np.random.default_rng(seed)
        errors_a = local_rng.normal(0, 1.0, 100)
        errors_b = local_rng.normal(0, 1.0, 100)  # Equal variance
        out = diebold_mariano(errors_a, errors_b, horizon=1)
        if out["p_value"] < 0.05:
            rejections += 1
    # Под H₀ ожидаем ~ 5 % отвержений; допустим до 15 % для устойчивости.
    assert rejections / n_trials < 0.15, (
        f"too many false rejections: {rejections}/{n_trials} = "
        f"{rejections / n_trials:.1%}"
    )


def test_absolute_loss_works(rng):
    """Абсолютная функция потерь должна тоже корректно отрабатывать."""
    n = 200
    errors_a = rng.normal(0, 0.5, n)
    errors_b = rng.normal(0, 1.0, n)
    out = diebold_mariano(errors_a, errors_b, horizon=1, loss="absolute")
    assert out["winner"] == "A"
    assert out["loss_kind"] == "absolute"


def test_hln_correction_is_more_conservative(rng):
    """На умеренной выборке коррекция Харви-Лейборна-Ньюбольда даёт
    более консервативный (больший) p-value, чем асимптотическая
    нормаль."""
    n = 50  # маленькая выборка — где разница ощутима
    errors_a = rng.normal(0, 0.7, n)
    errors_b = rng.normal(0, 1.0, n)
    out_with = diebold_mariano(errors_a, errors_b, horizon=1, use_hln_correction=True)
    out_without = diebold_mariano(errors_a, errors_b, horizon=1, use_hln_correction=False)
    # HLN-вариант должен давать p_value не меньше асимптотического
    # (на малых выборках обычно строго больше).
    assert out_with["p_value"] >= out_without["p_value"] - 1e-6


def test_newey_west_zero_lag_equals_variance(rng):
    """При max_lag=0 Newey-West оценка должна совпадать с обычной
    выборочной дисперсией."""
    n = 200
    d = rng.normal(0, 1, n)
    nw = _newey_west_long_run_variance(d, max_lag=0)
    sample_var = float(np.var(d))
    assert abs(nw - sample_var) < 1e-9


def test_newey_west_handles_autocorrelated_series():
    """На AR(1)-ряде Newey-West должна давать БОЛЬШУЮ оценку
    долгосрочной дисперсии, чем простая выборочная (т.к. есть
    положительная автокорреляция)."""
    rng = np.random.default_rng(0)
    n = 500
    eps = rng.normal(0, 1, n)
    # AR(1) с коэффициентом 0.7
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.7 * d[t - 1] + eps[t]
    sample_var = float(np.var(d))
    nw_lag10 = _newey_west_long_run_variance(d, max_lag=10)
    # Долгосрочная дисперсия для AR(1) с ρ=0.7: σ²_eps / (1 - ρ)² ≈ σ²_x · (1 + ρ) / (1 - ρ)
    # ≈ 5.67 × σ²_eps. Newey-West должна это улавливать.
    assert nw_lag10 > sample_var


def test_raises_on_unequal_length():
    a = np.zeros(10)
    b = np.zeros(11)
    with pytest.raises(ValueError, match="same length"):
        diebold_mariano(a, b, horizon=1)


def test_raises_on_too_small_sample():
    a = np.zeros(5)
    b = np.zeros(5)
    with pytest.raises(ValueError, match="too small"):
        diebold_mariano(a, b, horizon=1)
