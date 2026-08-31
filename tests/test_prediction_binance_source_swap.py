"""Tests for ``fetch_and_preprocess`` source-swap (release T.16).

Crypto tickers (BTC-USD / ETH-USD / etc.) now route through Binance
Spot first, falling back to yfinance only if Binance is unreachable.
Stocks/indices (AAPL / ^GSPC) keep using yfinance.

Tests pin:
1. Crypto ticker triggers Binance fetcher.
2. Stock ticker still uses yfinance.
3. Binance returning empty falls back to yfinance.
4. Cache hit short-circuits both paths.
5. Empty Binance + empty yfinance raises ValueError (no silent
   missing-data).
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# These tests exercise data-source selection and preprocessing only.  Provide
# a tiny TensorFlow surface so importing ``app.prediction`` does not pull the
# heavyweight training runtime into the unit-test job.
_tensorflow = ModuleType("tensorflow")
_tensorflow.config = SimpleNamespace(
    threading=SimpleNamespace(
        set_inter_op_parallelism_threads=lambda *_: None,
        set_intra_op_parallelism_threads=lambda *_: None,
    ),
)
_keras = ModuleType("tensorflow.keras")
_callbacks = ModuleType("tensorflow.keras.callbacks")
_models = ModuleType("tensorflow.keras.models")


class _Callback:
    pass


_callbacks.Callback = _Callback
_callbacks.EarlyStopping = _Callback
_callbacks.ReduceLROnPlateau = _Callback
_models.load_model = lambda *args, **kwargs: None
_models.clone_model = lambda model: model
_keras.callbacks = _callbacks
_keras.models = _models
_tensorflow.keras = _keras

sys.modules.setdefault("tensorflow", _tensorflow)
sys.modules.setdefault("tensorflow.keras", _keras)
sys.modules.setdefault("tensorflow.keras.callbacks", _callbacks)
sys.modules.setdefault("tensorflow.keras.models", _models)

_layers = ModuleType("app.layers")
_layers.CUSTOM_OBJECTS = {}
sys.modules.setdefault("app.layers", _layers)


def _make_ohlcv_df(n: int = 100, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    closes = 50000 + rng.normal(0, 500, size=n).cumsum()
    df = pd.DataFrame({
        "Open": closes,
        "High": closes * 1.005,
        "Low": closes * 0.995,
        "Close": closes,
        "Volume": rng.uniform(100, 200, size=n),
    }, index=idx)
    return df


def _make_naive_ohlcv_df(n: int = 100, start: str = "2025-01-01") -> pd.DataFrame:
    """yfinance-style: tz-naive index."""
    df = _make_ohlcv_df(n, start)
    df.index = df.index.tz_localize(None)
    return df


def test_crypto_ticker_uses_binance_first():
    """BTC-USD should hit Binance, not yfinance."""
    binance_df = _make_ohlcv_df(n=100)
    with patch("app.crypto_data.fetch_crypto_ohlcv") as mock_binance, \
         patch("yfinance.download") as mock_yf, \
         patch("app.prediction._fetch_macro_features") as mock_macro, \
         patch("app.prediction.preprocess") as mock_preprocess, \
         patch("app.prediction._cache_get", return_value=None), \
         patch("app.prediction._cache_set"):
        mock_binance.return_value = (binance_df, "binance")
        mock_macro.return_value = pd.DataFrame()
        # preprocess returns whatever — we just check upstream dispatch.
        mock_preprocess.return_value = pd.DataFrame({"x": [1]})

        from app.prediction import fetch_and_preprocess
        fetch_and_preprocess("BTC-USD", "2025-01-01", "2025-04-01")

        # Binance was called.
        assert mock_binance.called
        kwargs = mock_binance.call_args.kwargs
        assert kwargs["prefer"] == "binance"
        assert kwargs["interval"] == "1d"
        # yfinance MUST NOT have been called for the target ticker
        # (macro fetch uses _fetch_macro_features which is mocked).
        assert not mock_yf.called


def test_stock_ticker_still_uses_yfinance():
    """AAPL should NOT touch the Binance path."""
    yf_df = _make_naive_ohlcv_df(n=80)
    with patch("app.crypto_data.fetch_crypto_ohlcv") as mock_binance, \
         patch("yfinance.download") as mock_yf, \
         patch("app.prediction._fetch_macro_features") as mock_macro, \
         patch("app.prediction.preprocess") as mock_preprocess, \
         patch("app.prediction._cache_get", return_value=None), \
         patch("app.prediction._cache_set"):
        mock_yf.return_value = yf_df
        mock_macro.return_value = pd.DataFrame()
        mock_preprocess.return_value = pd.DataFrame({"x": [1]})

        from app.prediction import fetch_and_preprocess
        fetch_and_preprocess("AAPL", "2025-01-01", "2025-04-01")

        assert mock_yf.called, "AAPL should hit yfinance"
        assert not mock_binance.called, "AAPL must NOT hit Binance"


def test_binance_empty_falls_through_to_yfinance():
    """If Binance returns empty (geo-block, network fail), the function
    auto-falls-through to yfinance — preserves the original behaviour
    so we never regress on availability."""
    yf_df = _make_naive_ohlcv_df(n=60)
    with patch("app.crypto_data.fetch_crypto_ohlcv") as mock_binance, \
         patch("yfinance.download") as mock_yf, \
         patch("app.prediction._fetch_macro_features") as mock_macro, \
         patch("app.prediction.preprocess") as mock_preprocess, \
         patch("app.prediction._cache_get", return_value=None), \
         patch("app.prediction._cache_set"):
        # Binance returns empty (None df).
        mock_binance.return_value = (None, "fail")
        mock_yf.return_value = yf_df
        mock_macro.return_value = pd.DataFrame()
        mock_preprocess.return_value = pd.DataFrame({"x": [1]})

        from app.prediction import fetch_and_preprocess
        fetch_and_preprocess("ETH-USD", "2025-01-01", "2025-03-01")

        # Both were tried, in the right order.
        assert mock_binance.called
        assert mock_yf.called


def test_both_sources_empty_raises_value_error():
    """When Binance AND yfinance both come back empty, we want a
    loud ValueError — silent empty would let downstream model train
    on nothing."""
    with patch("app.crypto_data.fetch_crypto_ohlcv") as mock_binance, \
         patch("yfinance.download") as mock_yf, \
         patch("app.prediction._cache_get", return_value=None):
        mock_binance.return_value = (None, "fail")
        mock_yf.return_value = pd.DataFrame()  # also empty

        from app.prediction import fetch_and_preprocess
        with pytest.raises(ValueError, match="Нет данных"):
            fetch_and_preprocess("BTC-USD", "2025-01-01", "2025-04-01")


def test_cache_hit_short_circuits_both_paths():
    """A hit on the prediction-cache means neither Binance nor
    yfinance is called — the whole point of the cache."""
    cached_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
    with patch("app.prediction._cache_get", return_value=cached_df) as mock_cache, \
         patch("app.crypto_data.fetch_crypto_ohlcv") as mock_binance, \
         patch("yfinance.download") as mock_yf:
        from app.prediction import fetch_and_preprocess
        out = fetch_and_preprocess("BTC-USD", "2025-01-01", "2025-04-01")
        assert out is cached_df
        assert mock_cache.called
        assert not mock_binance.called
        assert not mock_yf.called


def test_binance_path_strips_tz_to_match_yfinance_naive_contract():
    """preprocess() expects tz-naive index (yfinance native shape).
    Binance returns tz-aware UTC — the swap must localize-to-None
    so downstream preprocessing doesn't break."""
    binance_df = _make_ohlcv_df(n=50)  # tz-aware UTC
    captured: dict = {}

    def _fake_preprocess(df_raw, macro_df):
        captured["index_tz"] = df_raw.index.tz
        return pd.DataFrame({"x": [1]})

    with patch("app.crypto_data.fetch_crypto_ohlcv", return_value=(binance_df, "binance")), \
         patch("app.prediction._fetch_macro_features", return_value=pd.DataFrame()), \
         patch("app.prediction.preprocess", side_effect=_fake_preprocess), \
         patch("app.prediction._cache_get", return_value=None), \
         patch("app.prediction._cache_set"):
        from app.prediction import fetch_and_preprocess
        fetch_and_preprocess("BTC-USD", "2025-01-01", "2025-02-15")
    assert captured.get("index_tz") is None, (
        "preprocess() must receive a tz-naive index (yfinance contract)"
    )
