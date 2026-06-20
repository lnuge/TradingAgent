"""Technical-indicator feature engineering.

Computes the feature set the strategy model is trained on:
RSI(14), MACD (line / signal / histogram), Bollinger-Band width, and a
volume z-score. Pure pandas/numpy so it has no model dependency and can be
reused by both the live signal tool and the offline training script.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "volume_zscore",
]


def bars_to_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty:
        raise ValueError("No bars provided")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return width.fillna(0.0)


def _volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mean = volume.rolling(period, min_periods=period).mean()
    std = volume.rolling(period, min_periods=period).std()
    z = (volume - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


def compute_feature_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with FEATURE_COLUMNS for every bar (NaN-safe)."""
    df = bars_to_frame(bars)
    close, volume = df["close"], df["volume"]
    macd_line, signal_line, hist = _macd(close)
    out = pd.DataFrame(
        {
            "rsi_14": _rsi(close),
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
            "bb_width": _bb_width(close),
            "volume_zscore": _volume_zscore(volume),
        }
    )
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def latest_features(bars: list[dict[str, Any]]) -> dict[str, float]:
    """Feature vector for the most recent bar."""
    frame = compute_feature_frame(bars)
    row = frame.iloc[-1]
    return {col: float(row[col]) for col in FEATURE_COLUMNS}
