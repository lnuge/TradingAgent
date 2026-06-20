"""Historical market data for model training, from a free source (yfinance).

Used by the training script to pull REAL historical bars so the model learns
genuine structure (the synthetic market is a random walk with no edge). This is
training data only — live execution still goes through the broker layer.

Note: yfinance reaches Yahoo Finance (query1/query2.finance.yahoo.com). In a
sandboxed/remote environment those hosts must be on the network egress
allowlist; otherwise run training locally.
"""
from __future__ import annotations

from typing import Any


class MarketDataError(RuntimeError):
    pass


def fetch_bars(symbol: str, period: str = "3y", interval: str = "1d") -> list[dict[str, Any]]:
    """Fetch historical OHLCV bars for a symbol in the project's bar format.

    period: e.g. '1y', '2y', '3y', '5y', 'max'. interval: '1d', '1h', etc.
    Returns [{begins_at, open, high, low, close, volume}] oldest-first.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise MarketDataError(
            "yfinance is required for real training data. Install with `pip install yfinance`."
        ) from exc

    df = yf.download(
        symbol, period=period, interval=interval, progress=False, auto_adjust=True
    )
    if df is None or len(df) == 0:
        raise MarketDataError(
            f"No data returned for {symbol}. If sandboxed, allowlist "
            "query1/query2.finance.yahoo.com or run locally."
        )

    # yfinance may return single- or multi-level columns depending on version.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    bars: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        bars.append(
            {
                "begins_at": str(ts),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if not _isnan(row["Volume"]) else 0,
            }
        )
    return bars


def _isnan(x: Any) -> bool:
    try:
        return x != x  # NaN is the only value not equal to itself
    except Exception:
        return False
