"""STRATEGY LAYER — model-driven signal generation.

``generate_signal`` pulls recent bars, computes the technical features, and
runs the trained XGBoost model (or the rule-based fallback) to return
BUY/SELL/HOLD + confidence. The agent is expected to layer qualitative
reasoning (news, macro context) *on top of* this output — the tool deliberately
returns the raw feature vector and probabilities so the agent can sanity-check
or override the quantitative call, rather than following it blindly.
"""
from __future__ import annotations

from typing import Any

from .broker import get_broker
from .context import SESSION_ID
from .features import latest_features
from .model import get_model
from .storage import get_storage


def generate_signal(symbol: str, interval: str = "5minute", span: str = "day") -> dict[str, Any]:
    symbol = symbol.upper().strip()
    bars = get_broker().get_bars(symbol, interval=interval, span=span)
    if len(bars) < 30:
        raise ValueError(f"Not enough bars to compute features for {symbol} (got {len(bars)})")

    features = latest_features(bars)
    prediction = get_model().predict(features)

    result = {
        "symbol": symbol,
        "signal": prediction["signal"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "model_source": prediction["source"],
        "features": features,
        "interpretation": _interpret(features, prediction),
        "agent_guidance": (
            "This is the QUANTITATIVE model output. Layer qualitative reasoning "
            "(news, earnings, macro) on top before acting. You must still call "
            "check_risk before any place_order, regardless of this signal."
        ),
    }
    get_storage().append_audit(
        SESSION_ID,
        "strategy.generate_signal",
        {"signal": result["signal"], "confidence": result["confidence"], "source": prediction["source"]},
        symbol,
    )
    return result


def _interpret(features: dict[str, float], prediction: dict[str, Any]) -> str:
    rsi = features["rsi_14"]
    rsi_note = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    macd_note = "bullish" if features["macd_hist"] > 0 else "bearish"
    return (
        f"RSI {rsi:.1f} ({rsi_note}); MACD histogram {macd_note}; "
        f"BB width {features['bb_width']:.4f}; volume z {features['volume_zscore']:.2f}. "
        f"Model → {prediction['signal']} @ {prediction['confidence']:.0%}."
    )
