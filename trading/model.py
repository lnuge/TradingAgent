"""Strategy model: an XGBoost classifier mapping the technical feature vector
to a BUY / SELL / HOLD decision plus a calibrated confidence.

If no trained model file exists, the module falls back to a transparent
rule-based scorer over the same features, so ``generate_signal`` is always
functional (and the system is runnable before any training has happened).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import get_config
from .features import FEATURE_COLUMNS

LABELS = ["SELL", "HOLD", "BUY"]  # class index 0,1,2
_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}


class SignalModel:
    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or get_config().model_path)
        self._booster = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import xgboost as xgb

            booster = xgb.XGBClassifier()
            booster.load_model(str(self.model_path))
            self._booster = booster
        except Exception:
            self._booster = None

    @property
    def is_trained(self) -> bool:
        return self._booster is not None

    # -- inference -----------------------------------------------------------
    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
        if self._booster is not None:
            proba = self._booster.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            return {
                "signal": LABELS[idx],
                "confidence": round(float(proba[idx]), 4),
                "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
                "source": "xgboost",
            }
        return self._rule_based(features)

    def _rule_based(self, features: dict[str, float]) -> dict[str, Any]:
        """Transparent fallback so the tool works before a model is trained."""
        rsi = features["rsi_14"]
        macd_hist = features["macd_hist"]
        score = 0.0
        # Mean-reversion on RSI extremes.
        if rsi < 30:
            score += (30 - rsi) / 30
        elif rsi > 70:
            score -= (rsi - 70) / 30
        # Trend confirmation from MACD histogram.
        score += float(np.tanh(macd_hist))
        signal = "BUY" if score > 0.5 else "SELL" if score < -0.5 else "HOLD"
        confidence = round(min(0.95, 0.5 + abs(score) / 2), 4)
        return {
            "signal": signal,
            "confidence": confidence,
            "probabilities": None,
            "source": "rule_based_fallback",
        }


def make_labels(close: np.ndarray, horizon: int = 5, threshold: float = 0.004) -> np.ndarray:
    """Forward-return labels for supervised training.

    Looks ``horizon`` bars ahead: > +threshold => BUY, < -threshold => SELL,
    otherwise HOLD. Returns integer class indices aligned to ``LABELS``.
    """
    n = len(close)
    labels = np.full(n, _LABEL_TO_IDX["HOLD"], dtype=int)
    for i in range(n - horizon):
        fwd = (close[i + horizon] - close[i]) / close[i]
        if fwd > threshold:
            labels[i] = _LABEL_TO_IDX["BUY"]
        elif fwd < -threshold:
            labels[i] = _LABEL_TO_IDX["SELL"]
    labels[n - horizon :] = _LABEL_TO_IDX["HOLD"]
    return labels


_model: SignalModel | None = None


def get_model() -> SignalModel:
    global _model
    if _model is None:
        _model = SignalModel()
    return _model
