"""Train the XGBoost signal model.

Builds a feature/label dataset from historical bars and fits a 3-class
classifier (SELL/HOLD/BUY) over RSI/MACD/BB-width/volume-z features, then saves
it to the path the strategy layer loads from (``data/signal_model.json``).

Data source:
* ``--symbols AAPL,MSFT`` with live Robinhood creds → real historicals.
* otherwise → the deterministic synthetic market (lets you train end-to-end
  with zero credentials, useful for smoke-testing the whole pipeline).

Usage:
    python scripts/train_model.py --symbols AAPL,MSFT,NVDA --span month
    python scripts/train_model.py --synthetic --n-symbols 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.broker import PaperBroker, RobinhoodBroker
from trading.config import get_config
from trading.features import FEATURE_COLUMNS, compute_feature_frame
from trading.model import LABELS, make_labels


def build_dataset(bars_by_symbol: dict[str, list], horizon: int, threshold: float):
    X, y = [], []
    for symbol, bars in bars_by_symbol.items():
        if len(bars) < 60:
            continue
        feats = compute_feature_frame(bars)
        close = np.array([b["close"] for b in bars], dtype=float)
        labels = make_labels(close, horizon=horizon, threshold=threshold)
        # Drop the warm-up region where indicators are not yet meaningful.
        warmup = 30
        X.append(feats.iloc[warmup:].to_numpy())
        y.append(labels[warmup:])
    if not X:
        raise SystemExit("No usable data collected.")
    return np.vstack(X), np.concatenate(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbols (live data).")
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data source.")
    ap.add_argument("--n-symbols", type=int, default=30, help="Synthetic symbol count.")
    ap.add_argument("--span", default="month")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.004)
    args = ap.parse_args()

    cfg = get_config()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if symbols and not args.synthetic:
        broker = RobinhoodBroker()
        print(f"Pulling live historicals for {symbols} ...")
        bars_by_symbol = {s: broker.get_bars(s, span=args.span) for s in symbols}
    else:
        broker = PaperBroker()
        symbols = symbols or [f"SYN{i:03d}" for i in range(args.n_symbols)]
        print(f"Generating synthetic bars for {len(symbols)} symbols ...")
        bars_by_symbol = {s: broker.get_bars(s, span=args.span) for s in symbols}

    X, y = build_dataset(bars_by_symbol, args.horizon, args.threshold)
    print(f"Dataset: X={X.shape}, label distribution={dict(zip(*np.unique(y, return_counts=True)))}")

    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(LABELS),
        eval_metric="mlogloss",
        n_jobs=4,
    )
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"Hold-out accuracy: {acc:.3f}")
    print("Feature importances:")
    for name, imp in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1]
    ):
        print(f"  {name:16s} {imp:.3f}")

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(cfg.model_path))
    print(f"Saved model → {cfg.model_path}")


if __name__ == "__main__":
    main()
