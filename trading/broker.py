"""Broker abstraction wrapping market data + execution.

Two implementations:

* :class:`PaperBroker` — default. Generates deterministic, realistic synthetic
  market data (so the whole system runs with zero credentials/network) and
  simulates fills against the paper account in :mod:`trading.storage`.
* :class:`RobinhoodBroker` — live/sandbox. Wraps ``robin_stocks`` for real
  quotes, bars, order book, and order placement. ``robin_stocks`` is imported
  lazily so it is only required when actually trading live.

Both expose the same interface, so the data/execution layers don't care which
one is active.
"""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from typing import Any

from .config import Config, get_config
from .storage import Storage, get_storage


class BrokerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Synthetic market: deterministic per-symbol geometric Brownian motion so that
# quotes/bars are stable within a run and reproducible across runs.
# ---------------------------------------------------------------------------
class SyntheticMarket:
    def __init__(self) -> None:
        self._epoch = time.time()

    def _seed(self, symbol: str) -> int:
        return int(hashlib.sha256(symbol.upper().encode()).hexdigest(), 16) % (2**32)

    def _base_price(self, symbol: str) -> float:
        # Map the seed into a plausible share price band [10, 510).
        return 10.0 + (self._seed(symbol) % 5000) / 10.0

    def bars(self, symbol: str, count: int = 200, interval_sec: int = 300) -> list[dict[str, Any]]:
        import numpy as np

        rng = np.random.default_rng(self._seed(symbol))
        base = self._base_price(symbol)
        drift, vol = 0.0001, 0.01
        prices = [base]
        for _ in range(count):
            shock = rng.normal(drift, vol)
            prices.append(max(0.5, prices[-1] * (1 + shock)))
        prices = prices[1:]
        now = int(time.time())
        bars = []
        for i, close in enumerate(prices):
            o = prices[i - 1] if i > 0 else close
            hi = max(o, close) * (1 + abs(rng.normal(0, vol / 2)))
            lo = min(o, close) * (1 - abs(rng.normal(0, vol / 2)))
            volume = int(abs(rng.normal(1_000_000, 250_000)))
            ts = now - (count - i) * interval_sec
            bars.append(
                {
                    "begins_at": ts,
                    "open": round(o, 4),
                    "high": round(hi, 4),
                    "low": round(lo, 4),
                    "close": round(close, 4),
                    "volume": volume,
                }
            )
        return bars

    def quote(self, symbol: str) -> dict[str, Any]:
        last_bar = self.bars(symbol, count=2)[-1]
        price = last_bar["close"]
        spread = max(0.01, price * 0.0005)
        return {
            "symbol": symbol.upper(),
            "last_price": round(price, 4),
            "bid_price": round(price - spread / 2, 4),
            "ask_price": round(price + spread / 2, 4),
            "previous_close": round(self.bars(symbol, count=2)[0]["close"], 4),
            "ts": time.time(),
        }

    def order_book(self, symbol: str, depth: int = 5) -> dict[str, Any]:
        q = self.quote(symbol)
        mid = q["last_price"]
        tick = max(0.01, mid * 0.0005)
        import numpy as np

        rng = np.random.default_rng(self._seed(symbol) + 7)
        bids = [
            {"price": round(mid - tick * (i + 1), 4), "size": int(abs(rng.normal(500, 150)))}
            for i in range(depth)
        ]
        asks = [
            {"price": round(mid + tick * (i + 1), 4), "size": int(abs(rng.normal(500, 150)))}
            for i in range(depth)
        ]
        return {"symbol": symbol.upper(), "bids": bids, "asks": asks, "ts": time.time()}


# ---------------------------------------------------------------------------
class BaseBroker:
    mode = "paper"

    def get_quote(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_bars(self, symbol: str, interval: str, span: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_order_book(self, symbol: str, depth: int = 5) -> dict[str, Any]:
        raise NotImplementedError

    def submit_order(self, symbol: str, side: str, quantity: float) -> dict[str, Any]:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_open_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class PaperBroker(BaseBroker):
    mode = "paper"

    def __init__(self, storage: Storage | None = None):
        self.market = SyntheticMarket()
        self.storage = storage or get_storage()

    def get_quote(self, symbol: str) -> dict[str, Any]:
        return self.market.quote(symbol)

    def get_bars(self, symbol: str, interval: str = "5minute", span: str = "day") -> list[dict[str, Any]]:
        count = {"day": 78, "week": 200, "month": 300}.get(span, 200)
        return self.market.bars(symbol, count=max(count, 60))

    def get_order_book(self, symbol: str, depth: int = 5) -> dict[str, Any]:
        return self.market.order_book(symbol, depth)

    def submit_order(self, symbol: str, side: str, quantity: float) -> dict[str, Any]:
        """Simulate an immediate fill at the current ask (buy) / bid (sell)."""
        q = self.get_quote(symbol)
        fill_price = q["ask_price"] if side == "buy" else q["bid_price"]
        notional = fill_price * quantity
        acct = self.storage.get_account()
        pos = self.storage.get_position(symbol)

        if side == "buy":
            if notional > acct["cash"] + 1e-6:
                raise BrokerError(
                    f"Insufficient paper cash: need ${notional:,.2f}, have ${acct['cash']:,.2f}"
                )
            self.storage.set_cash(acct["cash"] - notional)
            held = pos["quantity"] if pos else 0.0
            avg = pos["avg_price"] if pos else 0.0
            new_qty = held + quantity
            new_avg = (held * avg + quantity * fill_price) / new_qty
            self.storage.upsert_position(symbol, new_qty, new_avg)
        else:  # sell
            held = pos["quantity"] if pos else 0.0
            if quantity > held + 1e-9:
                raise BrokerError(
                    f"Cannot sell {quantity} {symbol}: only {held} held (no naked shorting in paper)"
                )
            self.storage.set_cash(acct["cash"] + notional)
            self.storage.upsert_position(symbol, held - quantity, pos["avg_price"] if pos else 0.0)

        return {
            "order_id": f"paper-{uuid.uuid4().hex[:16]}",
            "status": "filled",
            "filled_price": round(fill_price, 4),
            "filled_quantity": quantity,
            "notional": round(notional, 2),
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        # Paper orders fill instantly, so there is nothing live to cancel.
        return {"order_id": order_id, "status": "not_cancellable", "reason": "paper fills are immediate"}

    def get_open_orders(self) -> list[dict[str, Any]]:
        return []


class RobinhoodBroker(BaseBroker):
    mode = "live"

    def __init__(self) -> None:
        try:
            import robin_stocks.robinhood as rh  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only hit in live mode
            raise BrokerError(
                "robin_stocks is required for live trading. Install with `pip install robin_stocks`."
            ) from exc
        import os

        self.rh = rh
        user, pw = os.getenv("ROBINHOOD_USERNAME"), os.getenv("ROBINHOOD_PASSWORD")
        if not user or not pw:
            raise BrokerError("Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD for live trading.")
        self.rh.login(user, pw, mfa_code=os.getenv("ROBINHOOD_MFA") or None)

    def get_quote(self, symbol: str) -> dict[str, Any]:
        data = self.rh.stocks.get_quotes(symbol)
        if not data or data[0] is None:
            raise BrokerError(f"No quote for {symbol}")
        d = data[0]
        return {
            "symbol": symbol.upper(),
            "last_price": float(d["last_trade_price"]),
            "bid_price": float(d["bid_price"]),
            "ask_price": float(d["ask_price"]),
            "previous_close": float(d["previous_close"]),
            "ts": time.time(),
        }

    def get_bars(self, symbol: str, interval: str = "5minute", span: str = "day") -> list[dict[str, Any]]:
        hist = self.rh.stocks.get_stock_historicals(symbol, interval=interval, span=span)
        bars = []
        for h in hist or []:
            bars.append(
                {
                    "begins_at": h["begins_at"],
                    "open": float(h["open_price"]),
                    "high": float(h["high_price"]),
                    "low": float(h["low_price"]),
                    "close": float(h["close_price"]),
                    "volume": int(h.get("volume", 0)),
                }
            )
        return bars

    def get_order_book(self, symbol: str, depth: int = 5) -> dict[str, Any]:
        # Robinhood's retail API exposes only top-of-book; approximate around it.
        q = self.get_quote(symbol)
        return {
            "symbol": symbol.upper(),
            "bids": [{"price": q["bid_price"], "size": None}],
            "asks": [{"price": q["ask_price"], "size": None}],
            "note": "Robinhood retail API exposes only top-of-book.",
            "ts": time.time(),
        }

    def submit_order(self, symbol: str, side: str, quantity: float) -> dict[str, Any]:
        fn = self.rh.orders.order_buy_market if side == "buy" else self.rh.orders.order_sell_market
        res = fn(symbol, int(quantity))
        if not res or "id" not in res:
            raise BrokerError(f"Order rejected by broker: {res}")
        return {
            "order_id": res["id"],
            "status": res.get("state", "submitted"),
            "filled_price": None,
            "filled_quantity": quantity,
            "notional": None,
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.rh.orders.cancel_stock_order(order_id)
        return {"order_id": order_id, "status": "cancelled"}

    def get_open_orders(self) -> list[dict[str, Any]]:
        orders = self.rh.orders.get_all_open_stock_orders() or []
        return [{"order_id": o.get("id"), "state": o.get("state"), "raw": o} for o in orders]


_broker: BaseBroker | None = None


def get_broker(config: Config | None = None) -> BaseBroker:
    global _broker
    if _broker is not None:
        return _broker
    cfg = config or get_config()
    _broker = RobinhoodBroker() if cfg.is_live else PaperBroker()
    return _broker
