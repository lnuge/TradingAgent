"""DATA LAYER — read-only market data access.

Thin wrappers over the active broker (synthetic in paper mode, robin_stocks in
live mode). Every call is logged to the audit trail so the agent's view of the
market state is reconstructable after the fact.
"""
from __future__ import annotations

from typing import Any

from .broker import get_broker
from .context import SESSION_ID
from .storage import get_storage


def get_quote(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    quote = get_broker().get_quote(symbol)
    get_storage().append_audit(SESSION_ID, "data.get_quote", {"quote": quote}, symbol)
    return quote


def get_bars(symbol: str, interval: str = "5minute", span: str = "day") -> dict[str, Any]:
    symbol = symbol.upper().strip()
    bars = get_broker().get_bars(symbol, interval=interval, span=span)
    result = {
        "symbol": symbol,
        "interval": interval,
        "span": span,
        "count": len(bars),
        "bars": bars,
    }
    get_storage().append_audit(
        SESSION_ID, "data.get_bars", {"interval": interval, "span": span, "count": len(bars)}, symbol
    )
    return result


def get_order_book(symbol: str, depth: int = 5) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    book = get_broker().get_order_book(symbol, depth=depth)
    get_storage().append_audit(SESSION_ID, "data.get_order_book", {"depth": depth}, symbol)
    return book
