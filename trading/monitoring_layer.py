"""MONITORING LAYER — immutable logging and session introspection.

Every agent action across the other layers already appends to the hash-chained
audit log in :mod:`trading.storage`. This layer adds an explicit ``log_trade``
hook for the agent's own annotations and a ``get_session_summary`` /
``get_trade_history`` view so the agent can query what it has done mid-session
and feed that back into its next decision.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .context import SESSION_ID
from .risk_layer import get_portfolio_status
from .storage import get_storage


def log_trade(event_type: str, detail: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    """Append an agent annotation to the immutable audit log."""
    entry = get_storage().append_audit(
        SESSION_ID, f"agent.{event_type}", {"detail": detail}, symbol
    )
    return {"logged": True, "entry": entry}


def get_trade_history(limit: int = 50, event_type: str | None = None) -> dict[str, Any]:
    rows = get_storage().get_audit(session_id=SESSION_ID, event_type=event_type, limit=limit)
    return {"session_id": SESSION_ID, "count": len(rows), "events": rows}


def get_session_summary() -> dict[str, Any]:
    storage = get_storage()
    events = storage.get_audit(session_id=SESSION_ID, limit=10_000)
    orders = storage.get_orders(session_id=SESSION_ID)

    event_counts = Counter(e["event_type"] for e in events)
    placed = [o for o in orders if o["status"] not in {"RISK_REJECTED", "ERROR"}]
    rejected = [o for o in orders if o["status"] == "RISK_REJECTED"]

    buys = [o for o in placed if o["side"] == "BUY"]
    sells = [o for o in placed if o["side"] == "SELL"]

    portfolio = get_portfolio_status()
    chain = storage.verify_audit_chain()

    return {
        "session_id": SESSION_ID,
        "mode": portfolio["mode"],
        "event_counts": dict(event_counts),
        "orders": {
            "total": len(orders),
            "executed": len(placed),
            "risk_rejected": len(rejected),
            "buys": len(buys),
            "sells": len(sells),
            "buy_notional": round(sum(o["notional"] or 0 for o in buys), 2),
            "sell_notional": round(sum(o["notional"] or 0 for o in sells), 2),
        },
        "portfolio": {
            "equity": portfolio["equity"],
            "cash": portfolio["cash"],
            "day_pnl": portfolio["day_pnl"],
            "day_drawdown_pct": portfolio["day_drawdown_pct"],
            "open_positions": portfolio["open_positions"],
            "kill_switch_engaged": portfolio["kill_switch_engaged"],
        },
        "audit_chain_valid": chain["valid"],
        "audit_events_total": chain["total_rows"],
    }
