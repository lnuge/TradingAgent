"""EXECUTION LAYER — order placement, cancellation, and open-order queries.

Two safety properties are enforced here in code, not by agent cooperation:

1. **Non-bypassable risk.** ``place_order`` calls ``evaluate_risk`` itself and
   refuses to submit anything the risk layer rejects — even if the agent never
   called the ``check_risk`` tool first.
2. **Idempotency.** Every order requires an ``idempotency_key``. If the same key
   is seen again (e.g. an agent retry after a timeout), the original order is
   returned instead of submitting a duplicate.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .broker import get_broker
from .config import get_config
from .context import SESSION_ID
from .risk_layer import evaluate_risk
from .storage import get_storage


def place_order(
    symbol: str,
    side: str,
    quantity: float,
    idempotency_key: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    storage = get_storage()
    symbol = symbol.upper().strip()
    side = side.upper().strip()

    if not idempotency_key:
        return {"status": "rejected", "reason": "idempotency_key is required for every order."}

    # 1. Idempotency — return the prior result instead of duplicating.
    existing = storage.get_order_by_key(idempotency_key)
    if existing is not None:
        return {
            "status": "duplicate",
            "reason": "idempotency_key already used; returning original order (no duplicate sent).",
            "order": existing,
        }

    # 2. NON-BYPASSABLE risk re-check. This runs regardless of the agent.
    decision = evaluate_risk(side, symbol, quantity, confidence=confidence)
    if not decision.approved:
        record = _order_record(
            idempotency_key, "RISK_REJECTED", symbol, side, quantity, decision.price, cfg.mode,
            detail="; ".join(decision.reasons),
        )
        storage.record_order(record)
        storage.append_audit(
            SESSION_ID, "execution.rejected",
            {"reasons": decision.reasons, "idempotency_key": idempotency_key}, symbol,
        )
        return {
            "status": "rejected",
            "reason": "Risk layer rejected the order (enforced in code).",
            "risk_decision": decision.to_dict(),
        }

    # 3. Submit to the broker.
    try:
        result = get_broker().submit_order(symbol, side.lower(), quantity)
    except Exception as exc:
        record = _order_record(
            idempotency_key, "ERROR", symbol, side, quantity, decision.price, cfg.mode, detail=str(exc)
        )
        storage.record_order(record)
        storage.append_audit(
            SESSION_ID, "execution.error", {"error": str(exc), "idempotency_key": idempotency_key}, symbol
        )
        return {"status": "error", "reason": str(exc)}

    record = _order_record(
        idempotency_key,
        result.get("status", "submitted"),
        symbol,
        side,
        quantity,
        result.get("filled_price") or decision.price,
        cfg.mode,
        order_id=result["order_id"],
        notional=result.get("notional"),
        detail="executed",
    )
    storage.record_order(record)
    storage.append_audit(
        SESSION_ID,
        "execution.placed",
        {
            "order_id": result["order_id"],
            "side": side,
            "quantity": quantity,
            "price": record["price"],
            "idempotency_key": idempotency_key,
            "mode": cfg.mode,
        },
        symbol,
    )
    return {"status": "ok", "order": record, "broker_result": result}


def cancel_order(order_id: str) -> dict[str, Any]:
    result = get_broker().cancel_order(order_id)
    get_storage().append_audit(SESSION_ID, "execution.cancel", {"order_id": order_id, "result": result})
    return result


def get_open_orders() -> dict[str, Any]:
    broker_open = get_broker().get_open_orders()
    recorded = get_storage().get_orders(session_id=SESSION_ID)
    return {"broker_open_orders": broker_open, "session_orders": recorded}


def _order_record(
    idempotency_key: str,
    status: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float | None,
    mode: str,
    order_id: str | None = None,
    notional: float | None = None,
    detail: str = "",
) -> dict[str, Any]:
    px = price or 0.0
    return {
        "idempotency_key": idempotency_key,
        "order_id": order_id or f"local-{uuid.uuid4().hex[:12]}",
        "ts": time.time(),
        "session_id": SESSION_ID,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": px,
        "notional": notional if notional is not None else round(px * quantity, 2),
        "status": status,
        "mode": mode,
        "detail": detail,
    }
