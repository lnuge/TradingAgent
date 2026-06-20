"""Watchlist scanner — turns the universe into ranked buy/sell decisions.

For each symbol it runs the full per-name pipeline the way the agent would:
generate a signal, decide a proposed action (entry / exit / hold), size it, and
run it through the (non-bypassable) risk layer. By default it is a **dry run**
(``execute=False``): it returns ranked opportunities for the agent or human to
review. With ``execute=True`` it places the approved orders.

Exit logic (for positions already held):
  * stop-loss breached            -> SELL the full position
  * model says SELL w/ confidence -> SELL the full position
Entry logic (for symbols not held):
  * model says BUY w/ confidence  -> BUY a target-sized position

Sizing: a new position targets ``TARGET_POSITION_PCT`` of equity, capped by the
risk layer's ``max_position_pct`` and per-order notional limit.
"""
from __future__ import annotations

import time
from typing import Any

from .broker import get_broker
from .config import get_config
from .context import SESSION_ID
from .execution_layer import place_order
from .risk_layer import check_risk, get_portfolio_status
from .storage import get_storage
from .strategy_layer import generate_signal
from .watchlist import load_watchlist


def _size_new_position(equity: float, price: float, cfg) -> int:
    target_notional = min(cfg.target_position_pct * equity, cfg.risk.max_order_value)
    return int(target_notional // price) if price > 0 else 0


def scan_watchlist(
    symbols: list[str] | None = None,
    execute: bool = False,
    interval: str = "day",
    span: str = "month",
) -> dict[str, Any]:
    cfg = get_config()
    storage = get_storage()
    broker = get_broker()
    symbols = symbols or load_watchlist()
    if not symbols:
        return {"error": "Watchlist is empty. Add symbols to watchlist.txt."}

    scan_id = f"scan-{int(time.time())}"
    portfolio = get_portfolio_status()
    equity = portfolio["equity"]
    positions = {p["symbol"]: p for p in portfolio["positions"]}

    opportunities: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            sig = generate_signal(symbol, interval=interval, span=span)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
            continue

        signal = sig["signal"]
        confidence = sig["confidence"]
        try:
            price = broker.get_quote(symbol)["last_price"]
        except Exception:
            price = 0.0
        pos = positions.get(symbol)
        held = pos["quantity"] if pos else 0.0

        action, side, quantity, rationale = "none", None, 0.0, ""

        if held > 0:
            # --- exit logic for an existing position ---
            if pos and pos["stop_breached"]:
                action, side, quantity = "exit", "SELL", held
                rationale = f"stop-loss breached (last ${pos['last_price']:.2f} <= stop ${pos['stop_loss_price']:.2f})"
            elif signal == "SELL" and confidence >= cfg.risk.min_confidence:
                action, side, quantity = "exit", "SELL", held
                rationale = f"model SELL @ {confidence:.0%}"
            else:
                rationale = f"hold position ({signal} @ {confidence:.0%})"
        else:
            # --- entry logic for a symbol we don't hold ---
            if signal == "BUY" and confidence >= cfg.risk.min_confidence:
                qty = _size_new_position(equity, price, cfg)
                if qty >= 1:
                    action, side, quantity = "entry", "BUY", float(qty)
                    rationale = f"model BUY @ {confidence:.0%}, target {cfg.target_position_pct:.0%} of equity"
                else:
                    rationale = "BUY signal but sized below 1 share"
            else:
                rationale = f"no entry ({signal} @ {confidence:.0%})"

        record = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "last_price": round(price, 4),
            "action": action,
            "rationale": rationale,
            "held": held,
        }

        if action == "none":
            holds.append(record)
            continue

        # Every proposed order goes through the deterministic risk layer.
        decision = check_risk(side, symbol, quantity, confidence=confidence, price=price)
        record.update(
            {
                "side": side,
                "quantity": quantity,
                "notional": decision["notional"],
                "risk_approved": decision["approved"],
                "risk_reasons": decision["reasons"],
                "executed": False,
            }
        )

        if execute and decision["approved"]:
            res = place_order(
                symbol, side, quantity,
                idempotency_key=f"{scan_id}-{symbol}", confidence=confidence,
            )
            record["executed"] = res["status"] == "ok"
            record["exec_status"] = res["status"]

        opportunities.append(record)

    # Rank actionable, risk-approved ideas by confidence (highest conviction first).
    opportunities.sort(key=lambda r: (r["risk_approved"], r["confidence"]), reverse=True)

    summary = {
        "scan_id": scan_id,
        "mode": portfolio["mode"],
        "executed": execute,
        "equity": equity,
        "cash": portfolio["cash"],
        "open_positions": portfolio["open_positions"],
        "symbols_scanned": len(symbols),
        "opportunities": opportunities,
        "approved_count": sum(1 for o in opportunities if o["risk_approved"]),
        "executed_count": sum(1 for o in opportunities if o.get("executed")),
        "holds": [{"symbol": h["symbol"], "signal": h["signal"], "rationale": h["rationale"]} for h in holds],
        "errors": errors,
        "guidance": (
            "Dry run — review opportunities and call scan_watchlist(execute=true) or "
            "place_order to act."
            if not execute
            else "Approved orders were placed. Risk layer gatekept every one."
        ),
    }
    storage.append_audit(
        SESSION_ID,
        "scanner.scan",
        {
            "scan_id": scan_id,
            "execute": execute,
            "scanned": len(symbols),
            "approved": summary["approved_count"],
            "executed": summary["executed_count"],
        },
    )
    return summary
