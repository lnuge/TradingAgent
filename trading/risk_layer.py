"""RISK LAYER — deterministic, non-bypassable hard constraints.

Nothing here is an agent judgement call. ``evaluate_risk`` is pure, rule-based
code that approves or rejects a proposed trade against fixed limits:

* kill switch (manual file or auto-tripped on drawdown breach)
* daily drawdown limit (trips the kill switch when breached)
* max position size as a % of equity
* absolute per-order notional cap
* sufficient cash (buys) / sufficient holdings — no naked shorting (sells)
* minimum model confidence
* max number of concurrent open positions

The execution layer calls :func:`evaluate_risk` *itself* before every order, so
even if the agent skips the ``check_risk`` tool, an unsafe order is still
rejected. The risk layer is enforced in code, not by agent cooperation.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .broker import get_broker
from .config import get_config
from .context import SESSION_ID
from .storage import get_storage

VALID_SIDES = {"BUY", "SELL"}


@dataclass
class RiskDecision:
    approved: bool
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    portfolio_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- kill switch ------------------------------------------------------------
def is_kill_switch_engaged() -> bool:
    return get_config().kill_switch_path.exists()


def engage_kill_switch(reason: str) -> None:
    path = get_config().kill_switch_path
    path.write_text(f"{time.time()}\n{reason}\n")
    get_storage().append_audit(SESSION_ID, "risk.kill_switch_engaged", {"reason": reason})


def reset_kill_switch() -> bool:
    path = get_config().kill_switch_path
    if path.exists():
        path.unlink()
        get_storage().append_audit(SESSION_ID, "risk.kill_switch_reset", {})
        return True
    return False


# -- portfolio valuation ----------------------------------------------------
def get_portfolio_status() -> dict[str, Any]:
    cfg = get_config()
    storage = get_storage()
    storage.roll_day_if_needed()
    acct = storage.get_account()
    broker = get_broker()

    positions = []
    market_value = 0.0
    for p in storage.get_positions():
        try:
            last = broker.get_quote(p["symbol"])["last_price"]
        except Exception:
            last = p["avg_price"]
        value = last * p["quantity"]
        market_value += value
        unrealized = (last - p["avg_price"]) * p["quantity"]
        stop_price = round(p["avg_price"] * (1 - cfg.risk.stop_loss_pct), 4)
        positions.append(
            {
                "symbol": p["symbol"],
                "quantity": p["quantity"],
                "avg_price": p["avg_price"],
                "last_price": round(last, 4),
                "market_value": round(value, 2),
                "unrealized_pnl": round(unrealized, 2),
                "stop_loss_price": stop_price,
                "stop_breached": last <= stop_price,
            }
        )

    equity = acct["cash"] + market_value
    day_start = acct["day_start_equity"] or equity
    day_pnl = equity - day_start
    drawdown_pct = (day_start - equity) / day_start if day_start > 0 else 0.0

    return {
        "mode": cfg.describe_mode(),
        "cash": round(acct["cash"], 2),
        "market_value": round(market_value, 2),
        "equity": round(equity, 2),
        "starting_equity": round(acct["starting_equity"], 2),
        "day_start_equity": round(day_start, 2),
        "day_pnl": round(day_pnl, 2),
        "day_drawdown_pct": round(drawdown_pct, 4),
        "kill_switch_engaged": is_kill_switch_engaged(),
        "open_positions": len(positions),
        "positions": positions,
        "limits": {
            "max_position_pct": cfg.risk.max_position_pct,
            "max_order_value": cfg.risk.max_order_value,
            "stop_loss_pct": cfg.risk.stop_loss_pct,
            "daily_drawdown_limit_pct": cfg.risk.daily_drawdown_limit_pct,
            "min_confidence": cfg.risk.min_confidence,
            "max_open_positions": cfg.risk.max_open_positions,
        },
    }


# -- the core risk evaluation (called by both the tool AND execution) -------
def evaluate_risk(
    signal: str,
    symbol: str,
    quantity: float,
    confidence: float | None = None,
    price: float | None = None,
) -> RiskDecision:
    cfg = get_config()
    storage = get_storage()
    symbol = symbol.upper().strip()
    side = signal.upper().strip()

    if price is None:
        try:
            q = get_broker().get_quote(symbol)
            price = q["ask_price"] if side == "BUY" else q["bid_price"]
        except Exception:
            price = 0.0
    notional = round(price * quantity, 2)

    decision = RiskDecision(
        approved=True, symbol=symbol, side=side, quantity=quantity, price=price, notional=notional
    )
    portfolio = get_portfolio_status()
    decision.portfolio_snapshot = {
        "equity": portfolio["equity"],
        "cash": portfolio["cash"],
        "day_drawdown_pct": portfolio["day_drawdown_pct"],
        "open_positions": portfolio["open_positions"],
    }
    equity = portfolio["equity"]
    checks = decision.checks

    def reject(reason: str) -> None:
        decision.approved = False
        decision.reasons.append(reason)

    # 0. Valid actionable side. HOLD never produces an order.
    if side not in VALID_SIDES:
        checks["valid_side"] = False
        reject(f"Signal '{side}' is not actionable (only BUY/SELL place orders).")
        return _finalize(decision, storage)
    checks["valid_side"] = True

    if quantity <= 0:
        reject("Quantity must be positive.")
        return _finalize(decision, storage)

    # 1. Kill switch.
    if is_kill_switch_engaged():
        checks["kill_switch"] = "ENGAGED"
        reject("Kill switch is engaged — all trading halted.")
    else:
        checks["kill_switch"] = "ok"

    # 2. Daily drawdown limit — breaching it also trips the kill switch.
    dd = portfolio["day_drawdown_pct"]
    checks["daily_drawdown"] = {"current": dd, "limit": cfg.risk.daily_drawdown_limit_pct}
    if dd >= cfg.risk.daily_drawdown_limit_pct:
        reject(
            f"Daily drawdown {dd:.2%} ≥ limit {cfg.risk.daily_drawdown_limit_pct:.2%}; "
            "engaging kill switch."
        )
        if not is_kill_switch_engaged():
            engage_kill_switch(f"daily drawdown {dd:.2%} breached limit")

    # 3. Minimum confidence (only enforced when a confidence is supplied).
    if confidence is not None:
        checks["confidence"] = {"value": confidence, "min": cfg.risk.min_confidence}
        if confidence < cfg.risk.min_confidence:
            reject(f"Confidence {confidence:.2f} < minimum {cfg.risk.min_confidence:.2f}.")

    # 4. Absolute per-order notional cap.
    checks["order_notional"] = {"value": notional, "max": cfg.risk.max_order_value}
    if notional > cfg.risk.max_order_value:
        reject(f"Order notional ${notional:,.2f} exceeds cap ${cfg.risk.max_order_value:,.2f}.")

    position = storage.get_position(symbol)
    held = position["quantity"] if position else 0.0

    if side == "BUY":
        # 5. Resulting position size vs equity.
        resulting_value = (held + quantity) * price
        max_position_value = cfg.risk.max_position_pct * equity if equity > 0 else 0.0
        checks["position_size"] = {
            "resulting_value": round(resulting_value, 2),
            "max_allowed": round(max_position_value, 2),
        }
        if resulting_value > max_position_value:
            reject(
                f"Resulting {symbol} position ${resulting_value:,.2f} exceeds "
                f"{cfg.risk.max_position_pct:.0%} of equity (${max_position_value:,.2f})."
            )
        # 6. Sufficient cash.
        checks["cash"] = {"required": notional, "available": portfolio["cash"]}
        if notional > portfolio["cash"]:
            reject(f"Insufficient cash: need ${notional:,.2f}, have ${portfolio['cash']:,.2f}.")
        # 7. Max concurrent positions (only when opening a new symbol).
        if held == 0:
            checks["open_positions"] = {
                "current": portfolio["open_positions"],
                "max": cfg.risk.max_open_positions,
            }
            if portfolio["open_positions"] >= cfg.risk.max_open_positions:
                reject(
                    f"Already at max {cfg.risk.max_open_positions} open positions; "
                    "cannot open a new one."
                )
    else:  # SELL — no naked shorting.
        checks["holdings"] = {"held": held, "requested": quantity}
        if quantity > held + 1e-9:
            reject(f"Cannot SELL {quantity} {symbol}: only {held} held (no naked shorting).")

    return _finalize(decision, storage)


def _finalize(decision: RiskDecision, storage) -> RiskDecision:
    if decision.approved:
        decision.reasons.append("All hard risk checks passed.")
    storage.append_audit(
        SESSION_ID,
        "risk.check",
        {
            "approved": decision.approved,
            "side": decision.side,
            "quantity": decision.quantity,
            "notional": decision.notional,
            "reasons": decision.reasons,
        },
        decision.symbol,
    )
    return decision


def check_risk(
    signal: str,
    symbol: str,
    quantity: float,
    confidence: float | None = None,
    price: float | None = None,
) -> dict[str, Any]:
    """Public MCP-facing wrapper around :func:`evaluate_risk`."""
    return evaluate_risk(signal, symbol, quantity, confidence=confidence, price=price).to_dict()
