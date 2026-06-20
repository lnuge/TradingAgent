"""Tests for the RISK LAYER — the safety-critical, non-bypassable guards.

These verify the limits are enforced in *code* (not agent cooperation): the
execution layer must reject unsafe orders even when ``check_risk`` is never
called, and idempotency must prevent duplicate orders.
"""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture()
def fresh_env(tmp_path, monkeypatch):
    """Isolate each test with its own DB/model/kill-switch paths and limits."""
    monkeypatch.setenv("TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("PAPER_STARTING_CASH", "100000")
    monkeypatch.setenv("MAX_POSITION_PCT", "0.20")
    monkeypatch.setenv("MAX_ORDER_VALUE", "5000")
    monkeypatch.setenv("DAILY_DRAWDOWN_LIMIT_PCT", "0.03")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.55")
    monkeypatch.setenv("TRADING_SESSION_ID", "test-session")

    # Reload modules so they pick up the patched env and singletons reset.
    import trading.config as config
    import trading.storage as storage
    import trading.broker as broker
    import trading.context as context
    import trading.risk_layer as risk_layer
    import trading.execution_layer as execution_layer

    for mod in (config, context, storage, broker, risk_layer, execution_layer):
        importlib.reload(mod)
    # Reset singletons explicitly.
    config._config = None
    storage._storage = None
    broker._broker = None
    return {"risk": risk_layer, "execution": execution_layer, "broker": broker}


def test_hold_signal_is_not_actionable(fresh_env):
    d = fresh_env["risk"].evaluate_risk("HOLD", "AAPL", 1)
    assert d.approved is False
    assert any("not actionable" in r for r in d.reasons)


def test_order_notional_cap(fresh_env):
    # A huge quantity blows past the $5,000 per-order cap.
    risk = fresh_env["risk"]
    price = fresh_env["broker"].get_broker().get_quote("AAPL")["ask_price"]
    qty = (6000 / price) + 1
    d = risk.evaluate_risk("BUY", "AAPL", qty, confidence=0.9)
    assert d.approved is False
    assert any("notional" in r.lower() for r in d.reasons)


def test_low_confidence_rejected(fresh_env):
    d = fresh_env["risk"].evaluate_risk("BUY", "AAPL", 1, confidence=0.10)
    assert d.approved is False
    assert any("confidence" in r.lower() for r in d.reasons)


def test_no_naked_shorting(fresh_env):
    d = fresh_env["risk"].evaluate_risk("SELL", "AAPL", 10, confidence=0.9)
    assert d.approved is False
    assert any("naked shorting" in r.lower() or "held" in r.lower() for r in d.reasons)


def test_kill_switch_blocks_everything(fresh_env):
    risk = fresh_env["risk"]
    risk.engage_kill_switch("manual test")
    d = risk.evaluate_risk("BUY", "AAPL", 1, confidence=0.99)
    assert d.approved is False
    assert any("kill switch" in r.lower() for r in d.reasons)
    risk.reset_kill_switch()


def test_valid_small_buy_is_approved(fresh_env):
    d = fresh_env["risk"].evaluate_risk("BUY", "AAPL", 1, confidence=0.9)
    assert d.approved is True


def test_execution_enforces_risk_without_check_risk_call(fresh_env):
    """The crux: skip check_risk entirely and place a too-large order.
    Execution must still reject it."""
    execution = fresh_env["execution"]
    broker = fresh_env["broker"]
    price = broker.get_broker().get_quote("AAPL")["ask_price"]
    qty = (50000 / price)  # way over the $5k cap and 20% position limit
    res = execution.place_order("AAPL", "BUY", qty, idempotency_key="k1", confidence=0.99)
    assert res["status"] == "rejected"
    assert "risk_decision" in res


def test_idempotency_prevents_duplicate(fresh_env):
    execution = fresh_env["execution"]
    r1 = execution.place_order("MSFT", "BUY", 1, idempotency_key="dup-key", confidence=0.9)
    r2 = execution.place_order("MSFT", "BUY", 1, idempotency_key="dup-key", confidence=0.9)
    assert r1["status"] == "ok"
    assert r2["status"] == "duplicate"
    assert r2["order"]["order_id"] == r1["order"]["order_id"]


def test_position_size_limit(fresh_env):
    """Repeated buys cannot push a single position past 20% of equity."""
    execution = fresh_env["execution"]
    risk = fresh_env["risk"]
    price = fresh_env["broker"].get_broker().get_quote("NVDA")["bid_price"]
    # 20% of 100k = 20k. Each order capped at 5k notional. Try to reach 25k.
    n = 0
    for i in range(8):
        qty = int(4500 / price) or 1
        res = execution.place_order("NVDA", "BUY", qty, idempotency_key=f"sz-{i}", confidence=0.9)
        if res["status"] == "ok":
            n += 1
        elif res["status"] == "rejected":
            break
    status = risk.get_portfolio_status()
    pos = next((p for p in status["positions"] if p["symbol"] == "NVDA"), None)
    assert pos is not None
    assert pos["market_value"] <= 0.20 * status["equity"] + 1e-6


def test_audit_chain_integrity(fresh_env):
    execution = fresh_env["execution"]
    execution.place_order("AAPL", "BUY", 1, idempotency_key="chain-1", confidence=0.9)
    from trading.storage import get_storage

    assert get_storage().verify_audit_chain()["valid"] is True
