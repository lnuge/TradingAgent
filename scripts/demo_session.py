"""End-to-end paper-mode demo: a full agent trading session in one command.

Walks through every layer the way Claude would orchestrate them, and forces the
safety mechanics to fire so you can see them working without waiting for market
open (paper mode uses the synthetic market — runs 24/7):

  1. starting portfolio status
  2. data + signal + risk + execution loop over several symbols
  3. idempotency replay (no duplicate order)
  4. oversized order rejected by the risk cap
  5. naked-short rejected
  6. forced daily-drawdown breach -> auto kill switch -> all trading halted
  7. session summary + audit hash-chain verification

Runs against an isolated demo database so it is repeatable. Nothing here
touches real money.

Usage:
    python scripts/demo_session.py
    python scripts/demo_session.py --symbols AAPL,MSFT,TSLA,NVDA
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Configure an isolated, repeatable demo environment BEFORE importing trading.
DEMO_DIR = ROOT / "data" / "demo"
if DEMO_DIR.exists():
    shutil.rmtree(DEMO_DIR)
os.environ.setdefault("TRADING_DATA_DIR", str(DEMO_DIR))
os.environ.setdefault("TRADING_MODE", "paper")
os.environ.setdefault("PAPER_STARTING_CASH", "100000")
os.environ.setdefault("MIN_CONFIDENCE", "0.30")  # let synthetic signals through
os.environ.setdefault("TRADING_SESSION_ID", "demo-session")

from trading import (  # noqa: E402
    data_layer,
    execution_layer,
    monitoring_layer,
    risk_layer,
    strategy_layer,
)
from trading.storage import get_storage  # noqa: E402


def header(n: int, title: str) -> None:
    print(f"\n{'='*70}\n  STEP {n}: {title}\n{'='*70}")


def show_portfolio() -> dict:
    p = risk_layer.get_portfolio_status()
    print(
        f"  mode={p['mode']}\n  equity=${p['equity']:,.2f}  cash=${p['cash']:,.2f}  "
        f"positions={p['open_positions']}  day_pnl=${p['day_pnl']:,.2f}  "
        f"drawdown={p['day_drawdown_pct']:.2%}  kill_switch={p['kill_switch_engaged']}"
    )
    for pos in p["positions"]:
        print(
            f"    • {pos['symbol']}: {pos['quantity']:g} @ ${pos['avg_price']:.2f} "
            f"(last ${pos['last_price']:.2f}, uPnL ${pos['unrealized_pnl']:,.2f}, "
            f"stop ${pos['stop_loss_price']:.2f})"
        )
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,TSLA")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    header(1, "Starting portfolio")
    show_portfolio()

    header(2, "Agent loop: quote -> signal -> risk -> execute")
    for i, sym in enumerate(symbols):
        q = data_layer.get_quote(sym)
        sig = strategy_layer.generate_signal(sym)
        print(
            f"\n  {sym}: last=${q['last_price']:.2f}  "
            f"signal={sig['signal']} conf={sig['confidence']:.2f} ({sig['model_source']})"
        )
        print(f"    {sig['interpretation']}")
        if sig["signal"] == "HOLD":
            print("    -> HOLD: no order.")
            continue
        # Size a modest position (~$2k) within limits.
        qty = max(1, int(2000 / q["last_price"]))
        decision = risk_layer.check_risk(sig["signal"], sym, qty, confidence=sig["confidence"])
        if not decision["approved"]:
            print(f"    -> risk REJECTED: {decision['reasons'][0]}")
            continue
        res = execution_layer.place_order(
            sym, sig["signal"], qty, idempotency_key=f"loop-{sym}-{i}", confidence=sig["confidence"]
        )
        print(f"    -> place_order: {res['status']}", end="")
        if res["status"] == "ok":
            o = res["order"]
            print(f"  ({o['side']} {o['quantity']:g} @ ${o['price']:.2f}, ${o['notional']:,.2f})")
        else:
            print()

    header(3, "Idempotency: the same key never places a duplicate")
    first = symbols[0]
    key = "idem-demo"
    r1 = execution_layer.place_order(first, "BUY", 2, idempotency_key=key, confidence=0.9)
    r2 = execution_layer.place_order(first, "BUY", 2, idempotency_key=key, confidence=0.9)
    print(f"  first send  status={r1['status']}")
    print(f"  retry (same key) status={r2['status']} (expected 'duplicate' — original returned, nothing re-sent)")

    header(4, "Risk cap: oversized order (skips check_risk entirely)")
    big = execution_layer.place_order(first, "BUY", 100_000, idempotency_key="oversize", confidence=0.99)
    print(f"  status={big['status']}")
    if big.get("risk_decision"):
        print(f"  reason: {big['risk_decision']['reasons'][0]}")

    header(5, "No naked shorting: sell a symbol we don't hold")
    short = execution_layer.place_order("ZZZZ", "SELL", 5, idempotency_key="short", confidence=0.99)
    print(f"  status={short['status']}")
    if short.get("risk_decision"):
        print(f"  reason: {short['risk_decision']['reasons'][0]}")

    header(6, "Drawdown breach -> auto kill switch")
    storage = get_storage()
    equity = risk_layer.get_portfolio_status()["equity"]
    # Simulate prior intraday losses: pretend the day opened 5% higher than now.
    storage.set_day_start_equity(equity / (1 - 0.05))
    print(f"  Simulated day-start equity so current drawdown ≈ 5% (> 3% limit).")
    halted = execution_layer.place_order(first, "BUY", 1, idempotency_key="after-dd", confidence=0.99)
    print(f"  order status={halted['status']}")
    if halted.get("risk_decision"):
        for r in halted["risk_decision"]["reasons"]:
            print(f"    reason: {r}")
    print(f"  kill switch now engaged: {risk_layer.is_kill_switch_engaged()}")

    header(7, "Session summary + audit integrity")
    summ = monitoring_layer.get_session_summary()
    o = summ["orders"]
    print(
        f"  orders: total={o['total']} executed={o['executed']} "
        f"risk_rejected={o['risk_rejected']} buys={o['buys']} sells={o['sells']}"
    )
    print(f"  event_counts: {summ['event_counts']}")
    print(f"  audit chain valid: {summ['audit_chain_valid']}  ({summ['audit_events_total']} events)")

    # Clean up so trading can resume on the next run.
    risk_layer.reset_kill_switch()
    print(f"\n  (kill switch reset for next run)\n  Demo DB: {DEMO_DIR}")


if __name__ == "__main__":
    main()
