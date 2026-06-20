"""Run one watchlist scan from the command line.

Designed to be scheduled (cron, a systemd timer, or Claude Code's /loop skill)
so the agent periodically reviews the universe and acts. Paper-safe by default;
``--execute`` places the risk-approved orders.

Examples:
    python scripts/run_scan.py                 # dry run, print opportunities
    python scripts/run_scan.py --execute       # place approved orders (paper unless live env)
    python scripts/run_scan.py --symbols AAPL,MSFT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.config import get_config  # noqa: E402
from trading.scanner import scan_watchlist  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Place risk-approved orders.")
    ap.add_argument("--symbols", default="", help="Override watchlist (comma-separated).")
    ap.add_argument("--interval", default="day")
    ap.add_argument("--span", default="month")
    ap.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    result = scan_watchlist(
        symbols=symbols, execute=args.execute, interval=args.interval, span=args.span
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    cfg = get_config()
    print(f"\n=== Watchlist scan ({cfg.describe_mode()}) ===")
    print(
        f"equity=${result['equity']:,.2f}  cash=${result['cash']:,.2f}  "
        f"positions={result['open_positions']}  scanned={result['symbols_scanned']}  "
        f"approved={result['approved_count']}  executed={result['executed_count']}"
    )
    if result.get("opportunities"):
        print("\nOpportunities (ranked):")
        for o in result["opportunities"]:
            flag = "✅" if o["risk_approved"] else "⛔"
            ex = f"  -> {o.get('exec_status','')}" if o.get("executed") is not None and args.execute else ""
            print(
                f"  {flag} {o['action'].upper():5s} {o['side']} {o['quantity']:g} {o['symbol']:6s} "
                f"@ ${o['last_price']:.2f}  conf={o['confidence']:.2f}  ${o['notional']:,.0f}{ex}"
            )
            if not o["risk_approved"]:
                print(f"       risk: {o['risk_reasons'][0]}")
    else:
        print("\nNo actionable opportunities this scan.")
    if result.get("holds"):
        print(f"\nHolding/neutral: {', '.join(h['symbol'] for h in result['holds'])}")
    if result.get("errors"):
        print(f"\nErrors: {result['errors']}")


if __name__ == "__main__":
    main()
