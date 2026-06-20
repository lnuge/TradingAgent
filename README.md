# Agentic Trading Bot (MCP)

An autonomous trading bot where **Claude is the decision-making agent**, calling
tools over the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).
A local MCP server exposes a **5-layer toolkit**; Claude orchestrates the layers
to assess the market, generate signals, enforce risk, execute orders, and audit
its own behaviour.

> ⚠️ **Paper trading by default.** Live trading sends real orders to Robinhood
> and requires *two* explicit flags. There is no official Robinhood API — live
> mode wraps the unofficial [`robin_stocks`](https://github.com/jmfernandes/robin_stocks)
> library. Trade at your own risk.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Claude (orchestrating agent)│
                    └───────────────┬──────────────┘
                                    │ MCP (stdio)
                    ┌───────────────▼──────────────┐
                    │      server.py (FastMCP)     │
                    └───────────────┬──────────────┘
   ┌───────────┬──────────────┬─────┴───────┬──────────────┬───────────────┐
   ▼           ▼              ▼             ▼              ▼               ▼
 DATA      STRATEGY         RISK        EXECUTION      MONITORING
get_quote  generate_signal  check_risk  place_order    log_trade
get_bars   (XGBoost model)  get_portfolio_status       get_trade_history
get_order_book              reset_kill_switch          get_session_summary
                            cancel_order / get_open_orders
```

| Layer | Tools | Notes |
|-------|-------|-------|
| **1. Data** | `get_quote`, `get_bars`, `get_order_book` | Real-time/historical data. Synthetic in paper mode, `robin_stocks` in live mode. |
| **2. Strategy** | `generate_signal` | XGBoost over RSI / MACD / BB-width / volume z-score → BUY/SELL/HOLD + confidence. Returns the raw feature vector so the agent can reason *on top of* it. |
| **+ Orchestration** | `get_watchlist`, `scan_watchlist` | Scan the whole universe → ranked, sized, risk-checked decisions. Dry-run by default. |
| **3. Risk** | `check_risk`, `get_portfolio_status` | **Deterministic, non-bypassable** hard limits. Not agent judgement. |
| **4. Execution** | `place_order`, `cancel_order`, `get_open_orders` | Re-runs risk checks in code + idempotency keys. |
| **5. Monitoring** | `log_trade`, `get_trade_history`, `get_session_summary` | Immutable, hash-chained audit log. |

### Why the risk layer is *non-bypassable*

The risk limits are enforced in **code**, not by trusting the agent to behave:

1. `place_order` calls `evaluate_risk()` **itself** before submitting anything.
   Even if the agent never calls the `check_risk` tool, an unsafe order is
   still rejected.
2. Breaching the daily drawdown limit **auto-trips the kill switch**, halting
   all trading until an operator clears it.
3. The audit log is **hash-chained** (each row commits to the previous row's
   hash), so any tampering is detectable via `verify_audit_chain()`.

See `tests/test_risk_layer.py` — `test_execution_enforces_risk_without_check_risk_call`
is the proof: it skips `check_risk` entirely and the order is still blocked.

---

## Setup

```bash
pip install -r requirements.txt          # paper mode
pip install -r requirements.txt robin_stocks   # + live mode
cp .env.example .env                      # then edit limits / creds
```

### Pick stocks → scan → trade (the workflow)

This agent doesn't pick "good companies" — it times entries/exits on a
**watchlist you curate** using technical signals. The workflow:

1. **Curate the universe** in `watchlist.txt` (ships with ~17 liquid large-caps
   + ETFs across sectors). Keep names liquid; technicals are noise on microcaps.
2. **Train on real history** (see below) so signals have an edge.
3. **Scan** the watchlist — generates a signal per name, decides entry/exit/hold,
   sizes it, and runs it through the risk layer:
   ```bash
   python scripts/run_scan.py            # dry run — print ranked, risk-checked ideas
   python scripts/run_scan.py --execute  # place the approved orders (paper by default)
   ```
4. **Schedule it** for swing cadence (once or a few times a day) via cron, a
   systemd timer, or Claude Code's `/loop` skill. Or just ask Claude to call the
   `scan_watchlist` tool.

Sizing: a new position targets `TARGET_POSITION_PCT` of equity (default 10%),
capped by the risk layer's `MAX_POSITION_PCT` and per-order notional limit.
Exits trigger on a stop-loss breach or a confident SELL signal on a held name.

### Train the strategy model

The strategy layer works out-of-the-box with a transparent **rule-based
fallback**, but for real signals train the XGBoost model:

```bash
# Real historicals from a FREE source (yfinance), using watchlist.txt — recommended:
python scripts/train_model.py --source yfinance --period 3y --interval 1d

# Specific symbols instead of the watchlist:
python scripts/train_model.py --source yfinance --symbols AAPL,MSFT,NVDA

# End-to-end smoke test on synthetic data (no network/creds needed):
python scripts/train_model.py --synthetic --n-symbols 40
```

The model is saved to `data/signal_model.json`, which the strategy layer loads
automatically.

> **Note:** yfinance reaches `query1/query2.finance.yahoo.com`. In a sandboxed or
> remote environment those hosts must be on the network egress allowlist —
> otherwise run training locally. The synthetic path needs no network.

### Validate it — no need to wait for market open

Paper mode uses a self-contained synthetic market, so you can validate the
whole system **24/7 with zero credentials**:

```bash
python -m pytest tests/ -q          # 1) safety tests (risk, idempotency, kill switch, audit)
python scripts/demo_session.py      # 2) full agent session, end to end
```

The demo runs a complete loop the way Claude would (quote → signal → risk →
execute) and deliberately forces every guard to fire — idempotency replay, the
order-size cap, the no-naked-short rule, and a simulated drawdown that
auto-trips the kill switch — finishing with a session summary and audit
hash-chain check. It writes to an isolated `data/demo/` DB and is repeatable.

| Mode | Data | Fills | Market hours? | Risk |
|------|------|-------|---------------|------|
| **Paper** (default) | synthetic | simulated | ❌ no | none |
| **Paper + live data** (`LIVE_MARKET_DATA=1`) | real Robinhood quotes | simulated | ✅ yes | none |
| **Live** (`TRADING_MODE=live` + `ENABLE_LIVE_TRADING=1`) | real | **real orders** | ✅ yes | real money |

The middle row lets you validate against real quotes with no financial risk —
it needs Robinhood credentials and only returns fresh data during market hours.

---

## Register with Claude Code

Add the server to Claude Code (stdio transport):

```bash
claude mcp add trading-agent -- python /absolute/path/to/TradingAgent/server.py
```

Or in `.mcp.json` / Claude Desktop config:

```json
{
  "mcpServers": {
    "trading-agent": {
      "command": "python",
      "args": ["/absolute/path/to/TradingAgent/server.py"],
      "env": { "TRADING_MODE": "paper" }
    }
  }
}
```

Then ask Claude things like:

> *"Check AAPL: pull a quote, generate a signal, and if it's a confident BUY,
> size a position within risk limits and place the order. Then summarise the
> session."*

Claude will call `get_quote` → `generate_signal` → `check_risk` →
`place_order` → `get_session_summary`, with the risk layer enforcing limits
regardless of what it decides.

---

## Going live (real money)

Live trading is gated behind **both** flags — either one alone stays in paper:

```bash
TRADING_MODE=live
ENABLE_LIVE_TRADING=1
ROBINHOOD_USERNAME=...
ROBINHOOD_PASSWORD=...
ROBINHOOD_MFA=...        # if 2FA enabled
```

`get_portfolio_status` always reports the active mode so you can confirm before
trading.

---

## Risk limits (configurable in `.env`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `MAX_POSITION_PCT` | 0.20 | Max single position as % of equity |
| `MAX_ORDER_VALUE` | 5000 | Per-order notional cap (USD) |
| `STOP_LOSS_PCT` | 0.05 | Per-position stop loss |
| `DAILY_DRAWDOWN_LIMIT_PCT` | 0.03 | Trips the kill switch when breached |
| `MIN_CONFIDENCE` | 0.55 | Minimum model confidence to approve a trade |
| `MAX_OPEN_POSITIONS` | 10 | Max concurrent positions |

## Project layout

```
server.py                 MCP server (FastMCP) — exposes all 15 tools
watchlist.txt             The universe the agent scans (edit this)
trading/
  config.py               env-driven config + paper/live gating
  storage.py              SQLite: hash-chained audit log, orders, paper account
  broker.py               PaperBroker (synthetic) + RobinhoodBroker (live)
  features.py             RSI / MACD / BB-width / volume z-score
  model.py                XGBoost signal model + rule-based fallback
  market_data.py          Free historical data (yfinance) for training
  watchlist.py            Watchlist loader
  scanner.py              Watchlist scanner -> ranked, risk-checked decisions
  data_layer.py           Layer 1
  strategy_layer.py       Layer 2
  risk_layer.py           Layer 3 (non-bypassable guards)
  execution_layer.py      Layer 4 (idempotency + enforced risk)
  monitoring_layer.py     Layer 5
scripts/train_model.py    Offline XGBoost training (yfinance / synthetic)
scripts/run_scan.py       Run one watchlist scan (schedulable)
scripts/demo_session.py   One-command end-to-end paper-mode demo
tests/test_risk_layer.py  Safety-critical tests
```

## Disclaimer

This is an educational/research project. It is **not** financial advice. Markets
are risky; `robin_stocks` is unofficial and may break. Always validate in paper
mode first.
