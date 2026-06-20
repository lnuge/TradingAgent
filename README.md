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

### Train the strategy model

The strategy layer works out-of-the-box with a transparent **rule-based
fallback**, but for real signals train the XGBoost model:

```bash
# End-to-end smoke test on synthetic data (no creds needed):
python scripts/train_model.py --synthetic --n-symbols 40

# Real historicals (requires live Robinhood creds in .env):
python scripts/train_model.py --symbols AAPL,MSFT,NVDA --span month
```

The model is saved to `data/signal_model.json`, which the strategy layer loads
automatically.

### Run the tests

```bash
python -m pytest tests/ -q
```

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
server.py                 MCP server (FastMCP) — exposes all 13 tools
trading/
  config.py               env-driven config + paper/live gating
  storage.py              SQLite: hash-chained audit log, orders, paper account
  broker.py               PaperBroker (synthetic) + RobinhoodBroker (live)
  features.py             RSI / MACD / BB-width / volume z-score
  model.py                XGBoost signal model + rule-based fallback
  data_layer.py           Layer 1
  strategy_layer.py       Layer 2
  risk_layer.py           Layer 3 (non-bypassable guards)
  execution_layer.py      Layer 4 (idempotency + enforced risk)
  monitoring_layer.py     Layer 5
scripts/train_model.py    Offline XGBoost training
tests/test_risk_layer.py  Safety-critical tests
```

## Disclaimer

This is an educational/research project. It is **not** financial advice. Markets
are risky; `robin_stocks` is unofficial and may break. Always validate in paper
mode first.
