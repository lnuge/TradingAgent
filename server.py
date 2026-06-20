"""MCP server exposing the 5-layer trading toolkit to Claude.

Run it as a local stdio MCP server and register it with Claude Code (see
README). Claude is the orchestrating agent: it calls these tools to assess the
market, generate signals, check risk, place orders, and review its own history.

The risk layer tools are hard constraints — ``place_order`` re-runs the risk
checks in code, so the agent cannot bypass them by skipping ``check_risk``.

Defaults to PAPER mode. Live trading requires TRADING_MODE=live AND
ENABLE_LIVE_TRADING=1.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from trading import data_layer, execution_layer, monitoring_layer, risk_layer, strategy_layer
from trading.config import get_config
from trading.context import SESSION_ID

mcp = FastMCP("trading-agent")


# === DATA LAYER ============================================================
@mcp.tool()
def get_quote(symbol: str) -> dict[str, Any]:
    """Get the latest quote (last/bid/ask/prev-close) for a stock symbol."""
    return data_layer.get_quote(symbol)


@mcp.tool()
def get_bars(symbol: str, interval: str = "5minute", span: str = "day") -> dict[str, Any]:
    """Get historical OHLCV bars. interval: 5minute/10minute/hour/day. span: day/week/month."""
    return data_layer.get_bars(symbol, interval=interval, span=span)


@mcp.tool()
def get_order_book(symbol: str, depth: int = 5) -> dict[str, Any]:
    """Get the order book (top bids/asks) for a symbol."""
    return data_layer.get_order_book(symbol, depth=depth)


# === STRATEGY LAYER ========================================================
@mcp.tool()
def generate_signal(symbol: str, interval: str = "5minute", span: str = "day") -> dict[str, Any]:
    """Run the XGBoost model on RSI/MACD/BB-width/volume-z features and return
    BUY/SELL/HOLD + confidence + the raw feature vector. Layer your own
    qualitative reasoning on top; do not follow it blindly."""
    return strategy_layer.generate_signal(symbol, interval=interval, span=span)


# === RISK LAYER (hard constraints) =========================================
@mcp.tool()
def check_risk(
    signal: str,
    symbol: str,
    quantity: float,
    confidence: float | None = None,
    price: float | None = None,
) -> dict[str, Any]:
    """Deterministically check a proposed trade against hard risk limits
    (kill switch, drawdown, position size, order cap, cash, no naked shorting).
    MUST be called before place_order. Returns approved=true/false with reasons.
    Note: place_order also re-runs these checks, so they cannot be bypassed."""
    return risk_layer.check_risk(signal, symbol, quantity, confidence=confidence, price=price)


@mcp.tool()
def get_portfolio_status() -> dict[str, Any]:
    """Get current portfolio status: cash, equity, positions with unrealized
    P&L and stop levels, day drawdown, kill-switch state, and active limits."""
    return risk_layer.get_portfolio_status()


@mcp.tool()
def reset_kill_switch() -> dict[str, Any]:
    """Manually clear the kill switch (operator action). Trading resumes only
    after the underlying drawdown condition no longer holds."""
    return {"reset": risk_layer.reset_kill_switch()}


# === EXECUTION LAYER =======================================================
@mcp.tool()
def place_order(
    symbol: str,
    side: str,
    quantity: float,
    idempotency_key: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Place a market order (side: BUY/SELL). Requires a unique idempotency_key
    (retries with the same key never duplicate). Risk checks are re-run in code
    and an unsafe order is rejected regardless of prior check_risk calls."""
    return execution_layer.place_order(symbol, side, quantity, idempotency_key, confidence=confidence)


@mcp.tool()
def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel an open order by id."""
    return execution_layer.cancel_order(order_id)


@mcp.tool()
def get_open_orders() -> dict[str, Any]:
    """List open broker orders and all orders recorded this session."""
    return execution_layer.get_open_orders()


# === MONITORING LAYER ======================================================
@mcp.tool()
def log_trade(event_type: str, detail: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    """Append an annotation (rationale, observation) to the immutable audit log."""
    return monitoring_layer.log_trade(event_type, detail, symbol=symbol)


@mcp.tool()
def get_trade_history(limit: int = 50, event_type: str | None = None) -> dict[str, Any]:
    """Query this session's audit trail (optionally filtered by event_type)."""
    return monitoring_layer.get_trade_history(limit=limit, event_type=event_type)


@mcp.tool()
def get_session_summary() -> dict[str, Any]:
    """Get an aggregated summary of the session: event counts, order stats,
    portfolio state, and whether the audit hash-chain is intact."""
    return monitoring_layer.get_session_summary()


if __name__ == "__main__":
    cfg = get_config()
    print(f"[trading-agent] session={SESSION_ID} mode={cfg.describe_mode()}", flush=True)
    mcp.run()
