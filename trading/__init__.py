"""Agentic trading bot — layered trading toolkit exposed to Claude via MCP.

Layers:
    data_layer        market data (quotes, bars, order book)
    strategy_layer    XGBoost signal generation
    risk_layer        deterministic, non-bypassable risk guards
    execution_layer   order placement with idempotency + enforced risk re-check
    monitoring_layer  immutable hash-chained audit log + session summary
"""
from __future__ import annotations

__version__ = "0.1.0"
