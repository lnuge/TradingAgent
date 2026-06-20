"""Central configuration for the agentic trading bot.

All tunables are read from environment variables so the same code runs in
paper (default) and live modes without edits. Live trading is gated behind a
*double* flag (`TRADING_MODE=live` AND `ENABLE_LIVE_TRADING=1`) to make it
impossible to go live by accident.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Project paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRADING_DATA_DIR", ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk constraints. These are enforced in code, not by the agent."""

    # Max fraction of total equity allowed in a single position (notional).
    max_position_pct: float = field(default_factory=lambda: _env_float("MAX_POSITION_PCT", 0.20))
    # Absolute ceiling on a single order's notional value (USD).
    max_order_value: float = field(default_factory=lambda: _env_float("MAX_ORDER_VALUE", 5000.0))
    # Per-position stop loss as a fraction below average cost.
    stop_loss_pct: float = field(default_factory=lambda: _env_float("STOP_LOSS_PCT", 0.05))
    # Daily drawdown limit (fraction of day-start equity). Breaching it trips
    # the kill switch for the rest of the session.
    daily_drawdown_limit_pct: float = field(
        default_factory=lambda: _env_float("DAILY_DRAWDOWN_LIMIT_PCT", 0.03)
    )
    # Minimum model confidence required for the risk layer to approve a trade.
    min_confidence: float = field(default_factory=lambda: _env_float("MIN_CONFIDENCE", 0.55))
    # Max number of distinct open positions at once.
    max_open_positions: int = field(
        default_factory=lambda: int(_env_float("MAX_OPEN_POSITIONS", 10))
    )


@dataclass(frozen=True)
class Config:
    mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper").strip().lower())
    enable_live: bool = field(default_factory=lambda: _env_bool("ENABLE_LIVE_TRADING", False))
    # Hybrid: pull REAL market data from Robinhood but simulate fills (paper
    # account). Lets you validate against live quotes with zero financial risk.
    # Requires Robinhood credentials and is subject to market hours.
    live_data: bool = field(default_factory=lambda: _env_bool("LIVE_MARKET_DATA", False))

    # Paper account seed equity.
    paper_starting_cash: float = field(
        default_factory=lambda: _env_float("PAPER_STARTING_CASH", 100_000.0)
    )

    db_path: Path = field(default_factory=lambda: DATA_DIR / "trading.db")
    model_path: Path = field(default_factory=lambda: DATA_DIR / "signal_model.json")
    kill_switch_path: Path = field(default_factory=lambda: DATA_DIR / "KILL_SWITCH")

    risk: RiskLimits = field(default_factory=RiskLimits)

    @property
    def is_live(self) -> bool:
        """Live trading requires BOTH flags. Any other combination is paper."""
        return self.mode == "live" and self.enable_live

    def describe_mode(self) -> str:
        if self.is_live:
            return "LIVE (real money — orders sent to Robinhood)"
        if self.live_data:
            return "PAPER FILLS + LIVE DATA (real Robinhood quotes, simulated execution)"
        if self.mode == "live" and not self.enable_live:
            return "PAPER (TRADING_MODE=live ignored: ENABLE_LIVE_TRADING not set)"
        return "PAPER (synthetic market, simulated execution, no real orders)"


# Singleton-style accessor so every layer shares one config instance.
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
