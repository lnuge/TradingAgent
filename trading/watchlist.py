"""Watchlist loading — the universe of symbols the agent scans.

Reads ``watchlist.txt`` at the project root (override with ``WATCHLIST_PATH``).
Format: one symbol per line, optional ``, sector`` label; ``#`` comments and
blank lines ignored.
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT_DIR


def watchlist_path() -> Path:
    return Path(os.getenv("WATCHLIST_PATH", ROOT_DIR / "watchlist.txt"))


def load_watchlist() -> list[str]:
    """Return the list of symbols (uppercased, de-duplicated, order-preserving)."""
    path = watchlist_path()
    if not path.exists():
        return []
    seen: set[str] = set()
    symbols: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbol = line.split(",")[0].strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def load_watchlist_with_sectors() -> list[dict[str, str]]:
    """Return [{symbol, sector}] for display/grouping."""
    path = watchlist_path()
    if not path.exists():
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        symbol = parts[0].upper()
        sector = parts[1] if len(parts) > 1 else ""
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append({"symbol": symbol, "sector": sector})
    return out
