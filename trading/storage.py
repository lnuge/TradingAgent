"""Persistence layer: SQLite for the immutable audit log, order book, and the
paper-trading account/positions.

The audit log is *append-only and hash-chained*: every row stores the SHA-256
hash of the previous row plus its own payload, so any after-the-fact edit or
deletion breaks the chain and is detectable via :func:`verify_audit_chain`.
This is what makes the monitoring layer tamper-evident.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import get_config

_GENESIS_HASH = "0" * 64


class Storage:
    def __init__(self, db_path: Path | None = None):
        cfg = get_config()
        self.db_path = Path(db_path or cfg.db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()
        self._ensure_paper_account(cfg.paper_starting_cash)

    # -- schema --------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL NOT NULL,
                    session_id  TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    symbol      TEXT,
                    payload     TEXT NOT NULL,
                    prev_hash   TEXT NOT NULL,
                    hash        TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    idempotency_key TEXT PRIMARY KEY,
                    order_id        TEXT NOT NULL,
                    ts              REAL NOT NULL,
                    session_id      TEXT,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    price           REAL,
                    notional        REAL,
                    status          TEXT NOT NULL,
                    mode            TEXT NOT NULL,
                    detail          TEXT
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol     TEXT PRIMARY KEY,
                    quantity   REAL NOT NULL,
                    avg_price  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS account (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    cash            REAL NOT NULL,
                    starting_equity REAL NOT NULL,
                    day_start_equity REAL NOT NULL,
                    day_start_ts    REAL NOT NULL
                );
                """
            )

    def _ensure_paper_account(self, starting_cash: float) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id FROM account WHERE id = 1").fetchone()
            if row is None:
                now = time.time()
                self._conn.execute(
                    "INSERT INTO account (id, cash, starting_equity, day_start_equity, day_start_ts)"
                    " VALUES (1, ?, ?, ?, ?)",
                    (starting_cash, starting_cash, starting_cash, now),
                )

    # -- audit log -----------------------------------------------------------
    def append_audit(
        self, session_id: str, event_type: str, payload: dict[str, Any], symbol: str | None = None
    ) -> dict[str, Any]:
        with self._lock, self._conn:
            prev = self._conn.execute(
                "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev["hash"] if prev else _GENESIS_HASH
            ts = time.time()
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(
                f"{prev_hash}|{ts}|{session_id}|{event_type}|{symbol}|{payload_json}".encode()
            ).hexdigest()
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, session_id, event_type, symbol, payload, prev_hash, hash)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, session_id, event_type, symbol, payload_json, prev_hash, digest),
            )
            return {
                "id": cur.lastrowid,
                "ts": ts,
                "event_type": event_type,
                "symbol": symbol,
                "hash": digest,
            }

    def get_audit(
        self, session_id: str | None = None, event_type: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM audit_log"
        clauses, args = [], []
        if session_id:
            clauses.append("session_id = ?")
            args.append(session_id)
        if event_type:
            clauses.append("event_type = ?")
            args.append(event_type)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def verify_audit_chain(self) -> dict[str, Any]:
        """Recompute the hash chain end-to-end to detect tampering."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        prev_hash = _GENESIS_HASH
        for r in rows:
            expected = hashlib.sha256(
                f"{prev_hash}|{r['ts']}|{r['session_id']}|{r['event_type']}|{r['symbol']}|{r['payload']}".encode()
            ).hexdigest()
            if expected != r["hash"] or r["prev_hash"] != prev_hash:
                return {"valid": False, "broken_at_id": r["id"], "total_rows": len(rows)}
            prev_hash = r["hash"]
        return {"valid": True, "broken_at_id": None, "total_rows": len(rows)}

    # -- orders / idempotency ------------------------------------------------
    def get_order_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return dict(row) if row else None

    def record_order(self, order: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO orders (idempotency_key, order_id, ts, session_id, symbol,"
                " side, quantity, price, notional, status, mode, detail)"
                " VALUES (:idempotency_key, :order_id, :ts, :session_id, :symbol, :side, :quantity,"
                " :price, :notional, :status, :mode, :detail)",
                order,
            )

    def get_orders(self, session_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        q, clauses, args = "SELECT * FROM orders", [], []
        if session_id:
            clauses.append("session_id = ?")
            args.append(session_id)
        if status:
            clauses.append("status = ?")
            args.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY ts DESC"
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    # -- paper account + positions ------------------------------------------
    def get_account(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._conn.execute("SELECT * FROM account WHERE id = 1").fetchone())

    def set_cash(self, cash: float) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE account SET cash = ? WHERE id = 1", (cash,))

    def roll_day_if_needed(self) -> None:
        """Reset the day-start equity marker once per UTC day."""
        acct = self.get_account()
        if time.time() - acct["day_start_ts"] >= 86_400:
            equity = acct["cash"] + self.positions_market_value_fallback()
            with self._lock, self._conn:
                self._conn.execute(
                    "UPDATE account SET day_start_equity = ?, day_start_ts = ? WHERE id = 1",
                    (equity, time.time()),
                )

    def positions_market_value_fallback(self) -> float:
        """Cost-basis valuation (used only when live quotes are unavailable)."""
        total = 0.0
        for p in self.get_positions():
            total += p["quantity"] * p["avg_price"]
        return total

    def get_positions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM positions").fetchall()]

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_position(self, symbol: str, quantity: float, avg_price: float) -> None:
        with self._lock, self._conn:
            if quantity <= 1e-9:
                self._conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO positions (symbol, quantity, avg_price) VALUES (?, ?, ?)",
                    (symbol, quantity, avg_price),
                )


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
