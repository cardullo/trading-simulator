"""SQLite persistence layer for simulation runs and trade logs."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class DatabaseManager:
    """Manages SQLite database for persisting simulation runs."""

    def __init__(self, db_path: str | Path = "data/runs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    initial_balance REAL NOT NULL,
                    risk_per_trade REAL NOT NULL,
                    sl_pips REAL NOT NULL,
                    tp_pips REAL NOT NULL,
                    fee_pips REAL NOT NULL,
                    trades_per_day INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    days INTEGER NOT NULL,
                    compounding_mode TEXT NOT NULL,
                    seed INTEGER,
                    created_at TEXT NOT NULL,
                    final_balance REAL
                );
                CREATE TABLE IF NOT EXISTS trade_logs (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    trade_number INTEGER NOT NULL,
                    day INTEGER NOT NULL,
                    lots REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    pips_gross REAL NOT NULL,
                    pips_net REAL NOT NULL,
                    fee_pips REAL NOT NULL,
                    profit_euros REAL NOT NULL,
                    balance_before REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    position_id TEXT DEFAULT '',
                    fill_index INTEGER DEFAULT 0,
                    total_fills INTEGER DEFAULT 1,
                    pyramid_fill_index INTEGER DEFAULT 0,
                    pyramid_total_fills INTEGER DEFAULT 1,
                    entry_price REAL DEFAULT 0.0,
                    position_avg_price REAL DEFAULT 0.0,
                    trailing_stop_active INTEGER DEFAULT 0,
                    trailing_stop_pips REAL DEFAULT 0.0,
                    trailing_triggered INTEGER DEFAULT 0,
                    tp_level_hit INTEGER DEFAULT 1,
                    tp_level_pips REAL DEFAULT 0.0,
                    timestamp TEXT DEFAULT '',
                    FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    balance REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    peak_balance REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
                );
                """
            )
            # Migration check for existing trade_logs tables
            cursor = conn.cursor()
            existing_cols = {col[1] for col in cursor.execute("PRAGMA table_info(trade_logs)").fetchall()}
            m11_cols = [
                ("position_id", "TEXT DEFAULT ''"),
                ("fill_index", "INTEGER DEFAULT 0"),
                ("total_fills", "INTEGER DEFAULT 1"),
                ("pyramid_fill_index", "INTEGER DEFAULT 0"),
                ("pyramid_total_fills", "INTEGER DEFAULT 1"),
                ("entry_price", "REAL DEFAULT 0.0"),
                ("position_avg_price", "REAL DEFAULT 0.0"),
                ("trailing_stop_active", "INTEGER DEFAULT 0"),
                ("trailing_stop_pips", "REAL DEFAULT 0.0"),
                ("trailing_triggered", "INTEGER DEFAULT 0"),
                ("tp_level_hit", "INTEGER DEFAULT 1"),
                ("tp_level_pips", "REAL DEFAULT 0.0"),
                ("timestamp", "TEXT DEFAULT ''"),
            ]
            for col_name, col_type in m11_cols:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE trade_logs ADD COLUMN {col_name} {col_type}")

    def save_run(self, run_id: str, config: Any, result: Any) -> None:
        """Save a simulation run and its trades to the database."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                INSERT OR REPLACE INTO simulation_runs
                (run_id, initial_balance, risk_per_trade, sl_pips, tp_pips, fee_pips,
                 trades_per_day, win_rate, days, compounding_mode, seed, created_at, final_balance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config.account.initial_balance,
                    config.account.risk_per_trade,
                    config.risk.sl_pips,
                    config.risk.tp_pips,
                    config.risk.fee_pips,
                    config.activity.trades_per_day,
                    config.activity.win_rate,
                    config.activity.days,
                    config.risk.compounding_mode.value,
                    config.seed,
                    datetime.now().isoformat(),
                    result.final_balance,
                ),
            )

            # Batch insert trades using executemany
            trade_rows = [
                (
                    run_id,
                    t.trade_number,
                    t.day,
                    t.lots,
                    t.risk_amount,
                    t.pips_gross,
                    t.pips_net,
                    t.fee_pips,
                    t.profit_euros,
                    t.balance_before,
                    t.balance_after,
                    t.outcome,
                    getattr(t, "position_id", ""),
                    getattr(t, "fill_index", 0),
                    getattr(t, "total_fills", 1),
                    getattr(t, "pyramid_fill_index", 0),
                    getattr(t, "pyramid_total_fills", 1),
                    getattr(t, "entry_price", 0.0),
                    getattr(t, "position_avg_price", 0.0),
                    1 if getattr(t, "trailing_stop_active", False) else 0,
                    getattr(t, "trailing_stop_pips", 0.0),
                    1 if getattr(t, "trailing_triggered", False) else 0,
                    1 if getattr(t, "tp_level_hit", True) else 0,
                    getattr(t, "tp_level_pips", 0.0),
                    t.timestamp.isoformat() if hasattr(t, "timestamp") and hasattr(t.timestamp, "isoformat") else "",
                )
                for t in result.trades
            ]
            if trade_rows:
                conn.executemany(
                    """
                    INSERT INTO trade_logs
                    (run_id, trade_number, day, lots, risk_amount, pips_gross, pips_net,
                     fee_pips, profit_euros, balance_before, balance_after, outcome,
                     position_id, fill_index, total_fills, pyramid_fill_index, pyramid_total_fills,
                     entry_price, position_avg_price, trailing_stop_active, trailing_stop_pips,
                     trailing_triggered, tp_level_hit, tp_level_pips, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    trade_rows,
                )

            # Batch insert equity snapshots using executemany
            equity_rows = [
                (
                    run_id,
                    e.timestamp.isoformat(),
                    e.balance,
                    e.drawdown_pct,
                    e.peak_balance,
                )
                for e in result.equity_curve
            ]
            if equity_rows:
                conn.executemany(
                    """
                    INSERT INTO equity_snapshots
                    (run_id, timestamp, balance, drawdown_pct, peak_balance)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    equity_rows,
                )

            conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a simulation run by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM simulation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def list_runs(self) -> list[dict[str, Any]]:
        """List all simulation runs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM simulation_runs ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trades(self, run_id: str) -> list[dict[str, Any]]:
        """Retrieve all trades for a run."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_logs WHERE run_id = ? ORDER BY trade_number",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]