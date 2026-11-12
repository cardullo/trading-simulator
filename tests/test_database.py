"""Unit tests for database integration."""

import os
import tempfile
import uuid

import pytest

from trading_simulator.config import SimulationConfig
from trading_simulator.database import DatabaseManager
from trading_simulator.engine import SimulationEngine


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    def _cfg(self, **overrides):
        base = {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            "activity": {"trades_per_day": 5, "win_rate": 0.55, "days": 5},
            "seed": 42,
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base:
                base[key].update(value)
            else:
                base[key] = value
        return SimulationConfig(**base)

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    def test_save_and_retrieve_run(self, temp_db):
        """Test saving a run and retrieving it."""
        db = DatabaseManager(db_path=temp_db)
        cfg = self._cfg()
        engine = SimulationEngine(cfg, seed=42)
        result = engine.run(days=5)

        run_id = str(uuid.uuid4())[:8]
        db.save_run(run_id, cfg, result)

        # Retrieve
        stored = db.get_run(run_id)
        assert stored is not None
        assert stored["run_id"] == run_id
        assert stored["initial_balance"] == 1000.0
        assert stored["final_balance"] == result.final_balance
        assert stored["compounding_mode"] == "standard"

    def test_save_and_retrieve_trades(self, temp_db):
        """Test saving trades and retrieving them."""
        db = DatabaseManager(db_path=temp_db)
        cfg = self._cfg(activity={"trades_per_day": 3, "days": 2})
        engine = SimulationEngine(cfg, seed=42)
        result = engine.run(days=2)

        run_id = str(uuid.uuid4())[:8]
        db.save_run(run_id, cfg, result)

        trades = db.get_trades(run_id)
        assert len(trades) == 6  # 3 trades/day × 2 days
        assert trades[0]["run_id"] == run_id
        assert trades[0]["trade_number"] == 1
        assert trades[0]["outcome"] in ("WIN", "LOSS")

    def test_list_runs(self, temp_db):
        """Test listing all runs."""
        db = DatabaseManager(db_path=temp_db)

        # Initially empty
        assert db.list_runs() == []

        # Save two runs
        cfg = self._cfg()
        engine = SimulationEngine(cfg, seed=42)
        result = engine.run(days=1)

        run_id_1 = str(uuid.uuid4())[:8]
        db.save_run(run_id_1, cfg, result)

        run_id_2 = str(uuid.uuid4())[:8]
        db.save_run(run_id_2, cfg, result)

        runs = db.list_runs()
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert run_id_1 in run_ids
        assert run_id_2 in run_ids

    def test_get_nonexistent_run(self, temp_db):
        """Test retrieving a nonexistent run returns None."""
        db = DatabaseManager(db_path=temp_db)
        assert db.get_run("nonexistent") is None

    def test_get_trades_for_nonexistent_run(self, temp_db):
        """Test retrieving trades for a nonexistent run returns empty list."""
        db = DatabaseManager(db_path=temp_db)
        trades = db.get_trades("nonexistent")
        assert trades == []

    def test_database_creates_directory(self):
        """Test that DatabaseManager creates parent directories."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        # Use a path in a nonexistent subdirectory
        nested_path = os.path.join(tempfile.gettempdir(), "trading_sim_test_db", "subdir", "test.db")
        # Clean up if exists
        import shutil
        shutil.rmtree(os.path.dirname(nested_path), ignore_errors=True)

        db = DatabaseManager(db_path=nested_path)
        assert db.db_path.exists()
        # Cleanup
        shutil.rmtree(os.path.dirname(nested_path), ignore_errors=True)

    def test_end_to_end_run_save_export_cycle(self, temp_db):
        """Test full cycle: run simulation → save → retrieve → verify data integrity."""
        db = DatabaseManager(db_path=temp_db)
        cfg = self._cfg(
            risk={"compounding_mode": "high_water_mark"},
            activity={"trades_per_day": 4, "win_rate": 0.6, "days": 3},
        )
        engine = SimulationEngine(cfg, seed=123)
        result = engine.run(days=3)

        run_id = str(uuid.uuid4())[:8]
        db.save_run(run_id, cfg, result)

        # Retrieve and verify
        stored = db.get_run(run_id)
        assert stored["final_balance"] == result.final_balance
        assert stored["compounding_mode"] == "high_water_mark"

        trades = db.get_trades(run_id)
        assert len(trades) == 12  # 4 trades/day × 3 days

        # Verify trade data integrity
        for i, t in enumerate(trades):
            assert t["trade_number"] == i + 1
            assert t["day"] >= 1
            assert t["day"] <= 3
            assert t["lots"] > 0
            assert t["outcome"] in ("WIN", "LOSS")
            assert t["balance_after"] > 0