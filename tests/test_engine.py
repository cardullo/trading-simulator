"""Unit tests for the simulation engine."""

import pytest

from trading_simulator.config import SimulationConfig
from trading_simulator.engine import SimulationEngine, SimulationResult


def _cfg(**overrides):
    """Build a SimulationConfig with sensible defaults and optional overrides."""
    base = {
        "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
        "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
        "activity": {"trades_per_day": 10, "win_rate": 0.55, "days": 5},
        "seed": 42,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base:
            base[key].update(value)
        else:
            base[key] = value
    return SimulationConfig(**base)


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_engine_initialization(self):
        cfg = _cfg(account={"initial_balance": 1000.0, "risk_per_trade": 0.01}, seed=42)
        engine = SimulationEngine(cfg, seed=42)
        assert engine.account.balance == 1000.0
        assert engine.account.initial_balance == 1000.0
        assert engine.account.peak_balance == 1000.0
        assert engine.seed == 42
        assert len(engine.trades) == 0
        assert len(engine.equity_curve) == 0

    def test_engine_run_returns_result(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 5, "win_rate": 0.55, "days": 5},
            seed=42,
        )
        engine = SimulationEngine(cfg, seed=42)
        result = engine.run(days=5)
        assert isinstance(result, SimulationResult)
        assert result.initial_balance == 1000.0
        assert result.final_balance > 0
        assert len(result.trades) == 25
        assert len(result.equity_curve) == 5

    def test_deterministic_seed(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 10, "win_rate": 0.55, "days": 10},
            seed=42,
        )
        result_a = SimulationEngine(cfg, seed=42).run(days=10)
        result_b = SimulationEngine(cfg, seed=42).run(days=10)
        assert result_a.final_balance == result_b.final_balance
        assert [t.profit_euros for t in result_a.trades] == [t.profit_euros for t in result_b.trades]

    def test_different_seeds_produce_different_results(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 10, "win_rate": 0.5, "days": 30},
            seed=42,
        )
        result_a = SimulationEngine(cfg, seed=42).run(days=30)
        result_b = SimulationEngine(cfg, seed=99).run(days=30)
        assert result_a.final_balance != result_b.final_balance

    def test_win_rate_100_percent(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 5, "win_rate": 1.0, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        assert all(t.outcome == "WIN" for t in result.trades)
        assert result.final_balance > result.initial_balance
        assert result.win_rate_actual == 1.0

    def test_win_rate_0_percent(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 5, "win_rate": 0.0, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        assert all(t.outcome == "LOSS" for t in result.trades)
        assert result.final_balance < result.initial_balance
        assert result.win_rate_actual == 0.0

    def test_fee_applied_to_every_trade(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 2.0},
            activity={"trades_per_day": 3, "win_rate": 0.5, "days": 2},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=2)
        assert all(t.fee_pips == 2.0 for t in result.trades)
        assert all(t.pips_net == t.pips_gross - 2.0 for t in result.trades)

    def test_compounding_none_uses_fixed_risk_base(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0, "compounding_mode": "none"},
            activity={"trades_per_day": 5, "win_rate": 0.55, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        lot_sizes = {t.lots for t in result.trades}
        assert len(lot_sizes) == 1

    def test_compounding_standard_changes_lot_size(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.05},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0, "compounding_mode": "standard"},
            activity={"trades_per_day": 10, "win_rate": 0.8, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        lot_sizes = {t.lots for t in result.trades}
        assert len(lot_sizes) > 1

    def test_profit_factor_calculation(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 5, "win_rate": 0.5, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        wins = [t for t in result.trades if t.outcome == "WIN"]
        losses = [t for t in result.trades if t.outcome == "LOSS"]
        gross_profit = sum(t.profit_euros for t in wins)
        gross_loss = abs(sum(t.profit_euros for t in losses))
        expected_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        assert abs(result.profit_factor - expected_pf) < 1e-6

    def test_total_return_percentage(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 5, "win_rate": 0.55, "days": 5},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        expected_return = (result.final_balance - result.initial_balance) / result.initial_balance * 100
        assert abs(result.total_return_pct - expected_return) < 1e-6

    def test_trade_fields_populated(self):
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"trades_per_day": 3, "win_rate": 0.55, "days": 2},
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=2)
        for t in result.trades:
            assert t.trade_number >= 1
            assert t.day >= 1
            assert t.lots > 0
            assert t.risk_amount > 0
            assert t.balance_before > 0
            assert t.balance_after > 0
            assert t.outcome in ("WIN", "LOSS")
            assert t.timestamp is not None


class TestSimulationResult:
    def test_default_values(self):
        result = SimulationResult(config=SimulationConfig())
        assert result.trades == []
        assert result.equity_curve == []
        assert result.initial_balance == 0.0
        assert result.final_balance == 0.0
        assert result.total_return_pct == 0.0
        assert result.max_drawdown_pct == 0.0
        assert result.win_rate_actual == 0.0
        assert result.profit_factor == 0.0
