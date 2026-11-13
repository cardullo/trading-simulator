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

    def test_trailing_stop_stochastic_no_cascade(self):
        """Fix C-01: Ensure trailing stop with partial close does not cascade into 10+ fills."""
        from collections import Counter
        from trading_simulator.config import PricePathMode
        cfg = _cfg(
            account={"initial_balance": 5000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 12.0, "tp_pips": 40.0, "fee_pips": 1.0},
            activity={
                "trades_per_day": 10, "win_rate": 0.60, "days": 5,
                "price_path_mode": PricePathMode.STOCHASTIC,
                "pullback_prob": 0.35,
            },
            trailing_stop={
                "enabled": True, "activate_pips": 10.0, "distance_pips": 4.0, "partial_close_pct": 50.0
            },
            seed=777,
        )
        result = SimulationEngine(cfg, seed=777).run(days=5)
        pos_counts = Counter(t.position_id for t in result.trades)
        assert max(pos_counts.values()) <= 3, f"Expected at most 3 fills per position, got {max(pos_counts.values())}"
        # Verify balance chain holds across all fills
        prev = 5000.0
        for t in result.trades:
            assert abs(t.balance_before - prev) < 1e-4
            prev = t.balance_after

    def test_stochastic_win_pullback_triggers_sl(self):
        """Fix C-02: Stochastic mode trade with WIN drift that hits SL must exit as LOSS."""
        from trading_simulator.config import PricePathMode, OutcomeMode
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 5.0, "tp_pips": 100.0, "fee_pips": 0.0},
            activity={
                "trades_per_day": 1, "win_rate": 1.0, "days": 1,
                "outcome_mode": OutcomeMode.FIXED_WINS_FIRST,
                "price_path_mode": PricePathMode.STOCHASTIC,
                "pullback_prob": 0.48,
            },
            seed=12,
        )
        result = SimulationEngine(cfg, seed=12).run(days=1)
        assert len(result.trades) >= 1
        # Trade hit SL on pullback
        sl_hits = [t for t in result.trades if t.outcome == "LOSS"]
        assert len(sl_hits) > 0, "Expected trade to exit at SL on adverse stochastic excursion"
        for t in sl_hits:
            assert t.pips_gross <= -5.0

    def test_multi_night_holding_swap_calculation(self):
        """Fix H-03: Multi-night hold crossing Wednesday triple-swap calculates night-by-night."""
        from trading_simulator.config import OutcomeMode
        # Start day 2 (Tuesday), hold 2 nights (Tue night = 1x, Wed night = 3x -> 4x total)
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 0.0},
            activity={
                "trades_per_day": 1, "win_rate": 1.0, "days": 2,
                "outcome_mode": OutcomeMode.FIXED_WINS_FIRST,
            },
            execution={
                "enabled": True, "swap_enabled": True, "swap_long_pips": -0.5,
                "holding_nights": 2, "triple_swap_day": 3,
            },
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=2)
        # Day 2 trade spans Tuesday and Wednesday nights: (-0.5 * 1) + (-0.5 * 3) = -2.0 pips
        trade_day2 = [t for t in result.trades if t.day == 2][0]
        # Gross TP = 30.0, swap = -2.0 -> net = 28.0
        import math
        assert math.isclose(trade_day2.pips_net, 28.0, abs_tol=1e-3)

    def test_pyramid_floor_alignment(self):
        """Fix H-01: Pyramid additions use math.floor truncation matching broker conventions."""
        from trading_simulator.config import PyramidMode, OutcomeMode
        cfg = _cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.01},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={
                "trades_per_day": 1, "win_rate": 1.0, "days": 1,
                "outcome_mode": OutcomeMode.FIXED_WINS_FIRST,
            },
            pyramid={
                "enabled": True, "trigger_pips": 5.0, "max_adds": 2,
                "lot_mode": PyramidMode.PERCENT, "lot_pct": 25.0,
            },
            seed=42,
        )
        result = SimulationEngine(cfg, seed=42).run(days=1)
        for t in result.trades:
            # Each lot must be aligned to 0.01 step
            assert round(t.lots, 2) == t.lots
            assert t.lots >= 0.01


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
