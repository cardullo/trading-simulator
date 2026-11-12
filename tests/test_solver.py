"""Unit tests for the goal-seeking reverse solver."""

import pytest

from trading_simulator.solver import solve, SolverResult, SolverScenario


class TestSolverBasic:
    """Tests for the solve function basic behavior."""

    def test_basic_solve(self):
        result = solve(
            initial_balance=1000.0,
            target_balance=1000000.0,
            months=12,
        )
        assert isinstance(result, SolverResult)
        assert result.initial_balance == 1000.0
        assert result.target_balance == 1000000.0
        assert result.months == 12
        assert result.required_total_return_pct == 99900.0
        assert result.required_monthly_return_pct > 0
        assert len(result.scenarios) == 3
        assert result.recommendation is not None

    def test_required_monthly_return(self):
        initial = 1000.0
        target = 2000.0
        months = 12
        result = solve(initial, target, months)
        expected = ((target / initial) ** (1.0 / months) - 1.0) * 100.0
        assert abs(result.required_monthly_return_pct - expected) < 0.01

    def test_required_total_return(self):
        result = solve(1000.0, 5000.0, 12)
        expected = (5000.0 / 1000.0 - 1.0) * 100.0
        assert abs(result.required_total_return_pct - expected) < 0.01

    def test_invalid_initial_balance(self):
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            solve(0.0, 1000.0, 12)
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            solve(-100.0, 1000.0, 12)

    def test_invalid_target_balance(self):
        with pytest.raises(ValueError, match="target_balance must exceed"):
            solve(1000.0, 1000.0, 12)
        with pytest.raises(ValueError, match="target_balance must exceed"):
            solve(1000.0, 500.0, 12)

    def test_invalid_months(self):
        with pytest.raises(ValueError, match="months must be positive"):
            solve(1000.0, 2000.0, 0)
        with pytest.raises(ValueError, match="months must be positive"):
            solve(1000.0, 2000.0, -5)


class TestSolverScenarios:
    def test_three_scenarios(self):
        result = solve(1000.0, 1000000.0, 12)
        names = [s.name for s in result.scenarios]
        assert "Conservative" in names
        assert "Moderate" in names
        assert "Aggressive" in names

    def test_conservative_params(self):
        result = solve(1000.0, 1000000.0, 12)
        cons = [s for s in result.scenarios if s.name == "Conservative"][0]
        assert cons.risk_per_trade == 0.01
        assert cons.trades_per_day == 3
        assert cons.win_rate == 0.60

    def test_moderate_params(self):
        result = solve(1000.0, 1000000.0, 12)
        mod = [s for s in result.scenarios if s.name == "Moderate"][0]
        assert mod.risk_per_trade == 0.02
        assert mod.trades_per_day == 10
        assert mod.win_rate == 0.55

    def test_aggressive_params(self):
        result = solve(1000.0, 1000000.0, 12)
        agg = [s for s in result.scenarios if s.name == "Aggressive"][0]
        assert agg.risk_per_trade == 0.05
        assert agg.trades_per_day == 15
        assert agg.win_rate == 0.50

    def test_aggressive_highest_ror(self):
        result = solve(1000.0, 1000000.0, 12)
        rors = {s.name: s.risk_of_ruin for s in result.scenarios}
        assert rors["Aggressive"] > rors["Moderate"] > rors["Conservative"]


class TestSolverFeasibility:
    def test_feasible_when_scenario_meets_target(self):
        result = solve(1000.0, 2000.0, 12)
        assert result.feasible is True

    def test_not_feasible_when_impossible(self):
        result = solve(1000.0, 1e12, 1)
        assert result.feasible is False


class TestSolverMath:
    def test_expected_value_per_trade_r(self):
        from trading_simulator.solver import _expected_value_per_trade_r
        ev_r = _expected_value_per_trade_r(0.55, 3.0, 0.1)
        assert abs(ev_r - 1.10) < 0.001
        ev_r = _expected_value_per_trade_r(0.5, 1.0, 0.0)
        assert abs(ev_r - 0.0) < 0.001
        ev_r = _expected_value_per_trade_r(0.3, 2.0, 0.0)
        assert ev_r < 0

    def test_risk_of_ruin(self):
        from trading_simulator.solver import _risk_of_ruin
        ror = _risk_of_ruin(0.3, 2.0, 0.01, 1.0, 10.0)
        assert ror == 1.0
        ror = _risk_of_ruin(0.6, 3.0, 0.01, 1.0, 10.0)
        assert 0.0 < ror < 1.0

    def test_kelly_fraction(self):
        from trading_simulator.solver import _kelly_fraction
        kelly = _kelly_fraction(0.6, 2.0, 1.0, 10.0)
        assert 0.0 <= kelly <= 1.0
        kelly = _kelly_fraction(0.3, 2.0, 1.0, 10.0)
        assert kelly == 0.0

    def test_mathematical_proof_not_empty(self):
        result = solve(1000.0, 1000000.0, 12)
        for s in result.scenarios:
            assert s.mathematical_proof
            assert "Mathematical Proof" in s.mathematical_proof
            assert "EV_R" in s.mathematical_proof
            assert "Risk of Ruin" in s.mathematical_proof


class TestSolverDeterministic:
    def test_deterministic(self):
        result_a = solve(1000.0, 1000000.0, 12, fee_pips=1.0, sl_pips=10.0)
        result_b = solve(1000.0, 1000000.0, 12, fee_pips=1.0, sl_pips=10.0)
        assert result_a.required_monthly_return_pct == result_b.required_monthly_return_pct
        for sa, sb in zip(result_a.scenarios, result_b.scenarios):
            assert sa.expected_monthly_return_pct == sb.expected_monthly_return_pct
            assert sa.risk_of_ruin == sb.risk_of_ruin

    def test_recommendation_is_one_of_scenarios(self):
        result = solve(1000.0, 1000000.0, 12)
        assert result.recommendation in result.scenarios


class TestSolverCustomParams:
    def test_custom_fee_and_sl(self):
        result = solve(
            initial_balance=1000.0,
            target_balance=2000.0,
            months=6,
            fee_pips=2.0,
            sl_pips=15.0,
        )
        assert result is not None
        for s in result.scenarios:
            assert s.fee_pips == 2.0
            assert s.sl_pips == 15.0