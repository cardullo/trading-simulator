"""Unit tests for performance analytics."""

import pytest
from datetime import datetime

from trading_simulator.analytics import (
    PerformanceMetrics,
    QuantMetrics,
    compute_quant_metrics,
    compute_monte_carlo_drawdown,
    aggregate_daily,
    aggregate_weekly,
    aggregate_monthly,
    aggregate_all_periods,
    _compute_drawdown,
)


class FakeTrade:
    """Minimal fake trade for testing analytics."""

    def __init__(self, day, outcome, profit_euros, pips_net, balance_before, balance_after, fee_pips=1.0):
        self.day = day
        self.outcome = outcome
        self.profit_euros = profit_euros
        self.pips_net = pips_net
        self.balance_before = balance_before
        self.balance_after = balance_after
        self.fee_pips = fee_pips
        self.pips_gross = pips_net + fee_pips if outcome == "WIN" else pips_net - fee_pips


class FakeEquityPoint:
    """Minimal fake equity point for testing."""

    def __init__(self, timestamp, balance):
        self.timestamp = timestamp
        self.balance = balance


class TestComputeDrawdown:
    """Tests for _compute_drawdown helper."""

    def test_empty_balances(self):
        assert _compute_drawdown([], 1000.0) == 0.0

    def test_no_drawdown(self):
        # Balance only increases
        assert _compute_drawdown([1000, 1100, 1200, 1300], 1000.0) == 0.0

    def test_simple_drawdown(self):
        # 1000 -> 1100 -> 900 -> 1200
        # Peak at 1100, trough at 900: DD = (1100-900)/1100 = 18.18%
        result = _compute_drawdown([1000, 1100, 900, 1200], 1000.0)
        assert abs(result - 18.18) < 0.1

    def test_drawdown_from_start(self):
        # 1000 -> 800 -> 900
        # Peak stays at 1000, trough at 800: DD = (1000-800)/1000 = 20%
        result = _compute_drawdown([1000, 800, 900], 1000.0)
        assert abs(result - 20.0) < 0.1


class TestAggregateDaily:
    """Tests for daily aggregation."""

    def test_single_day_profitable(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1030.0, balance_after=1060.0),
            FakeTrade(day=1, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1060.0, balance_after=1050.0),
        ]
        metrics = aggregate_daily(trades)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.period == "daily"
        assert m.period_label == "Day 1"
        assert m.trade_count == 3
        assert m.win_count == 2
        assert m.loss_count == 1
        assert m.win_rate == 2 / 3
        assert m.starting_balance == 1000.0
        assert m.ending_balance == 1050.0
        assert m.net_profit == 50.0
        assert m.return_pct == 5.0
        assert m.total_pips == 29.0 + 29.0 + (-11.0)
        assert m.is_profitable is True

    def test_single_day_loss(self):
        trades = [
            FakeTrade(day=1, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1000.0, balance_after=990.0),
            FakeTrade(day=1, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=990.0, balance_after=980.0),
        ]
        metrics = aggregate_daily(trades)
        m = metrics[0]
        assert m.win_count == 0
        assert m.loss_count == 2
        assert m.win_rate == 0.0
        assert m.ending_balance == 980.0
        assert m.return_pct == -2.0
        assert m.is_profitable is False

    def test_multiple_days(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=2, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1030.0, balance_after=1020.0),
            FakeTrade(day=2, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1020.0, balance_after=1050.0),
        ]
        metrics = aggregate_daily(trades)
        assert len(metrics) == 2
        assert metrics[0].period_label == "Day 1"
        assert metrics[0].ending_balance == 1030.0
        assert metrics[1].period_label == "Day 2"
        assert metrics[1].starting_balance == 1030.0
        assert metrics[1].ending_balance == 1050.0

    def test_empty_trades(self):
        metrics = aggregate_daily([])
        assert metrics == []

    def test_profit_factor_all_wins(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1030.0, balance_after=1060.0),
        ]
        metrics = aggregate_daily(trades)
        m = metrics[0]
        assert m.gross_profit == 60.0
        assert m.gross_loss == 0.0
        assert m.profit_factor == float("inf")

    def test_profit_factor_all_losses(self):
        trades = [
            FakeTrade(day=1, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1000.0, balance_after=990.0),
            FakeTrade(day=1, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=990.0, balance_after=980.0),
        ]
        metrics = aggregate_daily(trades)
        m = metrics[0]
        assert m.gross_profit == 0.0
        assert m.gross_loss == 20.0
        assert m.profit_factor == 0.0


class TestAggregateWeekly:
    """Tests for weekly aggregation."""

    def test_single_week(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=5, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1030.0, balance_after=1020.0),
            FakeTrade(day=7, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1020.0, balance_after=1050.0),
        ]
        metrics = aggregate_weekly(trades)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.period == "weekly"
        assert m.period_label == "Week 1"
        assert m.trade_count == 3
        assert m.ending_balance == 1050.0

    def test_multiple_weeks(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=10, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1030.0, balance_after=1060.0),
            FakeTrade(day=15, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1060.0, balance_after=1050.0),
        ]
        metrics = aggregate_weekly(trades)
        # Day 1-7 = Week 1, Day 8-14 = Week 2, Day 15-21 = Week 3
        assert len(metrics) == 3
        assert metrics[0].trade_count == 1  # Day 1
        assert metrics[1].trade_count == 1  # Day 10
        assert metrics[2].trade_count == 1  # Day 15

    def test_empty_trades(self):
        metrics = aggregate_weekly([])
        assert metrics == []


class TestAggregateMonthly:
    """Tests for monthly aggregation."""

    def test_single_month(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=20, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1030.0, balance_after=1020.0),
        ]
        metrics = aggregate_monthly(trades)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.period == "monthly"
        assert m.period_label == "Month 1"
        assert m.trade_count == 2

    def test_multiple_months(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=35, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1030.0, balance_after=1060.0),
        ]
        metrics = aggregate_monthly(trades)
        # Day 1-30 = Month 1, Day 31-60 = Month 2
        assert len(metrics) == 2
        assert metrics[0].trade_count == 1
        assert metrics[1].trade_count == 1


class TestAggregateAllPeriods:
    """Tests for aggregate_all_periods convenience function."""

    def test_all_periods_returned(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=29.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=2, outcome="LOSS", profit_euros=-10.0, pips_net=-11.0, balance_before=1030.0, balance_after=1020.0),
        ]
        result = aggregate_all_periods(trades)
        assert "daily" in result
        assert "weekly" in result
        assert "monthly" in result
        assert isinstance(result["daily"], list)
        assert isinstance(result["weekly"], list)
        assert isinstance(result["monthly"], list)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_values(self):
        m = PerformanceMetrics(
            period="daily",
            period_label="Day 1",
            period_start=datetime.now(),
            period_end=datetime.now(),
            starting_balance=1000.0,
            ending_balance=1100.0,
            return_pct=10.0,
            total_pips=29.0,
            win_count=1,
            loss_count=0,
            win_rate=1.0,
            profit_factor=float("inf"),
            max_drawdown_pct=0.0,
            trade_count=1,
            gross_profit=100.0,
            gross_loss=0.0,
            net_profit=100.0,
        )
        assert m.is_profitable is True
        assert m.net_profit == 100.0

    def test_is_profitable_false(self):
        m = PerformanceMetrics(
            period="daily",
            period_label="Day 1",
            period_start=datetime.now(),
            period_end=datetime.now(),
            starting_balance=1000.0,
            ending_balance=900.0,
            return_pct=-10.0,
            total_pips=-11.0,
            win_count=0,
            loss_count=1,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=10.0,
            trade_count=1,
            gross_profit=0.0,
            gross_loss=100.0,
            net_profit=-100.0,
        )
        assert m.is_profitable is False


class TestQuantMetrics:
    """Tests for Sharpe, Sortino, Calmar, CAGR, and Monte Carlo Drawdown."""

    def test_quant_metrics_empty(self):
        qm = compute_quant_metrics([], initial_balance=1000.0)
        assert qm.total_trades == 0
        assert qm.sharpe_ratio == 0.0
        assert qm.win_rate == 0.0

    def test_quant_metrics_calculation(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=30.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=30.0, balance_before=1030.0, balance_after=1060.0),
            FakeTrade(day=2, outcome="LOSS", profit_euros=-10.0, pips_net=-10.0, balance_before=1060.0, balance_after=1050.0),
            FakeTrade(day=3, outcome="WIN", profit_euros=30.0, pips_net=30.0, balance_before=1050.0, balance_after=1080.0),
        ]
        qm = compute_quant_metrics(trades, initial_balance=1000.0, days=3)

        assert qm.total_trades == 4
        assert qm.win_rate == 0.75
        assert qm.average_win_euros == 30.0
        assert qm.average_loss_euros == 10.0
        assert qm.payoff_ratio == 3.0
        # Expectancy: (0.75 * 30) - (0.25 * 10) = 22.5 - 2.5 = 20.0 EUR
        assert abs(qm.expectancy_euros - 20.0) < 0.01
        assert qm.profit_factor == 9.0  # 90 / 10
        assert qm.net_profit_euros == 80.0
        assert qm.final_balance_euros == 1080.0
        # Positive returns produce positive Sharpe and Sortino
        assert qm.sharpe_ratio > 0.0
        assert qm.sortino_ratio > 0.0
        assert qm.calmar_ratio > 0.0
        assert qm.cagr_pct > 0.0

    def test_monte_carlo_drawdown_distribution(self):
        trades = [
            FakeTrade(day=1, outcome="WIN", profit_euros=30.0, pips_net=30.0, balance_before=1000.0, balance_after=1030.0),
            FakeTrade(day=2, outcome="LOSS", profit_euros=-50.0, pips_net=-50.0, balance_before=1030.0, balance_after=980.0),
            FakeTrade(day=3, outcome="LOSS", profit_euros=-50.0, pips_net=-50.0, balance_before=980.0, balance_after=930.0),
            FakeTrade(day=4, outcome="WIN", profit_euros=100.0, pips_net=100.0, balance_before=930.0, balance_after=1030.0),
        ]
        mc_dd = compute_monte_carlo_drawdown(trades, initial_balance=1000.0, n_simulations=500, seed=42)

        # Monotonicity: median <= p95 <= p99 <= max
        assert 0.0 <= mc_dd["median_dd_pct"] <= mc_dd["p95_dd_pct"]
        assert mc_dd["p95_dd_pct"] <= mc_dd["p99_dd_pct"]
        assert mc_dd["p99_dd_pct"] <= mc_dd["max_dd_pct"]

    def test_monte_carlo_empty(self):
        mc_dd = compute_monte_carlo_drawdown([], initial_balance=1000.0)
        assert mc_dd["median_dd_pct"] == 0.0
        assert mc_dd["max_dd_pct"] == 0.0
