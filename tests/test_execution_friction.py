"""Tests for execution friction: variable spread, asymmetric slippage, and overnight swaps."""

import math
import pytest
from trading_simulator.config import (
    AccountConfig,
    ActivityConfig,
    ExecutionConfig,
    OutcomeMode,
    RiskConfig,
    SimulationConfig,
)
from trading_simulator.engine import SimulationEngine


class TestExecutionFriction:
    """Test variable spread, asymmetric slippage, and overnight swap financing."""

    def test_variable_spread_increases_fee(self):
        """Variable spread should expand the effective fee beyond base fee_pips."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=1.0),
            activity=ActivityConfig(
                trades_per_day=5,
                days=2,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="WWLL",
            ),
            execution=ExecutionConfig(
                enabled=True,
                variable_spread_enabled=True,
                base_spread_pips=0.4,
                spread_std_pips=0.2,
            ),
            seed=42,
        )
        engine = SimulationEngine(config)
        result = engine.run()

        assert len(result.trades) == 10
        # Every trade fee should be at least base fee (1.0) + base spread (0.4) = 1.4 pips
        for trade in result.trades:
            assert trade.fee_pips >= 1.4

    def test_asymmetric_slippage_on_stop_loss(self):
        """Adverse slippage on SL market orders increases the gross loss in pips."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=1,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="L",
            ),
            execution=ExecutionConfig(
                enabled=True,
                slippage_enabled=True,
                sl_slippage_mean_pips=0.8,
                sl_slippage_std_pips=0.0,  # Zero variance for deterministic test
            ),
            seed=42,
        )
        engine = SimulationEngine(config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        # Nominal SL was -10 pips; with 0.8 pips slippage, gross pips should be -10.8 pips
        assert math.isclose(trade.pips_gross, -10.8, abs_tol=1e-3)
        assert trade.outcome == "LOSS"

    def test_positive_tp_slippage_price_improvement(self):
        """Limit TP orders can experience positive price improvement."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=1,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="W",
            ),
            execution=ExecutionConfig(
                enabled=True,
                slippage_enabled=True,
                tp_slippage_improvement_prob=1.0,  # Always improve
                tp_slippage_mean_pips=0.5,
            ),
            seed=42,
        )
        engine = SimulationEngine(config)
        result = engine.run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        # Nominal TP is 30.0 pips; with positive slippage it should exceed 30.0
        assert trade.pips_gross > 30.0

    def test_overnight_swap_and_triple_swap_wednesday(self):
        """Overnight swap is charged per night, with 3x triple swap on Wednesday (day 3)."""
        # Day 1 (Monday): 1 night swap = -0.5 pips
        config_mon = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=1,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="W",
            ),
            execution=ExecutionConfig(
                enabled=True,
                swap_enabled=True,
                swap_long_pips=-0.5,
                holding_nights=1,
                triple_swap_day=3,
            ),
            seed=42,
        )
        res_mon = SimulationEngine(config_mon).run()
        # TP = 30.0 gross, fee = 0.0, swap = -0.5 -> net pips = 29.5
        assert math.isclose(res_mon.trades[0].pips_net, 29.5, abs_tol=1e-3)

        # Multi-day run crossing Wednesday (day 3)
        config_multi = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=3,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="WWW",
            ),
            execution=ExecutionConfig(
                enabled=True,
                swap_enabled=True,
                swap_long_pips=-0.5,
                holding_nights=1,
                triple_swap_day=3,
            ),
            seed=42,
        )
        res_multi = SimulationEngine(config_multi).run()
        # Day 3 trade should have 3x swap (-1.5 pips) -> net pips = 30.0 - 1.5 = 28.5
        day3_trade = res_multi.trades[2]
        assert day3_trade.day == 3
        assert math.isclose(day3_trade.pips_net, 28.5, abs_tol=1e-3)

    def test_balance_chain_integrity_with_all_frictions(self):
        """With variable spread, slippage, and swaps active together, the balance chain
        invariant balance_after == balance_before + profit_euros must hold 100%."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=10000.0, risk_per_trade=0.01),
            risk=RiskConfig(sl_pips=15.0, tp_pips=45.0, fee_pips=1.0),
            activity=ActivityConfig(
                trades_per_day=10,
                days=5,
                outcome_mode=OutcomeMode.BERNOULLI,
                win_rate=0.55,
            ),
            execution=ExecutionConfig(
                enabled=True,
                variable_spread_enabled=True,
                base_spread_pips=0.2,
                spread_std_pips=0.1,
                slippage_enabled=True,
                sl_slippage_mean_pips=0.4,
                sl_slippage_std_pips=0.1,
                swap_enabled=True,
                swap_long_pips=-0.6,
                holding_nights=1,
                triple_swap_day=3,
            ),
            seed=123,
        )
        engine = SimulationEngine(config)
        result = engine.run()

        assert len(result.trades) == 50
        prev_balance = config.account.initial_balance
        for t in result.trades:
            assert math.isclose(t.balance_before, prev_balance, rel_tol=1e-9, abs_tol=1e-4)
            expected_after = t.balance_before + t.profit_euros
            assert math.isclose(t.balance_after, expected_after, rel_tol=1e-9, abs_tol=1e-4)
            prev_balance = t.balance_after
