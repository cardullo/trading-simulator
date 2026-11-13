"""Tests for leverage, margin requirements, margin call, and stop-out liquidation."""

import pytest
from trading_simulator.config import (
    AccountConfig,
    ActivityConfig,
    InstrumentConfig,
    LeverageConfig,
    OutcomeMode,
    PricePathMode,
    PyramidConfig,
    PyramidMode,
    RiskConfig,
    SimulationConfig,
)
from trading_simulator.engine import SimulationEngine


class TestLeverageAndMargin:
    """Test broker leverage and margin constraints."""

    def test_margin_rejection_insufficient_balance(self):
        """ESMA 1:30 leverage requires 3,333.33 EUR margin for 1.0 lot.
        A 500 EUR account attempting to trade 1.0 lot should be rejected."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=500.0, risk_per_trade=0.20),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(trades_per_day=1, days=1, outcome_mode=OutcomeMode.BERNOULLI),
            leverage=LeverageConfig(enabled=True, leverage=30.0),
            seed=42,
        )
        # 500 * 0.20 = 100 EUR risk. At 10 pip SL ($10/pip/lot), lots = 1.00 lot.
        # Required margin for 1.0 lot at 1:30 = 100,000 / 30 = 3,333.33 EUR.
        # Account balance is only 500 EUR, so free margin is insufficient.
        engine = SimulationEngine(config)
        result = engine.run()

        assert result.margin_rejections == 1
        assert len(result.trades) == 0
        assert result.final_balance == 500.0

    def test_max_lot_exposure_cap(self):
        """When max_lot_exposure is configured, any trade exceeding that cap is rejected."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=10000.0, risk_per_trade=0.05),
            risk=RiskConfig(sl_pips=10.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(trades_per_day=1, days=1),
            leverage=LeverageConfig(
                enabled=True,
                leverage=500.0,
                max_lot_exposure=2.0,  # Max 2.0 lots permitted
            ),
            seed=42,
        )
        # 10,000 * 0.05 = 500 EUR risk. At 10 pip SL, lots = 5.00 lots > 2.0 max cap.
        engine = SimulationEngine(config)
        result = engine.run()

        assert result.margin_rejections == 1
        assert len(result.trades) == 0

    def test_stop_out_liquidation_triggers_early(self):
        """Under high leverage and large position size, a losing trade should trigger
        cTrader stop-out liquidation at 50% margin level before hitting a deep SL."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=1000.0, risk_per_trade=0.80),
            risk=RiskConfig(sl_pips=80.0, tp_pips=30.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=1,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="L",
            ),
            leverage=LeverageConfig(
                enabled=True,
                leverage=100.0,
                margin_call_pct=100.0,
                stop_out_pct=50.0,
            ),
            seed=42,
        )
        # 1000 * 0.80 = 800 EUR risk. SL = 80 pips -> 1.00 lot.
        # Required margin for 1.00 lot at 1:100 = 100,000 / 100 = 1000.0 EUR.
        # At entry: equity = 1000 EUR, margin level = 100% -> margin call triggered!
        # As price drops, at -50 pips: floating loss = -50 * 1.0 * 10 = -500 EUR.
        # Equity = 500 EUR = 50% of 1000 EUR required margin -> Stop out liquidation!
        # Liquidation triggers at -50 pips, before hitting full -80 pips SL.
        engine = SimulationEngine(config)
        result = engine.run()

        assert result.stop_outs == 1
        assert result.margin_calls >= 1
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.outcome == "LOSS"
        # Trade closed early around -50 pips, not at full -80 pips SL
        assert abs(trade.pips_gross) <= 51.0
        # Account balance preserved at ~500 EUR, protecting from -800 EUR SL loss
        assert result.final_balance >= 490.0

    def test_pyramid_rejected_when_free_margin_insufficient(self):
        """When pyramiding is active, adding volume is rejected if free margin is insufficient."""
        config = SimulationConfig(
            account=AccountConfig(initial_balance=250.0, risk_per_trade=0.04),
            risk=RiskConfig(sl_pips=10.0, tp_pips=50.0, fee_pips=0.0),
            activity=ActivityConfig(
                trades_per_day=1,
                days=1,
                outcome_mode=OutcomeMode.CUSTOM_PATTERN,
                custom_pattern="W",
            ),
            pyramid=PyramidConfig(
                enabled=True,
                trigger_pips=5.0,
                max_adds=3,
                lot_mode=PyramidMode.FIXED,
                lot_size=0.10,  # 0.10 lot add requires 333.33 EUR margin at 1:30
            ),
            leverage=LeverageConfig(
                enabled=True,
                leverage=30.0,
            ),
            seed=42,
        )
        # Initial trade: 250 * 0.04 = 10 EUR risk / (10 * 10) = 0.10 lot?
        # Actually 10 / 100 = 0.10 lot. Margin for 0.10 lot = 333.33 > 250 -> rejected at start!
        # Let's use 0.02 lot initial:
        # Risk = 2 EUR / (10 * 10) = 0.02 lot.
        # Margin for 0.02 lot at 1:30 = 2,000 / 30 = 66.67 EUR.
        # Initial trade accepted! Free margin = 250 - 66.67 = 183.33 EUR.
        # But pyramid add asks for 0.10 lot (requires 333.33 EUR margin).
        # Free margin 183.33 < 333.33, so pyramid add cannot be accepted!
        config.account.risk_per_trade = 0.008  # 250 * 0.008 = 2.0 EUR -> 0.02 lot
        engine = SimulationEngine(config)
        result = engine.run()

        assert len(result.trades) == 1
        # No pyramid adds were added
        assert result.trades[0].lots == 0.02
        assert result.trades[0].pyramid_total_fills == 1
