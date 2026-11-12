"""
M11 Integration Tests — Pyramid Entry, Trailing Stop, TP Hit Simulation

Tests the advanced trade management features for mathematical correctness:
- Pyramid Entry with VWAP averaging (cTrader/IC Markets compatible)
- Trailing Stop loss protection
- TP Hit Simulation (fixed/random/percent modes)
- Combined M11 feature scenarios
"""
from __future__ import annotations

import pytest

from trading_simulator.config import (
    CompoundingMode,
    OutcomeMode,
    PyramidMode,
    SimulationConfig,
    TPHitConfig,
    TPHitMode,
)
from trading_simulator.engine import SimulationEngine
from trading_simulator.models import Trade


PIP_VALUE = 10.0  # EUR/USD: $10 per pip per lot


def run_engine(**overrides) -> SimulationEngine:
    """Build a SimulationConfig and run the engine."""
    base = {
        "instrument": {"symbol": "EUR/USD", "pip_size": 0.0001, "contract_size": 100_000},
        "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
        "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                 "compounding_mode": CompoundingMode.NONE},
        "activity": {"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                     "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        "pyramid": {"enabled": False},
        "trailing_stop": {"enabled": False},
        "tp_hit": {"enabled": False},
        "seed": 42,
    }
    base.update(overrides)
    cfg = SimulationConfig(**base)
    engine = SimulationEngine(cfg, seed=42)
    engine.run(days=cfg.activity.days)
    return engine


def verify_lot_alignment(trades: list[Trade]) -> list[str]:
    """All lots must be multiples of 0.01 and >= 0.01."""
    errors = []
    for t in trades:
        if t.lots < 0.01 - 1e-9:
            errors.append(f"Trade {t.trade_number}: lots {t.lots} < 0.01")
        if abs(t.lots / 0.01 - round(t.lots / 0.01)) > 1e-6:
            errors.append(f"Trade {t.trade_number}: lots {t.lots} not aligned to 0.01 step")
    return errors


def verify_balance_chain(trades: list[Trade], initial_balance: float) -> list[str]:
    """balance_before must equal previous balance_after."""
    errors = []
    prev = initial_balance
    for t in trades:
        if abs(t.balance_before - prev) > 1e-6:
            errors.append(f"Trade {t.trade_number}: bb {t.balance_before} != prev {prev}")
        ea = prev + t.profit_euros
        if abs(t.balance_after - ea) > 1e-6:
            errors.append(f"Trade {t.trade_number}: ba {t.balance_after} != {ea}")
        prev = t.balance_after
    return errors


def verify_profit_math(trades: list[Trade]) -> list[str]:
    """profit_euros = pips_net * lots * pip_value_per_lot."""
    errors = []
    for t in trades:
        ep = t.pips_net * t.lots * PIP_VALUE
        if abs(t.profit_euros - ep) > 1e-4:
            errors.append(
                f"Trade {t.trade_number}: profit {t.profit_euros} != "
                f"pn {t.pips_net} * lots {t.lots} * 10 = {ep}"
            )
    return errors


class TestPyramidEntry:
    """M11.1 Pyramid Entry — VWAP averaging, lot sizing, max adds."""

    def test_pyramid_fixed_lot_mode(self):
        """Fixed lot mode: each add uses same lot size, VWAP averaging correct."""
        engine = run_engine(
            pyramid={
                "enabled": True, "trigger_pips": 5.0, "max_adds": 3,
                "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                "lot_pct": 25.0, "be_trigger_pips": None,
            },
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades

        # 1 position × 4 fills (1 initial + 3 adds)
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]
        assert len(pos1_trades) == 4, f"Expected 4 fills, got {len(pos1_trades)}"

        # All lots must be aligned
        errors = verify_lot_alignment(pos1_trades)
        assert not errors, f"Lot alignment errors: {errors}"

        # Verify fill structure
        fill0 = pos1_trades[0]
        assert fill0.pyramid_fill_index == 0
        assert fill0.pyramid_total_fills == 4
        assert fill0.entry_price == 0.0
        assert fill0.position_avg_price == 0.0

        # Add 1 at 5 pips offset
        fill1 = pos1_trades[1]
        assert fill1.pyramid_fill_index == 1
        assert fill1.entry_price == 5.0
        # VWAP: (0×0.10 + 5×0.05) / 0.15 = 1.6667
        assert abs(fill1.position_avg_price - 1.6667) < 0.01

        # Add 2 trigger = current_avg + 5 = 1.667 + 5 = 6.667
        fill2 = pos1_trades[2]
        assert fill2.pyramid_fill_index == 2
        assert abs(fill2.entry_price - 6.6667) < 0.01
        # VWAP: (1.667×0.15 + 6.667×0.05) / 0.20 = 2.917
        assert abs(fill2.position_avg_price - 2.917) < 0.01

        # Add 3 trigger = current_avg + 5 = 2.917 + 5 = 7.917
        fill3 = pos1_trades[3]
        assert fill3.pyramid_fill_index == 3
        assert abs(fill3.entry_price - 7.917) < 0.01
        # VWAP: (2.917×0.20 + 7.917×0.05) / 0.25 = 3.917
        assert abs(fill3.position_avg_price - 3.917) < 0.01

        # All fills close at TP (30 pips from entry)
        assert fill0.pips_gross == 30.0
        assert fill1.pips_gross == 25.0  # 30 - 5
        assert abs(fill2.pips_gross - 23.333) < 0.01  # 30 - 6.667
        assert abs(fill3.pips_gross - 22.083) < 0.01  # 30 - 7.917

    def test_pyramid_percent_lot_mode(self):
        """Percent lot mode: each add = % of original position."""
        engine = run_engine(
            pyramid={
                "enabled": True, "trigger_pips": 5.0, "max_adds": 2,
                "lot_mode": PyramidMode.PERCENT, "lot_size": 0.01,
                "lot_pct": 50.0, "be_trigger_pips": None,
            },
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]

        # Initial: 0.10 lots (1% of 1000 / (10×10) = 0.10)
        assert pos1_trades[0].lots == 0.10

        # Add 1: 50% of 0.10 = 0.05 lots
        assert pos1_trades[1].lots == 0.05

        # Add 2: 50% of 0.10 = 0.05 lots
        assert pos1_trades[2].lots == 0.05

        # 3 fills total
        assert len(pos1_trades) == 3

    def test_pyramid_max_adds_limit(self):
        """Max adds enforcement: cannot add more than max_adds times."""
        engine = run_engine(
            pyramid={
                "enabled": True, "trigger_pips": 1.0, "max_adds": 2,
                "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                "lot_pct": 25.0, "be_trigger_pips": None,
            },
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]

        # 1 initial + 2 max adds = 3 fills
        assert len(pos1_trades) == 3
        assert pos1_trades[0].pyramid_total_fills == 3
        assert pos1_trades[1].pyramid_total_fills == 3
        assert pos1_trades[2].pyramid_total_fills == 3

    def test_pyramid_balance_chain_and_profit(self):
        """Balance chain and profit math correct for pyramid fills."""
        engine = run_engine(
            pyramid={
                "enabled": True, "trigger_pips": 5.0, "max_adds": 1,
                "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                "lot_pct": 25.0, "be_trigger_pips": None,
            },
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades

        errors = verify_balance_chain(trades, 1000.0)
        assert not errors, f"Balance chain errors: {errors}"

        errors = verify_profit_math(trades)
        assert not errors, f"Profit math errors: {errors}"


class TestTrailingStop:
    """M11.2 Trailing Stop — activation, hit vs TP, profit calculation."""

    def test_trailing_stop_level_calculation(self):
        """Trailing stop level = activate_pips - distance_pips."""
        engine = run_engine(
            trailing_stop={"enabled": True, "activate_pips": 10.0,
                           "distance_pips": 5.0, "partial_close_pct": 0.0},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades
        assert len(trades) == 1
        t = trades[0]
        assert t.trailing_stop_active is True
        assert t.trailing_stop_pips == 5.0  # 10 - 5

    def test_trailing_stop_reaches_tp(self):
        """When trailing stop not hit, trade reaches full TP."""
        engine = run_engine(
            trailing_stop={"enabled": True, "activate_pips": 10.0,
                           "distance_pips": 5.0, "partial_close_pct": 0.0},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
            seed=42,
        )
        trades = engine.trades
        t = trades[0]
        assert t.trailing_stop_active is True
        assert t.trailing_triggered is False
        assert t.pips_gross == 30.0  # Full TP

    def test_trailing_stop_balance_chain(self):
        """Balance chain correct with trailing stop."""
        engine = run_engine(
            trailing_stop={"enabled": True, "activate_pips": 10.0,
                           "distance_pips": 5.0, "partial_close_pct": 0.0},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
        )
        trades = engine.trades
        errors = verify_balance_chain(trades, 1000.0)
        assert not errors, f"Balance chain errors: {errors}"
        errors = verify_profit_math(trades)
        assert not errors, f"Profit math errors: {errors}"


class TestTPHitSimulation:
    """M11.3 TP Hit Simulation — fixed/random/percent modes."""

    def test_tp_hit_fixed_mode_all_hit(self):
        """FIXED mode: all TP levels always hit."""
        engine = run_engine(
            tp_hit={"enabled": True, "mode": TPHitMode.FIXED, "hit_pct": 100.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "tp_levels": [
                      {"pips": 10.0, "allocation_pct": 25.0},
                      {"pips": 20.0, "allocation_pct": 25.0},
                      {"pips": 30.0, "allocation_pct": 50.0},
                  ],
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]
        assert len(pos1_trades) == 3  # All 3 levels hit
        assert all(t.tp_level_hit for t in pos1_trades)

    def test_tp_hit_percent_mode_exact(self):
        """PERCENT mode: exactly hit_pct% of levels hit."""
        engine = run_engine(
            tp_hit={"enabled": True, "mode": TPHitMode.PERCENT, "hit_pct": 67.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "tp_levels": [
                      {"pips": 10.0, "allocation_pct": 25.0},
                      {"pips": 20.0, "allocation_pct": 25.0},
                      {"pips": 30.0, "allocation_pct": 50.0},
                  ],
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]

        # 67% of 3 levels = 2 levels hit (rounded)
        hit_count = sum(1 for t in pos1_trades if t.tp_level_hit)
        missed_count = sum(1 for t in pos1_trades if not t.tp_level_hit)
        assert hit_count == 2, f"Expected 2 hits, got {hit_count}"
        assert missed_count == 1, f"Expected 1 miss, got {missed_count}"
        assert len(pos1_trades) == 3  # All 3 levels recorded (2 hits + 1 miss)

    def test_tp_hit_missed_level_zero_profit(self):
        """Missed TP levels have zero profit and tp_level_hit=False."""
        engine = run_engine(
            tp_hit={"enabled": True, "mode": TPHitMode.PERCENT, "hit_pct": 33.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "tp_levels": [
                      {"pips": 10.0, "allocation_pct": 25.0},
                      {"pips": 20.0, "allocation_pct": 25.0},
                      {"pips": 30.0, "allocation_pct": 50.0},
                  ],
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]

        missed = [t for t in pos1_trades if not t.tp_level_hit]
        assert len(missed) == 2, f"Expected 2 misses, got {len(missed)}"
        for t in missed:
            assert t.pips_gross == 0.0
            assert t.pips_net == 0.0
            assert t.profit_euros == 0.0
            assert t.tp_level_pips > 0  # Target pips still recorded

    def test_tp_hit_disabled_backward_compatible(self):
        """TP hit disabled: all levels hit (original behavior)."""
        engine = run_engine(
            tp_hit={"enabled": False, "mode": TPHitMode.FIXED, "hit_pct": 100.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "tp_levels": [
                      {"pips": 10.0, "allocation_pct": 25.0},
                      {"pips": 20.0, "allocation_pct": 25.0},
                      {"pips": 30.0, "allocation_pct": 50.0},
                  ],
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades
        pos1_trades = [t for t in trades if t.position_id == "pos_1"]
        assert len(pos1_trades) == 3
        assert all(t.tp_level_hit for t in pos1_trades)


class TestM11CombinedFeatures:
    """M11 Combined — pyramid + trailing stop + TP hit together."""

    def test_all_m11_features_together(self):
        """All three M11 features enabled simultaneously.
        
        Note: Pyramid takes priority over TP levels in the engine, so when
        pyramid is enabled, TP levels are not used for that trade.
        """
        engine = run_engine(
            pyramid={"enabled": True, "trigger_pips": 5.0, "max_adds": 2,
                     "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                     "lot_pct": 25.0, "be_trigger_pips": None},
            trailing_stop={"enabled": True, "activate_pips": 10.0,
                           "distance_pips": 5.0, "partial_close_pct": 0.0},
            tp_hit={"enabled": True, "mode": TPHitMode.FIXED, "hit_pct": 100.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 1, "win_rate": 1.0, "days": 1,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades

        # Balance chain must be correct across all features
        errors = verify_balance_chain(trades, 1000.0)
        assert not errors, f"Balance chain errors: {errors}"

        # Profit math must be correct
        errors = verify_profit_math(trades)
        assert not errors, f"Profit math errors: {errors}"

        # Lot alignment
        errors = verify_lot_alignment(trades)
        assert not errors, f"Lot alignment errors: {errors}"

        # Verify pyramid fills exist
        pyramid_trades = [t for t in trades if t.pyramid_fill_index > 0]
        assert len(pyramid_trades) > 0, "Expected pyramid fills"

        # Note: pyramid takes priority over trailing stop, so trailing_stop_active
        # will be False when pyramid handles the trade

    def test_m11_features_no_negative_balance(self):
        """Balance never goes negative with M11 features."""
        engine = run_engine(
            pyramid={"enabled": True, "trigger_pips": 5.0, "max_adds": 3,
                     "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                     "lot_pct": 25.0, "be_trigger_pips": None},
            trailing_stop={"enabled": True, "activate_pips": 10.0,
                           "distance_pips": 5.0, "partial_close_pct": 0.0},
            tp_hit={"enabled": True, "mode": TPHitMode.RANDOM, "hit_pct": 50.0},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                  "tp_levels": [
                      {"pips": 10.0, "allocation_pct": 25.0},
                      {"pips": 20.0, "allocation_pct": 25.0},
                      {"pips": 30.0, "allocation_pct": 50.0},
                  ],
                  "compounding_mode": CompoundingMode.NONE},
            activity={"trades_per_day": 5, "win_rate": 0.5, "days": 5,
                      "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
        )
        trades = engine.trades
        for t in trades:
            assert t.balance_after >= 0, f"Trade {t.trade_number}: negative balance {t.balance_after}"