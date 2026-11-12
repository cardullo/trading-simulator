"""Unit tests for partial take-profit functionality."""

import pytest

from trading_simulator.config import SimulationConfig, TakeProfitLevel
from trading_simulator.engine import SimulationEngine


class TestTakeProfitLevel:
    """Tests for TakeProfitLevel model."""

    def test_valid_level(self):
        level = TakeProfitLevel(pips=10.0, allocation_pct=25.0)
        assert level.pips == 10.0
        assert level.allocation_pct == 25.0

    def test_zero_pips_rejected(self):
        with pytest.raises(ValueError):
            TakeProfitLevel(pips=0.0, allocation_pct=50.0)

    def test_zero_allocation_rejected(self):
        with pytest.raises(ValueError):
            TakeProfitLevel(pips=10.0, allocation_pct=0.0)

    def test_allocation_over_100_rejected(self):
        with pytest.raises(ValueError):
            TakeProfitLevel(pips=10.0, allocation_pct=150.0)


class TestPartialTPConfig:
    """Tests for RiskConfig with TP levels."""

    def test_tp_levels_validation_sums_to_100(self):
        cfg = SimulationConfig(
            account={'initial_balance': 1000.0, 'risk_per_trade': 0.01},
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 10.0, 'allocation_pct': 50.0},
                    {'pips': 20.0, 'allocation_pct': 50.0},
                ],
            },
            activity={'trades_per_day': 5, 'win_rate': 0.55, 'days': 5},
            seed=42,
        )
        assert len(cfg.risk.tp_levels) == 2
        assert cfg.risk.tp_levels[0].pips == 10.0
        assert cfg.risk.tp_levels[1].pips == 20.0

    def test_tp_levels_sorted_by_pips(self):
        """TP levels should be sorted by pips ascending regardless of input order."""
        cfg = SimulationConfig(
            account={'initial_balance': 1000.0, 'risk_per_trade': 0.01},
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 30.0, 'allocation_pct': 33.33},
                    {'pips': 10.0, 'allocation_pct': 33.33},
                    {'pips': 20.0, 'allocation_pct': 33.34},
                ],
            },
            activity={'trades_per_day': 5, 'win_rate': 0.55, 'days': 5},
            seed=42,
        )
        assert cfg.risk.tp_levels[0].pips == 10.0
        assert cfg.risk.tp_levels[1].pips == 20.0
        assert cfg.risk.tp_levels[2].pips == 30.0

    def test_tp_levels_allocation_mismatch_rejected(self):
        """Allocations that don't sum to 100% should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 100%"):
            SimulationConfig(
                account={'initial_balance': 1000.0, 'risk_per_trade': 0.01},
                risk={
                    'sl_pips': 10.0,
                    'tp_pips': 30.0,
                    'fee_pips': 1.0,
                    'tp_levels': [
                        {'pips': 10.0, 'allocation_pct': 30.0},
                        {'pips': 20.0, 'allocation_pct': 30.0},
                    ],
                },
                activity={'trades_per_day': 5, 'win_rate': 0.55, 'days': 5},
                seed=42,
            )


class TestPartialTPEngine:
    """Tests for the simulation engine with partial take profits."""

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

    def test_partial_tp_win_creates_multiple_fills(self):
        """A winning trade with TP levels should create multiple fill records."""
        cfg = self._cfg(
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 10.0, 'allocation_pct': 25.0},
                    {'pips': 20.0, 'allocation_pct': 25.0},
                    {'pips': 30.0, 'allocation_pct': 25.0},
                    {'pips': 40.0, 'allocation_pct': 25.0},
                ],
            },
            activity={'trades_per_day': 1, 'win_rate': 1.0, 'days': 1, 'outcome_mode': 'fixed_wins_first'},
        )
        result = SimulationEngine(cfg, seed=42).run(days=1)
        assert len(result.trades) == 4
        assert all(t.outcome == "WIN" for t in result.trades)
        position_ids = {t.position_id for t in result.trades}
        assert len(position_ids) == 1
        fill_indices = sorted([t.fill_index for t in result.trades])
        assert fill_indices == [0, 1, 2, 3]
        assert all(t.total_fills == 4 for t in result.trades)

    def test_partial_tp_lots_sum_to_total(self):
        """The sum of fill lots should equal the full position lot size."""
        cfg = self._cfg(
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 10.0, 'allocation_pct': 25.0},
                    {'pips': 20.0, 'allocation_pct': 25.0},
                    {'pips': 30.0, 'allocation_pct': 25.0},
                    {'pips': 40.0, 'allocation_pct': 25.0},
                ],
            },
            activity={'trades_per_day': 1, 'win_rate': 1.0, 'days': 1, 'outcome_mode': 'fixed_wins_first'},
        )
        result = SimulationEngine(cfg, seed=42).run(days=1)
        assert all(t.lots == result.trades[0].lots for t in result.trades)

    def test_loss_with_tp_levels_single_fill(self):
        """A losing trade with TP levels should create a single fill at SL."""
        cfg = self._cfg(
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 10.0, 'allocation_pct': 50.0},
                    {'pips': 20.0, 'allocation_pct': 50.0},
                ],
            },
            activity={'trades_per_day': 1, 'win_rate': 0.0, 'days': 1, 'outcome_mode': 'fixed_losses_first'},
        )
        result = SimulationEngine(cfg, seed=42).run(days=1)
        assert len(result.trades) == 1
        assert result.trades[0].outcome == "LOSS"
        assert result.trades[0].fill_index == 0
        assert result.trades[0].total_fills == 1
        assert result.trades[0].pips_gross == -10.0

    def test_no_tp_levels_backward_compatible(self):
        """Without tp_levels, engine behaves as before (single fill per trade)."""
        cfg = self._cfg(
            activity={'trades_per_day': 3, 'win_rate': 0.55, 'days': 2, 'outcome_mode': 'fixed_wins_first'},
        )
        result = SimulationEngine(cfg, seed=42).run(days=2)
        assert len(result.trades) == 6
        assert all(t.fill_index == 0 for t in result.trades)
        assert all(t.total_fills == 1 for t in result.trades)

    def test_mixed_wins_and_losses_with_tp_levels(self):
        """Mixed outcomes: wins get multiple fills, losses get single fill."""
        cfg = self._cfg(
            risk={
                'sl_pips': 10.0,
                'tp_pips': 30.0,
                'fee_pips': 1.0,
                'tp_levels': [
                    {'pips': 10.0, 'allocation_pct': 50.0},
                    {'pips': 20.0, 'allocation_pct': 50.0},
                ],
            },
            activity={
                'trades_per_day': 2,
                'win_rate': 0.5,
                'days': 2,
                'outcome_mode': 'fixed_wins_first',
            },
        )
        result = SimulationEngine(cfg, seed=42).run(days=2)
        wins = [t for t in result.trades if t.outcome == "WIN"]
        losses = [t for t in result.trades if t.outcome == "LOSS"]
        assert len(wins) == 4
        assert len(losses) == 2
        assert len(result.trades) == 6