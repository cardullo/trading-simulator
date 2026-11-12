"""Unit tests for deterministic win/loss outcome modes."""

import pytest

from trading_simulator.config import OutcomeMode, SimulationConfig
from trading_simulator.engine import SimulationEngine


class TestOutcomeModes:
    """Tests for deterministic outcome modes."""

    def _cfg(self, **overrides):
        base = {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            "activity": {"trades_per_day": 4, "win_rate": 0.5, "days": 3},
            "seed": 42,
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base:
                base[key].update(value)
            else:
                base[key] = value
        return SimulationConfig(**base)

    def test_bernoulli_default(self):
        """BERNOULLI is the default outcome mode."""
        cfg = self._cfg()
        assert cfg.activity.outcome_mode == OutcomeMode.BERNOULLI

    def test_alternating_produces_wl_pattern(self):
        """ALTERNATING mode produces W, L, W, L, ... pattern."""
        cfg = self._cfg(activity={"outcome_mode": "alternating"})
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        expected = ["WIN", "LOSS"] * 6
        assert outcomes == expected
        assert result.win_rate_actual == 0.5

    def test_fixed_wins_first(self):
        """FIXED_WINS_FIRST puts all wins before all losses."""
        cfg = self._cfg(activity={"outcome_mode": "fixed_wins_first"})
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        # 12 trades * 0.5 win rate = 6 wins first, then 6 losses
        assert outcomes == ["WIN"] * 6 + ["LOSS"] * 6
        assert result.win_rate_actual == 0.5

    def test_fixed_losses_first(self):
        """FIXED_LOSSES_FIRST puts all losses before all wins."""
        cfg = self._cfg(activity={"outcome_mode": "fixed_losses_first"})
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        # 12 trades * 0.5 loss rate = 6 losses first, then 6 wins
        assert outcomes == ["LOSS"] * 6 + ["WIN"] * 6
        assert result.win_rate_actual == 0.5

    def test_custom_pattern_repeats(self):
        """CUSTOM_PATTERN repeats the specified W/L pattern."""
        cfg = self._cfg(
            activity={
                "outcome_mode": "custom_pattern",
                "custom_pattern": "WWLL",
            }
        )
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        expected = ["WIN", "WIN", "LOSS", "LOSS"] * 3
        assert outcomes == expected

    def test_custom_pattern_single_char(self):
        """CUSTOM_PATTERN with single 'W' means all wins."""
        cfg = self._cfg(
            activity={
                "outcome_mode": "custom_pattern",
                "custom_pattern": "W",
            }
        )
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        assert all(o == "WIN" for o in outcomes)
        assert result.win_rate_actual == 1.0

    def test_custom_pattern_case_insensitive(self):
        """CUSTOM_PATTERN accepts lowercase w/l."""
        cfg = self._cfg(
            activity={
                "outcome_mode": "custom_pattern",
                "custom_pattern": "wlwl",
            }
        )
        result = SimulationEngine(cfg, seed=42).run(days=3)
        outcomes = [t.outcome for t in result.trades]
        expected = ["WIN", "LOSS"] * 6
        assert outcomes == expected

    def test_custom_pattern_requires_value(self):
        """CUSTOM_PATTERN mode requires custom_pattern to be set."""
        with pytest.raises(ValueError, match="custom_pattern must be specified"):
            cfg = self._cfg(activity={"outcome_mode": "custom_pattern"})
            SimulationEngine(cfg, seed=42)

    def test_custom_pattern_invalid_chars(self):
        """CUSTOM_PATTERN with invalid characters raises ValueError."""
        with pytest.raises(ValueError, match="custom_pattern must contain only"):
            self._cfg(
                activity={
                    "outcome_mode": "custom_pattern",
                    "custom_pattern": "WWXL",
                }
            )

    def test_outcome_mode_enum_values(self):
        """OutcomeMode enum has correct string values."""
        assert OutcomeMode.BERNOULLI.value == "bernoulli"
        assert OutcomeMode.ALTERNATING.value == "alternating"
        assert OutcomeMode.FIXED_WINS_FIRST.value == "fixed_wins_first"
        assert OutcomeMode.FIXED_LOSSES_FIRST.value == "fixed_losses_first"
        assert OutcomeMode.CUSTOM_PATTERN.value == "custom_pattern"

    def test_deterministic_seed_reproducibility(self):
        """Same seed produces identical results across runs."""
        cfg = self._cfg(activity={"outcome_mode": "fixed_wins_first"})
        result_a = SimulationEngine(cfg, seed=42).run(days=3)
        result_b = SimulationEngine(cfg, seed=42).run(days=3)
        assert [t.profit_euros for t in result_a.trades] == [t.profit_euros for t in result_b.trades]
        assert result_a.final_balance == result_b.final_balance

    def test_fixed_wins_first_profitable(self):
        """FIXED_WINS_FIRST with positive expectancy should be profitable."""
        cfg = self._cfg(
            account={"initial_balance": 1000.0, "risk_per_trade": 0.02},
            risk={"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            activity={"outcome_mode": "fixed_wins_first", "trades_per_day": 5, "win_rate": 0.6, "days": 5},
        )
        result = SimulationEngine(cfg, seed=42).run(days=5)
        # With 60% win rate and 1:3 R:R, should be profitable
        assert result.final_balance > result.initial_balance
        assert result.win_rate_actual == 0.6


class TestOutcomeModeIntegration:
    """Integration tests for outcome modes with the full engine."""

    def _cfg(self, **overrides):
        base = {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0},
            "activity": {"trades_per_day": 4, "win_rate": 0.5, "days": 3},
            "seed": 42,
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base:
                base[key].update(value)
            else:
                base[key] = value
        return SimulationConfig(**base)

    def test_all_modes_produce_valid_trades(self):
        """All outcome modes produce trades with valid fields."""
        for mode in ["bernoulli", "alternating", "fixed_wins_first", "fixed_losses_first"]:
            cfg = self._cfg(activity={"outcome_mode": mode})
            result = SimulationEngine(cfg, seed=42).run(days=3)
            for t in result.trades:
                assert t.lots > 0
                assert t.risk_amount > 0
                assert t.balance_before > 0
                assert t.balance_after > 0
                assert t.outcome in ("WIN", "LOSS")
                assert t.pips_net == t.pips_gross - t.fee_pips

    def test_custom_pattern_with_compounding(self):
        """CUSTOM_PATTERN works with STANDARD compounding."""
        cfg = self._cfg(
            risk={"compounding_mode": "standard"},
            activity={
                "outcome_mode": "custom_pattern",
                "custom_pattern": "WWL",
                "trades_per_day": 3,
                "days": 4,
            },
        )
        result = SimulationEngine(cfg, seed=42).run(days=4)
        outcomes = [t.outcome for t in result.trades]
        expected = ["WIN", "WIN", "LOSS"] * 4
        assert outcomes == expected
        # Lot sizes should vary with compounding
        lot_sizes = {t.lots for t in result.trades}
        assert len(lot_sizes) > 1  # compounding changes lot sizes
