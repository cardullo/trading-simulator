"""Unit tests for lot sizing engine."""

import pytest

from trading_simulator.lot_sizing import calculate_lot_size, validate_lot_size, LotSizingResult


class TestCalculateLotSize:
    """Tests for the calculate_lot_size function."""

    def test_standard_calculation(self):
        """Basic lot size calculation: €1000, 1% risk, 10 pips SL, $10/pip/lot."""
        result = calculate_lot_size(
            wallet_balance=1000.0,
            risk_per_trade=0.01,
            stop_loss_pips=10.0,
            pip_value_per_lot=10.0,
        )
        # risk_amount = 1000 * 0.01 = 10
        # raw_lots = 10 / (10 * 10) = 0.1
        assert result.risk_amount == 10.0
        assert result.raw_lots == 0.1
        assert result.lots == 0.1
        assert result.rounded is False  # 0.1 is already on 0.01 step

    def test_rounding_to_001_step(self):
        """Raw lots of 0.135 should truncate (floor) to 0.13 by default (cTrader/IC Markets),
        or round to 0.14 when rounding_mode='round'."""
        result_floor = calculate_lot_size(
            wallet_balance=1000.0,
            risk_per_trade=0.01,
            stop_loss_pips=7.4,  # raw = 10 / (7.4 * 10) = 0.1351...
            pip_value_per_lot=10.0,
            rounding_mode="floor",
        )
        assert result_floor.lots == 0.13
        assert result_floor.rounded is True

        result_round = calculate_lot_size(
            wallet_balance=1000.0,
            risk_per_trade=0.01,
            stop_loss_pips=7.4,
            pip_value_per_lot=10.0,
            rounding_mode="round",
        )
        assert result_round.lots == 0.14
        assert result_round.rounded is True

    def test_min_lot_enforcement(self):
        """Raw lots below 0.01 should be bumped to 0.01."""
        result = calculate_lot_size(
            wallet_balance=100.0,  # very small balance
            risk_per_trade=0.01,   # risk = 1
            stop_loss_pips=100.0,  # raw = 1 / (100 * 10) = 0.001
            pip_value_per_lot=10.0,
        )
        assert result.lots == 0.01
        assert result.min_lot_applied is True
        assert result.raw_lots < 0.01

    def test_max_lot_cap(self):
        """Raw lots above max_lot should be capped."""
        result = calculate_lot_size(
            wallet_balance=100000.0,
            risk_per_trade=0.05,  # risk = 5000
            stop_loss_pips=1.0,   # raw = 5000 / (1 * 10) = 500 lots
            pip_value_per_lot=10.0,
            max_lot=100.0,
        )
        assert result.lots == 100.0
        assert result.max_lot_applied is True

    def test_validate_lot_size_valid(self):
        """Valid lot sizes should pass validation."""
        assert validate_lot_size(0.01) is True
        assert validate_lot_size(0.50) is True
        assert validate_lot_size(1.00) is True
        assert validate_lot_size(0.01, max_lot=10.0) is True

    def test_validate_lot_size_invalid(self):
        """Invalid lot sizes should fail validation."""
        assert validate_lot_size(0.005) is False  # below min
        assert validate_lot_size(0.015) is False  # not aligned to 0.01 step
        assert validate_lot_size(0.01, max_lot=0.005) is False  # above max

    def test_input_validation_zero_balance(self):
        """Zero or negative balance should raise ValueError."""
        with pytest.raises(ValueError, match="wallet_balance must be positive"):
            calculate_lot_size(0.0, 0.01, 10.0, 10.0)
        with pytest.raises(ValueError, match="wallet_balance must be positive"):
            calculate_lot_size(-100.0, 0.01, 10.0, 10.0)

    def test_input_validation_zero_sl(self):
        """Zero or negative stop loss should raise ValueError."""
        with pytest.raises(ValueError, match="stop_loss_pips must be positive"):
            calculate_lot_size(1000.0, 0.01, 0.0, 10.0)

    def test_input_validation_invalid_risk(self):
        """Invalid risk percentage should raise ValueError."""
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            calculate_lot_size(1000.0, 0.0, 10.0, 10.0)
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            calculate_lot_size(1000.0, 1.5, 10.0, 10.0)

    def test_euro_usd_pip_value(self):
        """Standard EUR/USD: 1 lot = $10 per pip."""
        result = calculate_lot_size(
            wallet_balance=5000.0,
            risk_per_trade=0.02,  # risk = €100
            stop_loss_pips=20.0,
            pip_value_per_lot=10.0,  # EUR/USD standard
        )
        # raw = 100 / (20 * 10) = 0.5 lots
        assert result.lots == 0.5
        assert result.risk_amount == 100.0

    def test_rounding_edge_case_exact_half(self):
        """Raw lots exactly at 0.005 should floor to lower step by default, or round to even if rounding_mode='round'."""
        result_floor = calculate_lot_size(
            wallet_balance=1350.0,
            risk_per_trade=0.01,
            stop_loss_pips=10.0,
            pip_value_per_lot=10.0,
            rounding_mode="floor",
        )
        # raw = 13.5 / 100 = 0.135 -> floored to 0.13
        assert result_floor.lots == 0.13

        result_round = calculate_lot_size(
            wallet_balance=1350.0,
            risk_per_trade=0.01,
            stop_loss_pips=10.0,
            pip_value_per_lot=10.0,
            rounding_mode="round",
        )
        # raw = 0.135 -> rounded to 0.14 (nearest even)
        assert result_round.lots == 0.14

    def test_floor_vs_round_regression(self):
        """Task 5.2 regression test: verify floor never exceeds round and never over-allocates risk."""
        balances = [100.0, 523.0, 1000.0, 1350.0, 9999.0]
        sl_distances = [5.0, 7.3, 10.0, 15.6, 25.0]
        for b in balances:
            for sl in sl_distances:
                res_floor = calculate_lot_size(b, 0.01, sl, 10.0, rounding_mode="floor")
                res_round = calculate_lot_size(b, 0.01, sl, 10.0, rounding_mode="round")
                assert res_floor.lots <= res_round.lots
                # Risk incurred with floored lots should never exceed risk_amount (unless clamped to min_lot)
                actual_risk = res_floor.lots * sl * 10.0
                if not res_floor.min_lot_applied:
                    assert actual_risk <= res_floor.risk_amount + 1e-9


class TestLotSizingResult:
    """Tests for the LotSizingResult dataclass."""

    def test_result_defaults(self):
        """LotSizingResult should have sensible defaults."""
        result = LotSizingResult(
            lots=0.1,
            risk_amount=10.0,
            stop_loss_pips=10.0,
            pip_value_per_lot=10.0,
            raw_lots=0.1,
            rounded=False,
        )
        assert result.min_lot_applied is False
        assert result.max_lot_applied is False