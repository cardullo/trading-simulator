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
        """Raw lots of 0.135 should round to 0.14 (nearest 0.01 step)."""
        result = calculate_lot_size(
            wallet_balance=1000.0,
            risk_per_trade=0.01,
            stop_loss_pips=7.4,  # raw = 10 / (7.4 * 10) = 0.1351...
            pip_value_per_lot=10.0,
        )
        assert result.lots == 0.14
        assert result.rounded is True

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
        """Raw lots exactly at 0.005 should round to nearest even (banker's)."""
        # Use a balance that produces raw_lots = 0.135 (rounds to 0.14)
        # raw = balance * 0.01 / (sl * pip_value)
        # For raw = 0.135: balance = 0.135 * sl * pip_value / 0.01
        result = calculate_lot_size(
            wallet_balance=1350.0,
            risk_per_trade=0.01,
            stop_loss_pips=10.0,
            pip_value_per_lot=10.0,
        )
        # raw = 13.5 / 100 = 0.135
        # round(0.135 / 0.01) * 0.01 = round(13.5) * 0.01 = 14 * 0.01 = 0.14 (banker's: 14 is even)
        assert result.lots == 0.14


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