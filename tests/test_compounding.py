"""Unit tests for compounding modes."""

import pytest

from trading_simulator.compounding import (
    CompoundingMode,
    compute_risk_base,
    CompoundingResult,
)


class TestCompoundingModes:
    """Tests for the three compounding modes."""

    def test_none_mode_uses_initial_balance(self):
        """NONE mode always uses initial balance as risk base."""
        result = compute_risk_base(
            mode=CompoundingMode.NONE,
            current_balance=500.0,
            peak_balance=2000.0,
            initial_balance=1000.0,
        )
        assert result.risk_base == 1000.0
        assert result.mode == CompoundingMode.NONE
        assert result.current_balance == 500.0
        assert result.peak_balance == 2000.0

    def test_standard_mode_uses_current_balance(self):
        """STANDARD mode uses current balance as risk base."""
        result = compute_risk_base(
            mode=CompoundingMode.STANDARD,
            current_balance=1300.0,
            peak_balance=1500.0,
            initial_balance=1000.0,
        )
        assert result.risk_base == 1300.0
        assert result.mode == CompoundingMode.STANDARD

    def test_high_water_mark_uses_peak(self):
        """HIGH_WATER_MARK uses max(peak, current) as risk base."""
        # Peak higher than current
        result = compute_risk_base(
            mode=CompoundingMode.HIGH_WATER_MARK,
            current_balance=1300.0,
            peak_balance=1500.0,
            initial_balance=1000.0,
        )
        assert result.risk_base == 1500.0
        assert result.mode == CompoundingMode.HIGH_WATER_MARK

    def test_high_water_mark_current_higher_than_peak(self):
        """HIGH_WATER_MARK when current exceeds peak should use current."""
        result = compute_risk_base(
            mode=CompoundingMode.HIGH_WATER_MARK,
            current_balance=1600.0,
            peak_balance=1500.0,
            initial_balance=1000.0,
        )
        assert result.risk_base == 1600.0

    def test_high_water_mark_equal_values(self):
        """HIGH_WATER_MARK when current equals peak."""
        result = compute_risk_base(
            mode=CompoundingMode.HIGH_WATER_MARK,
            current_balance=1500.0,
            peak_balance=1500.0,
            initial_balance=1000.0,
        )
        assert result.risk_base == 1500.0

    def test_compounding_result_dataclass(self):
        """CompoundingResult dataclass fields are correct."""
        result = CompoundingResult(
            risk_base=1500.0,
            mode=CompoundingMode.HIGH_WATER_MARK,
            current_balance=1300.0,
            peak_balance=1500.0,
        )
        assert result.risk_base == 1500.0
        assert result.mode == CompoundingMode.HIGH_WATER_MARK
        assert result.current_balance == 1300.0
        assert result.peak_balance == 1500.0

    def test_compounding_mode_enum_values(self):
        """CompoundingMode enum has correct string values."""
        assert CompoundingMode.NONE.value == "none"
        assert CompoundingMode.STANDARD.value == "standard"
        assert CompoundingMode.HIGH_WATER_MARK.value == "high_water_mark"

    def test_unknown_mode_raises_valueerror(self):
        """Unknown compounding mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown compounding mode"):
            compute_risk_base(
                mode="invalid_mode",  # type: ignore[arg-type]
                current_balance=1000.0,
                peak_balance=1000.0,
                initial_balance=1000.0,
            )