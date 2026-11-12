"""Compounding modes for risk base calculation."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class CompoundingMode(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    HIGH_WATER_MARK = "high_water_mark"


@dataclass
class CompoundingResult:
    risk_base: float
    mode: CompoundingMode
    current_balance: float
    peak_balance: float


def compute_risk_base(
    mode: CompoundingMode,
    current_balance: float,
    peak_balance: float,
    initial_balance: float,
) -> CompoundingResult:
    """Compute the risk base for a trade based on compounding mode.

    - NONE: risk base = initial_balance (fixed)
    - STANDARD: risk base = current_balance (dynamic)
    - HIGH_WATER_MARK: risk base = max(peak_balance, current_balance)
    """
    if mode == CompoundingMode.NONE:
        risk_base = initial_balance
    elif mode == CompoundingMode.STANDARD:
        risk_base = current_balance
    elif mode == CompoundingMode.HIGH_WATER_MARK:
        risk_base = max(peak_balance, current_balance)
    else:
        raise ValueError(f"Unknown compounding mode: {mode}")

    return CompoundingResult(
        risk_base=risk_base,
        mode=mode,
        current_balance=current_balance,
        peak_balance=peak_balance,
    )