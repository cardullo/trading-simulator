"""Lot size calculation with broker rounding rules and constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LotSizingResult:
    lots: float
    risk_amount: float
    stop_loss_pips: float
    pip_value_per_lot: float
    raw_lots: float
    rounded: bool
    min_lot_applied: bool = False
    max_lot_applied: bool = False


def calculate_lot_size(
    wallet_balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value_per_lot: float,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
    max_lot: float | None = None,
) -> LotSizingResult:
    """Calculate lot size based on risk % and stop loss distance.

    Rules:
    - Risk amount = wallet_balance * risk_per_trade
    - Raw lots = risk_amount / (stop_loss_pips * pip_value_per_lot)
    - Round to nearest lot_step (0.01 lot granularity)
    - Enforce min_lot (default 0.01)
    - Optionally enforce max_lot cap

    Raises ValueError for invalid inputs (zero balance, zero SL, etc.).
    """
    # Validate inputs
    if wallet_balance <= 0:
        raise ValueError(f"wallet_balance must be positive, got {wallet_balance}")
    if risk_per_trade <= 0 or risk_per_trade > 1:
        raise ValueError(f"risk_per_trade must be in (0, 1], got {risk_per_trade}")
    if stop_loss_pips <= 0:
        raise ValueError(f"stop_loss_pips must be positive, got {stop_loss_pips}")
    if pip_value_per_lot <= 0:
        raise ValueError(f"pip_value_per_lot must be positive, got {pip_value_per_lot}")
    if min_lot <= 0:
        raise ValueError(f"min_lot must be positive, got {min_lot}")
    if lot_step <= 0:
        raise ValueError(f"lot_step must be positive, got {lot_step}")
    if max_lot is not None and max_lot <= 0:
        raise ValueError(f"max_lot must be positive, got {max_lot}")
    if max_lot is not None and max_lot < min_lot:
        raise ValueError(f"max_lot ({max_lot}) must be >= min_lot ({min_lot})")

    risk_amount = wallet_balance * risk_per_trade
    raw_lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

    # Round to nearest lot step using Decimal-like precision to avoid float artifacts
    rounded_lots = round(raw_lots / lot_step) * lot_step
    rounded_lots = round(rounded_lots, 2)  # 0.01 lot granularity

    # Enforce minimum
    min_lot_applied = False
    if rounded_lots < min_lot:
        rounded_lots = min_lot
        min_lot_applied = True

    # Enforce maximum cap
    max_lot_applied = False
    if max_lot is not None and rounded_lots > max_lot:
        rounded_lots = max_lot
        max_lot_applied = True

    # Final precision cleanup
    rounded_lots = round(rounded_lots, 2)

    was_rounded = abs(rounded_lots - raw_lots) > 1e-9

    return LotSizingResult(
        lots=rounded_lots,
        risk_amount=risk_amount,
        stop_loss_pips=stop_loss_pips,
        pip_value_per_lot=pip_value_per_lot,
        raw_lots=raw_lots,
        rounded=was_rounded,
        min_lot_applied=min_lot_applied,
        max_lot_applied=max_lot_applied,
    )


def validate_lot_size(lots: float, min_lot: float = 0.01, lot_step: float = 0.01, max_lot: float | None = None) -> bool:
    """Validate that a lot size conforms to broker constraints.

    Returns True if the lot size is valid:
    - >= min_lot
    - aligned to lot_step (multiple of 0.01)
    - <= max_lot (if specified)
    """
    if lots < min_lot:
        return False
    # Check alignment to lot_step
    remainder = lots % lot_step
    if remainder > 1e-9 and abs(remainder - lot_step) > 1e-9:
        return False
    if max_lot is not None and lots > max_lot + 1e-9:
        return False
    return True