"""Domain models for the Trading Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Instrument:
    symbol: str = "EUR/USD"
    pip_size: float = 0.0001
    contract_size: int = 100_000

    @property
    def pip_value_per_lot(self) -> float:
        """Value of 1 pip for 1 lot (in quote currency)."""
        return self.contract_size * self.pip_size


@dataclass
class Account:
    balance: float
    initial_balance: float
    peak_balance: float = 0.0

    def __post_init__(self):
        if self.peak_balance == 0.0:
            self.peak_balance = self.balance


@dataclass
class Trade:
    trade_number: int
    day: int
    lots: float
    risk_amount: float
    pips_gross: float
    pips_net: float
    fee_pips: float
    profit_euros: float
    balance_before: float
    balance_after: float
    outcome: Literal["WIN", "LOSS"]
    timestamp: datetime = field(default_factory=datetime.now)
    # Partial take-profit fill tracking
    position_id: str = ""
    fill_index: int = 0
    total_fills: int = 1
    # Pyramid entry tracking
    pyramid_fill_index: int = 0
    pyramid_total_fills: int = 1
    entry_price: float = 0.0
    position_avg_price: float = 0.0
    # Trailing stop tracking
    trailing_stop_active: bool = False
    trailing_stop_pips: float = 0.0
    trailing_triggered: bool = False
    # Take-profit hit tracking
    tp_level_hit: bool = True
    tp_level_pips: float = 0.0


@dataclass
class TradeResult:
    lots: float
    pips_net: float
    profit_euros: float
    balance_after: float
    outcome: Literal["WIN", "LOSS"]


@dataclass
class EquityPoint:
    timestamp: datetime
    balance: float
    drawdown_pct: float
    peak_balance: float