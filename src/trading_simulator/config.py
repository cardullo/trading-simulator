"""Configuration models and validation for the Trading Simulator."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CompoundingMode(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    HIGH_WATER_MARK = "high_water_mark"


class OutcomeMode(str, Enum):
    BERNOULLI = "bernoulli"
    ALTERNATING = "alternating"
    FIXED_WINS_FIRST = "fixed_wins_first"
    FIXED_LOSSES_FIRST = "fixed_losses_first"
    CUSTOM_PATTERN = "custom_pattern"


class TakeProfitLevel(BaseModel):
    """A single take-profit level with pips target and allocation percentage."""

    pips: float = Field(gt=0, description="Take-profit distance in pips from entry")
    allocation_pct: float = Field(gt=0, le=100, description="Percentage of position to close at this level")


class InstrumentConfig(BaseModel):
    symbol: str = "EUR/USD"
    pip_size: Decimal = Decimal("0.0001")
    contract_size: int = 100_000


class AccountConfig(BaseModel):
    initial_balance: float = Field(default=1000.0, gt=0)
    risk_per_trade: float = Field(default=0.01, gt=0, le=1)


class RiskConfig(BaseModel):
    sl_pips: float = Field(default=10.0, gt=0)
    tp_pips: float = Field(default=30.0, gt=0)
    fee_pips: float = Field(default=1.0, ge=0)
    compounding_mode: CompoundingMode = CompoundingMode.STANDARD
    tp_levels: Optional[list[TakeProfitLevel]] = None

    @field_validator("tp_levels")
    @classmethod
    def _validate_tp_levels(cls, v: Optional[list[TakeProfitLevel]]) -> Optional[list[TakeProfitLevel]]:
        if v is not None:
            if len(v) < 1:
                raise ValueError("tp_levels must contain at least one level")
            # Sort by pips ascending
            v.sort(key=lambda x: x.pips)
            # Validate allocations sum to 100%
            total_alloc = sum(t.allocation_pct for t in v)
            if abs(total_alloc - 100.0) > 0.01:
                raise ValueError(f"TP level allocations must sum to 100%, got {total_alloc}%")
        return v


class PyramidMode(str, Enum):
    FIXED = "fixed"
    PERCENT = "percent"


class PyramidConfig(BaseModel):
    enabled: bool = False
    trigger_pips: float = Field(default=5.0, gt=0, description="Unrealized profit in pips to trigger adding volume")
    max_adds: int = Field(default=3, gt=0, description="Maximum number of times to add to position")
    lot_mode: PyramidMode = PyramidMode.FIXED
    lot_size: float = Field(default=0.01, gt=0, description="Fixed lots to add per trigger (when lot_mode=fixed)")
    lot_pct: float = Field(default=25.0, gt=0, le=100, description="% of original position to add per trigger (when lot_mode=percent)")
    be_trigger_pips: Optional[float] = Field(default=None, gt=0, description="Unrealized profit to move SL to breakeven (None = disabled)")


class ActivityConfig(BaseModel):
    trades_per_day: int = Field(default=10, gt=0)
    win_rate: float = Field(default=0.55, ge=0, le=1)
    days: int = Field(default=30, gt=0)
    outcome_mode: OutcomeMode = OutcomeMode.BERNOULLI
    custom_pattern: Optional[str] = None

    @field_validator("custom_pattern")
    @classmethod
    def _validate_custom_pattern(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not v:
                raise ValueError("custom_pattern cannot be empty")
            if not all(c in ("W", "L") for c in v.upper()):
                raise ValueError("custom_pattern must contain only 'W' and 'L' characters")
        return v

    @model_validator(mode="after")
    def _validate_outcome_mode_consistency(self) -> "ActivityConfig":
        if self.outcome_mode == OutcomeMode.CUSTOM_PATTERN and not self.custom_pattern:
            raise ValueError("custom_pattern must be specified when outcome_mode is custom_pattern")
        return self


class TrailingStopConfig(BaseModel):
    enabled: bool = False
    activate_pips: float = Field(default=10.0, gt=0, description="Unrealized profit in pips to activate trailing stop")
    distance_pips: float = Field(default=5.0, gt=0, description="Trailing stop distance in pips from current price")
    partial_close_pct: float = Field(default=0.0, ge=0, le=100, description="% of position to close when trailing stop hits (0 = close all)")


class TPHitMode(str, Enum):
    FIXED = "fixed"       # All TP levels hit (current behavior)
    RANDOM = "random"     # Each TP level hit based on probability
    PERCENT = "percent"   # Exactly X% of TP levels hit (randomly selected)


class TPHitConfig(BaseModel):
    enabled: bool = False
    mode: TPHitMode = TPHitMode.FIXED
    hit_pct: float = Field(default=100.0, ge=0, le=100, description="% of TP levels to hit (for RANDOM: probability per level, for PERCENT: exact % of levels)")


class SimulationConfig(BaseModel):
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    account: AccountConfig = Field(default_factory=AccountConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    pyramid: PyramidConfig = Field(default_factory=PyramidConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    tp_hit: TPHitConfig = Field(default_factory=TPHitConfig)
    seed: Optional[int] = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SimulationConfig":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def load_config(
    file_path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> SimulationConfig:
    """Load configuration from JSON file with optional dictionary overrides."""
    base_data: dict[str, Any] = {}
    if file_path is not None:
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        import json
        with open(p, "r", encoding="utf-8") as f:
            base_data = json.load(f)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base_data[k] = v

    return SimulationConfig.from_dict(base_data)