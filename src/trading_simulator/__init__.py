"""Trading Simulation & Goal-Modeling Platform.

Core engine for simulating trading mechanics: risk, lot sizing, compounding,
spread/fees, and trade outcomes over time.
"""

from trading_simulator.config import (
    CompoundingMode,
    InstrumentConfig,
    RiskConfig,
    AccountConfig,
    SimulationConfig,
    OutcomeMode,
    PyramidConfig,
    PyramidMode,
    TrailingStopConfig,
    TPHitConfig,
    TPHitMode,
)
from trading_simulator.models import Account, Instrument, Trade, TradeResult, EquityPoint
from trading_simulator.lot_sizing import calculate_lot_size, validate_lot_size, LotSizingResult
from trading_simulator.compounding import compute_risk_base
from trading_simulator.engine import SimulationEngine, SimulationResult
from trading_simulator.analytics import (
    PerformanceMetrics,
    aggregate_daily,
    aggregate_weekly,
    aggregate_monthly,
    aggregate_all_periods,
)
from trading_simulator.database import DatabaseManager
from trading_simulator.export import export_csv, export_json
from trading_simulator.solver import solve, SolverResult, SolverScenario

__version__ = "0.1.5"

__all__ = [
    "CompoundingMode",
    "OutcomeMode",
    "PyramidMode",
    "InstrumentConfig",
    "RiskConfig",
    "AccountConfig",
    "SimulationConfig",
    "PyramidConfig",
    "TrailingStopConfig",
    "TPHitConfig",
    "TPHitMode",
    "Account",
    "Instrument",
    "Trade",
    "TradeResult",
    "EquityPoint",
    "calculate_lot_size",
    "validate_lot_size",
    "LotSizingResult",
    "compute_risk_base",
    "SimulationEngine",
    "SimulationResult",
    "PerformanceMetrics",
    "aggregate_daily",
    "aggregate_weekly",
    "aggregate_monthly",
    "aggregate_all_periods",
    "DatabaseManager",
    "export_csv",
    "export_json",
    "solve",
    "SolverResult",
    "SolverScenario",
]