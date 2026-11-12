# Trading Simulation & Goal-Modeling Platform

A mathematical trading simulation engine for modeling risk, lot sizing, compounding, spread/fees, and trade outcomes over time. Not a live trading system — strictly a simulation and goal-modeling tool.

## Features (v1)

### Core Engine
- **Account & instrument parameters**: configurable starting capital, EUR/USD instrument, risk per trade %
- **Lot sizing with broker constraints**: 0.01 lot minimum, 0.01 lot increment, correct rounding
- **Compounding modes**: none / standard (current balance) / high-water mark (peak balance)
- **Trade mechanics**: configurable R:R in pips, spread/fees as flat pip cost, activity volume (trades/day)
- **Outcome simulation**: stochastic Bernoulli trial model or deterministic distributions (alternating, fixed wins first, fixed losses first, custom pattern)
- **Performance analytics**: daily, weekly, monthly aggregation — drawdown, profit factor, win rate, return %
- **Export**: full trade log to CSV and JSON
- **Persistence**: SQLite storage of simulation runs, parameters, and trade logs
- **CLI**: Typer + Rich with `run`, `solve`, `export`, `inspect` subcommands
- **Goal-seeking reverse solver**: compute required trades/day, R:R, and win rate to reach a target balance

### M11 Advanced Trade Management
- **Pyramid Entry (M11.1)**: Scale into winning positions with VWAP averaging (cTrader/IC Markets compatible). Fixed lot size or % of original position. Maximum adds limit. Breakeven stop loss trigger.
- **Trailing Stop (M11.2)**: Trail stop loss on winning trades after activation threshold. Configurable activation pips, trailing distance, and partial close %.
- **TP Hit Simulation (M11.3)**: Simulate partial take-profit level hits — FIXED (all hit), RANDOM (probability per level), PERCENT (exact % of levels hit). Missed levels recorded as zero-profit fills for analytics.

### Web Interface (M11.4)
- **Streamlit dashboard**: Interactive simulation controls, equity curve charts, trade history tables, saved runs archive, multi-run comparison, goal-seeking solver tab
- All M11 features accessible via web UI sliders and checkboxes

## Installation

```bash
cd /Users/francesco/Documents/trading-simulator
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# For Streamlit web UI:
pip install streamlit plotly
```

## CLI Usage

### Run a simulation

```bash
trading-sim run \
  --initial-balance 1000 \
  --risk-per-trade 0.01 \
  --sl-pips 10 \
  --tp-pips 30 \
  --fee-pips 1 \
  --trades-per-day 10 \
  --win-rate 0.55 \
  --days 30 \
  --compounding standard \
  --seed 42
```

### With Pyramid Entry

```bash
trading-sim run \
  --initial-balance 1000 \
  --risk-per-trade 0.01 \
  --sl-pips 10 \
  --tp-pips 30 \
  --fee-pips 1 \
  --trades-per-day 5 \
  --win-rate 1.0 \
  --days 1 \
  --compounding none \
  --outcome-mode fixed_wins_first \
  --pyramid-enabled \
  --pyramid-trigger-pips 5 \
  --pyramid-max-adds 3 \
  --pyramid-lot-mode fixed \
  --pyramid-lot-size 0.05 \
  --seed 42
```

### With Trailing Stop

```bash
trading-sim run \
  --initial-balance 1000 \
  --risk-per-trade 0.01 \
  --sl-pips 10 \
  --tp-pips 30 \
  --fee-pips 1 \
  --trades-per-day 5 \
  --win-rate 1.0 \
  --days 1 \
  --compounding none \
  --outcome-mode fixed_wins_first \
  --trailing-enabled \
  --trailing-activate-pips 10 \
  --trailing-distance-pips 5 \
  --seed 42
```

### With TP Hit Simulation

```bash
trading-sim run \
  --initial-balance 1000 \
  --risk-per-trade 0.01 \
  --sl-pips 10 \
  --tp-pips 30 \
  --fee-pips 1 \
  --trades-per-day 2 \
  --win-rate 1.0 \
  --days 1 \
  --compounding none \
  --outcome-mode fixed_wins_first \
  --tp-levels '[{"pips":10,"allocation_pct":25},{"pips":20,"allocation_pct":25},{"pips":30,"allocation_pct":50}]' \
  --tp-hit-enabled \
  --tp-hit-mode random \
  --tp-hit-pct 70 \
  --seed 42
```

### Goal-seeking solver

```bash
trading-sim solve \
  --initial-balance 1000 \
  --target-balance 1000000 \
  --months 12
```

### Export trade log

```bash
trading-sim export --run-id <RUN_ID> --format csv --out data/exports/
```

### Inspect a stored run

```bash
trading-sim inspect --run-id <RUN_ID>
```

### Launch Web UI

```bash
streamlit run src/trading_simulator/web/streamlit_app.py
```

## Project Structure

```text
trading-simulator/
├── pyproject.toml
├── README.md
├── src/trading_simulator/     # Core package
│   ├── config.py              # Pydantic configuration models
│   ├── models.py              # Domain models (Account, Instrument, Trade, etc.)
│   ├── lot_sizing.py          # Lot size calculation with broker constraints
│   ├── compounding.py         # NONE / STANDARD / HIGH_WATER_MARK modes
│   ├── engine.py              # Core simulation loop
│   ├── analytics.py           # Daily/weekly/monthly performance aggregation
│   ├── database.py            # SQLite persistence
│   ├── export.py              # CSV and JSON trade log export
│   ├── solver.py              # Goal-seeking reverse solver
│   ├── cli/
│   │   ├── main.py            # Typer CLI entry point
│   │   ├── run.py             # Run command (placeholder)
│   │   ├── solve.py           # Solve command (placeholder)
│   │   ├── export.py          # Export command
│   │   └── inspect.py         # Inspect command
│   └── web/
│       └── streamlit_app.py   # Streamlit web interface
├── tests/                     # Pytest suite (150 tests)
├── config/                    # Configuration files
├── data/                      # SQLite DB and exports (gitignored)
├── docs/                      # Architecture & specs
└── notebooks/                 # Analysis notebooks
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=trading_simulator

# Lint
ruff check src/
black src/

# Type check
mypy src/
```

## Mathematical Correctness

All simulation math is verified by integration tests covering:

- **Lot sizing alignment**: 0.01 step, min 0.01 lot enforcement
- **Balance chain integrity**: balance_before = previous balance_after
- **Profit calculation**: profit = pips_net × lots × pip_value_per_lot
- **VWAP averaging for pyramid entries** (cTrader/IC Markets compatible)
- **Trailing stop activation and hit probability model**
- **TP hit mode correctness**: FIXED, RANDOM, PERCENT modes
- **No negative balances** under all feature combinations

Test count: 150 tests passing (137 core + 13 M11 feature tests)