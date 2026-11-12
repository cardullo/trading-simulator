# Architecture

## Overview
The Trading Simulation & Goal-Modeling Platform simulates trading mechanics:
risk, lot sizing, compounding, spread/fees, and trade outcomes over time.

## Modules
- `config.py` — Pydantic-based configuration models with validation
- `models.py` — Domain models (Account, Instrument, Trade, TradeResult, EquityPoint)
- `lot_sizing.py` — Lot size calculation with 0.01 lot rounding rules
- `compounding.py` — NONE / STANDARD / HIGH_WATER_MARK compounding modes
- `engine.py` — Core simulation loop (Bernoulli outcomes, fees, equity curve, pyramid, trailing stop, TP hit)
- `analytics.py` — Daily/weekly/monthly performance aggregation
- `database.py` — SQLite persistence for runs, trades, equity snapshots
- `export.py` — CSV and JSON trade log export
- `solver.py` — Goal-seeking reverse solver with mathematical proofs
- `cli/` — Typer + Rich CLI with `run`, `solve`, `export`, `inspect`
- `web/` — Streamlit web interface with all M11 features

## M11 Advanced Trade Management

### Pyramid Entry (M11.1)
- **VWAP averaging** (cTrader/IC Markets compatible): When adding volume to an existing position,
  the new average entry price is calculated as:
  `new_avg = (old_avg × old_lots + add_price × add_lots) / (old_lots + add_lots)`
- **Lot modes**: FIXED (constant lots per add) or PERCENT (% of original position)
- **Trigger**: Add when unrealized profit from current VWAP average reaches trigger_pips
- **Max adds**: Configurable maximum number of pyramid adds per position
- **Breakeven trigger**: Optional stop loss move to entry price after profit threshold

### Trailing Stop (M11.2)
- **Activation**: Trailing stop activates when unrealized profit reaches activate_pips
- **Trailing level**: `activate_pips - distance_pips`
- **Hit probability model**: P(hit trailing) = distance_pips / (remaining_to_tp + distance_pips)
- **Partial close**: Optional % of position to close when trailing stop hits

### TP Hit Simulation (M11.3)
- **FIXED mode**: All TP levels always hit (original behavior)
- **RANDOM mode**: Each TP level independently hit with probability = hit_pct/100
- **PERCENT mode**: Exactly hit_pct% of TP levels randomly selected to hit
- Missed levels recorded as zero-profit fills for analytics tracking

## Design Decisions
- Use `Decimal` for money where precision matters, `float` for performance in simulation loops
- Deterministic simulations via configurable random seed
- SQLite for local file-based persistence
- Broker lot constraints: min 0.01 lots, 0.01 lot increment
- Pyramid takes priority over TP levels and trailing stop in execution order