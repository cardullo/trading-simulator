"""
Batch simulation runner for M12 - Export System.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_simulator.config import (
    CompoundingMode,
    OutcomeMode,
    PyramidMode,
    SimulationConfig,
    TPHitMode,
)
from trading_simulator.engine import SimulationEngine
from trading_simulator.database import DatabaseManager
from trading_simulator.export import export_csv, export_json

SCENARIOS: dict[str, dict] = {
    "sim_001_baseline_bernoulli": {
        "name": "Baseline Bernoulli Simulation",
        "description": "Baseline simulation with standard compounding.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 10, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_002_compounding_hwm": {
        "name": "Compounding High-Water Mark",
        "description": "Tests high-water mark compounding.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.HIGH_WATER_MARK},
            "activity": {"trades_per_day": 10, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_003_compounding_none": {
        "name": "Compounding None (Fixed Risk)",
        "description": "Tests fixed risk sizing on initial balance.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.NONE},
            "activity": {"trades_per_day": 10, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_004_pyramid_fixed": {
        "name": "Pyramid Entry - Fixed Lot Mode",
        "description": "Tests pyramid entry with fixed lot size.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.60, "days": 30,
                         "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            "pyramid": {"enabled": True, "trigger_pips": 5.0, "max_adds": 3,
                        "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                        "lot_pct": 25.0, "be_trigger_pips": 5.0},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_005_pyramid_percent": {
        "name": "Pyramid Entry - Percent Lot Mode",
        "description": "Tests pyramid entry where each add is 25% of position.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.60, "days": 30,
                         "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            "pyramid": {"enabled": True, "trigger_pips": 5.0, "max_adds": 3,
                        "lot_mode": PyramidMode.PERCENT, "lot_size": 0.01,
                        "lot_pct": 25.0, "be_trigger_pips": 5.0},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_006_trailing_stop": {
        "name": "Trailing Stop",
        "description": "Tests trailing stop functionality.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 10, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": True, "activate_pips": 10.0,
                              "distance_pips": 5.0, "partial_close_pct": 0.0},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_007_tp_hit_random": {
        "name": "TP Hit Simulation - Random Mode",
        "description": "Tests partial take-profit where each TP level hit "
                       "with 70% probability.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "tp_levels": [
                         {"pips": 10.0, "allocation_pct": 25.0},
                         {"pips": 20.0, "allocation_pct": 25.0},
                         {"pips": 30.0, "allocation_pct": 50.0},
                     ],
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": True, "mode": TPHitMode.RANDOM, "hit_pct": 70.0},
            "seed": 42,
        },
    },
    "sim_008_tp_hit_percent": {
        "name": "TP Hit Simulation - Percent Mode",
        "description": "Tests partial take-profit where exactly 50% hit.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "tp_levels": [
                         {"pips": 10.0, "allocation_pct": 25.0},
                         {"pips": 20.0, "allocation_pct": 25.0},
                         {"pips": 30.0, "allocation_pct": 50.0},
                     ],
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.55, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": True, "mode": TPHitMode.PERCENT, "hit_pct": 50.0},
            "seed": 42,
        },
    },
    "sim_009_combined_all_features": {
        "name": "Combined All Features",
        "description": "Tests all features together.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.58, "days": 30,
                         "outcome_mode": OutcomeMode.FIXED_WINS_FIRST},
            "pyramid": {"enabled": True, "trigger_pips": 5.0, "max_adds": 2,
                        "lot_mode": PyramidMode.FIXED, "lot_size": 0.05,
                        "lot_pct": 25.0, "be_trigger_pips": None},
            "trailing_stop": {"enabled": True, "activate_pips": 10.0,
                              "distance_pips": 5.0, "partial_close_pct": 0.0},
            "tp_hit": {"enabled": True, "mode": TPHitMode.RANDOM, "hit_pct": 60.0},
            "seed": 42,
        },
    },
    "sim_010_high_activity_stress": {
        "name": "High Activity Stress Test",
        "description": "High-frequency simulation with 100 trades per day.",
        "config_overrides": {
            "account": {"initial_balance": 10000.0, "risk_per_trade": 0.005},
            "risk": {"sl_pips": 15.0, "tp_pips": 45.0, "fee_pips": 0.5,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 100, "win_rate": 0.52, "days": 30,
                         "outcome_mode": OutcomeMode.BERNOULLI},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 123,
        },
    },
    "sim_011_deterministic_alternating": {
        "name": "Deterministic Alternating Pattern",
        "description": "Tests deterministic alternating W/L pattern.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.50, "days": 10,
                         "outcome_mode": OutcomeMode.ALTERNATING},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
    "sim_012_deterministic_custom_pattern": {
        "name": "Deterministic Custom Pattern",
        "description": "Tests custom win/loss pattern WWLLWWLL.",
        "config_overrides": {
            "account": {"initial_balance": 1000.0, "risk_per_trade": 0.01},
            "risk": {"sl_pips": 10.0, "tp_pips": 30.0, "fee_pips": 1.0,
                     "compounding_mode": CompoundingMode.STANDARD},
            "activity": {"trades_per_day": 5, "win_rate": 0.50, "days": 10,
                         "outcome_mode": OutcomeMode.CUSTOM_PATTERN,
                         "custom_pattern": "WWLLWWLL"},
            "pyramid": {"enabled": False},
            "trailing_stop": {"enabled": False},
            "tp_hit": {"enabled": False},
            "seed": 42,
        },
    },
}

def run_all_simulations() -> None:
    db = DatabaseManager()
    print("=" * 70)
    print("TRADING SIMULATION - BATCH RUNNER")
    print("=" * 70)
    print(f"\nTotal scenarios to run: {len(SCENARIOS)}\n")
    for sim_id, sim_config in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"Running: {sim_config['name']} ({sim_id})")
        print(f"{'='*70}")
        try:
            cfg = SimulationConfig.from_dict(sim_config["config_overrides"])
            engine = SimulationEngine(cfg, seed=cfg.seed)
            result = engine.run(days=cfg.activity.days)
            run_dir = create_run_directory(sim_id)
            print(f"  Output directory: {run_dir}")
            write_simulation_config(sim_id, cfg, run_dir)
            write_simulation_readme(sim_id, cfg, result, run_dir)
            write_trade_logs(result, run_dir)
            write_equity_curve(result, run_dir)
            write_performance_summary(result, cfg, run_dir)
            db.save_run(sim_id, cfg, result)
            print(f"  ✓ Simulation complete: {len(result.trades)} trades, "
                  f"balance €{result.final_balance:,.2f}, "
                  f"return {result.total_return_pct:+.2f}%")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*70}")
    print("ALL SIMULATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: data/exports/simulations/")
    print(f"Database: data/runs.db")
    runs = db.list_runs()
    print(f"\nTotal runs in database: {len(runs)}")
    for run in runs:
        print(f"  - {run['run_id']}: balance €{run['final_balance']:,.2f}, "
              f"return {(run['final_balance'] / run['initial_balance'] - 1) * 100:+.2f}%")


def write_simulation_config(sim_id: str, config: SimulationConfig, run_dir: Path) -> Path:
    config_path = run_dir / "simulation_config.json"
    config_data = {
        "simulation_id": sim_id,
        "generated_at": datetime.now().isoformat(),
        "config": config.to_dict(),
    }
    config_path.write_text(json.dumps(config_data, indent=2, default=str))
    return config_path


def write_simulation_config(sim_id: str, config: SimulationConfig, run_dir: Path) -> Path:
    config_path = run_dir / "simulation_config.json"
    config_data = {
        "simulation_id": sim_id,
        "generated_at": datetime.now().isoformat(),
        "config": config.to_dict(),
    }
    config_path.write_text(json.dumps(config_data, indent=2, default=str))
    return config_path


def write_simulation_readme(sim_id: str, config: SimulationConfig, result: Any,
                            run_dir: Path) -> Path:
    readme_path = run_dir / "simulation_readme.md"
    winning = [t for t in result.trades if t.outcome == "WIN"]
    losing = [t for t in result.trades if t.outcome == "LOSS"]
    gross_profit = sum(t.profit_euros for t in winning)
    gross_loss = abs(sum(t.profit_euros for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    readme = f"""# Simulation Run: {sim_id}
## Overview
Simulation executed on {datetime.now().isoformat()}.
## Configuration Summary
| Parameter | Value |
|-----------|-------|
| Initial Balance | €{config.account.initial_balance:,.2f} |
| Risk per Trade | {config.account.risk_per_trade:.2%} |
| Stop Loss | {config.risk.sl_pips} pips |
| Take Profit | {config.risk.tp_pips} pips |
| Fee per Trade | {config.risk.fee_pips} pips |
| Compounding Mode | {config.risk.compounding_mode.value} |
| Trades per Day | {config.activity.trades_per_day} |
| Win Rate | {config.activity.win_rate:.2%} |
| Days | {config.activity.days} |
| Outcome Mode | {config.activity.outcome_mode.value} |
| Seed | {config.seed} |
## Results Summary
| Metric | Value |
|--------|-------|
| Total Trades | {len(result.trades)} |
| Winning Trades | {len(winning)} |
| Losing Trades | {len(losing)} |
| Win Rate | {result.win_rate_actual:.2%} |
| Initial Balance | €{config.account.initial_balance:,.2f} |
| Final Balance | €{result.final_balance:,.2f} |
| Total Return | {result.total_return_pct:+.2f}% |
| Net Profit | €{result.final_balance - config.account.initial_balance:,.2f} |
| Gross Profit | €{gross_profit:,.2f} |
| Gross Loss | €{gross_loss:,.2f} |
| Profit Factor | {profit_factor:.2f} |
| Max Drawdown | {result.max_drawdown_pct:.2f}% |
## Files in this Directory
- `simulation_config.json` - Complete simulation configuration
- `simulation_readme.md` - This file
- `trade_log.csv` - Complete trade-by-trade log
- `trade_log.json` - Trade log in JSON format
- `equity_curve.csv` - Balance, drawdown, and peak per day
- `performance_summary.md` - Detailed performance metrics
## ICMarket/cTrader Compatibility Notes
- Lot sizing uses 0.01 lot minimum and step (matches ICMarkets/cTrader)
- Pyramid entry uses VWAP averaging (matches cTrader position averaging)
- Partial TP closes use FIFO (matches cTrader)
- Fee applied per trade fill (consistent with ICMarkets commission model)
"""
    readme_path.write_text(readme)
    return readme_path


def write_trade_logs(result: Any, run_dir: Path) -> tuple[Path, Path]:
    trades_data = []
    for t in result.trades:
        trades_data.append({
            "trade_number": t.trade_number,
            "day": t.day,
            "lots": t.lots,
            "risk_amount": t.risk_amount,
            "pips_gross": t.pips_gross,
            "pips_net": t.pips_net,
            "fee_pips": t.fee_pips,
            "profit_euros": t.profit_euros,
            "balance_before": t.balance_before,
            "balance_after": t.balance_after,
            "outcome": t.outcome,
            "position_id": t.position_id,
            "fill_index": t.fill_index,
            "total_fills": t.total_fills,
            "pyramid_fill_index": t.pyramid_fill_index,
            "pyramid_total_fills": t.pyramid_total_fills,
            "entry_price": t.entry_price,
            "position_avg_price": t.position_avg_price,
            "trailing_stop_active": t.trailing_stop_active,
            "trailing_stop_pips": t.trailing_stop_pips,
            "trailing_triggered": t.trailing_triggered,
            "tp_level_hit": t.tp_level_hit,
            "tp_level_pips": t.tp_level_pips,
            "timestamp": t.timestamp.isoformat(),
        })
    csv_path = run_dir / "trade_log.csv"
    export_csv(trades_data, csv_path)
    json_path = run_dir / "trade_log.json"
    export_json(trades_data, json_path)
    return csv_path, json_path


def write_equity_curve(result: Any, run_dir: Path) -> Path:
    csv_path = run_dir / "equity_curve.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "balance", "drawdown_pct", "peak_balance"])
        writer.writeheader()
        for e in result.equity_curve:
            writer.writerow({
                "timestamp": e.timestamp.isoformat(),
                "balance": e.balance,
                "drawdown_pct": e.drawdown_pct,
                "peak_balance": e.peak_balance,
            })
    return csv_path


def write_performance_summary(result: Any, config: SimulationConfig, run_dir: Path) -> Path:
    readme_path = run_dir / "performance_summary.md"
    winning = [t for t in result.trades if t.outcome == "WIN"]
    losing = [t for t in result.trades if t.outcome == "LOSS"]
    gross_profit = sum(t.profit_euros for t in winning)
    gross_loss = abs(sum(t.profit_euros for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    content = f"""# Performance Summary - €{config.account.initial_balance:,.2f} → €{result.final_balance:,.2f}
## Trade Statistics
- Total Trades: {len(result.trades)}
- Winning Trades: {len(winning)}
- Losing Trades: {len(losing)}
- Win Rate: {result.win_rate_actual:.2%}
- Loss Rate: {(1 - result.win_rate_actual):.2%}
## Financial Metrics
- Initial Balance: €{config.account.initial_balance:,.2f}
- Final Balance: €{result.final_balance:,.2f}
- Net Profit: €{result.final_balance - config.account.initial_balance:,.2f}
- Total Return: {result.total_return_pct:+.2f}%
- Gross Profit: €{gross_profit:,.2f}
- Gross Loss: €{gross_loss:,.2f}
- Profit Factor: {profit_factor:.2f}
- Max Drawdown: {result.max_drawdown_pct:.2f}%
## Risk Metrics
- Risk per Trade: {config.account.risk_per_trade:.2%}
- Stop Loss: {config.risk.sl_pips} pips
- Take Profit: {config.risk.tp_pips} pips
- Fee per Trade: {config.risk.fee_pips} pips
## Trading Activity
- Trades per Day: {config.activity.trades_per_day}
- Total Days: {config.activity.days}
- Expected Trades: {config.activity.trades_per_day * config.activity.days}
- Actual Trades: {len(result.trades)}
## Compounding: {config.risk.compounding_mode.value}
- Initial Balance: €{config.account.initial_balance:,.2f}
- Risk Base: {{"current balance" if config.risk.compounding_mode.value == "standard" else "peak balance" if config.risk.compounding_mode.value == "high_water_mark" else "fixed initial balance"}}
## Outcome Mode: {config.activity.outcome_mode.value}
- Win Rate: {config.activity.win_rate:.2%}
- Seed: {config.seed}
"""
    readme_path.write_text(content)
    return readme_path


def run_all_simulations() -> None:
    db = DatabaseManager()
    print("=" * 70)
    print("TRADING SIMULATION - BATCH RUNNER")
    print("=" * 70)
    print(f"\nTotal scenarios to run: {len(SCENARIOS)}\n")
    for sim_id, sim_config in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"Running: {sim_config['name']} ({sim_id})")
        print(f"{'='*70}")
        try:
            cfg = SimulationConfig.from_dict(sim_config["config_overrides"])
            engine = SimulationEngine(cfg, seed=cfg.seed)
            result = engine.run(days=cfg.activity.days)
            run_dir = create_run_directory(sim_id)
            print(f"  Output directory: {run_dir}")
            write_simulation_config(sim_id, cfg, run_dir)
            write_simulation_readme(sim_id, cfg, result, run_dir)
            write_trade_logs(result, run_dir)
            write_equity_curve(result, run_dir)
            write_performance_summary(result, cfg, run_dir)
            db.save_run(sim_id, cfg, result)
            print(f"  ✓ Simulation complete: {len(result.trades)} trades, "
                  f"balance €{result.final_balance:,.2f}, "
                  f"return {result.total_return_pct:+.2f}%")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*70}")
    print("ALL SIMULATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: data/exports/simulations/")
    print(f"Database: data/runs.db")
    runs = db.list_runs()
    print(f"\nTotal runs in database: {len(runs)}")
    for run in runs:
        print(f"  - {run['run_id']}: balance €{run['final_balance']:,.2f}, return {run['final_balance'] / run.get('initial_balance', run['final_balance']) - 1:+.2%}")
    winning = [t for t in result.trades if t.outcome == "WIN"]
    losing = [t for t in result.trades if t.outcome == "LOSS"]
    gross_profit = sum(t.profit_euros for t in winning)
    gross_loss = abs(sum(t.profit_euros for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    readme = f"""# Simulation Run: {sim_id}
## Overview
Simulation executed on {datetime.now().isoformat()}.
## Configuration Summary
| Parameter | Value |
|-----------|-------|
| Initial Balance | €{config.account.initial_balance:,.2f} |
| Risk per Trade | {config.account.risk_per_trade:.2%} |
| Stop Loss | {config.risk.sl_pips} pips |
| Take Profit | {config.risk.tp_pips} pips |
| Fee per Trade | {config.risk.fee_pips} pips |
| Compounding Mode | {config.risk.compounding_mode.value} |
| Trades per Day | {config.activity.trades_per_day} |
| Win Rate | {config.activity.win_rate:.2%} |
| Days | {config.activity.days} |
| Outcome Mode | {config.activity.outcome_mode.value} |
| Seed | {config.seed} |
## Results Summary
| Metric | Value |
|--------|-------|
| Total Trades | {len(result.trades)} |
| Winning Trades | {len(winning)} |
| Losing Trades | {len(losing)} |
| Win Rate | {result.win_rate_actual:.2%} |
| Initial Balance | €{config.account.initial_balance:,.2f} |
| Final Balance | €{result.final_balance:,.2f} |
| Total Return | {result.total_return_pct:+.2f}% |
| Net Profit | €{result.final_balance - config.account.initial_balance:,.2f} |
| Gross Profit | €{gross_profit:,.2f} |
| Gross Loss | €{gross_loss:,.2f} |
| Profit Factor | {profit_factor:.2f} |
| Max Drawdown | {result.max_drawdown_pct:.2f}% |
## Files in this Directory
- `simulation_config.json` - Complete simulation configuration
- `simulation_readme.md` - This file
- `trade_log.csv` - Complete trade-by-trade log
- `trade_log.json` - Trade log in JSON format
- `equity_curve.csv` - Balance, drawdown, and peak per day
- `performance_summary.md` - Detailed performance metrics
## ICMarket/cTrader Compatibility Notes
- Lot sizing uses 0.01 lot minimum and step (matches ICMarkets/cTrader)
- Pyramid entry uses VWAP averaging (matches cTrader position averaging)
- Partial TP closes use FIFO (matches cTrader)
- Fee applied per trade fill (consistent with ICMarkets commission model)
"""
    readme_path.write_text(readme)
    return readme_path