"""CLI entry point using Typer + Rich."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_simulator.config import SimulationConfig

app = typer.Typer(
    name="trading-sim",
    help="Trading Simulation & Goal-Modeling Platform",
    no_args_is_help=True,
)
console = Console()


@app.command(name="run")
def run(
    initial_balance: float = typer.Option(1000.0, "--initial-balance", "-b", help="Starting wallet balance"),
    risk_per_trade: float = typer.Option(0.01, "--risk-per-trade", "-r", help="Risk per trade as fraction of balance"),
    sl_pips: float = typer.Option(10.0, "--sl-pips", help="Stop loss distance in pips"),
    tp_pips: float = typer.Option(30.0, "--tp-pips", help="Take profit distance in pips"),
    fee_pips: float = typer.Option(1.0, "--fee-pips", help="Flat fee per trade in pips"),
    trades_per_day: int = typer.Option(10, "--trades-per-day", "-t", help="Number of trades per day"),
    win_rate: float = typer.Option(0.55, "--win-rate", "-w", help="Win rate probability"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to simulate"),
    compounding: str = typer.Option("standard", "--compounding", "-c", help="Compounding mode: none, standard, high_water_mark"),
    outcome_mode: str = typer.Option("bernoulli", "--outcome-mode", "-m", help="Outcome mode: bernoulli, alternating, fixed_wins_first, fixed_losses_first, custom_pattern"),
    custom_pattern: Optional[str] = typer.Option(None, "--custom-pattern", "-p", help="Custom W/L pattern (e.g., 'WWLWLW')"),
    tp_levels: Optional[str] = typer.Option(None, "--tp-levels", help="Partial TP levels as JSON: '[{\"pips\":10,\"alloc\":25},{\"pips\":20,\"alloc\":25}]'"),
    pyramid_enabled: bool = typer.Option(False, "--pyramid-enabled", help="Enable pyramid entry (scale into winning positions)"),
    pyramid_trigger_pips: float = typer.Option(5.0, "--pyramid-trigger-pips", help="Unrealized profit in pips to trigger adding volume"),
    pyramid_max_adds: int = typer.Option(3, "--pyramid-max-adds", help="Maximum number of times to add to position"),
    pyramid_lot_mode: str = typer.Option("fixed", "--pyramid-lot-mode", help="Lot add mode: fixed or percent"),
    pyramid_lot_size: float = typer.Option(0.01, "--pyramid-lot-size", help="Fixed lots to add per trigger (when lot_mode=fixed)"),
    pyramid_lot_pct: float = typer.Option(25.0, "--pyramid-lot-pct", help="% of original position to add per trigger (when lot_mode=percent)"),
    pyramid_be_trigger_pips: Optional[float] = typer.Option(None, "--pyramid-be-trigger-pips", help="Unrealized profit to move SL to breakeven"),
    trailing_enabled: bool = typer.Option(False, "--trailing-enabled", help="Enable trailing stop loss"),
    trailing_activate_pips: float = typer.Option(10.0, "--trailing-activate-pips", help="Unrealized profit in pips to activate trailing stop"),
    trailing_distance_pips: float = typer.Option(5.0, "--trailing-distance-pips", help="Trailing stop distance in pips from current price"),
    trailing_partial_close_pct: float = typer.Option(0.0, "--trailing-partial-close-pct", help="% of position to close when trailing stop hits (0 = close all)"),
    tp_hit_enabled: bool = typer.Option(False, "--tp-hit-enabled", help="Enable TP hit simulation (not all TP levels always hit)"),
    tp_hit_mode: str = typer.Option("fixed", "--tp-hit-mode", help="TP hit mode: fixed, random, percent"),
    tp_hit_pct: float = typer.Option(100.0, "--tp-hit-pct", help="% of TP levels to hit (RANDOM: probability per level, PERCENT: exact % of levels)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to JSON config file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export format: csv, json, or none"),
    out_dir: str = typer.Option("data/exports", "--out-dir", help="Export directory"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save to SQLite"),
) -> None:
    """Run a trading simulation with the given parameters."""
    import uuid
    from datetime import datetime
    from pathlib import Path

    from trading_simulator.config import CompoundingMode, SimulationConfig, load_config
    from trading_simulator.database import DatabaseManager
    from trading_simulator.engine import SimulationEngine
    from trading_simulator.export import export_csv, export_json

    overrides: dict = {}
    if initial_balance is not None:
        overrides["account"] = {"initial_balance": initial_balance}
    if risk_per_trade is not None:
        overrides.setdefault("account", {})["risk_per_trade"] = risk_per_trade
    if sl_pips is not None:
        overrides["risk"] = {"sl_pips": sl_pips}
    if tp_pips is not None:
        overrides.setdefault("risk", {})["tp_pips"] = tp_pips
    if fee_pips is not None:
        overrides.setdefault("risk", {})["fee_pips"] = fee_pips
    if trades_per_day is not None:
        overrides["activity"] = {"trades_per_day": trades_per_day}
    if win_rate is not None:
        overrides.setdefault("activity", {})["win_rate"] = win_rate
    if days is not None:
        overrides.setdefault("activity", {})["days"] = days
    if compounding is not None:
        overrides.setdefault("risk", {})["compounding_mode"] = compounding
    if outcome_mode is not None:
        overrides.setdefault("activity", {})["outcome_mode"] = outcome_mode
    if custom_pattern is not None:
        overrides.setdefault("activity", {})["custom_pattern"] = custom_pattern
    if tp_levels is not None:
        import json as _json
        try:
            levels = _json.loads(tp_levels)
            overrides.setdefault("risk", {})["tp_levels"] = levels
        except _json.JSONDecodeError as exc:
            console.print(f"[red]Invalid --tp-levels JSON:[/red] {exc}")
            raise typer.Exit(code=2)
    if pyramid_enabled:
        overrides.setdefault("pyramid", {})["enabled"] = True
    if pyramid_trigger_pips is not None:
        overrides.setdefault("pyramid", {})["trigger_pips"] = pyramid_trigger_pips
    if pyramid_max_adds is not None:
        overrides.setdefault("pyramid", {})["max_adds"] = pyramid_max_adds
    if pyramid_lot_mode is not None:
        overrides.setdefault("pyramid", {})["lot_mode"] = pyramid_lot_mode
    if pyramid_lot_size is not None:
        overrides.setdefault("pyramid", {})["lot_size"] = pyramid_lot_size
    if pyramid_lot_pct is not None:
        overrides.setdefault("pyramid", {})["lot_pct"] = pyramid_lot_pct
    if pyramid_be_trigger_pips is not None:
        overrides.setdefault("pyramid", {})["be_trigger_pips"] = pyramid_be_trigger_pips
    if trailing_enabled:
        overrides.setdefault("trailing_stop", {})["enabled"] = True
    if trailing_activate_pips is not None:
        overrides.setdefault("trailing_stop", {})["activate_pips"] = trailing_activate_pips
    if trailing_distance_pips is not None:
        overrides.setdefault("trailing_stop", {})["distance_pips"] = trailing_distance_pips
    if trailing_partial_close_pct is not None:
        overrides.setdefault("trailing_stop", {})["partial_close_pct"] = trailing_partial_close_pct
    if tp_hit_enabled:
        overrides.setdefault("tp_hit", {})["enabled"] = True
    if tp_hit_mode is not None:
        overrides.setdefault("tp_hit", {})["mode"] = tp_hit_mode
    if tp_hit_pct is not None:
        overrides.setdefault("tp_hit", {})["hit_pct"] = tp_hit_pct
    if seed is not None:
        overrides["seed"] = seed

    try:
        cfg = load_config(file_path=config, overrides=overrides)
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        CompoundingMode(cfg.risk.compounding_mode.value)
    except ValueError:
        console.print(
            f"[red]Invalid compounding mode:[/red] {cfg.risk.compounding_mode}. "
            "Valid: none, standard, high_water_mark"
        )
        raise typer.Exit(code=2)

    _display_config(cfg)

    try:
        engine = SimulationEngine(cfg, seed=cfg.seed)
        result = engine.run(days=cfg.activity.days)
    except Exception as exc:
        console.print(f"[red]Simulation error:[/red] {exc}")
        raise typer.Exit(code=3)

    _display_results(result)

    run_id: Optional[str] = None
    if not no_save:
        try:
            run_id = str(uuid.uuid4())[:8]
            db = DatabaseManager()
            db.save_run(run_id, cfg, result)
            console.print(f"\n[green]Run saved to database:[/green] {run_id}")
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not save to database: {exc}")

    if output and output.lower() not in ("none", ""):
        try:
            trade_dicts = [
                {
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
                }
                for t in result.trades
            ]
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"run_{run_id or 'unsaved'}_{timestamp}"
            if output.lower() == "csv":
                export_csv(trade_dicts, out_path / f"{base_name}.csv")
                console.print(f"[green]Exported CSV:[/green] {out_path / f'{base_name}.csv'}")
            elif output.lower() == "json":
                export_json(trade_dicts, out_path / f"{base_name}.json")
                console.print(f"[green]Exported JSON:[/green] {out_path / f'{base_name}.json'}")
        except Exception as exc:
            console.print(f"[red]Export error:[/red] {exc}")
            raise typer.Exit(code=4)

    console.print(
        Panel.fit(
            f"[bold]Initial Balance:[/bold] \u20ac{result.initial_balance:,.2f}    "
            f"[bold]Final Balance:[/bold] \u20ac{result.final_balance:,.2f}    "
            f"[bold]Return:[/bold] {result.total_return_pct:+.2f}%    "
            f"[bold]Trades:[/bold] {len(result.trades)}    "
            f"[bold]Win Rate:[/bold] {result.win_rate_actual:.2%}",
            title="Simulation Complete",
            border_style="green",
        )
    )


def _display_config(cfg: SimulationConfig) -> None:
    """Display the effective configuration as a Rich table."""
    table = Table(title="Simulation Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Symbol", cfg.instrument.symbol)
    table.add_row("Pip Size", str(cfg.instrument.pip_size))
    table.add_row("Contract Size", f"{cfg.instrument.contract_size:,}")
    table.add_row("Initial Balance", f"\u20ac{cfg.account.initial_balance:,.2f}")
    table.add_row("Risk per Trade", f"{cfg.account.risk_per_trade:.2%}")
    table.add_row("Stop Loss (pips)", str(cfg.risk.sl_pips))
    table.add_row("Take Profit (pips)", str(cfg.risk.tp_pips))
    table.add_row("Fee per Trade (pips)", str(cfg.risk.fee_pips))
    table.add_row("Compounding Mode", cfg.risk.compounding_mode.value)
    table.add_row("Outcome Mode", cfg.activity.outcome_mode.value)
    if cfg.activity.custom_pattern:
        table.add_row("Custom Pattern", cfg.activity.custom_pattern)
    if cfg.risk.tp_levels:
        tp_desc = ", ".join(
            f"{t.pips}p@{t.allocation_pct:.0f}%" for t in cfg.risk.tp_levels
        )
        table.add_row("TP Levels", tp_desc)
    if cfg.pyramid.enabled:
        be_desc = f"{cfg.pyramid.be_trigger_pips} pips" if cfg.pyramid.be_trigger_pips else "disabled"
        table.add_row("Pyramid Enabled", "yes")
        table.add_row("Pyramid Trigger (pips)", str(cfg.pyramid.trigger_pips))
        table.add_row("Pyramid Max Adds", str(cfg.pyramid.max_adds))
        table.add_row("Pyramid Lot Mode", cfg.pyramid.lot_mode.value)
        if cfg.pyramid.lot_mode.value == "fixed":
            table.add_row("Pyramid Lot Size", str(cfg.pyramid.lot_size))
        else:
            table.add_row("Pyramid Lot %", f"{cfg.pyramid.lot_pct:.0f}%")
        table.add_row("Pyramid BE Trigger (pips)", be_desc)
    if cfg.trailing_stop.enabled:
        table.add_row("Trailing Stop Enabled", "yes")
        table.add_row("Trailing Activate (pips)", str(cfg.trailing_stop.activate_pips))
        table.add_row("Trailing Distance (pips)", str(cfg.trailing_stop.distance_pips))
        table.add_row("Trailing Partial Close (%)", f"{cfg.trailing_stop.partial_close_pct:.0f}%")
    if cfg.tp_hit.enabled:
        table.add_row("TP Hit Enabled", "yes")
        table.add_row("TP Hit Mode", cfg.tp_hit.mode.value)
        table.add_row("TP Hit %", f"{cfg.tp_hit.hit_pct:.0f}%")
    table.add_row("Trades per Day", str(cfg.activity.trades_per_day))
    table.add_row("Win Rate", f"{cfg.activity.win_rate:.2%}")
    table.add_row("Days", str(cfg.activity.days))
    table.add_row("Seed", str(cfg.seed))

    console.print(table)
    console.print()


def _display_results(result) -> None:
    """Display simulation results as a Rich table."""
    table = Table(title="Simulation Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    winning = [t for t in result.trades if t.outcome == "WIN"]
    losing = [t for t in result.trades if t.outcome == "LOSS"]
    gross_profit = sum(t.profit_euros for t in winning)
    gross_loss = abs(sum(t.profit_euros for t in losing))

    table.add_row("Initial Balance", f"\u20ac{result.initial_balance:,.2f}")
    table.add_row("Final Balance", f"\u20ac{result.final_balance:,.2f}")
    table.add_row("Total Return", f"{result.total_return_pct:+.2f}%")
    table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Total Trades", str(len(result.trades)))
    table.add_row("Winning Trades", str(len(winning)))
    table.add_row("Losing Trades", str(len(losing)))
    table.add_row("Actual Win Rate", f"{result.win_rate_actual:.2%}")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Gross Profit", f"\u20ac{gross_profit:,.2f}")
    table.add_row("Gross Loss", f"\u20ac{gross_loss:,.2f}")
    table.add_row("Net Profit", f"\u20ac{gross_profit - gross_loss:,.2f}")

    console.print(table)
    console.print()

    if result.equity_curve:
        eq_table = Table(title="Equity Curve (Last 5 Snapshots)", show_header=True, header_style="bold blue")
        eq_table.add_column("Day", justify="right")
        eq_table.add_column("Balance", justify="right")
        eq_table.add_column("Drawdown", justify="right")
        eq_table.add_column("Peak", justify="right")

        for point in result.equity_curve[-5:]:
            eq_table.add_row(
                str(point.timestamp.day),
                f"\u20ac{point.balance:,.2f}",
                f"{point.drawdown_pct:.2f}%",
                f"\u20ac{point.peak_balance:,.2f}",
            )
        console.print(eq_table)
        console.print()


@app.command(name="solve")
def solve_cmd(
    initial_balance: float = typer.Option(1000.0, "--initial-balance", "-b", help="Starting wallet balance"),
    target_balance: float = typer.Option(1000000.0, "--target-balance", "-t", help="Target wallet balance"),
    months: int = typer.Option(12, "--months", "-m", help="Timeframe in months"),
    fee_pips: float = typer.Option(1.0, "--fee-pips", help="Flat fee per trade in pips"),
    sl_pips: float = typer.Option(10.0, "--sl-pips", help="Stop loss distance in pips"),
) -> None:
    """Run the goal-seeking reverse solver."""
    from trading_simulator.solver import solve
    from rich.table import Table
    from rich.panel import Panel

    try:
        result = solve(
            initial_balance=initial_balance,
            target_balance=target_balance,
            months=months,
            fee_pips=fee_pips,
            sl_pips=sl_pips,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)

    # Summary panel
    console.print(
        Panel.fit(
            f"[bold]Initial Balance:[/bold] €{result.initial_balance:,.2f}    "
            f"[bold]Target Balance:[/bold] €{result.target_balance:,.2f}    "
            f"[bold]Timeframe:[/bold] {result.months} months    "
            f"[bold]Required Monthly Return:[/bold] {result.required_monthly_return_pct:.2f}%    "
            f"[bold]Feasible:[/bold] {'Yes' if result.feasible else 'No'}",
            title="Goal-Seeking Solver",
            border_style="green",
        )
    )
    console.print()

    # Scenarios table
    table = Table(title="Scenarios", show_header=True, header_style="bold cyan")
    table.add_column("Scenario", style="bold")
    table.add_column("Trades/Day", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Risk/Trade", justify="right")
    table.add_column("SL/TP (pips)", justify="right")
    table.add_column("Monthly Return", justify="right")
    table.add_column("Months to Target", justify="right")
    table.add_column("Risk of Ruin", justify="right")
    table.add_column("EV per Trade (R)", justify="right")

    for s in result.scenarios:
        is_rec = s is result.recommendation
        name_style = "bold green" if is_rec else "bold"
        months_str = f"{s.months_to_target:.1f}" if s.months_to_target != float("inf") else "∞"
        table.add_row(
            f"[{name_style}]{s.name}[/{name_style}]" + (" *" if is_rec else ""),
            str(s.trades_per_day),
            f"{s.win_rate:.2%}",
            f"{s.risk_per_trade:.2%}",
            f"{s.sl_pips}/{s.tp_pips}",
            f"{s.expected_monthly_return_pct:+.2f}%",
            months_str,
            f"{s.risk_of_ruin:.2%}",
            f"{s.expected_value_per_trade_r:+.4f}",
        )

    console.print(table)
    console.print()

    # Recommendation
    rec = result.recommendation
    rec_months_str = (
        f"{rec.months_to_target:.1f}"
        if rec.months_to_target != float("inf")
        else "∞"
    )
    console.print(
        Panel.fit(
            f"[bold]Recommended:[/bold] {rec.name} ({rec.description})\n"
            f"[bold]Trades/Day:[/bold] {rec.trades_per_day}    "
            f"[bold]Win Rate:[/bold] {rec.win_rate:.2%}    "
            f"[bold]Risk/Trade:[/bold] {rec.risk_per_trade:.2%}    "
            f"[bold]SL/TP:[/bold] {rec.sl_pips}/{rec.tp_pips} pips    "
            f"[bold]Monthly Return:[/bold] {rec.expected_monthly_return_pct:+.2f}%    "
            f"[bold]Months to Target:[/bold] {rec_months_str}    "
            f"[bold]Risk of Ruin:[/bold] {rec.risk_of_ruin:.2%}",
            title="Recommendation",
            border_style="blue",
        )
    )
    console.print()

    # Mathematical proof for recommendation
    console.print("[bold]Mathematical Proof (Recommended Scenario):[/bold]")
    console.print(rec.mathematical_proof)


# Import subcommand modules to register their @app.command() decorators
from trading_simulator.cli import export as _export_module  # noqa: F401
from trading_simulator.cli import inspect as _inspect_module  # noqa: F401


if __name__ == "__main__":
    app()