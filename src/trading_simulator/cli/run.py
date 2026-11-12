"""`run` subcommand: execute a trading simulation with full parameter overrides."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_simulator.cli.main import app
from trading_simulator.config import CompoundingMode, SimulationConfig, load_config
from trading_simulator.database import DatabaseManager
from trading_simulator.engine import SimulationEngine
from trading_simulator.export import export_csv, export_json

console = Console()


@app.command(name="run")
def run(
    initial_balance: Optional[float] = typer.Option(
        None, "--initial-balance", "-b", help="Starting wallet balance (e.g., 1000)"
    ),
    risk_per_trade: Optional[float] = typer.Option(
        None, "--risk-per-trade", "-r",
        help="Risk per trade as fraction of balance (e.g., 0.01 for 1%)",
    ),
    sl_pips: Optional[float] = typer.Option(
        None, "--sl-pips", help="Stop loss distance in pips (e.g., 10)"
    ),
    tp_pips: Optional[float] = typer.Option(
        None, "--tp-pips", help="Take profit distance in pips (e.g., 30)"
    ),
    fee_pips: Optional[float] = typer.Option(
        None, "--fee-pips", help="Flat fee per trade in pips (e.g., 1)"
    ),
    trades_per_day: Optional[int] = typer.Option(
        None, "--trades-per-day", "-t", help="Number of trades per day (e.g., 10)"
    ),
    win_rate: Optional[float] = typer.Option(
        None, "--win-rate", "-w", help="Win rate probability (e.g., 0.55 for 55%)"
    ),
    days: Optional[int] = typer.Option(
        None, "--days", "-d", help="Number of days to simulate (e.g., 30)"
    ),
    compounding: Optional[str] = typer.Option(
        None, "--compounding", "-c",
        help="Compounding mode: none, standard, high_water_mark",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (e.g., 42)"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to JSON config file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Export format: csv, json, or none"
    ),
    out_dir: str = typer.Option("data/exports", "--out-dir", help="Export directory"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save to SQLite"),
) -> None:
    """Run a trading simulation with the given parameters."""
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
            else:
                console.print(f"[yellow]Unknown output format:[/yellow] {output}. Use csv or json.")
        except Exception as exc:
            console.print(f"[red]Export error:[/red] {exc}")
            raise typer.Exit(code=4)

    console.print(
        Panel.fit(
            f"[bold]Initial Balance:[/bold] €{result.initial_balance:,.2f}    "
            f"[bold]Final Balance:[/bold] €{result.final_balance:,.2f}    "
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