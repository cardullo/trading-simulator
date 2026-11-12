"""`inspect` subcommand: view stored simulation runs and their details."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_simulator.cli.main import app
from trading_simulator.database import DatabaseManager

console = Console()


@app.command(name="inspect")
def inspect_cmd(
    run_id: Optional[str] = typer.Argument(None, help="Run ID to inspect (omit to list all runs)"),
) -> None:
    """Inspect stored simulation runs. List all runs or show details for a specific run."""
    db = DatabaseManager()

    if run_id is None:
        _list_runs(db)
    else:
        _show_run(db, run_id)


def _list_runs(db: DatabaseManager) -> None:
    """List all stored simulation runs."""
    runs = db.list_runs()

    if not runs:
        console.print("[yellow]No simulation runs found.[/yellow]")
        console.print("Run a simulation first with 'trading-sim run'.")
        return

    table = Table(title="Stored Simulation Runs", show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="bold")
    table.add_column("Created", justify="right")
    table.add_column("Initial Balance", justify="right")
    table.add_column("Final Balance", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Compounding", justify="right")
    table.add_column("Outcome Mode", justify="right")

    for r in runs:
        created = r["created_at"][:19].replace("T", " ") if r["created_at"] else "N/A"
        initial = f"€{r['initial_balance']:,.2f}"
        final = f"€{r['final_balance']:,.2f}" if r["final_balance"] is not None else "N/A"
        ret_pct = (
            f"{(r['final_balance'] / r['initial_balance'] - 1) * 100:+.2f}%"
            if r["final_balance"] is not None and r["initial_balance"] > 0
            else "N/A"
        )
        trades_count = len(db.get_trades(r["run_id"]))

        table.add_row(
            r["run_id"],
            created,
            initial,
            final,
            ret_pct,
            str(trades_count),
            r["compounding_mode"],
            r.get("outcome_mode", "bernoulli"),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]Tip:[/dim] Use 'trading-sim inspect <RUN_ID>' to see detailed run information."
    )
    console.print(
        "[dim]Tip:[/dim] Use 'trading-sim export <RUN_ID> --format csv' to export trades."
    )


def _show_run(db: DatabaseManager, run_id: str) -> None:
    """Show detailed information for a specific run."""
    run = db.get_run(run_id)
    if run is None:
        console.print(f"[red]Run not found:[/red] {run_id}")
        console.print("Use 'trading-sim inspect' to list available runs.")
        raise typer.Exit(code=2)

    trades = db.get_trades(run_id)

    # Run info panel
    info = (
        f"[bold]Run ID:[/bold] {run['run_id']}\n"
        f"[bold]Created:[/bold] {run['created_at'][:19].replace('T', ' ') if run['created_at'] else 'N/A'}\n"
        f"[bold]Initial Balance:[/bold] €{run['initial_balance']:,.2f}\n"
    )
    if run["final_balance"] is not None:
        info += f"[bold]Final Balance:[/bold] €{run['final_balance']:,.2f}\n"
    else:
        info += "[bold]Final Balance:[/bold] N/A\n"
    ret_pct = (
        f"{(run['final_balance'] / run['initial_balance'] - 1) * 100:+.2f}%"
        if run["final_balance"] is not None and run["initial_balance"] > 0
        else "N/A"
    )
    info += f"[bold]Total Return:[/bold] {ret_pct}\n"
    info += f"[bold]Compounding:[/bold] {run['compounding_mode']}\n"
    info += f"[bold]Outcome Mode:[/bold] {run.get('outcome_mode', 'bernoulli')}\n"
    info += f"[bold]Risk per Trade:[/bold] {run['risk_per_trade']:.2%}\n"
    info += f"[bold]Stop Loss:[/bold] {run['sl_pips']} pips\n"
    info += f"[bold]Take Profit:[/bold] {run['tp_pips']} pips\n"
    info += f"[bold]Fee per Trade:[/bold] {run['fee_pips']} pips\n"
    info += f"[bold]Trades/Day:[/bold] {run['trades_per_day']}\n"
    info += f"[bold]Win Rate:[/bold] {run['win_rate']:.2%}\n"
    info += f"[bold]Days:[/bold] {run['days']}\n"
    info += f"[bold]Seed:[/bold] {run['seed']}\n"
    info += f"[bold]Total Trades Logged:[/bold] {len(trades)}"

    console.print(Panel(info, title=f"Run Details: {run_id}", border_style="green"))
    console.print()

    # Recent trades table
    if trades:
        table = Table(title=f"Recent Trades (Last 10 of {len(trades)})", show_header=True, header_style="bold blue")
        table.add_column("#", justify="right")
        table.add_column("Day", justify="right")
        table.add_column("Lots", justify="right")
        table.add_column("Pips Net", justify="right")
        table.add_column("Profit (€)", justify="right")
        table.add_column("Balance (€)", justify="right")
        table.add_column("Outcome", justify="right")

        for t in trades[-10:]:
            outcome_style = "green" if t["outcome"] == "WIN" else "red"
            table.add_row(
                str(t["trade_number"]),
                str(t["day"]),
                f"{t['lots']:.4f}",
                f"{t['pips_net']:+.2f}",
                f"{t['profit_euros']:+.2f}",
                f"{t['balance_after']:,.2f}",
                f"[{outcome_style}]{t['outcome']}[/{outcome_style}]",
            )
        console.print(table)
    else:
        console.print("[yellow]No trades logged for this run.[/yellow]")