"""`export` subcommand: export trade logs from the database to CSV or JSON."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from trading_simulator.cli.main import app
from trading_simulator.database import DatabaseManager
from trading_simulator.export import export_csv as _export_csv, export_json as _export_json

console = Console()


@app.command(name="export")
def export_cmd(
    run_id: str = typer.Argument(..., help="Run ID to export (use 'trading-sim inspect' to list runs)"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv or json"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file path"),
) -> None:
    """Export a stored simulation run's trade log to CSV or JSON."""
    db = DatabaseManager()

    run = db.get_run(run_id)
    if run is None:
        console.print(f"[red]Run not found:[/red] {run_id}")
        console.print("Use 'trading-sim inspect' to list available runs.")
        raise typer.Exit(code=2)

    trades = db.get_trades(run_id)
    if not trades:
        console.print(f"[yellow]No trades found for run:[/yellow] {run_id}")
        raise typer.Exit(code=1)

    if out is None:
        out_dir = Path("data/exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "csv" if format.lower() == "csv" else "json"
        out_path = out_dir / f"run_{run_id}.{ext}"
    else:
        out_path = Path(out)

    try:
        if format.lower() == "csv":
            _export_csv(trades, out_path)
        elif format.lower() == "json":
            _export_json(trades, out_path)
        else:
            console.print(f"[red]Unknown format:[/red] {format}. Use 'csv' or 'json'.")
            raise typer.Exit(code=2)

        console.print(f"[green]Exported {len(trades)} trades:[/green] {out_path}")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Format: {format.upper()}")
        console.print(f"  Initial Balance: €{run['initial_balance']:,.2f}")
        console.print(f"  Final Balance: €{run['final_balance']:,.2f}")

    except Exception as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise typer.Exit(code=4)