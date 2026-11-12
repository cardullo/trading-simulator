"""Trade log export to CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_csv(trades: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Export trade log to CSV format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not trades:
        output_path.write_text("")
        return output_path

    fieldnames = list(trades[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    return output_path


def export_json(trades: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Export trade log to JSON format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, default=str)
    return output_path