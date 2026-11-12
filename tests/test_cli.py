"""CLI integration tests using Typer's CliRunner."""

import json
import os
import tempfile

import pytest
from typer.testing import CliRunner

from trading_simulator.cli.main import app


class TestCliHelp:
    """Tests for CLI help output."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_main_help(self):
        """Main help should show all commands."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "export" in result.output
        assert "inspect" in result.output

    def test_run_help(self):
        """Run help should show all options."""
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--initial-balance" in result.output
        assert "--sl-pips" in result.output
        assert "--tp-pips" in result.output
        assert "--fee-pips" in result.output
        assert "--trades-per-day" in result.output
        assert "--win-rate" in result.output
        assert "--days" in result.output
        assert "--compounding" in result.output
        assert "--outcome-mode" in result.output
        assert "--custom-pattern" in result.output
        assert "--tp-levels" in result.output
        assert "--seed" in result.output
        assert "--output" in result.output
        assert "--no-save" in result.output

    def test_inspect_help(self):
        """Inspect help should work."""
        result = self.runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_export_help(self):
        """Export help should work."""
        result = self.runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0


class TestCliRun:
    def setup_method(self):
        self.runner = CliRunner()

    def test_run_basic(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--compounding", "standard",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "Simulation Complete" in result.output
        assert "Final Balance" in result.output

    def test_run_high_water_mark(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--compounding", "high_water_mark",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "high_water_mark" in result.output

    def test_run_none_compounding(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--compounding", "none",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "none" in result.output

    def test_run_alternating_outcome(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "4", "--win-rate", "0.5",
                "--days", "3", "--outcome-mode", "alternating",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "alternating" in result.output

    def test_run_custom_pattern(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "4", "--win-rate", "0.5",
                "--days", "3", "--outcome-mode", "custom_pattern",
                "--custom-pattern", "WWLL", "--seed", "42",
                "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "WWLL" in result.output

    def test_run_tp_levels(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "3", "--win-rate", "0.6",
                "--days", "3", "--outcome-mode", "fixed_wins_first",
                "--tp-levels",
                '[{"pips":10,"allocation_pct":25},{"pips":20,"allocation_pct":25},{"pips":30,"allocation_pct":25},{"pips":40,"allocation_pct":25}]',
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 0
        assert "TP Levels" in result.output

    def test_run_invalid_compounding(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--compounding", "invalid_mode",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 2

    def test_run_invalid_tp_levels_json(self):
        result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--tp-levels", "not-valid-json",
                "--seed", "42", "--no-save",
            ],
        )
        assert result.exit_code == 2


class TestCliInspect:
    def setup_method(self):
        self.runner = CliRunner()
        # Clean up any existing database to ensure isolated test
        import os
        db_path = "data/runs.db"
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_inspect_no_runs(self):
        result = self.runner.invoke(app, ["inspect"])
        assert result.exit_code == 0
        assert "No simulation runs found" in result.output

    def test_inspect_invalid_run_id(self):
        result = self.runner.invoke(app, ["inspect", "nonexistent"])
        assert result.exit_code == 2
        assert "Run not found" in result.output


class TestCliExport:
    def setup_method(self):
        self.runner = CliRunner()

    def test_export_invalid_run_id(self):
        result = self.runner.invoke(app, ["export", "nonexistent"])
        assert result.exit_code == 2
        assert "Run not found" in result.output

    def test_export_invalid_format(self):
        result = self.runner.invoke(
            app,
            ["export", "nonexistent", "--format", "xml"],
        )
        assert result.exit_code == 2


class TestCliInspectWithRun:
    """Tests for inspect with actual run data in the database."""

    def setup_method(self):
        self.runner = CliRunner()
        import os
        db_path = "data/runs.db"
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_inspect_with_run(self):
        """Inspect should list a saved run."""
        # First, run a simulation and save it
        run_result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "5", "--win-rate", "0.55",
                "--days", "5", "--compounding", "standard",
                "--seed", "42",
            ],
        )
        assert run_result.exit_code == 0

        # Extract run ID from output
        import re
        match = re.search(r"Run saved to database:\s+(\w+)", run_result.output)
        assert match is not None, f"Run ID not found in output: {run_result.output}"
        run_id = match.group(1)

        # Now inspect
        result = self.runner.invoke(app, ["inspect"])
        assert result.exit_code == 0
        # Run ID is truncated in Rich table, check for partial match
        assert run_id[:6] in result.output
        assert "+37.04%" in result.output

    def test_inspect_detail(self):
        """Inspect <RUN_ID> should show detailed run info."""
        import os, re
        db_path = "data/runs.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        run_result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "3", "--win-rate", "0.55",
                "--days", "2", "--compounding", "standard",
                "--seed", "42",
            ],
        )
        assert run_result.exit_code == 0

        match = re.search(r"Run saved to database:\s+(\w+)", run_result.output)
        assert match is not None
        run_id = match.group(1)

        result = self.runner.invoke(app, ["inspect", run_id])
        assert result.exit_code == 0
        assert f"Run Details: {run_id}" in result.output
        assert "€1,000.00" in result.output
        assert "Recent Trades" in result.output


class TestCliExportWithRun:
    """Tests for export with actual run data."""

    def setup_method(self):
        self.runner = CliRunner()
        import os, shutil
        db_path = "data/runs.db"
        export_dir = "data/exports"
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

    def test_export_csv(self):
        """Export to CSV should create a file."""
        import os, re, shutil

        run_result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "3", "--win-rate", "0.55",
                "--days", "2", "--compounding", "standard",
                "--seed", "42",
            ],
        )
        assert run_result.exit_code == 0

        match = re.search(r"Run saved to database:\s+(\w+)", run_result.output)
        assert match is not None
        run_id = match.group(1)

        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        csv_path = os.path.join(export_dir, f"run_{run_id}.csv")

        result = self.runner.invoke(
            app,
            ["export", run_id, "--format", "csv"],
        )
        assert result.exit_code == 0
        assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

        with open(csv_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1
            assert "trade_id" in lines[0]
            assert "profit_euros" in lines[0]

        shutil.rmtree(export_dir)

    def test_export_json(self):
        """Export to JSON should create a file."""
        import os, re, shutil

        run_result = self.runner.invoke(
            app,
            [
                "run", "--initial-balance", "1000",
                "--risk-per-trade", "0.01", "--sl-pips", "10",
                "--tp-pips", "30", "--fee-pips", "1",
                "--trades-per-day", "3", "--win-rate", "0.55",
                "--days", "2", "--compounding", "standard",
                "--seed", "42",
            ],
        )
        assert run_result.exit_code == 0

        match = re.search(r"Run saved to database:\s+(\w+)", run_result.output)
        assert match is not None
        run_id = match.group(1)

        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        json_path = os.path.join(export_dir, f"run_{run_id}.json")

        result = self.runner.invoke(
            app,
            ["export", run_id, "--format", "json"],
        )
        assert result.exit_code == 0
        assert os.path.exists(json_path), f"JSON file not created: {json_path}"

        with open(json_path, "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "trade_id" in data[0]

        shutil.rmtree(export_dir)