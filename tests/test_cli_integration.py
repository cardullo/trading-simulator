"""Independent CLI integration tests for trading simulation correctness."""
from __future__ import annotations
import csv, json, math, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any
import pytest
from typer.testing import CliRunner
from trading_simulator.cli.main import app

PIP = 10.0; STEP = 0.01; MIN_LOT = 0.01

def round_lot(v: float) -> float:
    return round(v / STEP) * STEP

def exp_lots(bal: float, risk: float, sl: float) -> float:
    ra = bal * risk; rl = ra / (sl * PIP); return round_lot(rl)

class SimVerifier:
    def __init__(self, cfg: dict[str, Any], trades: list[dict[str, Any]], eq: list[dict[str, Any]]):
        self.cfg = cfg; self.trades = trades; self.eq = eq; self.errs: list[str] = []
    def _e(self, m: str) -> None: self.errs.append(m)
    def verify_all(self) -> bool:
        self._vlot(); self._vfee(); self._vchain(); self._vcomp(); self._vout(); self._vanalytics()
        return len(self.errs) == 0
    def _vlot(self) -> None:
        for t in self.trades:
            if t["lots"] < MIN_LOT - 1e-9:
                self._e(f"T{t['trade_number']}: lots {t['lots']} < min 0.01")
    def _vfee(self) -> None:
        for t in self.trades:
            epn = t["pips_gross"] - t["fee_pips"]
            if abs(t["pips_net"] - epn) > 1e-6:
                self._e(f"T{t['trade_number']}: pips_net {t['pips_net']} != pg {t['pips_gross']} - fee {t['fee_pips']}")
            ep = t["pips_net"] * t["lots"] * PIP
            if abs(t["profit_euros"] - ep) > 1e-4:
                self._e(f"T{t['trade_number']}: profit {t['profit_euros']} != pn {t['pips_net']} * lots {t['lots']} * 10 = {ep}")
    def _vchain(self) -> None:
        ib = self.cfg["initial_balance"]; pb = ib
        for t in self.trades:
            if abs(t["balance_before"] - pb) > 1e-4 and not math.isclose(t["balance_before"], pb, rel_tol=1e-9):
                self._e(f"T{t['trade_number']}: bb {t['balance_before']} != prev ba {pb}")
            ea = round(pb + t["profit_euros"], 4)
            if abs(t["balance_after"] - ea) > 1e-4 and not math.isclose(t["balance_after"], ea, rel_tol=1e-9):
                self._e(f"T{t['trade_number']}: ba {t['balance_after']} != bb {t['balance_before']} + profit {t['profit_euros']} = {ea}")
            pb = t["balance_after"]
    def _vcomp(self) -> None:
        cm = self.cfg.get("compounding_mode", "standard")
        if cm == "none":
            ls = {round(t["lots"], 4) for t in self.trades}
            if len(ls) > 1: self._e(f"Compounding NONE: lot sizes not constant: {sorted(ls)}")
    def _vout(self) -> None:
        wins = [t for t in self.trades if t["outcome"] == "WIN"]
        total = len(self.trades)
        if total == 0: return
        awr = len(wins) / total; ewr = self.cfg["win_rate"]
        if self.cfg.get("outcome_mode") != "bernoulli" and abs(awr - ewr) > 1e-9:
            self._e(f"WR {awr:.4%} != expected {ewr:.4%}")
    def _vanalytics(self) -> None:
        if not self.trades: return
        ib = self.cfg["initial_balance"]; fb = self.trades[-1]["balance_after"]
        tr = (fb - ib) / ib * 100
        peak = ib; md = 0.0
        for t in self.trades:
            if t["balance_after"] > peak: peak = t["balance_after"]
            dd = (peak - t["balance_after"]) / peak * 100 if peak > 0 else 0.0
            if dd > md: md = dd
        if not (abs(tr) < 1e12): self._e(f"Total return {tr}% unrealistic")
        if md < 0 or md > 100: self._e(f"Max drawdown {md}% out of [0,100]")

def make_cfg(**overrides) -> dict[str, Any]:
    base = {
        "initial_balance": 1000.0, "risk_per_trade": 0.01, "sl_pips": 10.0,
        "tp_pips": 30.0, "fee_pips": 1.0, "trades_per_day": 5,
        "win_rate": 0.55, "days": 5, "compounding_mode": "standard",
        "outcome_mode": "bernoulli", "seed": 42,
        "custom_pattern": None, "tp_levels": None,
    }
    base.update(overrides)
    return base

class TestCLISimulation:
    def _run_cli(self, cfg: dict[str, Any], tmp_path: Path) -> tuple[dict, list[dict], list[dict]]:
        runner = CliRunner()
        args = [
            "run", "--initial-balance", str(cfg["initial_balance"]),
            "--risk-per-trade", str(cfg["risk_per_trade"]),
            "--sl-pips", str(cfg["sl_pips"]), "--tp-pips", str(cfg["tp_pips"]),
            "--fee-pips", str(cfg["fee_pips"]),
            "--trades-per-day", str(cfg["trades_per_day"]),
            "--win-rate", str(cfg["win_rate"]), "--days", str(cfg["days"]),
            "--compounding", cfg["compounding_mode"],
            "--outcome-mode", cfg["outcome_mode"], "--seed", str(cfg["seed"]),
            "--no-save", "--output", "csv", "--out-dir", str(tmp_path),
        ]
        if cfg["custom_pattern"]: args.extend(["--custom-pattern", cfg["custom_pattern"]])
        if cfg["tp_levels"]:
            import json as _json
            args.extend(["--tp-levels", _json.dumps(cfg["tp_levels"])])
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        csv_files = list(tmp_path.glob("*.csv"))
        assert csv_files, f"No CSV found in {tmp_path}"
        csv_file = csv_files[0]
        with open(csv_file, newline="") as f:
            trades = list(csv.DictReader(f))
        for t in trades:
            t["trade_number"] = int(t["trade_number"]); t["day"] = int(t["day"])
            t["lots"] = float(t["lots"]); t["risk_amount"] = float(t["risk_amount"])
            t["pips_gross"] = float(t["pips_gross"]); t["pips_net"] = float(t["pips_net"])
            t["fee_pips"] = float(t["fee_pips"]); t["profit_euros"] = float(t["profit_euros"])
            t["balance_before"] = float(t["balance_before"]); t["balance_after"] = float(t["balance_after"])
        return cfg, trades, []
    def test_t1_baseline(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=5, trades_per_day=5, seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        assert len(trades) == 25

    def test_t2_compounding_none(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=10, trades_per_day=5, compounding_mode="none", outcome_mode="fixed_wins_first", seed=42, win_rate=0.56)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        ls = {round(t["lots"], 4) for t in trades}
        assert len(ls) == 1, f"Expected constant lots, got {sorted(ls)}"

    def test_t3_compounding_standard(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=10, trades_per_day=5, compounding_mode="standard", outcome_mode="fixed_wins_first", seed=42, win_rate=0.56)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        ls = {round(t["lots"], 4) for t in trades}
        assert len(ls) > 1, f"Expected varying lots, got {sorted(ls)}"

    def test_t4_compounding_hwm(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=10, trades_per_day=5, compounding_mode="high_water_mark", outcome_mode="fixed_wins_first", seed=42, risk_per_trade=0.02, win_rate=0.56)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        ls = {round(t["lots"], 4) for t in trades}
        assert len(ls) > 1, f"Expected varying lots with HWM, got {sorted(ls)}"
    def test_t5_min_lot(self, tmp_path: Path) -> None:
        cfg = make_cfg(initial_balance=50.0, risk_per_trade=0.005, sl_pips=50.0, tp_pips=100.0,
            fee_pips=1.0, trades_per_day=1, win_rate=1.0, days=3,
            compounding_mode="none", outcome_mode="fixed_wins_first", seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        for t in trades:
            assert t["lots"] == 0.01, f"Expected min lot 0.01, got {t['lots']}"

    def test_t6_rounding(self, tmp_path: Path) -> None:
        cfg = make_cfg(initial_balance=1234.0, risk_per_trade=0.01, sl_pips=7.0, tp_pips=21.0,
            fee_pips=1.0, trades_per_day=1, win_rate=1.0, days=5,
            compounding_mode="none", outcome_mode="fixed_wins_first", seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        for t in trades:
            assert abs(t["lots"] % STEP) < 1e-6 or abs(t["lots"] % STEP - STEP) < 1e-6

    def test_t7_alternating(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=3, trades_per_day=4, win_rate=0.5, compounding_mode="none", outcome_mode="alternating", seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        outcomes = [t["outcome"] for t in trades]
        assert outcomes == ["WIN", "LOSS"] * 6, f"Expected alternating WL, got {outcomes}"

    def test_t8_fixed_wins_first(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=3, trades_per_day=4, win_rate=0.5, compounding_mode="none", outcome_mode="fixed_wins_first", seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        outcomes = [t["outcome"] for t in trades]
        assert outcomes == ["WIN"] * 6 + ["LOSS"] * 6, f"Expected wins first, got {outcomes}"

    def test_t9_custom_pattern(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=4, trades_per_day=3, win_rate=2/3, compounding_mode="none",
            outcome_mode="custom_pattern", custom_pattern="WWL", seed=42)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
    def test_t10_partial_tp(self, tmp_path: Path) -> None:
        cfg = make_cfg(
            days=1, trades_per_day=1, win_rate=1.0, compounding_mode="none",
            outcome_mode="fixed_wins_first", seed=42,
            tp_levels=[
                {"pips": 10.0, "allocation_pct": 25.0},
                {"pips": 20.0, "allocation_pct": 25.0},
                {"pips": 30.0, "allocation_pct": 25.0},
                {"pips": 40.0, "allocation_pct": 25.0},
            ],
        )
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        assert len(trades) == 4, f"Expected 4 fills, got {len(trades)}"
        assert all(t["outcome"] == "WIN" for t in trades), f"Not all WIN: {[t['outcome'] for t in trades]}"
        fee_fills = [f for f in trades if f["fee_pips"] > 0]
        assert len(fee_fills) == 1, f"Expected 1 fill with fee, got {len(fee_fills)}"
        assert fee_fills[0]["fee_pips"] == 1.0
        no_fee_fills = [f for f in trades if f["fee_pips"] == 0]
        assert len(no_fee_fills) == 3, f"Expected 3 zero-fee fills, got {len(no_fee_fills)}"
        total_lots = sum(f["lots"] for f in trades)
        assert total_lots > 0, "Total lots should be > 0"
        for f in trades:
            expected_profit = f["pips_net"] * f["lots"] * PIP
            assert abs(f["profit_euros"] - expected_profit) < 1e-4, f"Fill profit {f['profit_euros']} != expected {expected_profit}"
        total_profit = sum(f["profit_euros"] for f in trades)
        bc = trades[-1]["balance_after"] - trades[0]["balance_before"]
        assert abs(total_profit - bc) < 1e-6

    def test_t11_deterministic(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=10, trades_per_day=10, seed=42)
        _, trades_a, _ = self._run_cli(cfg, tmp_path)
        _, trades_b, _ = self._run_cli(cfg, tmp_path)
        assert len(trades_a) == len(trades_b)
        for ta, tb in zip(trades_a, trades_b):
            assert ta["outcome"] == tb["outcome"]
            assert abs(ta["lots"] - tb["lots"]) < 1e-9
            assert abs(ta["profit_euros"] - tb["profit_euros"]) < 1e-9
            assert abs(ta["balance_after"] - tb["balance_after"]) < 1e-9

    def test_t12_high_activity(self, tmp_path: Path) -> None:
        cfg = make_cfg(initial_balance=10000.0, risk_per_trade=0.005,
            sl_pips=15.0, tp_pips=45.0, fee_pips=0.5,
            trades_per_day=100, win_rate=0.52, days=30,
            compounding_mode="standard", outcome_mode="bernoulli", seed=123)
        _, trades, _ = self._run_cli(cfg, tmp_path)
        v = SimVerifier(cfg, trades, [])
        assert v.verify_all(), "\n".join(v.errs)
        assert len(trades) == 3000
        assert all(t["balance_after"] > 0 for t in trades), "Balance went negative"

    def test_t13_database_persistence(self, tmp_path: Path) -> None:
        cfg = make_cfg(days=3, trades_per_day=5, seed=7)
        runner = CliRunner()
        args = [
            "run", "--initial-balance", str(cfg["initial_balance"]),
            "--risk-per-trade", str(cfg["risk_per_trade"]),
            "--sl-pips", str(cfg["sl_pips"]), "--tp-pips", str(cfg["tp_pips"]),
            "--fee-pips", str(cfg["fee_pips"]),
            "--trades-per-day", str(cfg["trades_per_day"]),
            "--win-rate", str(cfg["win_rate"]), "--days", str(cfg["days"]),
            "--compounding", cfg["compounding_mode"],
            "--outcome-mode", cfg["outcome_mode"], "--seed", str(cfg["seed"]),
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        match = re.search(r"Run saved to database:\s+(\w+)", result.output)
        assert match is not None, f"Run ID not found in output:\n{result.output}"
        run_id = match.group(1)
        insp_result = runner.invoke(app, ["inspect", run_id])
        assert insp_result.exit_code == 0, f"Inspect failed: {insp_result.output}"
        assert run_id in insp_result.output
        csv_path = tmp_path / "exported.csv"
        exp_result = runner.invoke(app, ["export", run_id, "--format", "csv", "--out", str(csv_path)])
        assert exp_result.exit_code == 0, f"Export failed: {exp_result.output}"
        assert csv_path.exists(), f"CSV not created at {csv_path}"
        with open(csv_path, newline="") as f:
            trades = list(csv.DictReader(f))
        assert len(trades) == 15
        first = trades[0]
        assert abs(float(first["balance_before"]) - cfg["initial_balance"]) < 1e-6
        json_path = tmp_path / "exported.json"
        exp_result2 = runner.invoke(app, ["export", run_id, "--format", "json", "--out", str(json_path)])
        assert exp_result2.exit_code == 0
        assert json_path.exists()
        with open(json_path) as f:
            json_data = json.load(f)
        assert len(json_data) == 15
        assert json_data[0]["trade_number"] == 1