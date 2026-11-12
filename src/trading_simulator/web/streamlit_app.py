"""Streamlit web interface for Trading Simulation & Goal-Modeling Platform."""
from __future__ import annotations
import json, uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
from trading_simulator.config import SimulationConfig, load_config, OutcomeMode
from trading_simulator.database import DatabaseManager
from trading_simulator.engine import SimulationEngine
from trading_simulator.export import export_csv, export_json
from trading_simulator.solver import solve
st.set_page_config(page_title="Trading Simulator", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
DB_PATH = Path("data/runs.db")
def get_db():
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager(db_path=DB_PATH)
    return st.session_state.db
def build_config(initial_balance, risk_per_trade, sl_pips, tp_pips, fee_pips, trades_per_day, win_rate, days, compounding, outcome_mode, custom_pattern, seed, pyramid_enabled=False, pyramid_trigger_pips=5.0, pyramid_max_adds=3, pyramid_lot_mode="fixed", pyramid_lot_size=0.01, pyramid_lot_pct=25.0, pyramid_be_trigger_pips=None, trailing_enabled=False, trailing_activate_pips=10.0, trailing_distance_pips=5.0, trailing_partial_close_pct=0.0, tp_hit_enabled=False, tp_hit_mode="fixed", tp_hit_pct=100.0):
    overrides = {}
    if initial_balance: overrides["account"] = {"initial_balance": initial_balance}
    if risk_per_trade: overrides.setdefault("account", {})["risk_per_trade"] = risk_per_trade
    overrides["risk"] = {"sl_pips": sl_pips}
    overrides.setdefault("risk", {})["tp_pips"] = tp_pips
    overrides.setdefault("risk", {})["fee_pips"] = fee_pips
    overrides.setdefault("risk", {})["compounding_mode"] = compounding
    overrides["activity"] = {"trades_per_day": trades_per_day}
    overrides.setdefault("activity", {})["win_rate"] = win_rate
    overrides.setdefault("activity", {})["days"] = days
    overrides.setdefault("activity", {})["outcome_mode"] = outcome_mode
    if custom_pattern: overrides.setdefault("activity", {})["custom_pattern"] = custom_pattern
    if pyramid_enabled:
        overrides.setdefault("pyramid", {})["enabled"] = True
        overrides.setdefault("pyramid", {})["trigger_pips"] = pyramid_trigger_pips
        overrides.setdefault("pyramid", {})["max_adds"] = pyramid_max_adds
        overrides.setdefault("pyramid", {})["lot_mode"] = pyramid_lot_mode
        overrides.setdefault("pyramid", {})["lot_size"] = pyramid_lot_size
        overrides.setdefault("pyramid", {})["lot_pct"] = pyramid_lot_pct
        if pyramid_be_trigger_pips:
            overrides.setdefault("pyramid", {})["be_trigger_pips"] = pyramid_be_trigger_pips
    if trailing_enabled:
        overrides.setdefault("trailing_stop", {})["enabled"] = True
        overrides.setdefault("trailing_stop", {})["activate_pips"] = trailing_activate_pips
        overrides.setdefault("trailing_stop", {})["distance_pips"] = trailing_distance_pips
        overrides.setdefault("trailing_stop", {})["partial_close_pct"] = trailing_partial_close_pct
    if tp_hit_enabled:
        overrides.setdefault("tp_hit", {})["enabled"] = True
        overrides.setdefault("tp_hit", {})["mode"] = tp_hit_mode
        overrides.setdefault("tp_hit", {})["hit_pct"] = tp_hit_pct
    if seed is not None: overrides["seed"] = seed
    return load_config(overrides=overrides)
def run_simulation(config, seed):
    engine = SimulationEngine(config, seed=seed)
    result = engine.run(days=config.activity.days)
    run_id = str(uuid.uuid4())[:8]
    db = get_db()
    db.save_run(run_id, config, result)
    return engine, result, run_id
def make_equity_chart(result, title="Equity Curve"):
    if not result.equity_curve:
        fig = go.Figure()
        fig.add_annotation(text="No equity data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    ts = [e.timestamp for e in result.equity_curve]
    bal = [e.balance for e in result.equity_curve]
    peak = [e.peak_balance for e in result.equity_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=bal, mode="lines", name="Wallet Balance", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=ts, y=peak, mode="lines", name="Peak Balance", line=dict(color="#2ca02c", width=1, dash="dot")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Balance (€)", hovermode="x unified", template="plotly_white", height=500)
    fig.update_yaxes(tickformat=",.0f")
    return fig
def make_comparison_chart(results, title="Multi-Run Comparison"):
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, (label, (_c, r)) in enumerate(results.items()):
        if not r.equity_curve: continue
        ts = [e.timestamp for e in r.equity_curve]
        bal = [e.balance for e in r.equity_curve]
        fig.add_trace(go.Scatter(x=ts, y=bal, mode="lines", name=label, line=dict(color=colors[idx % len(colors)], width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Balance (€)", hovermode="x unified", template="plotly_white", height=500)
    fig.update_yaxes(tickformat=",.0f")
    return fig
def make_trades_table(trades):
    if not trades: return []
    rows = []
    for t in trades:
        if isinstance(t, dict):
            rows.append({
                "Trade #": t.get("trade_number", t.get("id", "")),
                "Day": t.get("day", ""),
                "Lots": f"{t.get('lots', 0):.4f}",
                "Pips Net": f"{t.get('pips_net', 0):+.2f}",
                "Profit (€)": f"{t.get('profit_euros', 0):+.2f}",
                "Balance (€)": f"{t.get('balance_after', 0):,.2f}",
                "Outcome": t.get("outcome", ""),
            })
        else:
            rows.append({
                "Trade #": t.trade_number,
                "Day": t.day,
                "Lots": f"{t.lots:.4f}",
                "Pips Net": f"{t.pips_net:+.2f}",
                "Profit (€)": f"{t.profit_euros:+.2f}",
                "Balance (€)": f"{t.balance_after:,.2f}",
                "Outcome": t.outcome,
            })
    return rows
def highlight_outcome(val):
    if val == "WIN": return "color: #2ca02c; font-weight: bold"
    if val == "LOSS": return "color: #d62728; font-weight: bold"
    return ""


def render_sidebar():
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Account")
    initial_balance = st.sidebar.number_input("Initial Balance (€)", min_value=1.0, value=1000.0, step=100.0, format="%.2f")
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 10.0, 1.0, 0.1) / 100.0
    st.sidebar.subheader("Instrument (EUR/USD)")
    sl_pips = st.sidebar.number_input("Stop Loss (pips)", min_value=1.0, value=10.0, step=1.0)
    tp_pips = st.sidebar.number_input("Take Profit (pips)", min_value=1.0, value=30.0, step=1.0)
    fee_pips = st.sidebar.number_input("Fee per Trade (pips)", min_value=0.0, value=1.0, step=0.1)
    st.sidebar.subheader("Activity")
    trades_per_day = st.sidebar.number_input("Trades per Day", min_value=1, max_value=500, value=10, step=1)
    win_rate = st.sidebar.slider("Win Rate (%)", 0.0, 100.0, 55.0, 1.0) / 100.0
    days = st.sidebar.number_input("Days to Simulate", min_value=1, max_value=3650, value=30, step=1)
    st.sidebar.subheader("Compounding")
    compounding = st.sidebar.selectbox("Compounding Mode", ["none", "standard", "high_water_mark"], index=1, format_func=lambda x: {"none": "None (Fixed Risk)", "standard": "Standard (Current Balance)", "high_water_mark": "High-Water Mark (Peak Balance)"}[x])
    st.sidebar.subheader("Outcome Model")
    outcome_mode = st.sidebar.selectbox("Outcome Mode", ["bernoulli", "alternating", "fixed_wins_first", "fixed_losses_first", "custom_pattern"], index=0, format_func=lambda x: {"bernoulli": "Bernoulli (Stochastic)", "alternating": "Alternating W/L", "fixed_wins_first": "Fixed Wins First", "fixed_losses_first": "Fixed Losses First", "custom_pattern": "Custom W/L Pattern"}[x])
    custom_pattern = ""
    if outcome_mode == "custom_pattern":
        custom_pattern = st.sidebar.text_input("Custom Pattern (W/L)", value="WWLWLW", help="e.g., WWLWLW")
    seed = st.sidebar.number_input("Random Seed (optional)", min_value=0, value=42, step=1)

    # --- Pyramid Entry ---
    st.sidebar.subheader("📈 Pyramid Entry")
    pyramid_enabled = st.sidebar.checkbox("Enable Pyramid Entry", value=False, help="Scale into winning positions")
    pyramid_trigger_pips = st.sidebar.number_input("Pyramid Trigger (pips)", min_value=0.1, value=5.0, step=0.5, disabled=not pyramid_enabled)
    pyramid_max_adds = st.sidebar.number_input("Pyramid Max Adds", min_value=1, max_value=10, value=3, step=1, disabled=not pyramid_enabled)
    pyramid_lot_mode = st.sidebar.selectbox("Pyramid Lot Mode", ["fixed", "percent"], index=0, disabled=not pyramid_enabled)
    if pyramid_lot_mode == "fixed":
        pyramid_lot_size = st.sidebar.number_input("Pyramid Lot Size", min_value=0.01, value=0.01, step=0.01, disabled=not pyramid_enabled)
        pyramid_lot_pct = 25.0
    else:
        pyramid_lot_size = 0.01
        pyramid_lot_pct = st.sidebar.number_input("Pyramid Lot %", min_value=1.0, max_value=100.0, value=25.0, step=5.0, disabled=not pyramid_enabled)
    pyramid_be_trigger = st.sidebar.number_input("Pyramid BE Trigger (pips)", min_value=0.0, value=0.0, step=1.0, disabled=not pyramid_enabled, help="0 = disabled")

    # --- Trailing Stop ---
    st.sidebar.subheader("📉 Trailing Stop")
    trailing_enabled = st.sidebar.checkbox("Enable Trailing Stop", value=False, help="Trail stop loss on winning trades")
    trailing_activate_pips = st.sidebar.number_input("Trailing Activate (pips)", min_value=0.1, value=10.0, step=0.5, disabled=not trailing_enabled)
    trailing_distance_pips = st.sidebar.number_input("Trailing Distance (pips)", min_value=0.1, value=5.0, step=0.5, disabled=not trailing_enabled)
    trailing_partial_close = st.sidebar.number_input("Trailing Partial Close (%)", min_value=0.0, max_value=100.0, value=0.0, step=10.0, disabled=not trailing_enabled)

    # --- TP Hit Simulation ---
    st.sidebar.subheader("🎯 TP Hit Simulation")
    tp_hit_enabled = st.sidebar.checkbox("Enable TP Hit Simulation", value=False, help="Not all TP levels may be hit")
    tp_hit_mode = st.sidebar.selectbox("TP Hit Mode", ["fixed", "random", "percent"], index=0, disabled=not tp_hit_enabled)
    tp_hit_pct = st.sidebar.number_input("TP Hit %", min_value=0.0, max_value=100.0, value=100.0, step=5.0, disabled=not tp_hit_enabled)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    return {
        "initial_balance": initial_balance,
        "risk_per_trade": risk_per_trade,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "fee_pips": fee_pips,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "days": days,
        "compounding": compounding,
        "outcome_mode": outcome_mode,
        "custom_pattern": custom_pattern,
        "seed": seed,
        "pyramid_enabled": pyramid_enabled,
        "pyramid_trigger_pips": pyramid_trigger_pips,
        "pyramid_max_adds": pyramid_max_adds,
        "pyramid_lot_mode": pyramid_lot_mode,
        "pyramid_lot_size": pyramid_lot_size,
        "pyramid_lot_pct": pyramid_lot_pct,
        "pyramid_be_trigger_pips": pyramid_be_trigger if pyramid_be_trigger > 0 else None,
        "trailing_enabled": trailing_enabled,
        "trailing_activate_pips": trailing_activate_pips,
        "trailing_distance_pips": trailing_distance_pips,
        "trailing_partial_close_pct": trailing_partial_close,
        "tp_hit_enabled": tp_hit_enabled,
        "tp_hit_mode": tp_hit_mode,
        "tp_hit_pct": tp_hit_pct,
        "run_button": run_button,
    }


def render_run_tab(inputs):
    st.header("🏃 Run Simulation")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Configuration")
        for label, value in [("Symbol", "EUR/USD"), ("Initial Balance", f"€{inputs['initial_balance']:,.2f}"), ("Risk per Trade", f"{inputs['risk_per_trade']:.2%}"), ("Stop Loss", f"{inputs['sl_pips']} pips"), ("Take Profit", f"{inputs['tp_pips']} pips"), ("Fee per Trade", f"{inputs['fee_pips']} pips"), ("Compounding", inputs["compounding"].replace("_", " ").title()), ("Outcome Mode", inputs["outcome_mode"].replace("_", " ").title()), ("Trades per Day", str(inputs["trades_per_day"])), ("Win Rate", f"{inputs['win_rate']:.2%}"), ("Days", str(inputs["days"])), ("Seed", str(inputs["seed"]))] + ([("Pyramid Trigger", f"{inputs['pyramid_trigger_pips']} pips"), ("Pyramid Max Adds", str(inputs["pyramid_max_adds"])), ("Pyramid Lot Mode", inputs["pyramid_lot_mode"]), ("Pyramid BE Trigger", f"{inputs['pyramid_be_trigger_pips']} pips" if inputs.get("pyramid_be_trigger_pips") else "disabled")] if inputs.get("pyramid_enabled") else []) + ([("Trailing Activate", f"{inputs['trailing_activate_pips']} pips"), ("Trailing Distance", f"{inputs['trailing_distance_pips']} pips"), ("Trailing Partial Close", f"{inputs['trailing_partial_close_pct']:.0f}%")] if inputs.get("trailing_enabled") else []) + ([("TP Hit Mode", inputs["tp_hit_mode"]), ("TP Hit %", f"{inputs['tp_hit_pct']:.0f}%")] if inputs.get("tp_hit_enabled") else []):
            st.write(f"**{label}:** {value}")
    with col2:
        st.subheader("Action")
        if inputs["run_button"]:
            try:
                config = build_config(inputs["initial_balance"], inputs["risk_per_trade"], inputs["sl_pips"], inputs["tp_pips"], inputs["fee_pips"], inputs["trades_per_day"], inputs["win_rate"], inputs["days"], inputs["compounding"], inputs["outcome_mode"], inputs["custom_pattern"], inputs["seed"], pyramid_enabled=inputs.get("pyramid_enabled", False), pyramid_trigger_pips=inputs.get("pyramid_trigger_pips", 5.0), pyramid_max_adds=inputs.get("pyramid_max_adds", 3), pyramid_lot_mode=inputs.get("pyramid_lot_mode", "fixed"), pyramid_lot_size=inputs.get("pyramid_lot_size", 0.01), pyramid_lot_pct=inputs.get("pyramid_lot_pct", 25.0), pyramid_be_trigger_pips=inputs.get("pyramid_be_trigger_pips"), trailing_enabled=inputs.get("trailing_enabled", False), trailing_activate_pips=inputs.get("trailing_activate_pips", 10.0), trailing_distance_pips=inputs.get("trailing_distance_pips", 5.0), trailing_partial_close_pct=inputs.get("trailing_partial_close_pct", 0.0), tp_hit_enabled=inputs.get("tp_hit_enabled", False), tp_hit_mode=inputs.get("tp_hit_mode", "fixed"), tp_hit_pct=inputs.get("tp_hit_pct", 100.0))
                _, result, run_id = run_simulation(config, inputs["seed"])
                st.session_state["last_run_id"] = run_id
                st.session_state["last_result"] = result
                st.session_state["last_config"] = config
                st.success(f"✅ Simulation complete! Run ID: `{run_id}`")
                st.balloons()
            except Exception as exc:
                st.error(f"❌ Simulation error: {exc}")
        if st.button("💾 Save Current Result to DB"):
            if "last_result" in st.session_state:
                run_id = str(uuid.uuid4())[:8]
                db = get_db()
                db.save_run(run_id, st.session_state["last_config"], st.session_state["last_result"])
                st.session_state["last_run_id"] = run_id
                st.success(f"✅ Saved to database as `{run_id}`")
            else:
                st.warning("No result to save. Run a simulation first.")
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.divider()
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Balance", f"€{result.initial_balance:,.2f}")
        c2.metric("Final Balance", f"€{result.final_balance:,.2f}")
        c3.metric("Total Return", f"{result.total_return_pct:+.2f}%")
        c4.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Trades", str(len(result.trades)))
        c6.metric("Win Rate", f"{result.win_rate_actual:.2%}")
        c7.metric("Profit Factor", f"{result.profit_factor:.2f}")
        c8.metric("Net Profit", f"€{(result.final_balance - result.initial_balance):,.2f}")

        # M11 Feature Statistics
        pyramid_trades = [t for t in result.trades if getattr(t, 'pyramid_total_fills', 1) > 1]
        trailing_trades = [t for t in result.trades if getattr(t, 'trailing_stop_active', False)]
        tp_missed = [t for t in result.trades if getattr(t, 'tp_level_hit', True) is False]
        if pyramid_trades or trailing_trades or tp_missed:
            st.divider()
            st.subheader("📊 Advanced Feature Statistics")
            s1, s2, s3 = st.columns(3)
            if pyramid_trades:
                s1.metric("Pyramid Positions", str(len(set(t.position_id for t in pyramid_trades if t.position_id))))
                s1.caption(f"{len(pyramid_trades)} total pyramid fills")
            if trailing_trades:
                s2.metric("Trailing Stop Trades", str(len(trailing_trades)))
                trailing_hit = sum(1 for t in trailing_trades if getattr(t, 'trailing_triggered', False))
                s2.caption(f"{trailing_hit} hit trailing stop, {len(trailing_trades) - trailing_hit} reached TP")
            if tp_missed:
                s3.metric("Missed TP Levels", str(len(tp_missed)))
                s3.caption(f"{len(tp_missed)} TP fills not reached")
        st.plotly_chart(make_equity_chart(result, title="Equity Curve Over Time"), use_container_width=True)
        st.subheader("Trade History")
        trades_data = make_trades_table(result.trades)
        if trades_data:
            import pandas as pd
            df = pd.DataFrame(trades_data)
            styled = df.style.map(highlight_outcome, subset=["Outcome"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            e1, e2 = st.columns(2)
            with e1:
                rid = st.session_state.get("last_run_id", "temp")
                cd = export_csv([asdict(t) for t in result.trades], Path(f"/tmp/run_{rid}.csv"))
                cb = cd.read_bytes() if hasattr(cd, "read_bytes") else cd
                st.download_button("⬇️ Export CSV", data=cb, file_name=f"run_{rid}.csv", mime="text/csv")
            with e2:
                jd = export_json([asdict(t) for t in result.trades], Path(f"/tmp/run_{rid}.json"))
                jb = jd.read_bytes() if hasattr(jd, "read_bytes") else jd
                st.download_button("⬇️ Export JSON", data=jb, file_name=f"run_{rid}.json", mime="application/json")


def render_saved_runs_tab():
    st.header("📁 Saved Runs Archive")
    db = get_db()
    runs = db.list_runs()
    if not runs:
        st.info("No saved runs found. Run a simulation first to save results.")
        return
    st.subheader("All Runs")
    import pandas as pd
    runs_df = pd.DataFrame([{"Run ID": r["run_id"], "Created": r["created_at"][:19].replace("T", " ") if r["created_at"] else "N/A", "Initial Balance": f"€{r['initial_balance']:,.2f}", "Final Balance": f"€{r['final_balance']:,.2f}" if r["final_balance"] else "N/A", "Return": f"{(r['final_balance'] / r['initial_balance'] - 1) * 100:+.2f}%" if r["final_balance"] else "N/A", "Compounding": r["compounding_mode"].replace("_", " ").title(), "Outcome Mode": r.get("outcome_mode", "bernoulli").replace("_", " ").title()} for r in runs])
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    st.subheader("Run Details")
    run_id_input = st.text_input("Enter Run ID to inspect:", value=st.session_state.get("last_run_id", ""))
    if run_id_input:
        run = db.get_run(run_id_input)
        if run is None:
            st.error(f"Run `{run_id_input}` not found.")
        else:
            trades = db.get_trades(run_id_input)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Run ID:** {run['run_id']}")
                st.markdown(f"**Created:** {run['created_at'][:19].replace('T', ' ') if run['created_at'] else 'N/A'}")
                st.markdown(f"**Initial Balance:** €{run['initial_balance']:,.2f}")
                if run["final_balance"] is not None:
                    st.markdown(f"**Final Balance:** €{run['final_balance']:,.2f}")
                    ret = (run["final_balance"] / run["initial_balance"] - 1) * 100
                    st.markdown(f"**Total Return:** {ret:+.2f}%")
                else:
                    st.markdown("**Final Balance:** N/A")
                st.markdown(f"**Compounding:** {run['compounding_mode'].replace('_', ' ').title()}")
                st.markdown(f"**Outcome Mode:** {run.get('outcome_mode', 'bernoulli').replace('_', ' ').title()}")
                st.markdown(f"**Risk per Trade:** {run['risk_per_trade']:.2%}")
                st.markdown(f"**Stop Loss:** {run['sl_pips']} pips")
                st.markdown(f"**Take Profit:** {run['tp_pips']} pips")
                st.markdown(f"**Fee per Trade:** {run['fee_pips']} pips")
                st.markdown(f"**Trades per Day:** {run['trades_per_day']}")
                st.markdown(f"**Win Rate:** {run['win_rate']:.2%}")
                st.markdown(f"**Days:** {run['days']}")
                st.markdown(f"**Seed:** {run['seed']}")
            with c2:
                if trades:
                    st.metric("Total Trades Logged", len(trades))
                    wins = [t for t in trades if t["outcome"] == "WIN"]
                    losses = [t for t in trades if t["outcome"] == "LOSS"]
                    st.metric("Winning Trades", len(wins))
                    st.metric("Losing Trades", len(losses))
            if trades:
                st.subheader("Recent Trades (Last 20)")
                trades_data = make_trades_table(trades[-20:])
                df = pd.DataFrame(trades_data)
                styled = df.style.map(highlight_outcome, subset=["Outcome"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                e1, e2 = st.columns(2)
                with e1:
                    csv_data = export_csv(trades, Path(f"/tmp/run_{run_id_input}.csv"))
                    cb = csv_data.read_bytes() if hasattr(csv_data, "read_bytes") else csv_data
                    st.download_button("⬇️ Export CSV", data=cb, file_name=f"run_{run_id_input}.csv", mime="text/csv")
                with e2:
                    jd = export_json(trades, Path(f"/tmp/run_{run_id_input}.json"))
                    jb = jd.read_bytes() if hasattr(jd, "read_bytes") else jd
                    st.download_button("⬇️ Export JSON", data=jb, file_name=f"run_{run_id_input}.json", mime="application/json")


def render_comparison_tab():
    st.header("📊 Run Comparison")
    db = get_db()
    runs = db.list_runs()
    if not runs:
        st.info("No saved runs found. Run and save simulations first.")
        return
    run_options = {r["run_id"]: f"{r['run_id']} ({r['created_at'][:10]})" for r in runs}
    selected_ids = st.multiselect("Select runs to compare (max 6):", options=list(run_options.keys()), format_func=lambda x: run_options[x], max_selections=6)
    if not selected_ids:
        st.warning("Select at least one run to compare.")
        return
    results_dict = {}
    comparison_rows = []
    for run_id in selected_ids:
        run = db.get_run(run_id)
        trades = db.get_trades(run_id)
        from trading_simulator.models import Trade, EquityPoint
        trade_objs = []
        equity_objs = []
        balance = run["initial_balance"]
        peak = balance
        for t in trades:
            trade_objs.append(Trade(trade_number=t["trade_number"], day=t["day"], lots=t["lots"], risk_amount=t["risk_amount"], pips_gross=t["pips_gross"], pips_net=t["pips_net"], fee_pips=t["fee_pips"], profit_euros=t["profit_euros"], balance_before=t["balance_before"], balance_after=t["balance_after"], outcome=t["outcome"], timestamp=datetime.fromisoformat(t["timestamp"]) if t.get("timestamp") else datetime.now()))
            balance = t["balance_after"]
            if balance > peak:
                peak = balance
            dd = ((peak - balance) / peak * 100) if peak > 0 else 0.0
            equity_objs.append(EquityPoint(timestamp=datetime.fromisoformat(t["timestamp"]) if t.get("timestamp") else datetime.now(), balance=balance, drawdown_pct=dd, peak_balance=peak))
        class _Result:
            pass
        result = _Result()
        result.trades = trade_objs
        result.equity_curve = equity_objs
        result.initial_balance = run["initial_balance"]
        result.final_balance = run["final_balance"]
        result.total_return_pct = ((run["final_balance"] - run["initial_balance"]) / run["initial_balance"] * 100) if run["final_balance"] else 0.0
        result.max_drawdown_pct = max((e.drawdown_pct for e in equity_objs), default=0.0)
        result.win_rate_actual = len([t for t in trade_objs if t.outcome == "WIN"]) / len(trade_objs) if trade_objs else 0.0
        wins = [t for t in trade_objs if t.outcome == "WIN"]
        losses = [t for t in trade_objs if t.outcome == "LOSS"]
        gross_profit = sum(t.profit_euros for t in wins)
        gross_loss = abs(sum(t.profit_euros for t in losses))
        result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        label = f"{run_id[:8]} ({run['created_at'][:10]})"
        results_dict[label] = (None, result)
        comparison_rows.append({"Run ID": run_id[:8], "Date": run["created_at"][:10] if run["created_at"] else "N/A", "Initial": f"€{run['initial_balance']:,.2f}", "Final": f"€{run['final_balance']:,.2f}" if run["final_balance"] else "N/A", "Return": f"{result.total_return_pct:+.2f}%", "Trades": len(trade_objs), "Win Rate": f"{result.win_rate_actual:.2%}", "Max DD": f"{result.max_drawdown_pct:.2f}%", "Profit Factor": f"{result.profit_factor:.2f}"})
    import pandas as pd
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    if results_dict:
        st.plotly_chart(make_comparison_chart(results_dict, title="Equity Curve Comparison"), use_container_width=True)


def render_solver_tab():
    st.header("🎯 Goal-Seeking Reverse Solver")
    st.markdown("Enter your starting balance, target balance, and timeframe. The system will compute optimal scenarios with mathematical proofs.")
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_balance = st.number_input("Starting Balance (€)", min_value=1.0, value=1000.0, step=100.0, format="%.2f")
    with col2:
        target_balance = st.number_input("Target Balance (€)", min_value=2.0, value=1000000.0, step=10000.0, format="%.2f")
    with col3:
        months = st.number_input("Timeframe (months)", min_value=1, max_value=600, value=12, step=1)
    col4, col5 = st.columns(2)
    with col4:
        fee_pips = st.number_input("Fee per Trade (pips)", min_value=0.0, value=1.0, step=0.1)
    with col5:
        sl_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=10.0, step=1.0)
    if st.button("🔍 Compute Scenarios", type="primary", use_container_width=True):
        try:
            result = solve(initial_balance=initial_balance, target_balance=target_balance, months=months, fee_pips=fee_pips, sl_pips=sl_pips)
            st.session_state["solver_result"] = result
        except ValueError as exc:
            st.error(f"❌ {exc}")
    if "solver_result" in st.session_state:
        result = st.session_state["solver_result"]
        st.subheader("Solver Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Initial Balance", f"€{result.initial_balance:,.2f}")
        s2.metric("Target Balance", f"€{result.target_balance:,.2f}")
        s3.metric("Timeframe", f"{result.months} months")
        s4.metric("Required Monthly Return", f"{result.required_monthly_return_pct:+.2f}%")
        st.metric("Required Total Return", f"{result.required_total_return_pct:+.2f}%")
        st.metric("Feasible", "✅ Yes" if result.feasible else "❌ No")
        st.subheader("Scenarios")
        import pandas as pd
        scenarios_data = []
        for s in result.scenarios:
            is_rec = s is result.recommendation
            ms = f"{s.months_to_target:.1f}" if s.months_to_target != float("inf") else "∞"
            scenarios_data.append({"Scenario": f"{'🏆 ' if is_rec else ''}{s.name}", "Description": s.description, "Trades/Day": s.trades_per_day, "Win Rate": f"{s.win_rate:.2%}", "Risk/Trade": f"{s.risk_per_trade:.2%}", "SL/TP": f"{s.sl_pips}/{s.tp_pips}", "Monthly Return": f"{s.expected_monthly_return_pct:+.2f}%", "Months to Target": ms, "Risk of Ruin": f"{s.risk_of_ruin:.2%}", "EV per Trade (R)": f"{s.expected_value_per_trade_r:+.4f}", "Kelly Fraction": f"{s.kelly_fraction:.2%}"})
        df = pd.DataFrame(scenarios_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        rec = result.recommendation
        rms = f"{rec.months_to_target:.1f}" if rec.months_to_target != float("inf") else "∞"
        st.subheader("Recommended Scenario")
        st.info(f"**{rec.name}** — {rec.description}\n\n• Trades/Day: {rec.trades_per_day}\n• Win Rate: {rec.win_rate:.2%}\n• Risk per Trade: {rec.risk_per_trade:.2%}\n• Stop Loss / Take Profit: {rec.sl_pips}/{rec.tp_pips} pips\n• Monthly Return: {rec.expected_monthly_return_pct:+.2f}%\n• Months to Target: {rms}\n• Risk of Ruin: {rec.risk_of_ruin:.2%}\n• Kelly Fraction: {rec.kelly_fraction:.2%}")
        st.subheader("Mathematical Proof")
        st.code(rec.mathematical_proof, language="text")


def main():
    st.title("📈 Trading Simulation & Goal-Modeling Platform")
    st.markdown("A mathematical trading simulation engine for modeling risk, lot sizing, compounding, spread/fees, and trade outcomes over time.")
    inputs = render_sidebar()
    tab1, tab2, tab3, tab4 = st.tabs(["🏃 Run Simulation", "📁 Saved Runs", "📊 Compare Runs", "🎯 Goal-Seeking Solver"])
    with tab1:
        render_run_tab(inputs)
    with tab2:
        render_saved_runs_tab()
    with tab3:
        render_comparison_tab()
    with tab4:
        render_solver_tab()
    st.divider()
    st.caption("Trading Simulation & Goal-Modeling Platform — Strictly a simulation and goal-modeling tool. Not a live trading system.")


if __name__ == "__main__":
    main()