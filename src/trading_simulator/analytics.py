"""Performance analytics: daily, weekly, monthly aggregation and quantitative metrics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    period: str
    period_label: str
    period_start: datetime
    period_end: datetime
    starting_balance: float
    ending_balance: float
    return_pct: float
    total_pips: float
    win_count: int
    loss_count: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    trade_count: int
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0

    @property
    def is_profitable(self) -> bool:
        return self.ending_balance > self.starting_balance


def _compute_drawdown(balances: list[float], starting_balance: float) -> float:
    """Compute max drawdown % from sequential balances."""
    if not balances:
        return 0.0

    peak = starting_balance
    max_dd = 0.0

    for balance in balances:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak * 100 if peak > 0 else 0.0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _build_metrics(
    period: str,
    label: str,
    start: datetime,
    end: datetime,
    trades: list,
    equity_snapshots: list | None = None,
) -> PerformanceMetrics:
    """Build a PerformanceMetrics object from a list of trades."""
    if not trades:
        return PerformanceMetrics(
            period=period,
            period_label=label,
            period_start=start,
            period_end=end,
            starting_balance=0.0,
            ending_balance=0.0,
            return_pct=0.0,
            total_pips=0.0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            trade_count=0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
        )

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]

    gross_profit = sum(t.profit_euros for t in wins)
    gross_loss = abs(sum(t.profit_euros for t in losses))
    net_profit = gross_profit - gross_loss

    total_pips = sum(t.pips_net for t in trades)

    starting = trades[0].balance_before
    ending = trades[-1].balance_after

    return_pct = ((ending - starting) / starting * 100) if starting > 0 else 0.0
    win_rate = len(wins) / len(trades) if trades else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if equity_snapshots:
        period_balances = [
            s.balance
            for s in equity_snapshots
            if start <= s.timestamp <= end
        ]
        if period_balances:
            max_drawdown_pct = _compute_drawdown(period_balances, starting)
        else:
            balances = [starting] + [t.balance_after for t in trades]
            max_drawdown_pct = _compute_drawdown(balances, starting)
    else:
        balances = [starting] + [t.balance_after for t in trades]
        max_drawdown_pct = _compute_drawdown(balances, starting)

    return PerformanceMetrics(
        period=period,
        period_label=label,
        period_start=start,
        period_end=end,
        starting_balance=starting,
        ending_balance=ending,
        return_pct=return_pct,
        total_pips=total_pips,
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=len(trades),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
    )


def aggregate_daily(trades, equity_curve=None) -> list[PerformanceMetrics]:
    """Aggregate trades into daily performance metrics."""
    by_day: dict[int, list] = defaultdict(list)
    for t in trades:
        by_day[t.day].append(t)

    metrics = []
    for day in sorted(by_day.keys()):
        day_trades = by_day[day]
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day - 1)
        end = start + timedelta(days=1) - timedelta(seconds=1)
        metrics.append(
            _build_metrics("daily", f"Day {day}", start, end, day_trades, equity_curve)
        )
    return metrics


def aggregate_weekly(trades, equity_curve=None) -> list[PerformanceMetrics]:
    """Aggregate trades into weekly performance metrics."""
    by_week: dict[int, list] = defaultdict(list)
    for t in trades:
        week_num = (t.day - 1) // 7 + 1
        by_week[week_num].append(t)

    metrics = []
    for week_num in sorted(by_week.keys()):
        week_trades = by_week[week_num]
        label = f"Week {week_num}"
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(weeks=week_num - 1)
        end = start + timedelta(weeks=1) - timedelta(seconds=1)
        metrics.append(
            _build_metrics("weekly", label, start, end, week_trades, equity_curve)
        )
    return metrics


def aggregate_monthly(trades, equity_curve=None) -> list[PerformanceMetrics]:
    """Aggregate trades into monthly performance metrics."""
    by_month: dict[int, list] = defaultdict(list)
    for t in trades:
        month_num = (t.day - 1) // 30 + 1
        by_month[month_num].append(t)

    metrics = []
    for month_num in sorted(by_month.keys()):
        month_trades = by_month[month_num]
        label = f"Month {month_num}"
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=(month_num - 1) * 30)
        end = start + timedelta(days=30) - timedelta(seconds=1)
        metrics.append(
            _build_metrics("monthly", label, start, end, month_trades, equity_curve)
        )
    return metrics


def aggregate_all_periods(trades, equity_curve=None) -> dict[str, list[PerformanceMetrics]]:
    """Aggregate trades into daily, weekly, and monthly metrics at once."""
    return {
        "daily": aggregate_daily(trades, equity_curve),
        "weekly": aggregate_weekly(trades, equity_curve),
        "monthly": aggregate_monthly(trades, equity_curve),
    }


@dataclass
class QuantMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    cagr_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    downside_volatility_pct: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate: float
    expectancy_euros: float
    average_win_euros: float
    average_loss_euros: float
    payoff_ratio: float
    recovery_factor: float
    total_trades: int
    net_profit_euros: float
    final_balance_euros: float


def compute_monte_carlo_drawdown(
    trades: list,
    initial_balance: float = 1000.0,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap resampling of trade returns to compute distribution of maximum drawdowns."""
    if not trades:
        return {"median_dd_pct": 0.0, "p95_dd_pct": 0.0, "p99_dd_pct": 0.0, "max_dd_pct": 0.0}

    pnl_list = [t.profit_euros for t in trades]
    k = len(pnl_list)
    rng = random.Random(seed)
    max_drawdowns: list[float] = []

    for _ in range(n_simulations):
        sampled_pnls = rng.choices(pnl_list, k=k)
        bal = initial_balance
        peak = initial_balance
        dd_max = 0.0
        for pnl in sampled_pnls:
            bal += pnl
            if bal > peak:
                peak = bal
            dd = ((peak - bal) / peak * 100.0) if peak > 0 else 0.0
            if dd > dd_max:
                dd_max = dd
        max_drawdowns.append(round(dd_max, 4))

    max_drawdowns.sort()
    n = len(max_drawdowns)
    return {
        "median_dd_pct": max_drawdowns[int(n * 0.50)],
        "p95_dd_pct": max_drawdowns[int(n * 0.95)],
        "p99_dd_pct": max_drawdowns[int(n * 0.99)],
        "max_dd_pct": max_drawdowns[-1],
    }


def compute_quant_metrics(
    trades: list,
    equity_curve: list | None = None,
    initial_balance: float = 1000.0,
    days: int | None = None,
    risk_free_rate: float = 0.0,
) -> QuantMetrics:
    """Compute comprehensive quantitative analytics (Sharpe, Sortino, Calmar, CAGR, Expectancy, etc.)."""
    if not trades:
        return QuantMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            cagr_pct=0.0,
            annualized_return_pct=0.0,
            annualized_volatility_pct=0.0,
            downside_volatility_pct=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            expectancy_euros=0.0,
            average_win_euros=0.0,
            average_loss_euros=0.0,
            payoff_ratio=0.0,
            recovery_factor=0.0,
            total_trades=0,
            net_profit_euros=0.0,
            final_balance_euros=initial_balance,
        )

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]

    gross_profit = sum(t.profit_euros for t in wins)
    gross_loss = abs(sum(t.profit_euros for t in losses))
    net_profit = gross_profit - gross_loss

    total_trades = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
    avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    final_balance = trades[-1].balance_after

    # Compute daily balances and returns
    daily_metrics = aggregate_daily(trades, equity_curve)
    n_days = days if days is not None else len(daily_metrics)
    n_days = max(1, n_days)

    daily_returns: list[float] = []
    daily_balances = [initial_balance] + [m.ending_balance for m in daily_metrics]
    for i in range(1, len(daily_balances)):
        prev = daily_balances[i - 1]
        curr = daily_balances[i]
        r = ((curr - prev) / prev) if prev > 0 else 0.0
        daily_returns.append(r)

    # Max Drawdown
    all_balances = [initial_balance] + [t.balance_after for t in trades]
    peak = initial_balance
    max_dd_amount = 0.0
    max_dd_pct = 0.0
    for b in all_balances:
        if b > peak:
            peak = b
        dd_amt = peak - b
        dd_pct = (dd_amt / peak * 100.0) if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        if dd_amt > max_dd_amount:
            max_dd_amount = dd_amt

    recovery_factor = (net_profit / max_dd_amount) if max_dd_amount > 0 else (float("inf") if net_profit > 0 else 0.0)

    # Returns statistics
    if daily_returns:
        n_ret = len(daily_returns)
        mean_r = sum(daily_returns) / n_ret
        rf_daily = risk_free_rate / 252.0

        if n_ret > 1:
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (n_ret - 1)
            std_r = math.sqrt(var_r)
        else:
            std_r = 0.0

        downside_sq = sum(min(0.0, r - rf_daily) ** 2 for r in daily_returns)
        downside_std = math.sqrt(downside_sq / n_ret) if n_ret > 0 else 0.0

        ann_return = mean_r * 252.0
        ann_vol = std_r * math.sqrt(252.0)
        ann_downside_vol = downside_std * math.sqrt(252.0)

        sharpe = ((mean_r - rf_daily) / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0

        if downside_std > 0:
            sortino = (mean_r - rf_daily) / downside_std * math.sqrt(252.0)
        elif (mean_r - rf_daily) > 0:
            sortino = float("inf")
        else:
            sortino = 0.0
    else:
        ann_return = 0.0
        ann_vol = 0.0
        ann_downside_vol = 0.0
        sharpe = 0.0
        sortino = 0.0

    # CAGR: (V_final / V_initial) ** (252 / n_days) - 1
    years = n_days / 252.0
    if final_balance > 0 and initial_balance > 0 and years > 0:
        try:
            cagr = ((final_balance / initial_balance) ** (1.0 / years)) - 1.0
        except (OverflowError, ZeroDivisionError):
            cagr = float("inf")
    else:
        cagr = -1.0 if final_balance <= 0 else 0.0

    ann_return_pct = ann_return * 100.0
    cagr_pct_val = cagr * 100.0 if cagr != float("inf") else float("inf")
    if max_dd_pct > 0:
        calmar = cagr_pct_val / max_dd_pct if cagr_pct_val != float("inf") else float("inf")
    elif cagr_pct_val > 0:
        calmar = float("inf")
    else:
        calmar = 0.0

    return QuantMetrics(
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4) if sortino != float("inf") else float("inf"),
        calmar_ratio=round(calmar, 4) if calmar != float("inf") else float("inf"),
        cagr_pct=round(cagr * 100.0, 2) if cagr != float("inf") else float("inf"),
        annualized_return_pct=round(ann_return_pct, 2),
        annualized_volatility_pct=round(ann_vol * 100.0, 2),
        downside_volatility_pct=round(ann_downside_vol * 100.0, 2),
        max_drawdown_pct=round(max_dd_pct, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
        win_rate=round(win_rate, 4),
        expectancy_euros=round(expectancy, 4),
        average_win_euros=round(avg_win, 4),
        average_loss_euros=round(avg_loss, 4),
        payoff_ratio=round(payoff_ratio, 4) if payoff_ratio != float("inf") else float("inf"),
        recovery_factor=round(recovery_factor, 4) if recovery_factor != float("inf") else float("inf"),
        total_trades=total_trades,
        net_profit_euros=round(net_profit, 4),
        final_balance_euros=round(final_balance, 4),
    )
