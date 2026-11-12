"""Performance analytics: daily, weekly, monthly aggregation."""

from __future__ import annotations

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
