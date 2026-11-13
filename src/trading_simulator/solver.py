"""Goal-seeking reverse solver with mathematical proofs.

Computes required parameters to reach a target balance from a starting
balance within a given timeframe. Supports comparative scenarios
(Conservative, Moderate, Aggressive) with mathematical backing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SolverScenario:
    """A complete trading scenario with parameters and expected outcomes."""

    name: str
    trades_per_day: int
    win_rate: float
    risk_per_trade: float
    sl_pips: float
    tp_pips: float
    fee_pips: float
    expected_monthly_return_pct: float
    months_to_target: float
    risk_of_ruin: float
    expected_value_per_trade_r: float
    kelly_fraction: float
    description: str
    mathematical_proof: str = ""


@dataclass
class SolverResult:
    """Result of the goal-seeking reverse solver."""

    initial_balance: float
    target_balance: float
    months: int
    required_monthly_return_pct: float
    required_total_return_pct: float
    scenarios: list[SolverScenario]
    recommendation: SolverScenario
    feasible: bool


def _expected_value_per_trade_r(
    win_rate: float, tp_ratio: float, fee_r: float
) -> float:
    """Expected value per trade expressed in multiples of R (risk amount).

    EV_R = p × b - (1-p) - fee_r

    where:
    - p = win rate
    - b = take-profit / stop-loss ratio (tp_pips / sl_pips)
    - fee_r = fee as fraction of R (fee_pips / sl_pips)
    """
    return win_rate * tp_ratio - (1.0 - win_rate) - fee_r


def _monthly_return_pct(
    trades_per_day: int,
    win_rate: float,
    tp_ratio: float,
    risk_per_trade: float,
    fee_pips: float,
    sl_pips: float,
    trading_days_per_month: int = 21,
) -> float:
    """Expected monthly return as percentage.

    Compounds trade-by-trade: (1 + EV_pct_per_trade)^trades_per_month - 1

    EV_pct_per_trade = risk_per_trade × EV_R
    """
    fee_r = fee_pips / sl_pips if sl_pips > 0 else 0.0
    ev_r = _expected_value_per_trade_r(win_rate, tp_ratio, fee_r)
    ev_pct_per_trade = risk_per_trade * ev_r
    trades_per_month = trades_per_day * trading_days_per_month

    monthly_return = (1.0 + ev_pct_per_trade) ** trades_per_month - 1.0
    return monthly_return * 100.0


def _risk_of_ruin(
    win_rate: float,
    tp_ratio: float,
    risk_per_trade: float,
    fee_pips: float,
    sl_pips: float,
) -> float:
    """Risk of ruin over infinite time using the standard formula.

    RoR = ((1 - EV_R) / (1 + EV_R))^(1 / risk_per_trade)

    If EV_R <= 0, ruin is certain (RoR = 1.0).
    """
    fee_r = fee_pips / sl_pips if sl_pips > 0 else 0.0
    ev_r = _expected_value_per_trade_r(win_rate, tp_ratio, fee_r)

    if ev_r <= 0:
        return 1.0

    if risk_per_trade <= 0:
        return 1.0

    units = 1.0 / risk_per_trade
    p = win_rate
    b = tp_ratio

    # Solve Ralph Vince / Nauzer Balsara characteristic equation for asymmetric payoffs:
    # p * z^(b + 1) - z + (1 - p) = 0 for z in (0, 1)
    # Using bisection for guaranteed numerical stability and monotonic convergence
    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = p * (mid ** (b + 1.0)) - mid + (1.0 - p)
        if val > 0:
            low = mid
        else:
            high = mid
    z = (low + high) / 2.0

    try:
        ror = z ** units
    except (OverflowError, ValueError, ZeroDivisionError):
        ror = 0.0

    return max(0.0, min(1.0, float(ror)))


def _kelly_fraction(
    win_rate: float,
    tp_ratio: float,
    fee_pips: float,
    sl_pips: float,
) -> float:
    """Kelly optimal fraction to risk per trade.

    f* = (p × b - (1-p) - fee_r) / b
    """
    b = tp_ratio
    if b <= 0:
        return 0.0
    fee_r = fee_pips / sl_pips if sl_pips > 0 else 0.0
    numerator = win_rate * b - (1.0 - win_rate) - fee_r
    kelly = numerator / b
    return max(0.0, min(1.0, kelly))


def _mathematical_proof(
    scenario_name: str,
    trades_per_day: int,
    win_rate: float,
    risk_per_trade: float,
    sl_pips: float,
    tp_pips: float,
    fee_pips: float,
    monthly_return_pct: float,
    months_to_target: float,
    ror: float,
    ev_r: float,
    initial_balance: float,
    target_balance: float,
) -> str:
    """Generate a mathematical proof string for a scenario."""
    tp_ratio = tp_pips / sl_pips if sl_pips > 0 else 0.0
    fee_r = fee_pips / sl_pips if sl_pips > 0 else 0.0
    ev_pct_per_trade = risk_per_trade * ev_r
    trades_per_month = trades_per_day * 21
    required_total = (target_balance / initial_balance - 1) * 100

    lines = [
        f"=== Mathematical Proof: {scenario_name} ===",
        "",
        "1. Required Growth:",
        f"   Target: €{initial_balance:,.2f} -> €{target_balance:,.2f}",
        f"   Total return needed: {required_total:.2f}%",
        f"   Monthly return needed: {monthly_return_pct:.2f}% (compounded)",
        f"   Months available: {months_to_target:.1f}",
        "",
        "2. Expected Value per Trade (in R units):",
        f"   Win rate (p):         {win_rate:.2%}",
        f"   Win/Loss ratio (b):   {tp_ratio:.2f} ({tp_pips}/{sl_pips} pips)",
        f"   Fee as % of R:        {fee_r:.4f}",
        f"   EV_R = p*b - (1-p) - fee_r",
        f"   EV_R = {win_rate:.2f} * {tp_ratio:.2f} - (1-{win_rate:.2f}) - {fee_r:.4f}",
        f"   EV_R = {ev_r:.4f}",
        "",
        "3. Expected Return per Trade (% of balance):",
        f"   Risk per trade:       {risk_per_trade:.2%}",
        f"   EV_pct = risk * EV_R = {risk_per_trade:.4f} * {ev_r:.4f}",
        f"   EV_pct = {ev_pct_per_trade:.6%}",
        "",
        "4. Monthly Compounded Return:",
        f"   Trades/month:         {trades_per_day} * 21 = {trades_per_month}",
        f"   Monthly = (1 + EV_pct)^trades - 1",
        f"   Monthly = (1 + {ev_pct_per_trade:.6f})^{trades_per_month} - 1",
        f"   Monthly = {monthly_return_pct:.2f}%",
        "",
        "5. Risk of Ruin:",
        f"   RoR = ((1 - EV_R) / (1 + EV_R))^(1/risk)",
        f"   RoR = ((1 - {ev_r:.4f}) / (1 + {ev_r:.4f}))^(1/{risk_per_trade})",
        f"   RoR = {ror:.4%}",
        "",
        "6. Verification:",
        f"   €{initial_balance:,.2f} * (1 + {monthly_return_pct/100:.6f})^months",
        f"   ≈ €{target_balance:,.2f}",
        "",
    ]
    return "\n".join(lines)


def solve(
    initial_balance: float,
    target_balance: float,
    months: int,
    fee_pips: float = 1.0,
    sl_pips: float = 10.0,
) -> SolverResult:
    """Compute required parameters to reach target balance.

    Args:
        initial_balance: Starting capital (e.g., 1000.0)
        target_balance: Target balance (e.g., 1000000.0)
        months: Timeframe in months (e.g., 12)
        fee_pips: Flat fee per trade in pips (default: 1.0)
        sl_pips: Stop loss distance in pips (default: 10.0)

    Returns:
        SolverResult with required monthly return, scenarios, and recommendation.
    """
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    if target_balance <= initial_balance:
        raise ValueError("target_balance must exceed initial_balance")
    if months <= 0:
        raise ValueError("months must be positive")

    required_total_return = (target_balance / initial_balance - 1) * 100
    required_monthly = (
        (target_balance / initial_balance) ** (1.0 / months) - 1.0
    ) * 100.0

    scenarios_data = [
        {
            "name": "Conservative",
            "trades_per_day": 3,
            "win_rate": 0.60,
            "risk_per_trade": 0.01,
            "tp_pips": 20.0,
            "description": "Low risk, steady compounding",
        },
        {
            "name": "Moderate",
            "trades_per_day": 10,
            "win_rate": 0.55,
            "risk_per_trade": 0.02,
            "tp_pips": 30.0,
            "description": "Balanced growth and risk",
        },
        {
            "name": "Aggressive",
            "trades_per_day": 15,
            "win_rate": 0.50,
            "risk_per_trade": 0.05,
            "tp_pips": 30.0,
            "description": "High growth, high risk of ruin",
        },
    ]

    scenarios = []
    for data in scenarios_data:
        tp_ratio = data["tp_pips"] / sl_pips if sl_pips > 0 else 0.0
        monthly_return = _monthly_return_pct(
            trades_per_day=data["trades_per_day"],
            win_rate=data["win_rate"],
            tp_ratio=tp_ratio,
            risk_per_trade=data["risk_per_trade"],
            fee_pips=fee_pips,
            sl_pips=sl_pips,
        )

        if monthly_return > 0:
            months_to_target = math.log(
                target_balance / initial_balance
            ) / math.log(1.0 + monthly_return / 100.0)
        else:
            months_to_target = float("inf")

        ror = _risk_of_ruin(
            win_rate=data["win_rate"],
            tp_ratio=tp_ratio,
            risk_per_trade=data["risk_per_trade"],
            fee_pips=fee_pips,
            sl_pips=sl_pips,
        )

        fee_r = fee_pips / sl_pips if sl_pips > 0 else 0.0
        ev_r = _expected_value_per_trade_r(
            win_rate=data["win_rate"],
            tp_ratio=tp_ratio,
            fee_r=fee_r,
        )

        kelly = _kelly_fraction(
            win_rate=data["win_rate"],
            tp_ratio=tp_ratio,
            fee_pips=fee_pips,
            sl_pips=sl_pips,
        )

        proof = _mathematical_proof(
            scenario_name=data["name"],
            trades_per_day=data["trades_per_day"],
            win_rate=data["win_rate"],
            risk_per_trade=data["risk_per_trade"],
            sl_pips=sl_pips,
            tp_pips=data["tp_pips"],
            fee_pips=fee_pips,
            monthly_return_pct=monthly_return,
            months_to_target=months_to_target,
            ror=ror,
            ev_r=ev_r,
            initial_balance=initial_balance,
            target_balance=target_balance,
        )

        scenario = SolverScenario(
            name=data["name"],
            trades_per_day=data["trades_per_day"],
            win_rate=data["win_rate"],
            risk_per_trade=data["risk_per_trade"],
            sl_pips=sl_pips,
            tp_pips=data["tp_pips"],
            fee_pips=fee_pips,
            expected_monthly_return_pct=monthly_return,
            months_to_target=months_to_target,
            risk_of_ruin=ror,
            expected_value_per_trade_r=ev_r,
            kelly_fraction=kelly,
            description=data["description"],
            mathematical_proof=proof,
        )
        scenarios.append(scenario)

    feasible = any(s.months_to_target <= months for s in scenarios)

    viable = [s for s in scenarios if s.months_to_target <= months]
    if viable:
        recommendation = min(viable, key=lambda s: s.risk_of_ruin)
    else:
        recommendation = min(scenarios, key=lambda s: s.risk_of_ruin)

    return SolverResult(
        initial_balance=initial_balance,
        target_balance=target_balance,
        months=months,
        required_monthly_return_pct=required_monthly,
        required_total_return_pct=required_total_return,
        scenarios=scenarios,
        recommendation=recommendation,
        feasible=feasible,
    )