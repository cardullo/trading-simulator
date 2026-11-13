"""
Core simulation engine — price-path based simulation.

Simulates trade outcomes using price-path modeling instead of simple Bernoulli
win/loss trials. This enables proper modeling of pyramiding, trailing stops,
partial take profits, and breakeven stops as they would occur in real markets.

Each trade simulates price movement step by step (±1 pip per step). At each
step, the engine checks exit conditions in order:
1. Stop Loss hit → close at SL (LOSS)
2. Take Profit hit → close at TP (WIN)
3. Trailing Stop hit → close at trailing level (WIN)
4. Pyramid trigger → add to position (VWAP averaging)
5. Breakeven trigger → move SL to entry
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from trading_simulator.config import (
    CompoundingMode,
    OutcomeMode,
    PricePathMode,
    PyramidMode,
    SimulationConfig,
    TPHitMode,
)
from trading_simulator.compounding import compute_risk_base
from trading_simulator.lot_sizing import calculate_lot_size
from trading_simulator.models import Account, Instrument, Trade, EquityPoint
@dataclass
class SimulationResult:
    config: SimulationConfig
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_actual: float = 0.0
    profit_factor: float = 0.0
    margin_calls: int = 0
    stop_outs: int = 0
    margin_rejections: int = 0


class SimulationEngine:
    """Core simulation engine using price-path modeling.

    Each trade simulates price movement step by step (±1 pip per step). At each
    step, the engine checks exit conditions in order:
    1. Stop Loss hit → close at SL (LOSS)
    2. Take Profit hit → close at TP (WIN)
    3. Trailing Stop hit → close at trailing level (WIN)
    4. Pyramid trigger → add to position (VWAP averaging)
    5. Breakeven trigger → move SL to entry
    """

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        self.seed = seed if seed is not None else config.seed
        self.rng = random.Random(self.seed)
        self.account = Account(
            balance=config.account.initial_balance,
            initial_balance=config.account.initial_balance,
            peak_balance=config.account.initial_balance,
        )
        self.instrument = Instrument(
            symbol=config.instrument.symbol,
            pip_size=float(config.instrument.pip_size),
            contract_size=config.instrument.contract_size,
            quote_to_account_rate=config.instrument.quote_to_account_rate,
        )
        # Cache frequently accessed values
        self._pip_value_per_lot = self.instrument.pip_value_per_lot
        self._fee_pips = config.risk.fee_pips
        self._sl_pips = config.risk.sl_pips
        self._tp_pips = config.risk.tp_pips
        self._risk_per_trade = config.account.risk_per_trade
        self._compounding_mode = config.risk.compounding_mode
        self._win_rate = config.activity.win_rate
        self._trades_per_day = config.activity.trades_per_day
        self._tp_levels = config.risk.tp_levels
        self._price_path_mode = config.activity.price_path_mode
        self._pullback_prob = config.activity.pullback_prob

        # Pyramid entry configuration
        self._pyramid_enabled = config.pyramid.enabled
        self._pyramid_trigger_pips = config.pyramid.trigger_pips
        self._pyramid_max_adds = config.pyramid.max_adds
        self._pyramid_lot_mode = config.pyramid.lot_mode
        self._pyramid_lot_size = config.pyramid.lot_size
        self._pyramid_lot_pct = config.pyramid.lot_pct
        self._pyramid_be_trigger_pips = config.pyramid.be_trigger_pips
        self._pyramid_be_offset_pips = config.pyramid.be_offset_pips

        # Trailing stop configuration
        self._trailing_enabled = config.trailing_stop.enabled
        self._trailing_activate_pips = config.trailing_stop.activate_pips
        self._trailing_distance_pips = config.trailing_stop.distance_pips
        self._trailing_partial_close_pct = config.trailing_stop.partial_close_pct

        # TP hit tracking configuration
        self._tp_hit_enabled = config.tp_hit.enabled
        self._tp_hit_mode = config.tp_hit.mode
        self._tp_hit_pct = config.tp_hit.hit_pct
        self._tp_levels_to_hit = set()

        # Leverage & Margin configuration
        self._leverage_enabled = config.leverage.enabled
        self._leverage = config.leverage.leverage
        self._margin_call_pct = config.leverage.margin_call_pct
        self._stop_out_pct = config.leverage.stop_out_pct
        self._max_lot_exposure = config.leverage.max_lot_exposure

        self.margin_calls: int = 0
        self.stop_outs: int = 0
        self.margin_rejections: int = 0

        # Execution Friction (spread, slippage, swaps)
        self._execution_enabled = config.execution.enabled
        self._variable_spread_enabled = config.execution.variable_spread_enabled
        self._base_spread_pips = config.execution.base_spread_pips
        self._spread_std_pips = config.execution.spread_std_pips
        self._slippage_enabled = config.execution.slippage_enabled
        self._sl_slippage_mean_pips = config.execution.sl_slippage_mean_pips
        self._sl_slippage_std_pips = config.execution.sl_slippage_std_pips
        self._tp_slippage_improvement_prob = config.execution.tp_slippage_improvement_prob
        self._tp_slippage_mean_pips = config.execution.tp_slippage_mean_pips
        self._swap_enabled = config.execution.swap_enabled
        self._swap_long_pips = config.execution.swap_long_pips
        self._holding_nights = config.execution.holding_nights
        self._triple_swap_day = config.execution.triple_swap_day

        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self._day = 1
        self._position_counter = 0
        self._current_position_trades: list[Trade] = []

    def run(self, days: int | None = None) -> SimulationResult:
        """Run the simulation for the given number of days."""
        days = days or self.config.activity.days
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        timestamps = [
            start_date + timedelta(days=d) for d in range(1, days + 1)
        ]

        trades_per_day = self._trades_per_day

        for day in range(1, days + 1):
            self._day = day
            for _ in range(trades_per_day):
                self._execute_trade()

            self._record_equity(timestamps[day - 1])

        return self._build_result()

    def _record_equity(self, timestamp: datetime) -> None:
        """Record an equity snapshot."""
        peak = self.account.peak_balance
        drawdown_pct = ((peak - self.account.balance) / peak * 100) if peak > 0 else 0.0
        self.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                balance=self.account.balance,
                drawdown_pct=drawdown_pct,
                peak_balance=peak,
            )
        )

    def _determine_outcome(self) -> Literal["WIN", "LOSS"]:
        """Determine trade outcome based on configured outcome mode."""
        rng = self.rng.random()

        if self.config.activity.outcome_mode == OutcomeMode.BERNOULLI:
            return "WIN" if rng < self._win_rate else "LOSS"

        elif self.config.activity.outcome_mode == OutcomeMode.ALTERNATING:
            if not hasattr(self, "_alternating_state"):
                self._alternating_state = True
            outcome = "WIN" if self._alternating_state else "LOSS"
            self._alternating_state = not self._alternating_state
            return outcome

        elif self.config.activity.outcome_mode == OutcomeMode.FIXED_WINS_FIRST:
            total = self._trades_per_day * self.config.activity.days
            wins_count = int(total * self._win_rate)
            trade_idx = self._position_counter - 1
            return "WIN" if trade_idx < wins_count else "LOSS"

        elif self.config.activity.outcome_mode == OutcomeMode.FIXED_LOSSES_FIRST:
            total = self._trades_per_day * self.config.activity.days
            losses_count = int(total * (1 - self._win_rate))
            trade_idx = self._position_counter - 1
            return "LOSS" if trade_idx < losses_count else "WIN"

        elif self.config.activity.outcome_mode == OutcomeMode.CUSTOM_PATTERN:
            if not self.config.activity.custom_pattern:
                raise ValueError("custom_pattern must be specified when outcome_mode is custom_pattern")
            pattern = self.config.activity.custom_pattern.upper()
            trade_idx = self._position_counter - 1
            result = pattern[trade_idx % len(pattern)]
            return "WIN" if result == "W" else "LOSS"

        return "WIN" if rng < self._win_rate else "LOSS"

    def _record_fill(
        self,
        position_id: str,
        fill_index: int,
        total_fills: int,
        lots: float,
        pips_gross: float,
        pips_net: float,
        fee_pips: float,
        profit_euros: float,
        entry_price: float,
        position_avg_price: float,
        outcome: Literal["WIN", "LOSS"],
        close_price: float,
        tp_level_hit: bool = True,
        tp_level_pips: float = 0.0,
        balance_before_override: float | None = None,
        trailing_stop_active: bool = False,
        trailing_stop_pips: float = 0.0,
        trailing_triggered: bool = False,
    ) -> None:
        """Record a single fill (partial or full position exit/add)."""
        balance_before = balance_before_override if balance_before_override is not None else self.account.balance

        # Round balance_before to prevent floating point drift
        balance_before = round(balance_before, 4)
        profit_euros = round(profit_euros, 4)
        balance_after = round(balance_before + profit_euros, 4)

        trade = Trade(
            trade_number=len(self.trades) + 1,
            day=self._day,
            lots=round(lots, 2),
            risk_amount=round(lots * self._sl_pips * self._pip_value_per_lot, 4),
            pips_gross=round(pips_gross, 4),
            pips_net=round(pips_net, 4),
            fee_pips=round(fee_pips, 4),
            profit_euros=profit_euros,
            balance_before=balance_before,
            balance_after=balance_after,
            outcome=outcome,
            position_id=position_id,
            fill_index=fill_index,
            total_fills=total_fills,
            entry_price=round(entry_price, 4),
            position_avg_price=round(position_avg_price, 4),
            tp_level_hit=tp_level_hit,
            tp_level_pips=round(tp_level_pips, 4),
            trailing_stop_active=trailing_stop_active,
            trailing_stop_pips=round(trailing_stop_pips, 4),
            trailing_triggered=trailing_triggered,
        )
        self.trades.append(trade)
        self._current_position_trades.append(trade)

        # Update fill indices for all trades in this position
        total_pos_fills = len(self._current_position_trades)
        for idx, t in enumerate(self._current_position_trades):
            t.fill_index = idx
            t.total_fills = total_pos_fills
            t.pyramid_fill_index = idx
            t.pyramid_total_fills = total_pos_fills

    def _execute_trade(self) -> None:
        # Stop trading if balance is zero or negative
        if self.account.balance <= 0:
            return

        risk_base_result = compute_risk_base(
            mode=self._compounding_mode,
            current_balance=self.account.balance,
            peak_balance=self.account.peak_balance,
            initial_balance=self.config.account.initial_balance,
        )
        risk_base = risk_base_result.risk_base
        risk_amount = risk_base * self._risk_per_trade
        lots = calculate_lot_size(
            wallet_balance=risk_base,
            risk_per_trade=self._risk_per_trade,
            stop_loss_pips=self._sl_pips,
            pip_value_per_lot=self._pip_value_per_lot,
            min_lot=0.01,
            lot_step=0.01,
            max_lot=None,
        ).lots

        # Leverage & Margin checks on position entry
        if self._leverage_enabled:
            if self._max_lot_exposure is not None and lots > self._max_lot_exposure:
                self.margin_rejections += 1
                return
            required_margin = (lots * self.config.instrument.contract_size) / self._leverage
            if required_margin > self.account.balance:
                self.margin_rejections += 1
                return

        self._position_counter += 1
        self._current_position_trades = []
        position_id = f"pos_{self._position_counter}"
        base_outcome = self._determine_outcome()

        # Base entry price in absolute terms (EUR/USD simulation)
        base_entry_price = 1.0800
        entry_price = base_entry_price
        current_price = base_entry_price
        remaining_lots = lots
        total_lots = lots
        total_cost_basis = base_entry_price * lots
        sl_price = base_entry_price - (self._sl_pips * 0.0001)
        position_balance_before = self.account.balance

        # Execution friction parameters for this trade
        effective_fee = self._fee_pips
        if self._execution_enabled and self._variable_spread_enabled:
            spread_jitter = abs(self.rng.gauss(0.0, self._spread_std_pips))
            effective_fee += self._base_spread_pips + spread_jitter

        swap_pips = 0.0
        if self._execution_enabled and self._swap_enabled and self._holding_nights > 0:
            for night in range(self._holding_nights):
                dow = ((self._day - 1 + night) % 7) + 1
                multiplier = 3 if dow == self._triple_swap_day else 1
                swap_pips += self._swap_long_pips * multiplier

        sl_slip = 0.0
        if self._execution_enabled and self._slippage_enabled:
            sl_slip = max(0.0, self.rng.gauss(self._sl_slippage_mean_pips, self._sl_slippage_std_pips))

        tp_slip = 0.0
        if self._execution_enabled and self._slippage_enabled:
            if self.rng.random() < self._tp_slippage_improvement_prob:
                tp_slip = max(0.0, self.rng.gauss(self._tp_slippage_mean_pips, self._tp_slippage_mean_pips * 0.5))

        tp_levels = list(self._tp_levels) if self._tp_levels else []
        tp_hits = {i: False for i in range(len(tp_levels))}

        # Pre-determine TP hit pattern for PERCENT mode
        self._tp_levels_to_hit = set()
        if self._tp_hit_enabled and self._tp_hit_mode == TPHitMode.PERCENT and tp_levels:
            n_hits = round(len(tp_levels) * self._tp_hit_pct / 100.0)
            n_hits = max(0, min(len(tp_levels), n_hits))
            # Select first N levels in order for deterministic behavior
            self._tp_levels_to_hit = set(range(n_hits))

        trailing_active = False
        trailing_stop_price = None
        trailing_triggered = False
        trailing_partial_done = False
        be_moved = False
        pyramid_adds = 0
        pyramid_last_add_price = base_entry_price
        margin_call_triggered = False

        max_steps = 10000
        step = 0
        position_closed = False
        close_price = base_entry_price

        # Deterministic price movement toward target based on outcome
        if base_outcome == "WIN":
            # When TP levels are configured, target is the highest HIT TP level
            # (for PERCENT mode, some levels may be missed)
            if tp_levels:
                if self._tp_hit_enabled and self._tp_hit_mode == TPHitMode.PERCENT and self._tp_levels_to_hit:
                    max_hit_pips = max(tp_levels[i].pips for i in self._tp_levels_to_hit)
                    target_price = base_entry_price + (max_hit_pips * 0.0001)
                else:
                    max_tp_pips = max(tl.pips for tl in tp_levels)
                    target_price = base_entry_price + (max_tp_pips * 0.0001)
            else:
                target_price = base_entry_price + (self._tp_pips * 0.0001)
            direction = 1  # upward toward TP
        else:
            target_price = base_entry_price - (self._sl_pips * 0.0001)
            direction = -1  # downward toward SL

        # Track whether we need deferred P&L calculation (pyramid only)
        # When pyramid is enabled, it handles the position with deferred P&L.
        # Trailing stop without pyramiding records closing deals sequentially.
        self._deferred_pnl = self._pyramid_enabled

        # Record initial position open fill when M11 features are active
        if self._deferred_pnl:
            self._record_fill(
                position_id=position_id,
                fill_index=0,
                total_fills=0,
                lots=lots,
                pips_gross=0.0,
                pips_net=0.0,
                fee_pips=0.0,
                profit_euros=0.0,
                entry_price=0.0,  # pips offset from base_entry_price
                position_avg_price=0.0,
                outcome=base_outcome,
                close_price=base_entry_price,
            )

        while not position_closed and step < max_steps:
            step += 1

            if self._price_path_mode == PricePathMode.STOCHASTIC:
                # With pullback_prob, move against direction; otherwise with direction
                step_dir = -direction if self.rng.random() < self._pullback_prob else direction
                current_price += step_dir * 0.0001
            else:
                current_price += direction * 0.0001

            # Cap at target to prevent overshoot
            if direction == 1 and current_price > target_price:
                current_price = target_price
            elif direction == -1 and current_price < target_price:
                current_price = target_price

            unrealized_pips = (current_price - entry_price) / 0.0001

            # Check Leverage & Margin (Margin Call & Stop-Out Liquidation)
            if self._leverage_enabled:
                total_margin = (total_lots * self.config.instrument.contract_size) / self._leverage
                floating_pnl = unrealized_pips * total_lots * self._pip_value_per_lot
                equity = self.account.balance + floating_pnl
                margin_level_pct = (equity / total_margin) * 100.0 if total_margin > 0 else float("inf")

                if margin_level_pct <= self._margin_call_pct and not margin_call_triggered:
                    self.margin_calls += 1
                    margin_call_triggered = True

                if margin_level_pct <= self._stop_out_pct:
                    self.stop_outs += 1
                    close_price = current_price
                    so_slip = sl_slip if (self._execution_enabled and self._slippage_enabled) else 0.0
                    close_pips = ((close_price - base_entry_price) / 0.0001 - so_slip) if self._deferred_pnl else ((close_price - entry_price) / 0.0001 - so_slip)

                    if self._deferred_pnl:
                        position_trades = list(self._current_position_trades)
                        position_trades.sort(key=lambda t: t.fill_index)
                        balance_before_position = self.account.balance
                        for idx, t in enumerate(position_trades):
                            t.pips_gross = round(close_pips - t.entry_price, 4)
                            t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                            t.pips_net = round(
                                t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                            )
                            t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                            t.balance_before = round(balance_before_position, 4)
                            t.balance_after = round(balance_before_position + t.profit_euros, 4)
                            t.outcome = "LOSS"
                            balance_before_position = t.balance_after
                    else:
                        existing_fills = len(self._current_position_trades)
                        fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                        close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)
                        profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                        self._record_fill(
                            position_id=position_id,
                            fill_index=existing_fills,
                            total_fills=0,
                            lots=total_lots,
                            pips_gross=close_pips,
                            pips_net=close_pips_net,
                            fee_pips=fee_applied,
                            profit_euros=profit,
                            entry_price=entry_price,
                            position_avg_price=total_cost_basis / total_lots if total_lots > 0 else entry_price,
                            outcome="LOSS",
                            close_price=close_price,
                            balance_before_override=position_balance_before,
                        )
                        position_balance_before = round(position_balance_before + profit, 4)
                    position_closed = True
                    break

            # Check Take Profit levels
            # Each TP level closes its allocated lots (hit=profit, miss=0 profit)
            # All levels are processed regardless of current price
            # Check Take Profit levels (only when pyramid is NOT enabled, since pyramid takes priority)
            if not position_closed and base_outcome == "WIN" and tp_levels and not self._pyramid_enabled:
                # Track running balance within the position for sequential balance chaining
                position_balance_before = self.account.balance

                for i, tp_level in enumerate(tp_levels):
                    if tp_hits[i]:
                        continue  # Already processed

                    tp_target_price = base_entry_price + (tp_level.pips * 0.0001)
                    # Check if this TP level should be hit
                    hit_this_level = True
                    if self._tp_hit_enabled:
                        if self._tp_hit_mode == TPHitMode.RANDOM:
                            if self.rng.random() > (self._tp_hit_pct / 100.0):
                                hit_this_level = False
                        elif self._tp_hit_mode == TPHitMode.PERCENT:
                            hit_this_level = i in self._tp_levels_to_hit

                    # Calculate fill lots: exact allocation
                    fill_lots = round(lots * (tp_level.allocation_pct / 100.0), 2)

                    fill_fee_pips = round(effective_fee, 4) if i == 0 else 0.0
                    if hit_this_level:
                        tp_hits[i] = True
                        fill_pips = round(tp_level.pips + tp_slip - fill_fee_pips + (swap_pips if i == 0 else 0.0), 4)
                        profit = round(fill_pips * fill_lots * self._pip_value_per_lot, 4)
                    else:
                        tp_hits[i] = True
                        fill_fee_pips = 0.0
                        fill_pips = 0.0
                        profit = 0.0

                    existing_fills = len(self._current_position_trades)
                    self._record_fill(
                        position_id=position_id,
                        fill_index=existing_fills,
                        total_fills=0,
                        lots=fill_lots,
                        pips_gross=tp_level.pips if hit_this_level else 0.0,
                        pips_net=fill_pips,
                        fee_pips=fill_fee_pips if hit_this_level else 0.0,
                        profit_euros=profit,
                        entry_price=entry_price,
                        position_avg_price=total_cost_basis / total_lots if total_lots > 0 else entry_price,
                        outcome="WIN",
                        tp_level_hit=hit_this_level,
                        tp_level_pips=tp_level.pips,
                        close_price=tp_target_price,
                        balance_before_override=position_balance_before,
                    )

                    # Update running balance for next fill in this position
                    position_balance_before = round(position_balance_before + profit, 4)

                    remaining_lots -= fill_lots
                    total_cost_basis -= fill_lots * (total_cost_basis / total_lots)
                    total_lots -= fill_lots

                    if i == len(tp_levels) - 1 or remaining_lots <= 0.001:
                        close_price = tp_target_price
                        position_closed = True
                        break

            if position_closed:
                break

            # If reached target price, close position (for non-TP trades or remaining lots)
            if direction == 1 and current_price >= target_price:
                close_price = current_price
                close_pips = (close_price - base_entry_price) / 0.0001 + tp_slip

                if self._deferred_pnl:
                    # Update all fills' pips_gross and balance chain based on close price
                    position_trades = list(self._current_position_trades)
                    # Sort by fill_index to process in order
                    position_trades.sort(key=lambda t: t.fill_index)
                    balance_before_position = self.account.balance
                    # Now apply profits in order
                    for idx, t in enumerate(position_trades):
                        t.pips_gross = round(close_pips - t.entry_price, 4)
                        t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                        t.pips_net = round(
                            t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                        )
                        t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                        t.balance_before = round(balance_before_position, 4)
                        t.balance_after = round(balance_before_position + t.profit_euros, 4)
                        balance_before_position = t.balance_after
                    # Balance is updated at end of _execute_trade (line 669)
                    # to avoid double counting
                else:
                    existing_fills = len(self._current_position_trades)
                    fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                    close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)
                    profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                    self._record_fill(
                        position_id=position_id,
                        fill_index=existing_fills,
                        total_fills=0,
                        lots=total_lots,
                        pips_gross=close_pips,
                        pips_net=close_pips_net,
                        fee_pips=fee_applied,
                        profit_euros=profit,
                        entry_price=base_entry_price,
                        position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                        outcome=base_outcome,
                        close_price=close_price,
                        trailing_stop_active=trailing_active,
                        trailing_stop_pips=self._trailing_distance_pips if trailing_active else 0.0,
                        trailing_triggered=trailing_triggered,
                        balance_before_override=position_balance_before,
                    )
                    position_balance_before = round(position_balance_before + profit, 4)
                position_closed = True
                break

            unrealized_pips = (current_price - entry_price) / 0.0001

            # Check Stop Loss hit (for LOSS trades moving down or stochastic pullbacks hitting SL)
            if current_price <= sl_price and not be_moved:
                close_pips = (sl_price - base_entry_price) / 0.0001 - sl_slip
                if self._deferred_pnl and not trailing_partial_done:
                    position_trades = list(self._current_position_trades)
                    position_trades.sort(key=lambda t: t.fill_index)
                    balance_before_position = self.account.balance
                    for idx, t in enumerate(position_trades):
                        t.pips_gross = round(close_pips - t.entry_price, 4)
                        t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                        t.pips_net = round(
                            t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                        )
                        t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                        t.balance_before = round(balance_before_position, 4)
                        t.balance_after = round(balance_before_position + t.profit_euros, 4)
                        t.outcome = "LOSS"
                        balance_before_position = t.balance_after
                else:
                    existing_fills = len(self._current_position_trades)
                    fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                    close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)
                    profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                    self._record_fill(
                        position_id=position_id,
                        fill_index=existing_fills,
                        total_fills=0,
                        lots=total_lots,
                        pips_gross=close_pips,
                        pips_net=close_pips_net,
                        fee_pips=fee_applied,
                        profit_euros=profit,
                        entry_price=base_entry_price,
                        position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                        outcome="LOSS",
                        close_price=sl_price,
                        trailing_stop_active=trailing_active,
                        trailing_stop_pips=self._trailing_distance_pips if trailing_active else 0.0,
                        trailing_triggered=trailing_triggered,
                        balance_before_override=position_balance_before,
                    )
                    position_balance_before = round(position_balance_before + profit, 4)
                close_price = sl_price
                position_closed = True
                break

            # Check breakeven SL after BE move
            if be_moved and trailing_stop_price is not None:
                if current_price <= trailing_stop_price:
                    close_pips = (trailing_stop_price - base_entry_price) / 0.0001 - sl_slip
                    if self._deferred_pnl and not trailing_partial_done:
                        position_trades = list(self._current_position_trades)
                        position_trades.sort(key=lambda t: t.fill_index)
                        balance_before_position = self.account.balance
                        for idx, t in enumerate(position_trades):
                            t.pips_gross = round(close_pips - t.entry_price, 4)
                            t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                            t.pips_net = round(
                                t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                            )
                            t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                            t.balance_before = round(balance_before_position, 4)
                            t.balance_after = round(balance_before_position + t.profit_euros, 4)
                            t.outcome = "WIN" if t.profit_euros >= 0 else "LOSS"
                            balance_before_position = t.balance_after
                    else:
                        existing_fills = len(self._current_position_trades)
                        fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                        close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)
                        profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                        self._record_fill(
                            position_id=position_id,
                            fill_index=existing_fills,
                            total_fills=0,
                            lots=total_lots,
                            pips_gross=close_pips,
                            pips_net=close_pips_net,
                            fee_pips=fee_applied,
                            profit_euros=profit,
                            entry_price=base_entry_price,
                            position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                            outcome="WIN" if profit >= 0 else "LOSS",
                            close_price=trailing_stop_price,
                            trailing_stop_active=trailing_active,
                            trailing_stop_pips=self._trailing_distance_pips if trailing_active else 0.0,
                            trailing_triggered=trailing_triggered,
                            balance_before_override=position_balance_before,
                        )
                        position_balance_before = round(position_balance_before + profit, 4)
                    close_price = trailing_stop_price
                    position_closed = True
                    break

            if position_closed:
                break

            if not position_closed and base_outcome == "WIN" and self._trailing_enabled:
                if not trailing_active and unrealized_pips >= self._trailing_activate_pips:
                    trailing_active = True
                    trailing_stop_price = current_price - (self._trailing_distance_pips * 0.0001)
                    # Update all existing fills in this position to mark trailing active
                    for t in self._current_position_trades:
                        t.trailing_stop_active = True
                        t.trailing_stop_pips = self._trailing_distance_pips

                if trailing_active:
                    new_trailing = current_price - (self._trailing_distance_pips * 0.0001)
                    if new_trailing > trailing_stop_price:
                        trailing_stop_price = new_trailing

                    if current_price <= trailing_stop_price:
                        close_pips = (trailing_stop_price - base_entry_price) / 0.0001
                        existing_fills = len(self._current_position_trades)
                        fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                        close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)

                        # Mark all trades in this position as trailing_triggered
                        for t in self._current_position_trades:
                            t.trailing_triggered = True

                        if self._trailing_partial_close_pct > 0 and not trailing_partial_done:
                            close_lots = round(total_lots * (self._trailing_partial_close_pct / 100.0), 2)
                            if close_lots < 0.01:
                                close_lots = 0.01
                            if close_lots > total_lots:
                                close_lots = total_lots
                            profit = round(close_pips_net * close_lots * self._pip_value_per_lot, 4)
                            existing_fills = len(self._current_position_trades)
                            self._record_fill(
                                position_id=position_id,
                                fill_index=existing_fills,
                                total_fills=0,
                                lots=close_lots,
                                pips_gross=close_pips,
                                pips_net=close_pips_net,
                                fee_pips=fee_applied,
                                profit_euros=profit,
                                entry_price=base_entry_price,
                                position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                                outcome="WIN" if profit >= 0 else "LOSS",
                                close_price=trailing_stop_price,
                                trailing_stop_active=True,
                                trailing_stop_pips=self._trailing_distance_pips,
                                trailing_triggered=True,
                                balance_before_override=position_balance_before,
                            )
                            position_balance_before = round(position_balance_before + profit, 4)
                            remaining_lots -= close_lots
                            total_lots -= close_lots
                            trailing_partial_done = True
                            # For the remaining lots, adjust stop price so it doesn't immediately re-trigger
                            trailing_stop_price = current_price - (self._trailing_distance_pips * 0.0001)
                            if remaining_lots <= 0.001:
                                close_price = trailing_stop_price
                                position_closed = True
                                break
                        else:
                            if self._deferred_pnl and not trailing_partial_done:
                                position_trades = list(self._current_position_trades)
                                position_trades.sort(key=lambda t: t.fill_index)
                                balance_before_position = self.account.balance
                                for idx, t in enumerate(position_trades):
                                    t.pips_gross = round(close_pips - t.entry_price, 4)
                                    t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                                    t.pips_net = round(
                                        t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                                    )
                                    t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                                    t.balance_before = round(balance_before_position, 4)
                                    t.balance_after = round(balance_before_position + t.profit_euros, 4)
                                    t.trailing_stop_active = True
                                    t.trailing_stop_pips = self._trailing_distance_pips
                                    t.trailing_triggered = True
                                    t.outcome = "WIN" if t.profit_euros >= 0 else "LOSS"
                                    balance_before_position = t.balance_after
                            else:
                                profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                                existing_fills = len(self._current_position_trades)
                                self._record_fill(
                                    position_id=position_id,
                                    fill_index=existing_fills,
                                    total_fills=0,
                                    lots=total_lots,
                                    pips_gross=close_pips,
                                    pips_net=close_pips_net,
                                    fee_pips=fee_applied,
                                    profit_euros=profit,
                                    entry_price=base_entry_price,
                                    position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                                    outcome="WIN" if profit >= 0 else "LOSS",
                                    close_price=trailing_stop_price,
                                    trailing_stop_active=True,
                                    trailing_stop_pips=self._trailing_distance_pips,
                                    trailing_triggered=True,
                                    balance_before_override=position_balance_before,
                                )
                                position_balance_before = round(position_balance_before + profit, 4)
                            close_price = trailing_stop_price
                            position_closed = True
                            break

            if not position_closed and base_outcome == "WIN" and self._pyramid_enabled:
                if (
                    pyramid_adds < self._pyramid_max_adds
                    and unrealized_pips >= self._pyramid_trigger_pips - 1e-9
                ):
                    if self._pyramid_lot_mode == PyramidMode.PERCENT:
                        add_lots = lots * (self._pyramid_lot_pct / 100.0)
                    else:
                        add_lots = self._pyramid_lot_size

                    # Enforce broker minimum lot (0.01) and step alignment (0.01) with floor truncation
                    add_lots = math.floor(add_lots / 0.01 + 1e-9) * 0.01
                    add_lots = round(add_lots, 2)
                    if add_lots < 0.01:
                        add_lots = 0.01

                    # Leverage & Margin check for pyramid add
                    if self._leverage_enabled:
                        if self._max_lot_exposure is not None and (total_lots + add_lots) > self._max_lot_exposure:
                            continue
                        add_margin = (add_lots * self.config.instrument.contract_size) / self._leverage
                        current_margin = (total_lots * self.config.instrument.contract_size) / self._leverage
                        floating_pnl = unrealized_pips * total_lots * self._pip_value_per_lot
                        equity = self.account.balance + floating_pnl
                        free_margin = equity - current_margin
                        if free_margin < add_margin:
                            continue

                    # Save old entry price (VWAP average before this add)
                    old_entry_price = entry_price

                    # Compute trigger price based on old VWAP average
                    trigger_price = old_entry_price + self._pyramid_trigger_pips * 0.0001
                    add_entry_pips = (trigger_price - base_entry_price) / 0.0001

                    new_avg = (
                        (total_cost_basis + add_lots * trigger_price)
                        / (total_lots + add_lots)
                    )
                    total_cost_basis += add_lots * trigger_price
                    total_lots += add_lots
                    remaining_lots += add_lots
                    entry_price = new_avg
                    pyramid_last_add_price = current_price
                    pyramid_adds += 1

                    existing_fills = len(self._current_position_trades)
                    # entry_price as pips offset from base_entry_price
                    avg_offset = (new_avg - base_entry_price) / 0.0001
                    self._record_fill(
                        position_id=position_id,
                        fill_index=existing_fills,
                        total_fills=0,
                        lots=add_lots,
                        pips_gross=0.0,
                        pips_net=0.0,
                        fee_pips=0.0,
                        profit_euros=0.0,
                        entry_price=add_entry_pips,
                        position_avg_price=avg_offset,
                        outcome="WIN",
                        close_price=current_price,
                    )

                    if (
                        self._pyramid_be_trigger_pips is not None
                        and not be_moved
                        and unrealized_pips >= self._pyramid_be_trigger_pips
                    ):
                        sl_price = entry_price + (self._pyramid_be_offset_pips * 0.0001)
                        be_moved = True
                        trailing_stop_price = sl_price

            if not position_closed and not be_moved and self._pyramid_be_trigger_pips is not None:
                if unrealized_pips >= self._pyramid_be_trigger_pips:
                    sl_price = entry_price + (self._pyramid_be_offset_pips * 0.0001)
                    be_moved = True
                    trailing_stop_price = sl_price

        if not position_closed:
            close_pips = (current_price - base_entry_price) / 0.0001
            if self._deferred_pnl and not trailing_partial_done:
                position_trades = list(self._current_position_trades)
                position_trades.sort(key=lambda t: t.fill_index)
                balance_before_position = self.account.balance
                for idx, t in enumerate(position_trades):
                    t.pips_gross = round(close_pips - t.entry_price, 4)
                    t.fee_pips = round(effective_fee, 4) if idx == 0 else 0.0
                    t.pips_net = round(
                        t.pips_gross - t.fee_pips + (swap_pips if idx == 0 else 0.0), 4
                    )
                    t.profit_euros = round(t.pips_net * t.lots * self._pip_value_per_lot, 4)
                    t.balance_before = round(balance_before_position, 4)
                    t.balance_after = round(balance_before_position + t.profit_euros, 4)
                    t.outcome = "WIN" if t.profit_euros >= 0 else "LOSS"
                    balance_before_position = t.balance_after
            else:
                existing_fills = len(self._current_position_trades)
                fee_applied = round(effective_fee, 4) if existing_fills == 0 else 0.0
                close_pips_net = round(close_pips - fee_applied + (swap_pips if existing_fills == 0 else 0.0), 4)
                profit = round(close_pips_net * total_lots * self._pip_value_per_lot, 4)
                self._record_fill(
                    position_id=position_id,
                    fill_index=existing_fills,
                    total_fills=0,
                    lots=total_lots,
                    pips_gross=close_pips,
                    pips_net=close_pips_net,
                    fee_pips=fee_applied,
                    profit_euros=profit,
                    entry_price=base_entry_price,
                    position_avg_price=total_cost_basis / total_lots if total_lots > 0 else base_entry_price,
                    outcome=base_outcome,
                    close_price=current_price,
                    trailing_stop_active=trailing_active,
                    trailing_stop_pips=self._trailing_distance_pips if trailing_active else 0.0,
                    trailing_triggered=trailing_triggered,
                    balance_before_override=position_balance_before,
                )
                position_balance_before = round(position_balance_before + profit, 4)
            close_price = current_price

        # Always update account balance with position profit (win or loss)
        position_profit = sum(
            t.profit_euros
            for t in self._current_position_trades
        )
        self.account.balance += position_profit
        if self.account.balance > self.account.peak_balance:
            self.account.peak_balance = self.account.balance

    def _build_result(self) -> SimulationResult:
        """Build the final simulation result."""
        winning_trades = [t for t in self.trades if t.outcome == "WIN"]
        losing_trades = [t for t in self.trades if t.outcome == "LOSS"]

        gross_profit = sum(t.profit_euros for t in winning_trades)
        gross_loss = abs(sum(t.profit_euros for t in losing_trades))

        result = SimulationResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.config.account.initial_balance,
            final_balance=self.account.balance,
            total_return_pct=(
                (self.account.balance - self.config.account.initial_balance)
                / self.config.account.initial_balance
                * 100
            ),
            max_drawdown_pct=max((e.drawdown_pct for e in self.equity_curve), default=0.0),
            win_rate_actual=len(winning_trades) / len(self.trades) if self.trades else 0.0,
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
            margin_calls=self.margin_calls,
            stop_outs=self.stop_outs,
            margin_rejections=self.margin_rejections,
        )
        return result