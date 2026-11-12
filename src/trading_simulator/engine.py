"""Core simulation engine — optimized for performance."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import cycle
from typing import Literal

from trading_simulator.config import CompoundingMode, OutcomeMode, PyramidMode, TPHitMode, SimulationConfig
from trading_simulator.compounding import compute_risk_base
from trading_simulator.lot_sizing import calculate_lot_size
from trading_simulator.models import Account, Instrument, Trade, TradeResult, EquityPoint


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


class SimulationEngine:
    """Runs a trading simulation with configurable parameters."""

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
        )
        # Cache frequently accessed values to avoid repeated attribute lookups
        self._pip_value_per_lot = self.instrument.pip_value_per_lot
        self._fee_pips = config.risk.fee_pips
        self._sl_pips = config.risk.sl_pips
        self._tp_pips = config.risk.tp_pips
        self._risk_per_trade = config.account.risk_per_trade
        self._compounding_mode = config.risk.compounding_mode
        self._outcome_mode = config.activity.outcome_mode
        self._win_rate = config.activity.win_rate
        self._trades_per_day = config.activity.trades_per_day
        self._tp_levels = config.risk.tp_levels

        # Pyramid entry configuration
        self._pyramid_enabled = config.pyramid.enabled
        self._pyramid_trigger_pips = config.pyramid.trigger_pips
        self._pyramid_max_adds = config.pyramid.max_adds
        self._pyramid_lot_mode = config.pyramid.lot_mode
        self._pyramid_lot_size = config.pyramid.lot_size
        self._pyramid_lot_pct = config.pyramid.lot_pct
        self._pyramid_be_trigger_pips = config.pyramid.be_trigger_pips

        # Trailing stop configuration
        self._trailing_enabled = config.trailing_stop.enabled
        self._trailing_activate_pips = config.trailing_stop.activate_pips
        self._trailing_distance_pips = config.trailing_stop.distance_pips
        self._trailing_partial_close_pct = config.trailing_stop.partial_close_pct

        # TP hit tracking configuration
        self._tp_hit_enabled = config.tp_hit.enabled
        self._tp_hit_mode = config.tp_hit.mode
        self._tp_hit_pct = config.tp_hit.hit_pct

        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self._day = 1
        self._pattern_cycle: cycle[str] | None = None
        self._pattern_index = 0
        self._position_counter = 0
        self._init_pattern()

    def _init_pattern(self) -> None:
        """Initialize the deterministic outcome pattern based on config."""
        mode = self.config.activity.outcome_mode
        total_trades = self.config.activity.trades_per_day * self.config.activity.days
        win_rate = self.config.activity.win_rate

        if mode == OutcomeMode.BERNOULLI:
            self._pattern_cycle = None
            self._pattern_index = 0
        elif mode == OutcomeMode.ALTERNATING:
            pattern = ["W", "L"] * (total_trades // 2 + 1)
            self._pattern_cycle = cycle(pattern[:total_trades])
            self._pattern_index = 0
        elif mode == OutcomeMode.FIXED_WINS_FIRST:
            n_wins = int(round(total_trades * win_rate))
            pattern = ["W"] * n_wins + ["L"] * (total_trades - n_wins)
            self._pattern_cycle = cycle(pattern)
            self._pattern_index = 0
        elif mode == OutcomeMode.FIXED_LOSSES_FIRST:
            n_losses = int(round(total_trades * (1 - win_rate)))
            pattern = ["L"] * n_losses + ["W"] * (total_trades - n_losses)
            self._pattern_cycle = cycle(pattern)
            self._pattern_index = 0
        elif mode == OutcomeMode.CUSTOM_PATTERN:
            if not self.config.activity.custom_pattern:
                raise ValueError(
                    "custom_pattern must be specified when outcome_mode is custom_pattern"
                )
            pattern = list(self.config.activity.custom_pattern.upper())
            self._pattern_cycle = cycle(pattern)
            self._pattern_index = 0
        else:
            raise ValueError(f"Unknown outcome mode: {mode}")

    def _determine_outcome(self) -> bool:
        """Determine if the next trade is a win based on the outcome mode.

        Returns True for WIN, False for LOSS.
        """
        if self._outcome_mode == OutcomeMode.BERNOULLI:
            return self.rng.random() < self._win_rate

        if self._pattern_cycle is not None:
            result = next(self._pattern_cycle)
            return result == "W"

        return self.rng.random() < self._win_rate

    def _execute_trade(self) -> None:
        """Execute a single simulated trade (position)."""
        compounding_result = compute_risk_base(
            mode=self._compounding_mode,
            current_balance=self.account.balance,
            peak_balance=self.account.peak_balance,
            initial_balance=self.account.initial_balance,
        )
        self.account.peak_balance = compounding_result.peak_balance

        lot_result = calculate_lot_size(
            wallet_balance=compounding_result.risk_base,
            risk_per_trade=self._risk_per_trade,
            stop_loss_pips=self._sl_pips,
            pip_value_per_lot=self._pip_value_per_lot,
        )

        win = self._determine_outcome()

        self._position_counter += 1
        position_id = f"pos_{self._position_counter}"

        if self._pyramid_enabled and win:
            self._execute_pyramid_win(lot_result, position_id)
        elif win and self._tp_levels:
            self._execute_partial_tp_win(lot_result, position_id)
        elif self._trailing_enabled and win:
            self._execute_trailing_stop_win(lot_result, position_id)
        else:
            if win:
                pips_gross = self._tp_pips
            else:
                pips_gross = -self._sl_pips

            pips_net = pips_gross - self._fee_pips
            profit_euros = pips_net * lot_result.lots * self._pip_value_per_lot

            balance_before = self.account.balance
            self.account.balance += profit_euros

            if self.account.balance > self.account.peak_balance:
                self.account.peak_balance = self.account.balance

            trade = Trade(
                trade_number=len(self.trades) + 1,
                day=self._day,
                lots=lot_result.lots,
                risk_amount=lot_result.risk_amount,
                pips_gross=pips_gross,
                pips_net=pips_net,
                fee_pips=self._fee_pips,
                profit_euros=profit_euros,
                balance_before=balance_before,
                balance_after=self.account.balance,
                outcome="WIN" if win else "LOSS",
                position_id=position_id,
                fill_index=0,
                total_fills=1,
                pyramid_fill_index=0,
                pyramid_total_fills=1,
                entry_price=0.0,
                position_avg_price=0.0,
            )
            self.trades.append(trade)

    def _execute_pyramid_win(
        self, lot_result, position_id: str
    ) -> None:
        """Execute a winning trade with pyramid entry (scaling into position).

        Uses VWAP averaging (cTrader/IC Markets approach) when adding volume.
        Each fill closes at the same Take Profit level.

        Pyramid triggers when unrealized profit from current VWAP average
        reaches trigger_pips. Each add recalculates the average entry price.
        """
        initial_lots = lot_result.lots
        risk_amount = lot_result.risk_amount
        fee_pips = self._fee_pips
        pip_value = self._pip_value_per_lot
        tp_pips = self._tp_pips

        # Pyramid parameters
        trigger_pips = self._pyramid_trigger_pips
        max_adds = self._pyramid_max_adds
        lot_mode = self._pyramid_lot_mode
        add_lots_size = self._pyramid_lot_size
        add_lots_pct = self._pyramid_lot_pct
        be_trigger_pips = self._pyramid_be_trigger_pips

        balance_before = self.account.balance
        running_profit = 0.0

        # Track position state using pip offsets from entry (entry = 0 pips)
        current_avg_offset = 0.0  # VWAP average entry offset in pips from entry
        current_lots = initial_lots
        adds_done = 0

        # --- Initial fill (position open) ---
        pips_gross = tp_pips
        pips_net = pips_gross - fee_pips
        profit_euros = pips_net * initial_lots * pip_value

        self.account.balance += profit_euros
        running_profit += profit_euros

        if self.account.balance > self.account.peak_balance:
            self.account.peak_balance = self.account.balance

        fill_balance_before = balance_before

        trade = Trade(
            trade_number=len(self.trades) + 1,
            day=self._day,
            lots=initial_lots,
            risk_amount=risk_amount,
            pips_gross=pips_gross,
            pips_net=pips_net,
            fee_pips=fee_pips,
            profit_euros=profit_euros,
            balance_before=fill_balance_before,
            balance_after=self.account.balance,
            outcome="WIN",
            position_id=position_id,
            fill_index=0,
            total_fills=1,  # Will be updated after all adds
            pyramid_fill_index=0,
            pyramid_total_fills=1,  # Will be updated
            entry_price=0.0,  # Entry at 0 pips offset
            position_avg_price=0.0,
        )
        self.trades.append(trade)

        # --- Pyramid adds: scale into position as price moves toward TP ---
        # Each add requires trigger_pips of unrealized profit from current VWAP average
        # Price moves from entry (0 pips) toward TP (tp_pips)
        for add_idx in range(1, max_adds + 1):
            # Next trigger: price must be trigger_pips above current VWAP average
            next_trigger_offset = current_avg_offset + trigger_pips

            # Cannot add beyond TP (no room left)
            if next_trigger_offset >= tp_pips:
                break

            add_price_offset = next_trigger_offset

            # Calculate add lot size
            if lot_mode == PyramidMode.FIXED:
                add_lots = add_lots_size
            else:  # PyramidMode.PERCENT
                add_lots = initial_lots * (add_lots_pct / 100.0)

            # Round to valid lot size (0.01 step, min 0.01)
            add_lots = round(add_lots / 0.01) * 0.01
            if add_lots < 0.01:
                add_lots = 0.01

            # VWAP average entry price (cTrader/IC Markets approach)
            # new_avg = (old_avg × old_lots + add_price × add_lots) / (old_lots + add_lots)
            new_avg_offset = (
                (current_avg_offset * current_lots + add_price_offset * add_lots)
                / (current_lots + add_lots)
            )

            # Profit for this add fill: closes at TP from add entry price
            pips_gross_add = tp_pips - add_price_offset
            pips_net_add = pips_gross_add - fee_pips
            profit_euros_add = pips_net_add * add_lots * pip_value

            self.account.balance += profit_euros_add
            running_profit += profit_euros_add

            if self.account.balance > self.account.peak_balance:
                self.account.peak_balance = self.account.balance

            # Update position state
            current_avg_offset = new_avg_offset
            current_lots += add_lots
            adds_done += 1

            fill_balance_before = balance_before + (running_profit - profit_euros_add)

            # Breakeven stop loss tracking (recorded but doesn't affect WIN outcome)
            be_triggered = be_trigger_pips is not None and add_price_offset >= be_trigger_pips

            trade = Trade(
                trade_number=len(self.trades) + 1,
                day=self._day,
                lots=add_lots,
                risk_amount=risk_amount,
                pips_gross=pips_gross_add,
                pips_net=pips_net_add,
                fee_pips=fee_pips,
                profit_euros=profit_euros_add,
                balance_before=fill_balance_before,
                balance_after=self.account.balance,
                outcome="WIN",
                position_id=position_id,
                fill_index=add_idx,
                total_fills=1,  # Will be updated
                pyramid_fill_index=add_idx,
                pyramid_total_fills=1 + adds_done,
                entry_price=add_price_offset,
                position_avg_price=new_avg_offset,
            )
            self.trades.append(trade)

        # Update total_fills on all trades for this position
        total_fills = 1 + adds_done
        for trade in self.trades:
            if trade.position_id == position_id:
                trade.pyramid_total_fills = total_fills
                trade.total_fills = total_fills

    def _execute_trailing_stop_win(
        self, lot_result, position_id: str
    ) -> None:
        """Execute a winning trade with trailing stop protection.

        Trailing stop activates when unrealized profit reaches activate_pips.
        Trailing stop trails at distance_pips behind current price.
        If price retraces and hits trailing stop, trade closes early at that level.
        Otherwise, trade reaches Take Profit.

        Uses probability model: P(hit trailing) = distance_pips / (remaining_to_tp + distance_pips)
        where remaining_to_tp = tp_pips - activate_pips.
        """
        initial_lots = lot_result.lots
        risk_amount = lot_result.risk_amount
        fee_pips = self._fee_pips
        pip_value = self._pip_value_per_lot
        tp_pips = self._tp_pips

        activate_pips = self._trailing_activate_pips
        distance_pips = self._trailing_distance_pips
        partial_close_pct = self._trailing_partial_close_pct

        balance_before = self.account.balance

        # --- Initial fill (position open) ---
        # Trailing stop activates when price reaches activate_pips
        # Trailing stop level = activate_pips - distance_pips
        trailing_stop_level = activate_pips - distance_pips

        # Probability of price retracing to hit trailing stop vs reaching TP
        remaining_to_tp = tp_pips - activate_pips
        if remaining_to_tp > 0:
            prob_hit_trailing = distance_pips / (remaining_to_tp + distance_pips)
        else:
            prob_hit_trailing = 0.0

        hit_trailing = self.rng.random() < prob_hit_trailing

        if hit_trailing:
            # Price retraced and hit trailing stop before TP
            pips_gross = trailing_stop_level
            pips_net = pips_gross - fee_pips
            profit_euros = pips_net * initial_lots * pip_value

            self.account.balance += profit_euros

            if self.account.balance > self.account.peak_balance:
                self.account.peak_balance = self.account.balance

            trade = Trade(
                trade_number=len(self.trades) + 1,
                day=self._day,
                lots=initial_lots,
                risk_amount=risk_amount,
                pips_gross=pips_gross,
                pips_net=pips_net,
                fee_pips=fee_pips,
                profit_euros=profit_euros,
                balance_before=balance_before,
                balance_after=self.account.balance,
                outcome="WIN",
                position_id=position_id,
                fill_index=0,
                total_fills=1,
                pyramid_fill_index=0,
                pyramid_total_fills=1,
                entry_price=0.0,
                position_avg_price=0.0,
                trailing_stop_active=True,
                trailing_stop_pips=trailing_stop_level,
                trailing_triggered=True,
            )
            self.trades.append(trade)
        else:
            # Price reached Take Profit (trailing stop was active but not hit)
            pips_gross = tp_pips
            pips_net = pips_gross - fee_pips
            profit_euros = pips_net * initial_lots * pip_value

            self.account.balance += profit_euros

            if self.account.balance > self.account.peak_balance:
                self.account.peak_balance = self.account.balance

            trade = Trade(
                trade_number=len(self.trades) + 1,
                day=self._day,
                lots=initial_lots,
                risk_amount=risk_amount,
                pips_gross=pips_gross,
                pips_net=pips_net,
                fee_pips=fee_pips,
                profit_euros=profit_euros,
                balance_before=balance_before,
                balance_after=self.account.balance,
                outcome="WIN",
                position_id=position_id,
                fill_index=0,
                total_fills=1,
                pyramid_fill_index=0,
                pyramid_total_fills=1,
                entry_price=0.0,
                position_avg_price=0.0,
                trailing_stop_active=True,
                trailing_stop_pips=trailing_stop_level,
                trailing_triggered=False,
            )
            self.trades.append(trade)

    def _execute_partial_tp_win(
        self, lot_result, position_id: str
    ) -> None:
        """Execute a winning trade with partial take-profit levels with TP hit simulation.

        Each TP level may or may not be hit based on TPHitConfig:
        - FIXED: All TP levels are hit (original behavior)
        - RANDOM: Each TP level independently hit with probability = hit_pct/100
        - PERCENT: Exactly hit_pct% of TP levels are hit (randomly selected)

        Tracks tp_level_hit and tp_level_pips on each Trade for analytics.
        """
        tp_levels = self._tp_levels
        total_lots = lot_result.lots
        fee_pips = self._fee_pips
        pip_value = self._pip_value_per_lot

        # Determine which TP levels are hit based on mode
        tp_hit_flags = self._determine_tp_hits(len(tp_levels))

        balance_before = self.account.balance
        running_profit = 0.0
        fills_executed = 0

        for idx, tp_level in enumerate(tp_levels):
            # Skip TP levels that were not hit
            if not tp_hit_flags[idx]:
                continue

            allocated_lots = total_lots * (tp_level.allocation_pct / 100.0)
            pips_gross = tp_level.pips

            fill_fee_pips = fee_pips if fills_executed == 0 else 0.0
            pips_net = pips_gross - fill_fee_pips
            profit_euros = pips_net * allocated_lots * pip_value

            self.account.balance += profit_euros
            running_profit += profit_euros

            if self.account.balance > self.account.peak_balance:
                self.account.peak_balance = self.account.balance

            fill_balance_before = balance_before + (running_profit - profit_euros)

            fills_executed += 1

            trade = Trade(
                trade_number=len(self.trades) + 1,
                day=self._day,
                lots=allocated_lots,
                risk_amount=lot_result.risk_amount,
                pips_gross=pips_gross,
                pips_net=pips_net,
                fee_pips=fill_fee_pips,
                profit_euros=profit_euros,
                balance_before=fill_balance_before,
                balance_after=self.account.balance,
                outcome="WIN",
                position_id=position_id,
                fill_index=fills_executed - 1,
                total_fills=len(tp_levels),
                tp_level_hit=True,
                tp_level_pips=pips_gross,
            )
            self.trades.append(trade)

        # Record missed TP levels as zero-profit fills for analytics
        for idx, tp_level in enumerate(tp_levels):
            if not tp_hit_flags[idx]:
                trade = Trade(
                    trade_number=len(self.trades) + 1,
                    day=self._day,
                    lots=total_lots * (tp_level.allocation_pct / 100.0),
                    risk_amount=lot_result.risk_amount,
                    pips_gross=0.0,
                    pips_net=0.0,
                    fee_pips=0.0,
                    profit_euros=0.0,
                    balance_before=self.account.balance,
                    balance_after=self.account.balance,
                    outcome="WIN",
                    position_id=position_id,
                    fill_index=idx,
                    total_fills=len(tp_levels),
                    tp_level_hit=False,
                    tp_level_pips=tp_level.pips,
                )
                self.trades.append(trade)

    def _determine_tp_hits(self, num_levels: int) -> list[bool]:
        """Determine which TP levels are hit based on TPHitConfig mode.

        Returns a list of booleans indicating hit/miss for each TP level.
        """
        if not self._tp_hit_enabled:
            return [True] * num_levels

        mode = self._tp_hit_mode
        hit_pct = self._tp_hit_pct
        hit_prob = hit_pct / 100.0

        if mode == TPHitMode.FIXED:
            return [True] * num_levels

        elif mode == TPHitMode.RANDOM:
            return [self.rng.random() < hit_prob for _ in range(num_levels)]

        elif mode == TPHitMode.PERCENT:
            # Exactly hit_pct% of levels hit (rounded), randomly selected
            num_hits = max(0, min(num_levels, round(num_levels * hit_prob)))
            hit_flags = [False] * num_levels
            hit_indices = self.rng.sample(range(num_levels), num_hits)
            for idx in hit_indices:
                hit_flags[idx] = True
            return hit_flags

        else:
            return [True] * num_levels

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
        )
        return result