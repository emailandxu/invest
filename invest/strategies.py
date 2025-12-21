"""
Example Trading Strategies for Backtesting

This module provides several example strategies that demonstrate
how to implement the Strategy interface. These can be used as
templates for creating custom strategies.

All strategies in this module are auto-registered with StrategyRegistry.
"""

from typing import Dict, List
import numpy as np

from .backtest import (
    Strategy, 
    BacktestContext, 
    Bar, 
    register_strategy,
    OrderSide,
    StrategyParameter
)


@register_strategy
class BuyAndHoldStrategy(Strategy):
    """
    Simple buy-and-hold strategy.
    
    Buys equal amounts of all assets at the start and holds until the end.
    This is often used as a benchmark for other strategies.
    """
    
    name = "Buy and Hold"
    description = "Buy equal amounts of all assets at start and hold"
    parameters = []  # No configurable parameters
    
    def __init__(self, allocation: Dict[str, float] = None):
        """
        Args:
            allocation: Optional dict mapping symbol to target weight (0-1).
                       If None, equal weight across all symbols.
        """
        self.allocation = allocation
    
    def on_init(self, context: BacktestContext) -> None:
        context.set_state("initialized_symbols", [])
    
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        initialized_symbols = context.get_state("initialized_symbols", [])
        # Calculate allocation
        if self.allocation:
            alloc = self.allocation
        else:
            # Equal weight
            n = len(context.symbols)
            alloc = {s: 1.0 / n for s in context.symbols}
        
        # Buy according to allocation
        updated_initialized = False
        for symbol, weight in alloc.items():
            if symbol in initialized_symbols:
                continue
            
            if symbol in bars:
                # Buy based on INITIAL capital to preserve strict buy-and-hold ratios
                # regardless of current portfolio value (drifting).
                target_value = context.initial_capital * weight
                price = bars[symbol].close
                
                if price > 0:
                    quantity = target_value / price
                    context.buy(symbol, quantity)
                    initialized_symbols.append(symbol)
                    updated_initialized = True
        
        if updated_initialized:
            context.set_state("initialized_symbols", initialized_symbols)


@register_strategy
class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signals when short-term MA crosses above long-term MA,
    and sell signals when it crosses below.
    """
    
    name = "MA Crossover"
    description = "Buy when short MA crosses above long MA, sell on cross below"
    parameters = [
        StrategyParameter("short_period", "Short MA Period", "int", 20, 5, 100, 5),
        StrategyParameter("long_period", "Long MA Period", "int", 50, 10, 300, 10),
    ]
    
    def __init__(self, short_period: int = 20, long_period: int = 50, allocation: Dict[str, float] = None):
        """
        Args:
            short_period: Period for short moving average (default: 20)
            long_period: Period for long moving average (default: 50)
            allocation: Optional dict mapping symbol to target weight
        """
        self.short_period = short_period
        self.long_period = long_period
        self.allocation = allocation
    
    def on_init(self, context: BacktestContext) -> None:
        # Track previous MA relationship for crossover detection
        context.set_state("prev_above", {s: None for s in context.symbols})
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate simple moving average."""
        if len(prices) < period:
            return float('nan')
        return np.mean(prices[-period:])
    
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        prev_above = context.get_state("prev_above")
        n_symbols = len([s for s in context.symbols if s in bars])
        
        # Calculate target allocation
        target_per_symbol = 1.0 / n_symbols if n_symbols > 0 else 0
        
        for symbol in context.symbols:
            if symbol not in bars:
                continue
            
            # Determine target weight for this symbol
            weight = target_per_symbol
            if self.allocation and symbol in self.allocation:
                weight = self.allocation[symbol]
            
            closes = context.get_closes(symbol)
            if len(closes) < self.long_period:
                continue
            
            short_ma = self._calculate_sma(closes, self.short_period)
            long_ma = self._calculate_sma(closes, self.long_period)
            
            if np.isnan(short_ma) or np.isnan(long_ma):
                continue
            
            currently_above = short_ma > long_ma
            was_above = prev_above.get(symbol)
            
            position = context.get_position(symbol)
            
            # Crossover detection
            if was_above is not None:
                if currently_above and not was_above:
                    # Golden cross - buy signal
                    if position.quantity == 0:
                        context.target_percent(symbol, weight)
                elif not currently_above and was_above:
                    # Death cross - sell signal
                    if position.quantity > 0:
                        context.sell_all(symbol)
            
            prev_above[symbol] = currently_above
        
        context.set_state("prev_above", prev_above)


@register_strategy
class PeriodicRebalanceStrategy(Strategy):
    """
    Periodic Rebalancing Strategy.
    
    Maintains target allocation weights by rebalancing at fixed intervals.
    """
    
    name = "Periodic Rebalance"
    description = "Rebalance to target weights at fixed intervals"
    parameters = [
        StrategyParameter("rebalance_days", "Rebalance Period (Days)", "int", 252, 5, 9999, 5),
    ]
    
    def __init__(self, 
                 allocation: Dict[str, float] = None,
                 rebalance_days: int = 252 * 3):
        """
        Args:
            allocation: Dict mapping symbol to target weight (0-1).
                       If None, equal weight.
            rebalance_days: Days between rebalancing (default: 30)
        """
        self.allocation = allocation
        self.rebalance_days = rebalance_days
    
    def on_init(self, context: BacktestContext) -> None:
        context.set_state("days_since_rebalance", self.rebalance_days)  # Trigger initial buy
    
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        days = context.get_state("days_since_rebalance", 0)
        days += 1
        
        if days >= self.rebalance_days:
            self._rebalance(context, bars)
            days = 0
        
        context.set_state("days_since_rebalance", days)
    
    def _rebalance(self, context: BacktestContext, bars: Dict[str, Bar]):
        """Execute rebalancing trades."""
        if self.allocation:
            alloc = self.allocation
        else:
            n = len(context.symbols)
            alloc = {s: 1.0 / n for s in context.symbols}
        
        for symbol in context.symbols:
            if symbol in bars:
                target_pct = alloc.get(symbol, 0)
                context.target_percent(symbol, target_pct)


@register_strategy
class MomentumStrategy(Strategy):
    """
    Momentum Strategy.
    
    Buys assets with positive momentum (price above N-day average)
    and sells those with negative momentum.
    """
    
    name = "Momentum"
    description = "Buy winners with positive momentum, sell losers"
    parameters = [
        StrategyParameter("lookback", "Lookback Period (Days)", "int", 60, 10, 252, 10),
        StrategyParameter("hold_top_n", "Hold Top N Assets", "int", 1, 1, 10, 1),
        StrategyParameter("rebalance_period", "Rebalance Period (Days)", "int", 30, 1, 365, 1),
    ]
    
    def __init__(self, lookback: int = 60, hold_top_n: int = 1, rebalance_period: int = 30):
        """
        Args:
            lookback: Days to look back for momentum calculation (default: 60)
            hold_top_n: Number of top momentum assets to hold (default: 1)
            rebalance_period: Days between rebalancing (default: 30)
        """
        self.lookback = lookback
        self.hold_top_n = hold_top_n
        self.rebalance_period = rebalance_period
    
    def on_init(self, context: BacktestContext) -> None:
        context.set_state("rebalance_counter", 0)

    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        # Check rebalance period
        counter = context.get_state("rebalance_counter", 0)
        counter += 1
        context.set_state("rebalance_counter", counter)
        
        # Only rebalance on specific days (and first day)
        if counter < self.rebalance_period and counter != 1:
            return
            
        if counter >= self.rebalance_period:
            context.set_state("rebalance_counter", 0)

        # Calculate momentum for each symbol
        momentum_scores = {}
        
        for symbol in context.symbols:
            if symbol not in bars:
                continue
            
            closes = context.get_closes(symbol)
            # Need lookback + 1 points to calculate N-day return
            if len(closes) <= self.lookback:
                continue
            
            # Momentum = current price / price N days ago - 1
            past_price = closes[-(self.lookback + 1)]
            current_price = closes[-1]
            
            if past_price > 0:
                momentum = (current_price / past_price) - 1
                momentum_scores[symbol] = momentum
        
        if not momentum_scores:
            return
        
        # Rank by momentum
        sorted_symbols = sorted(
            momentum_scores.keys(),
            key=lambda s: momentum_scores[s],
            reverse=True
        )
        
        # Select top N (only from assets with data today)
        top_symbols = [s for s in sorted_symbols[:self.hold_top_n] if s in bars]
        
        # If no valid top symbols, keep current positions
        if not top_symbols:
            return
        
        target_pct = 1.0 / len(top_symbols) if top_symbols else 0
        
        # Sell symbols not in top N
        for symbol in context.symbols:
            position = context.get_position(symbol)
            if position.quantity > 0 and symbol not in top_symbols:
                context.sell_all(symbol)
        
        # Buy top N symbols
        for symbol in top_symbols:
            context.target_percent(symbol, target_pct)


@register_strategy
class DollarCostAverageStrategy(Strategy):
    """
    Dollar Cost Averaging Strategy.
    
    Invests a fixed amount at regular intervals regardless of price.
    """
    
    name = "Dollar Cost Average"
    description = "Invest fixed amount at regular intervals"
    parameters = [
        StrategyParameter("amount_per_period", "Amount Per Period ($)", "float", 1000.0, 100, 100000, 100),
        StrategyParameter("period_days", "Investment Period (Days)", "int", 30, 1, 365, 1),
    ]
    
    def __init__(self, 
                 amount_per_period: float = 1000.0,
                 period_days: int = 30,
                 allocation: Dict[str, float] = None):
        """
        Args:
            amount_per_period: Dollar amount to invest each period
            period_days: Days between investments
            allocation: Optional dict mapping symbol to target weight (0-1)
        """
        self.amount_per_period = amount_per_period
        self.period_days = period_days
        self.allocation = allocation
    
    def on_init(self, context: BacktestContext) -> None:
        context.set_state("days_counter", self.period_days)
    
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        days = context.get_state("days_counter", 0)
        days += 1
        
        if days >= self.period_days:
            self._invest(context, bars)
            days = 0
        
        context.set_state("days_counter", days)
    
    def _invest(self, context: BacktestContext, bars: Dict[str, Bar]):
        """Invest the fixed amount across symbols."""
        # Use provided allocation if available
        if self.allocation:
            for symbol, weight in self.allocation.items():
                if symbol in bars:
                    amount = self.amount_per_period * weight
                    if context.cash >= amount:
                        context.buy_value(symbol, amount)
            return

        # Default to equal weight if no allocation provided
        available_symbols = [s for s in context.symbols if s in bars]
        if not available_symbols:
            return
        
        amount_per_symbol = self.amount_per_period / len(available_symbols)
        
        for symbol in available_symbols:
            if context.cash >= amount_per_symbol:
                context.buy_value(symbol, amount_per_symbol)


@register_strategy
class ThresholdRebalanceStrategy(Strategy):
    """
    Threshold Rebalancing Strategy.
    
    Rebalances to target weights only when any asset's weight 
    deviates from the target by more than a specified threshold.
    """
    
    name = "Threshold Rebalance"
    description = "Rebalance only when weights deviate by more than threshold"
    parameters = [
        StrategyParameter("threshold", "Deviation Threshold", "float", 0.05, 0.01, 0.3, 0.01),
        StrategyParameter("check_period", "Check Period (Days)", "int", 1, 1, 252, 1),
    ]
    
    def __init__(self, 
                 allocation: Dict[str, float] = None,
                 threshold: float = 0.05,
                 check_period: int = 1):
        """
        Args:
            allocation: Dict mapping symbol to target weight (0-1).
                       If None, equal weight.
            threshold: Maximum allowed deviation (e.g. 0.05 for 5%)
            check_period: How often to check for rebalancing (default: daily)
        """
        self.allocation = allocation
        self.threshold = threshold
        self.check_period = check_period
    
    def on_init(self, context: BacktestContext) -> None:
        context.set_state("days_since_check", self.check_period)  # Trigger initial buy
    
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        days = context.get_state("days_since_check", 0)
        days += 1
        
        if days >= self.check_period:
            self._check_and_rebalance(context, bars)
            days = 0
        
        context.set_state("days_since_check", days)
    
    def _check_and_rebalance(self, context: BacktestContext, bars: Dict[str, Bar]):
        """Check deviations and execute rebalancing if needed."""
        if self.allocation:
            alloc = self.allocation
        else:
            n = len(context.symbols)
            alloc = {s: 1.0 / n for s in context.symbols}
        
        total_equity = context.equity
        if total_equity <= 0:
            return

        should_rebalance = False
        
        # Check current weights
        for symbol, target_pct in alloc.items():
            pos = context.get_position(symbol)
            price = context.get_price(symbol)
            current_val = pos.quantity * price
            current_pct = current_val / total_equity
            
            if abs(current_pct - target_pct) > self.threshold:
                should_rebalance = True
                break
        
        if should_rebalance:
            for symbol in context.symbols:
                if symbol in bars:
                    target_pct = alloc.get(symbol, 0)
                    context.target_percent(symbol, target_pct)
