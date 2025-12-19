"""
Backtesting Framework for Algorithmic Trading Strategies

This module provides a flexible backtesting engine that supports:
- Day-level price data
- Extensible strategy interface via Strategy ABC
- Strategy registry for GUI discovery
- Performance metrics calculation

Example usage:
    from invest.backtest import Backtester, BacktestContext
    from invest.strategies import BuyAndHoldStrategy
    
    backtester = Backtester(
        strategy=BuyAndHoldStrategy(),
        symbols=["VTI", "SP500"],
        start_date="2010-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0
    )
    result = backtester.run()
    print(result.summary())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Callable, Type
import numpy as np


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Bar:
    """OHLCV data for a single trading day."""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    @property
    def typical_price(self) -> float:
        """Return typical price (average of high, low, close)."""
        return (self.high + self.low + self.close) / 3.0


def create_backtester_from_config(config: 'PortfolioConfig', start_date, end_date, initial_capital, 
                                  withdrawal_amount=0, withdrawal_period_days=30, 
                                  withdrawal_method="proportional", adjust_for_inflation=False) -> 'Backtester':
    """
    Helper to create a Backtester instance from a PortfolioConfig.
    
    Args:
        config: PortfolioConfig object defining strategy and assets
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        withdrawal_amount: Amount to withdraw periodically
        withdrawal_period_days: How often to withdraw
        withdrawal_method: 'proportional', 'rebalance', 'sell_winners', 'sell_losers'
        adjust_for_inflation: Whether to adjust nominal withdrawal amount for inflation
    """
    # Import locally to avoid circular import issues if Config imports Backtester/StrategyRegistry
    from .portfolio_manager import PortfolioConfig
    
    strategy = config.create_strategy()
    
    return Backtester(
        strategy=strategy,
        symbols=config.assets,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        withdrawal_amount=withdrawal_amount,
        withdrawal_period_days=withdrawal_period_days,
        withdrawal_method=withdrawal_method,
        target_allocation=config.allocation,
        adjust_for_inflation=adjust_for_inflation
    )



@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    
    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for limit orders")


@dataclass
class Trade:
    """Executed trade record."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    date: date
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        """Total trade value including commission."""
        base_value = self.quantity * self.price
        if self.side == OrderSide.BUY:
            return base_value + self.commission
        else:
            return base_value - self.commission


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Return market value at average cost."""
        return self.quantity * self.avg_cost
    
    def update(self, quantity_delta: float, price: float):
        """Update position with a new trade."""
        if quantity_delta > 0:  # Buying
            total_cost = self.quantity * self.avg_cost + quantity_delta * price
            self.quantity += quantity_delta
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0
        else:  # Selling
            self.quantity += quantity_delta  # quantity_delta is negative
            # avg_cost remains the same when selling


@dataclass
class EquityPoint:
    """Single point in equity curve."""
    date: date
    equity: float
    cash: float
    positions_value: float
    allocation: Dict[str, float] = field(default_factory=dict)  # symbol -> percentage (0-1)


class TransactionType(Enum):
    """Types of cash transactions."""
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIV"
    WITHDRAWAL = "SPEND"


@dataclass
class Transaction:
    """Represents a cash transaction for tracking and logging."""
    date: date
    type: TransactionType
    symbol: str = ""
    quantity: float = 0.0
    price: float = 0.0
    cash_change: float = 0.0  # Positive = cash in, negative = cash out
    note: str = ""
    
    def log_line(self, balance: float) -> str:
        """Generate log line for this transaction."""
        type_str = self.type.value
        if self.type == TransactionType.BUY:
            return f"{self.date} | {type_str:5} | {self.symbol:6} | {self.quantity:8.2f} @ ${self.price:8.2f} | {self.cash_change:+12,.0f} | ${balance:12,.0f}"
        elif self.type == TransactionType.SELL:
            return f"{self.date} | {type_str:5} | {self.symbol:6} | {self.quantity:8.2f} @ ${self.price:8.2f} | {self.cash_change:+12,.0f} | ${balance:12,.0f}"
        elif self.type == TransactionType.DIVIDEND:
            return f"{self.date} | {type_str:5} | {self.symbol:6} | {self.quantity:8.2f} x ${self.price:8.4f} | {self.cash_change:+12,.0f} | ${balance:12,.0f}"
        else:  # WITHDRAWAL
            return f"{self.date} | {type_str:5} | {'CASH':6} |  {self.quantity:8.2f} x ${1.0:8.4f}| {self.cash_change:+12,.0f} | ${balance:12,.0f}"


class BacktestContext:
    """
    Context object provided to strategies during backtesting.
    
    Provides access to:
    - Current date and bar data
    - Portfolio state (cash, positions)
    - Historical price data
    - Order submission methods
    """
    
    def __init__(self, initial_capital: float, symbols: List[str]):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.symbols = symbols
        self.positions: Dict[str, Position] = {s: Position(symbol=s) for s in symbols}
        self.current_date: Optional[date] = None
        self.current_bars: Dict[str, Bar] = {}
        self.historical_bars: Dict[str, List[Bar]] = {s: [] for s in symbols}
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []
        self._state: Dict = {}  # Strategy state storage
        self.transaction_log: List[str] = []  # Cash-centric transaction log
        self.transactions: List[Transaction] = []  # Transaction objects
    
    def record_transaction(self, txn: Transaction) -> None:
        """Execute a transaction: update cash and log."""
        self.cash += txn.cash_change
        self.transaction_log.append(txn.log_line(self.cash))
        self.transactions.append(txn)
    
    def record_buy(self, symbol: str, quantity: float, price: float, commission: float = 0.0) -> Transaction:
        """Record a buy transaction."""
        cash_change = -(quantity * price + commission)
        txn = Transaction(
            date=self.current_date,
            type=TransactionType.BUY,
            symbol=symbol,
            quantity=quantity,
            price=price,
            cash_change=cash_change
        )
        self.record_transaction(txn)
        return txn
    
    def record_sell(self, symbol: str, quantity: float, price: float, commission: float = 0.0) -> Transaction:
        """Record a sell transaction."""
        cash_change = quantity * price - commission
        txn = Transaction(
            date=self.current_date,
            type=TransactionType.SELL,
            symbol=symbol,
            quantity=quantity,
            price=price,
            cash_change=cash_change
        )
        self.record_transaction(txn)
        return txn
    
    def record_dividend(self, symbol: str, shares: float, per_share: float) -> Transaction:
        """Record dividend income."""
        cash_change = shares * per_share
        txn = Transaction(
            date=self.current_date,
            type=TransactionType.DIVIDEND,
            symbol=symbol,
            quantity=shares,
            price=per_share,
            cash_change=cash_change
        )
        self.record_transaction(txn)
        return txn
    
    def record_withdrawal(self, amount: float, note: str = "") -> Transaction:
        """Record a withdrawal/consumption."""
        txn = Transaction(
            date=self.current_date,
            type=TransactionType.WITHDRAWAL,
            price=1.0,
            quantity=amount,
            cash_change=-amount
        )
        self.record_transaction(txn)
        return txn
    
    @property
    def equity(self) -> float:
        """Current total equity (cash + positions value)."""
        positions_value = sum(
            pos.quantity * self.get_price(pos.symbol)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def positions_value(self) -> float:
        """Current total value of all positions."""
        return sum(
            pos.quantity * self.get_price(pos.symbol)
            for pos in self.positions.values()
        )
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if symbol in self.current_bars:
            return self.current_bars[symbol].close
        return 0.0
    
    def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol."""
        return self.positions.get(symbol, Position(symbol=symbol))
    
    def get_history(self, symbol: str, lookback: int = 0) -> List[Bar]:
        """
        Get historical bars for a symbol.
        
        Args:
            symbol: Asset symbol
            lookback: Number of bars to return (0 = all available)
        """
        history = self.historical_bars.get(symbol, [])
        if lookback > 0:
            return history[-lookback:]
        return history
    
    def get_closes(self, symbol: str, lookback: int = 0) -> np.ndarray:
        """Get array of close prices for a symbol."""
        history = self.get_history(symbol, lookback)
        return np.array([bar.close for bar in history])
    
    def buy(self, symbol: str, quantity: float, 
            order_type: OrderType = OrderType.MARKET,
            limit_price: Optional[float] = None) -> Order:
        """Submit a buy order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        self.pending_orders.append(order)
        return order
    
    def sell(self, symbol: str, quantity: float,
             order_type: OrderType = OrderType.MARKET,
             limit_price: Optional[float] = None) -> Order:
        """Submit a sell order."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        self.pending_orders.append(order)
        return order
    
    def buy_value(self, symbol: str, value: float) -> Optional[Order]:
        """Buy a specific dollar value of an asset."""
        price = self.get_price(symbol)
        if price <= 0:
            return None
        quantity = value / price
        return self.buy(symbol, quantity)
    
    def sell_all(self, symbol: str) -> Optional[Order]:
        """Sell entire position in an asset."""
        position = self.get_position(symbol)
        if position.quantity > 0:
            return self.sell(symbol, position.quantity)
        return None
    
    def target_percent(self, symbol: str, target_pct: float) -> Optional[Order]:
        """
        Adjust position to target percentage of portfolio.
        
        Args:
            symbol: Asset symbol
            target_pct: Target percentage (0.0 to 1.0)
        """
        target_value = self.equity * target_pct
        current_value = self.get_position(symbol).quantity * self.get_price(symbol)
        diff_value = target_value - current_value
        
        price = self.get_price(symbol)
        if price <= 0:
            return None
            
        quantity = abs(diff_value) / price
        if quantity < 0.0001:  # Skip tiny orders
            return None
            
        if diff_value > 0:
            return self.buy(symbol, quantity)
        else:
            return self.sell(symbol, quantity)
    
    def set_state(self, key: str, value):
        """Store state that persists across bars."""
        self._state[key] = value
    
    def get_state(self, key: str, default=None):
        """Retrieve stored state."""
        return self._state.get(key, default)


@dataclass
class StrategyParameter:
    """
    Declaration of a configurable strategy parameter.
    
    Used by GUI to dynamically create parameter controls.
    """
    name: str  # Parameter name (must match __init__ argument name)
    label: str  # Display label in GUI
    param_type: str  # 'int', 'float', 'bool', or 'choice'
    default: any  # Default value
    min_value: float = None  # For int/float types
    max_value: float = None  # For int/float types
    step: float = 1  # Step size for sliders
    choices: List[str] = None  # For choice type


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    To create a custom strategy:
    1. Inherit from Strategy
    2. Set the `name` class attribute
    3. Define `parameters` list for GUI configuration
    4. Implement `on_bar()` method
    5. Optionally override `on_init()` and `on_end()`
    
    Example:
        class MyStrategy(Strategy):
            name = "My Custom Strategy"
            parameters = [
                StrategyParameter("threshold", "Threshold", "float", 0.5, 0, 1, 0.1)
            ]
            
            def __init__(self, threshold=0.5):
                self.threshold = threshold
            
            def on_bar(self, context, bars):
                if context.cash > 1000:
                    context.buy("VTI", 10)
    """
    
    name: str = "Unnamed Strategy"
    description: str = ""
    parameters: List[StrategyParameter] = []  # Override in subclasses
    
    def on_init(self, context: BacktestContext) -> None:
        """
        Called once at the start of backtesting.
        Use this to initialize strategy state.
        """
        pass
    
    @abstractmethod
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        """
        Called for each trading day with current bar data.
        
        Args:
            context: BacktestContext with portfolio state and order methods
            bars: Dict mapping symbol to current Bar data
        """
        pass
    
    def on_end(self, context: BacktestContext) -> None:
        """
        Called once at the end of backtesting.
        Use this for cleanup or final calculations.
        """
        pass


class StrategyRegistry:
    """
    Global registry for strategy discovery.
    
    Strategies can be registered using the @register decorator:
        
        @StrategyRegistry.register
        class MyStrategy(Strategy):
            name = "My Strategy"
            ...
    """
    
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, strategy_class: Type[Strategy]) -> Type[Strategy]:
        """Register a strategy class. Can be used as a decorator."""
        cls._strategies[strategy_class.name] = strategy_class
        return strategy_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Strategy]]:
        """Get a strategy class by name."""
        return cls._strategies.get(name)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_all(cls) -> Dict[str, Type[Strategy]]:
        """Get all registered strategies."""
        return cls._strategies.copy()
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[Strategy]:
        """Create an instance of a registered strategy."""
        strategy_class = cls.get(name)
        if strategy_class:
            return strategy_class(**kwargs)
        return None


@dataclass
class BacktestResult:
    """
    Results from a backtest run.
    
    Contains equity curve, trades, and calculated metrics.
    """
    strategy_name: str
    symbols: List[str]
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    equity_curve: List[EquityPoint]
    trades: List[Trade]
    total_dividends: float = 0.0
    transaction_log: List[str] = field(default_factory=list)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    
    @property
    def total_return(self) -> float:
        """Total return as a decimal (e.g., 0.25 for 25%)."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_equity - self.initial_capital) / self.initial_capital
    
    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage."""
        return self.total_return * 100
    
    @property
    def days(self) -> int:
        """Number of trading days in backtest."""
        return len(self.equity_curve)
    
    @property
    def years(self) -> float:
        """Number of years in backtest."""
        if not self.equity_curve:
            return 0.0
        delta = self.end_date - self.start_date
        return delta.days / 365.25
    
    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        years = self.years
        if years <= 0 or self.initial_capital <= 0:
            return 0.0
        return (self.final_equity / self.initial_capital) ** (1 / years) - 1
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a decimal (negative value)."""
        if not self.equity_curve:
            return 0.0
        
        equities = [ep.equity for ep in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
        
        return max_dd
    
    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as a percentage (negative value)."""
        return self.max_drawdown * 100
    
    @property
    def sharpe_ratio(self) -> float:
        """
        Sharpe ratio using daily risk-free rate from interest data.
        Annualized based on 252 trading days.
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate Strategy Returns
        equities = np.array([ep.equity for ep in self.equity_curve])
        returns = np.diff(equities) / equities[:-1] # Daily returns
        
        if len(returns) == 0:
            return 0.0

        # Load Risk Free Rates (Annualized)
        try:
            from .read_data import load_rf_rates
            rf_data = load_rf_rates()
            if not rf_data:
                raise ValueError("No RF data")
                
            # Create sorted list of dates and rates for lookup
            rf_dates_sorted = sorted(rf_data.keys())
            
            # Map daily returns to daily RF rate
            # Strategy Dates: equity_curve[0].date is start, equity_curve[1].date is first return day
            # returns[i] corresponds to period from date[i] to date[i+1]
            return_dates = [ep.date for ep in self.equity_curve[1:]]
            
            excess_returns = []
            
            for i, r_date in enumerate(return_dates):
                # Find latest RF rate available on or before r_date
                # Simple linear scan or bisect. Given monthly data and ordered dates, 
                # we can just find the max date <= r_date
                
                # Optimized: Since data is monthly, just get Y-M
                # But efficient lookup:
                # Use the rate from the start of the month/previous month
                
                # Let's simple check: Find last date in rf_dates <= r_date
                # Ideally use bisect, but for simplicity:
                current_rf = 0.0
                
                # Find most recent rate (poor man's ffill)
                # Since rf_dates might be few (months) and backtest (days), 
                # Bisect is better.
                import bisect
                idx = bisect.bisect_right(rf_dates_sorted, r_date)
                if idx > 0:
                    last_date = rf_dates_sorted[idx-1]
                    current_rf = rf_data[last_date]
                
                # Convert Annual RF to Daily RF
                # Daily RF = (1 + Annual)^(1/252) - 1  OR approx Annual / 252
                daily_rf = current_rf / 252.0
                
                excess_returns.append(returns[i] - daily_rf)
                
            excess_returns = np.array(excess_returns)
            
            if np.std(excess_returns) == 0:
                return 0.0
                
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except ImportError:
            # Fallback to RF=0 if module issue
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        except Exception:
            # Fallback if logic mismatch or empty data
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    @property
    def sortino_ratio(self) -> float:
        """
        Sortino ratio (downside deviation only).
        Annualized based on 252 trading days.
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        equities = np.array([ep.equity for ep in self.equity_curve])
        returns = np.diff(equities) / equities[:-1]
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.mean(returns) / downside_std * np.sqrt(252)
    
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self.trades:
            return 0.0
        
        # Group trades by symbol to calculate P&L
        # For simplicity, count buys followed by sells
        wins = 0
        total = 0
        
        # Simple approach: compare consecutive buy/sell pairs
        symbol_trades: Dict[str, List[Trade]] = {}
        for trade in self.trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        for symbol, trades in symbol_trades.items():
            buy_price = 0
            for trade in trades:
                if trade.side == OrderSide.BUY:
                    buy_price = trade.price
                elif trade.side == OrderSide.SELL and buy_price > 0:
                    total += 1
                    if trade.price > buy_price:
                        wins += 1
                    buy_price = 0
        
        return (wins / total * 100) if total > 0 else 0.0
    
    @property
    def trade_count(self) -> int:
        """Total number of trades."""
        return len(self.trades)
    
    def get_equity_series(self) -> tuple:
        """Return dates and equity values as separate lists."""
        dates = [ep.date for ep in self.equity_curve]
        equities = [ep.equity for ep in self.equity_curve]
        return dates, equities
    
    @property
    def volatility(self) -> float:
        """Annualized volatility (standard deviation of returns)."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equities = np.array([ep.equity for ep in self.equity_curve])
        returns = np.diff(equities) / equities[:-1]
        
        if len(returns) == 0:
            return 0.0
            
        return np.std(returns) * np.sqrt(252)

    def calculate_benchmark_metrics(self, benchmark_prices: List[float], benchmark_dates: List[date]):
        """
        Calculate Beta and Alpha against a benchmark series.
        
        Args:
            benchmark_prices: List of benchmark closing prices
            benchmark_dates: List of corresponding dates
        """
        if not self.equity_curve or not benchmark_prices:
            return

        # Create dictionaries for alignment
        strategy_series = {ep.date: ep.equity for ep in self.equity_curve}
        benchmark_series = {d: p for d, p in zip(benchmark_dates, benchmark_prices)}
        
        # Find common dates
        common_dates = sorted(list(set(strategy_series.keys()) & set(benchmark_series.keys())))
        
        if len(common_dates) < 2:
            return
            
        # Extract aligned prices
        strat_prices = np.array([strategy_series[d] for d in common_dates])
        bench_prices = np.array([benchmark_series[d] for d in common_dates])
        
        # Calculate returns
        strat_returns = np.diff(strat_prices) / strat_prices[:-1]
        bench_returns = np.diff(bench_prices) / bench_prices[:-1]
        
        if len(strat_returns) == 0 or np.var(bench_returns) == 0:
            return
            
        # Calculate Beta: Covariance / Variance
        # Use ddof=1 for unbiased estimator to match np.cov default
        covariance = np.cov(strat_returns, bench_returns)[0][1]
        variance = np.var(bench_returns, ddof=1)
        self.beta = covariance / variance
        
        # Calculate Alpha (Annualized): Strategy Return - Beta * Benchmark Return
        # Assuming Risk Free Rate approx 0 for simplicity in this contest
        strat_ann_ret = np.mean(strat_returns) * 252
        bench_ann_ret = np.mean(bench_returns) * 252
        
        self.alpha = strat_ann_ret - (self.beta * bench_ann_ret)

    def summary(self, benchmark: 'BacktestResult' = None) -> str:
        """Generate a summary report string."""
        
        # Helper to format diff
        def fmt_diff(val1, val2, is_pct=False):
            diff = val1 - val2
            sign = "+" if diff >= 0 else ""
            if is_pct:
                return f"{sign}{diff*100:.2f}%"
            return f"{sign}{diff:.2f}"

        lines = [
            f"📊 BACKTEST RESULTS: {self.strategy_name}",
            f"{'='*65}",
            f"",
            f"📅 PERIOD:",
            f"  • Start Date:      {self.start_date}",
            f"  • End Date:        {self.end_date}",
            f"  • Duration:        {self.years:.1f} years ({self.days} days)",
            f"",
        ]

        if benchmark and benchmark.strategy_name != self.strategy_name:
            # Comparison Mode
            lines.append(f"🆚 COMPARISON: vs {benchmark.strategy_name}")
            lines.append(f"{'-'*70}")
            # Truncate names to 15 chars to fit column
            s_name = self.strategy_name[:15]
            b_name = benchmark.strategy_name[:15]
            lines.append(f"{'METRIC':<20} | {s_name:<15} | {b_name:<15} | {'DIFF':<8}")
            lines.append(f"{'-'*70}")
            
            # Performance
            lines.append(f"{'Total Return':<20} | {self.total_return_pct:>11.2f}% | {benchmark.total_return_pct:>11.2f}% | {fmt_diff(self.total_return_pct, benchmark.total_return_pct, False)}%")
            lines.append(f"{'CAGR':<20} | {self.cagr*100:>11.2f}% | {benchmark.cagr*100:>11.2f}% | {fmt_diff(self.cagr, benchmark.cagr, True)}")
            lines.append(f"{'Final Equity':<20} | ${self.final_equity:>11,.0f} | ${benchmark.final_equity:>11,.0f} | ${self.final_equity - benchmark.final_equity:,.0f}")
            lines.append(f"",)
            
            # Risk
            lines.append(f"{'Max Drawdown':<20} | {self.max_drawdown_pct:>11.2f}% | {benchmark.max_drawdown_pct:>11.2f}% | {fmt_diff(self.max_drawdown_pct, benchmark.max_drawdown_pct, False)}%")
            lines.append(f"{'Volatility (Ann)':<20} | {self.volatility*100:>11.2f}% | {benchmark.volatility*100:>11.2f}% | {fmt_diff(self.volatility, benchmark.volatility, True)}")
            lines.append(f"{'Sharpe Ratio':<20} | {self.sharpe_ratio:>12.2f} | {benchmark.sharpe_ratio:>12.2f} | {fmt_diff(self.sharpe_ratio, benchmark.sharpe_ratio)}")
            lines.append(f"{'Sortino Ratio':<20} | {self.sortino_ratio:>12.2f} | {benchmark.sortino_ratio:>12.2f} | {fmt_diff(self.sortino_ratio, benchmark.sortino_ratio)}")
            
            # Stats
            lines.append(f"{'Win Rate':<20} | {self.win_rate:>11.1f}% | {benchmark.win_rate:>11.1f}% | {fmt_diff(self.win_rate, benchmark.win_rate, False)}%")
            lines.append(f"{'Trades':<20} | {self.trade_count:>12} | {benchmark.trade_count:>12} | {self.trade_count - benchmark.trade_count:+d}")

            # Beta/Alpha (Benchmark is usually Market, so Alpha/Beta relative to it is strictly 1 and 0 if it IS the market, 
            # but if we calculated beta/alpha against VTI, we show that separately below)
            
            lines.append(f"")
            if hasattr(self, 'beta') and self.beta is not None:
                lines.append(f"📉 ALPHA / BETA (vs VTI):")
                lines.append(f"  • Beta:  {self.beta:.2f}")
                lines.append(f"  • Alpha: {self.alpha*100:+.2f}%")

        else:
            # Standard Mode (No Benchmark comparison)
            lines.append(f"💰 PERFORMANCE:")
            lines.append(f"  • Initial Capital: ${self.initial_capital:,.2f}")
            lines.append(f"  • Final Equity:    ${self.final_equity:,.2f}")
            lines.append(f"  • Total Return:    {self.total_return_pct:+.2f}%")
            lines.append(f"  • CAGR:            {self.cagr*100:+.2f}%")
            lines.append(f"  • Total Dividends: ${self.total_dividends:,.2f}")
            lines.append(f"")
            lines.append(f"📉 RISK METRICS:")
            lines.append(f"  • Volatility (Ann):{self.volatility*100:.2f}%")
            lines.append(f"  • Max Drawdown:    {self.max_drawdown_pct:.2f}%")
            lines.append(f"  • Sharpe Ratio:    {self.sharpe_ratio:.2f}")
            lines.append(f"  • Sortino Ratio:   {self.sortino_ratio:.2f}")
            
            if hasattr(self, 'beta') and self.beta is not None:
                 lines.append(f"  • Beta (vs VTI):   {self.beta:.2f}")
                 lines.append(f"  • Alpha (vs VTI):  {self.alpha*100:+.2f}%")
                 
            lines.extend([
                f"",
                f"📈 TRADES:",
                f"  • Total Trades:    {self.trade_count}",
                f"  • Win Rate:        {self.win_rate:.1f}%",
                f"  • Assets:          {', '.join(self.symbols)}",
            ])
            
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.summary()


class Backtester:
    """
    Main backtesting engine.
    
    Runs a strategy against historical price data and produces results.
    
    Example:
        backtester = Backtester(
            strategy=BuyAndHoldStrategy(),
            symbols=["VTI"],
            start_date="2010-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0
        )
        result = backtester.run()
    """
    
    def __init__(
        self,
        strategy: Strategy,
        symbols: List[str],
        start_date: str | date,
        end_date: str | date,
        initial_capital: float = 100000.0,
        commission: float = 0.0,
        data_loader: Optional[Callable[[str], List[Bar]]] = None,
        withdrawal_amount: float = 0.0,
        withdrawal_period_days: int = 30,
        withdrawal_method: str = "proportional",  # proportional, rebalance, sell_winners, sell_losers
        target_allocation: Optional[Dict[str, float]] = None,
        adjust_for_inflation: bool = False
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_loader = data_loader or self._default_data_loader
        self.withdrawal_amount = withdrawal_amount
        self.withdrawal_period_days = withdrawal_period_days
        self.withdrawal_method = withdrawal_method
        self.target_allocation = target_allocation or {}
        self.adjust_for_inflation = adjust_for_inflation
    
    @staticmethod
    def _parse_date(d: str | date) -> date:
        """Parse date string to date object."""
        if isinstance(d, date):
            return d
        return datetime.strptime(d, "%Y-%m-%d").date()
    
    def _default_data_loader(self, symbol: str) -> List[Bar]:
        """Load daily bar data from project data files."""
        from .read_data import stock_data_daily
        return stock_data_daily(symbol)
    
    def _load_data(self) -> Dict[str, List[Bar]]:
        """Load price data for all symbols."""
        data = {}
        for symbol in self.symbols:
            bars = self.data_loader(symbol)
            # Filter to date range
            filtered = [
                bar for bar in bars
                if self.start_date <= bar.date <= self.end_date
            ]
            data[symbol] = filtered
        return data
    
    def _execute_orders(self, context: BacktestContext):
        """Execute pending orders at current prices."""
        orders_to_process = context.pending_orders.copy()
        context.pending_orders.clear()
        
        # Execute SELL orders first, then BUY orders
        # This ensures cash from sales is available for purchases
        orders_to_process.sort(key=lambda o: (0 if o.side == OrderSide.SELL else 1))
        
        for order in orders_to_process:
            price = context.get_price(order.symbol)
            if price <= 0:
                continue
            
            # Check limit orders
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price > order.limit_price:
                    continue
                if order.side == OrderSide.SELL and price < order.limit_price:
                    continue
            
            # Calculate trade value
            trade_value = order.quantity * price
            commission = self.commission * trade_value
            
            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission
                if total_cost > context.cash:
                    # Reduce quantity to fit available cash
                    available = context.cash - commission
                    if available <= 0:
                        continue
                    order.quantity = available / price
                    trade_value = order.quantity * price
                    commission = self.commission * trade_value
                
                if order.quantity > 0:
                    context.record_buy(order.symbol, order.quantity, price, commission)
                    context.positions[order.symbol].update(order.quantity, price)
                
            else:  # SELL
                position = context.positions.get(order.symbol)
                if not position or position.quantity < order.quantity:
                    # Reduce to available quantity
                    order.quantity = position.quantity if position else 0
                
                if order.quantity > 0:
                    context.record_sell(order.symbol, order.quantity, price, commission)
                    context.positions[order.symbol].update(-order.quantity, price)
            
            # Record trade
            if order.quantity > 0:
                trade = Trade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=price,
                    date=context.current_date,
                    commission=commission
                )
                context.trades.append(trade)
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Load data
        all_data = self._load_data()
        
        # Get all unique dates across all symbols
        all_dates = set()
        for bars in all_data.values():
            for bar in bars:
                all_dates.add(bar.date)
        sorted_dates = sorted(all_dates)
        
        if not sorted_dates:
            # No data available
            return BacktestResult(
                strategy_name=self.strategy.name,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
                equity_curve=[],
                trades=[]
            )
        
        # Create date-indexed data for fast lookup
        date_bars: Dict[date, Dict[str, Bar]] = {}
        for d in sorted_dates:
            date_bars[d] = {}
        for symbol, bars in all_data.items():
            for bar in bars:
                date_bars[bar.date][symbol] = bar
        
        # Initialize context
        context = BacktestContext(self.initial_capital, self.symbols)
        
        # Initialize strategy
        self.strategy.on_init(context)
        
        # Load dividend data for each symbol
        from .read_data import load_dividends
        dividend_data: Dict[str, Dict[date, float]] = {}
        for symbol in self.symbols:
            dividend_data[symbol] = load_dividends(symbol)
        total_dividends = 0.0
        
        # Withdrawal tracking
        days_since_withdrawal = self.withdrawal_period_days  # Trigger first withdrawal immediately if enabled
        total_withdrawn = 0.0
        
        # Run through each trading day
        for current_date in sorted_dates:
            context.current_date = current_date
            bars = date_bars[current_date]
            context.current_bars = bars
            
            # Add bars to history
            for symbol, bar in bars.items():
                context.historical_bars[symbol].append(bar)
            
            # Call strategy
            self.strategy.on_bar(context, bars)
            
            # Execute orders
            self._execute_orders(context)
            
            # Process dividends: add dividend income to cash
            for symbol in self.symbols:
                pos = context.positions.get(symbol)
                if pos and pos.quantity > 0:
                    div = dividend_data.get(symbol, {}).get(current_date, 0)
                    if div > 0:
                        context.record_dividend(symbol, pos.quantity, div)
                        total_dividends += pos.quantity * div
            
            # Periodic withdrawal (consumption)
            if self.withdrawal_amount > 0:
                days_since_withdrawal += 1
                if days_since_withdrawal >= self.withdrawal_period_days:
                    withdrawal_needed = self.withdrawal_amount
                    
                    # Apply inflation adjustment if enabled
                    if self.adjust_for_inflation:
                        from .read_data import get_inflation_multiplier_for_date
                        inflation_mult = get_inflation_multiplier_for_date(
                            self.start_date, current_date
                        )
                        withdrawal_needed *= inflation_mult
                    
                    # Step 1: Use available cash first
                    cash_used = min(context.cash, withdrawal_needed)
                    remaining = withdrawal_needed - cash_used
                    
                    # Step 2: If still need more, sell assets
                    if remaining > 0:
                        # Get positions with values
                        position_values = []
                        for symbol in self.symbols:
                            pos = context.positions.get(symbol)
                            if pos and pos.quantity > 0:
                                price = context.get_price(symbol)
                                if price > 0:
                                    value = pos.quantity * price
                                    position_values.append((symbol, pos, price, value))
                        
                        if position_values:
                            total_position_value = sum(pv[3] for pv in position_values)
                            
                            if self.withdrawal_method == "proportional":
                                # Sell proportionally from each position
                                for symbol, pos, price, value in position_values:
                                    sell_ratio = min(remaining / total_position_value, 1.0)
                                    sell_value = value * sell_ratio
                                    sell_qty = min(sell_value / price, pos.quantity)
                                    if sell_qty > 0:
                                        context.record_sell(symbol, sell_qty, price)
                                        pos.quantity -= sell_qty
                            
                            elif self.withdrawal_method == "rebalance":
                                # First sell from overweight positions (based on target allocation)
                                overweight = []
                                for symbol, pos, price, value in position_values:
                                    # Use custom target allocation or equal weight as fallback
                                    target_weight = self.target_allocation.get(symbol, 1.0 / len(position_values))
                                    target_value = total_position_value * target_weight
                                    if value > target_value:
                                        excess = value - target_value
                                        overweight.append((symbol, pos, price, value, excess))
                                
                                # Sort by absolute excess (largest first)
                                overweight.sort(key=lambda x: x[4], reverse=True)
                                
                                for symbol, pos, price, value, excess in overweight:
                                    sell_value = min(excess, remaining)
                                    sell_qty = min(sell_value / price, pos.quantity)
                                    if sell_qty > 0:
                                        context.record_sell(symbol, sell_qty, price)
                                        pos.quantity -= sell_qty
                                        remaining -= sell_qty * price
                                    if remaining <= 0:
                                        break
                                
                                # If still need more, fall back to proportional selling
                                if remaining > 0:
                                    # Recalculate position values after selling overweight
                                    remaining_positions = []
                                    for symbol, pos, price, value in position_values:
                                        if pos.quantity > 0:
                                            current_value = pos.quantity * price
                                            remaining_positions.append((symbol, pos, price, current_value))
                                    
                                    if remaining_positions:
                                        total_remaining_value = sum(pv[3] for pv in remaining_positions)
                                        for symbol, pos, price, value in remaining_positions:
                                            sell_ratio = min(remaining / total_remaining_value, 1.0)
                                            sell_value = value * sell_ratio
                                            sell_qty = min(sell_value / price, pos.quantity)
                                            if sell_qty > 0:
                                                context.record_sell(symbol, sell_qty, price)
                                                pos.quantity -= sell_qty
                                        remaining = 0
                            
                            elif self.withdrawal_method == "sell_winners":
                                # Sell from highest return positions first
                                # Calculate return rate: (current_price / avg_cost) - 1
                                positions_with_return = []
                                for symbol, pos, price, value in position_values:
                                    if pos.avg_cost > 0:
                                        return_rate = (price / pos.avg_cost) - 1
                                    else:
                                        return_rate = 0
                                    positions_with_return.append((symbol, pos, price, value, return_rate))
                                
                                sorted_positions = sorted(positions_with_return, key=lambda x: x[4], reverse=True)
                                for symbol, pos, price, value, ret in sorted_positions:
                                    sell_value = min(value, remaining)
                                    sell_qty = min(sell_value / price, pos.quantity)
                                    if sell_qty > 0:
                                        context.record_sell(symbol, sell_qty, price)
                                        pos.quantity -= sell_qty
                                        remaining -= sell_qty * price
                                    if remaining <= 0:
                                        break
                            
                            elif self.withdrawal_method == "sell_losers":
                                # Sell from lowest return (biggest losers) positions first
                                positions_with_return = []
                                for symbol, pos, price, value in position_values:
                                    if pos.avg_cost > 0:
                                        return_rate = (price / pos.avg_cost) - 1
                                    else:
                                        return_rate = 0
                                    positions_with_return.append((symbol, pos, price, value, return_rate))
                                
                                sorted_positions = sorted(positions_with_return, key=lambda x: x[4])
                                for symbol, pos, price, value, ret in sorted_positions:
                                    sell_value = min(value, remaining)
                                    sell_qty = min(sell_value / price, pos.quantity)
                                    if sell_qty > 0:
                                        context.record_sell(symbol, sell_qty, price)
                                        pos.quantity -= sell_qty
                                        remaining -= sell_qty * price
                                    if remaining <= 0:
                                        break
                    
                    days_since_withdrawal = 0
                    
                    # Step 3: Deduct total withdrawal from cash
                    context.record_withdrawal(withdrawal_needed)
                    total_withdrawn += withdrawal_needed
            
            # Record equity with allocation
            equity = context.equity
            allocation = {"CASH": context.cash / equity if equity > 0 else 0}
            for symbol in self.symbols:
                pos = context.positions.get(symbol)
                if pos and pos.quantity > 0:
                    price = context.get_price(symbol)
                    value = pos.quantity * price
                    allocation[symbol] = value / equity if equity > 0 else 0
                else:
                    allocation[symbol] = 0
            
            context.equity_curve.append(EquityPoint(
                date=current_date,
                equity=equity,
                cash=context.cash,
                positions_value=context.positions_value,
                allocation=allocation
            ))
        
        # End strategy
        self.strategy.on_end(context)
        
        # Build result
        return BacktestResult(
            strategy_name=self.strategy.name,
            symbols=self.symbols,
            start_date=sorted_dates[0] if sorted_dates else self.start_date,
            end_date=sorted_dates[-1] if sorted_dates else self.end_date,
            initial_capital=self.initial_capital,
            final_equity=context.equity,
            equity_curve=context.equity_curve,
            trades=context.trades,
            total_dividends=total_dividends,
            transaction_log=context.transaction_log
        )


def register_strategy(cls: Type[Strategy]) -> Type[Strategy]:
    """Decorator to register a strategy with StrategyRegistry."""
    return StrategyRegistry.register(cls)
