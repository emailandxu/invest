
from invest.backtest import Backtester, Strategy, StrategyRegistry, BacktestContext, Bar
from invest import strategies
from invest.read_data import stock_data_daily, load_dividends
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict

# 1. Define Reinvest Strategy
@StrategyRegistry.register
class ReinvestStrategy(Strategy):
    name = "Reinvest Divs"
    
    def on_init(self, context: BacktestContext) -> None:
        pass
        
    def on_bar(self, context: BacktestContext, bars: Dict[str, Bar]) -> None:
        # Buy with all available cash every day
        for symbol in context.symbols:
            if symbol in bars and context.cash > 10:
                context.buy_value(symbol, context.cash)

# 2. Run Backtest
backtester = Backtester(
    strategy=ReinvestStrategy(),
    symbols=["VTI"],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    initial_capital=100000.0
)
res = backtester.run()

# 3. Create Total Return Benchmark
days, equities = res.get_equity_series()
vti_bars = stock_data_daily("VTI")
bench_series = pd.Series({b.date: b.close for b in vti_bars})
divs = load_dividends("VTI")

# Construct TR Index
tr_index = [100.0]
tr_dates = [days[0]]
prev_price = bench_series[days[0]]

for d in days[1:]:
    if d not in bench_series:
        tr_index.append(tr_index[-1])
        tr_dates.append(d)
        continue
        
    price = bench_series[d]
    div = divs.get(d, 0.0)
    
    # Return = (Price + Div - PrevPrice) / PrevPrice
    ret = (price + div - prev_price) / prev_price
    
    new_val = tr_index[-1] * (1 + ret)
    tr_index.append(new_val)
    tr_dates.append(d)
    prev_price = price

# 4. Calculate Beta
res.calculate_benchmark_metrics(tr_index, tr_dates)
print(f"Beta (Reinvested): {res.beta:.4f}")
print(f"Alpha (Reinvested): {res.alpha:.4f}")

# Also calculate against Raw Price just to see
prices_aligned = [bench_series[d] for d in tr_dates]
backtester.strategy = strategies.BuyAndHoldStrategy() # Dummy
res.calculate_benchmark_metrics(prices_aligned, tr_dates)
print(f"Beta (vs Raw Price): {res.beta:.4f}")
