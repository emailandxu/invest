
from invest.backtest import Backtester, StrategyRegistry, BacktestResult
from invest import strategies
from invest.read_data import stock_data_daily, load_dividends
import pandas as pd
import numpy as np
from datetime import date

# 1. Backtest
strategy = StrategyRegistry.create("Buy and Hold", allocation={"VTI": 1.0})
backtester = Backtester(
    strategy=strategy,
    symbols=["VTI"],
    start_date=date(2015, 1, 1),
    end_date=date(2023, 12, 31),
    initial_capital=100000.0
)
res = backtester.run()

# 2. Get Dividend Dates
divs = load_dividends("VTI")
div_dates = sorted([d for d in divs.keys() if date(2015, 1, 1) <= d <= date(2023, 12, 31)])

# 3. Analyze Returns around Dividend Dates
dates, equities = res.get_equity_series()
strat_series = pd.Series(equities, index=dates)
strat_ret = strat_series.pct_change()

vti_bars = stock_data_daily("VTI")
bench_series = pd.Series({b.date: b.close for b in vti_bars})
bench_ret = bench_series.pct_change()

print(f"{'Date':<12} | {'Div':<6} | {'Strat Ret':<10} | {'Bench Ret':<10} | {'Diff':<10}")
print("-" * 60)

for d in div_dates:
    if d in strat_ret.index and d in bench_ret.index:
        s_r = strat_ret[d]
        b_r = bench_ret[d]
        print(f"{d} | {divs[d]:<6.3f} | {s_r:>9.4%} | {b_r:>9.4%} | {s_r - b_r:>9.4%}")

# 4. Try adjusting Benchmark for Dividends (Approximation)
# Reconstruct Total Return Index for Benchmark
aligned_bench = bench_series[strat_series.index] 
tr_bench = aligned_bench.copy()
shares = 1.0
cash = 0.0
# Simple TR: Add dividend to price? No.
# TR Index = Price * AccumFactor.
# AccumFactor *= (1 + Div/Price) on Ex-Date.

# Let's try to pass a "Total Return" benchmark to calculate_benchmark_metrics
# We need to construct a Total Return Series for VTI
# TR_t = TR_{t-1} * (Price_t + Div_t) / Price_{t-1} ? 
# Actually simpler: Daily Return = (Price_t + Div_t - Price_{t-1}) / Price_{t-1}
# Then Index_t = Index_{t-1} * (1 + Daily Return)

tr_index = [100.0]
tr_dates = [dates[0]]
prev_price = bench_series[dates[0]]

for d in dates[1:]:
    price = bench_series[d]
    div = divs.get(d, 0.0)
    
    # Return = (Price + Div - PrevPrice) / PrevPrice
    ret = (price + div - prev_price) / prev_price
    
    new_val = tr_index[-1] * (1 + ret)
    tr_index.append(new_val)
    tr_dates.append(d)
    prev_price = price

res.calculate_benchmark_metrics(tr_index, tr_dates)
print(f"\nBeta against Total Return Index: {res.beta:.4f}")
print(f"Alpha against Total Return Index: {res.alpha:.4f}")
