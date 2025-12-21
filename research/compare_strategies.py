import datetime
import invest.strategies  # Trigger registration
from invest.backtest import Backtester, StrategyRegistry

def run_comparison():
    # Configuration
    assets = ["GLD", "QQQ"]
    initial_capital = 100000.0
    allocation = {"GLD": 0.5, "QQQ": 0.5}
    
    # Momentum Params (from JSON)
    strategy_params = {
        "lookback": 60,
        "hold_top_n": 1,
        "rebalance_period": 60
    }

    # Define Periods
    periods = [
        ("Full History (2006-2024)", "2006-01-01", "2024-12-16"),
        ("2008 Crisis (2007-2009)", "2007-01-01", "2009-12-31"),
        ("Bull Market (2010-2019)", "2010-01-01", "2019-12-31"),
        ("COVID & Recent (2020-2024)", "2020-01-01", "2024-12-16"),
        ("Last 3 Years (2022-2024)", "2022-01-01", "2024-12-16"),
    ]

    print(f"{'Period':<30} | {'Momentum CAGR':<15} | {'Buy&Hold CAGR':<15} | {'Diff':<10}")
    print("-" * 80)

    for name, start, end in periods:
        try:
            # Run Momentum
            mom_strategy = StrategyRegistry.create("Momentum", **strategy_params)
            mom_backtester = Backtester(
                strategy=mom_strategy,
                symbols=assets,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
                target_allocation=allocation # Ignored by Momentum but good practice
            )
            mom_result = mom_backtester.run()
            
            # Run Buy and Hold
            bh_strategy = StrategyRegistry.create("Buy and Hold", allocation=allocation)
            bh_backtester = Backtester(
                strategy=bh_strategy,
                symbols=assets,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
                target_allocation=allocation
            )
            bh_result = bh_backtester.run()
            
            diff = (mom_result.cagr - bh_result.cagr) * 100
            print(f"{name:<30} | {mom_result.cagr*100:6.2f}%         | {bh_result.cagr*100:6.2f}%         | {diff:+6.2f}%")
            
        except Exception as e:
            print(f"{name:<30} | ERROR: {str(e)}")

if __name__ == "__main__":
    run_comparison()
