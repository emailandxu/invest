
import datetime
import invest.strategies # Register strategies
from invest.backtest import Backtester, StrategyRegistry

def diagnose():
    assets = ["GLD", "QQQ"]
    initial_capital = 100000.0
    allocation = {"GLD": 0.5, "QQQ": 0.5}
    
    # Focused Period
    start_date = "2020-01-01"
    end_date = "2024-12-16"
    
    params = {
        "lookback": 60,
        "hold_top_n": 1,
        "rebalance_period": 60
    }

    print(f"🔬 DIAGNOSTIC: Momentum (Lookback={params['lookback']}, Rebalance={params['rebalance_period']})")
    print(f"📅 Period: {start_date} to {end_date}")
    print("-" * 60)

    # Run Momentum
    strategy = StrategyRegistry.create("Momentum", **params)
    backtester = Backtester(
        strategy=strategy,
        symbols=assets,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    result = backtester.run()

    print(f"Final CAGR: {result.cagr*100:.2f}%")
    print(f"Total Trades: {len(result.trades)}")
    print("-" * 60)
    print("📜 TRADE LOG (First 20 trades):")
    for t in result.trades[:20]:
         print(f"{t.date} | {t.side.value.upper():<4} {t.symbol} | Qty: {t.quantity:8.2f} | Price: ${t.price:7.2f} | Val: ${t.value:,.0f}")
    
    print("-" * 60)
    print("📉 MAJOR DRAWDOWN EVENTS:")
    # Simple check of equity curve drops
    peak = 0
    drawdown_start = None
    for ep in result.equity_curve:
        if ep.equity > peak:
            if drawdown_start and (peak - min_equity) / peak > 0.15:
                 print(f"Drawdown > 15%: {drawdown_start} to {ep.date}")
            peak = ep.equity
            drawdown_start = ep.date
            min_equity = peak
        else:
            min_equity = min(min_equity, ep.equity)

    # Inspect Holdings at key dates
    print("-" * 60)
    print("🔍 HOLDINGS SNAPSHOT (Quarterly):")
    for ep in result.equity_curve:
        if ep.date.day == 1 and ep.date.month % 3 == 0: # Quarterly roughly
             # Infer holding from allocation dict in EquityPoint
             holding = max(ep.allocation, key=ep.allocation.get) if ep.allocation else "CASH"
             if holding == "CASH" and len(ep.allocation) > 1:
                 # If cash is max, check if there is a close second (meaning barely invested?)
                 # Actually Strategy puts 100% in one asset (hold_top_n=1)
                 # So if allocated, it should be the key.
                 pass
             
             print(f"{ep.date} | Holding: {holding:<4} ({ep.allocation.get(holding,0)*100:.1f}%) | Equity: ${ep.equity:,.0f}")

if __name__ == "__main__":
    diagnose()
