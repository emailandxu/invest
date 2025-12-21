# save_vbtlx.py
import argparse
import datetime

import yfinance as yf

from ._paths import data_path
import pandas as pd
def cli_parser():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance and save to CSV.")

    # 添加参数
    parser.add_argument("--code", help="Stock code to download (e.g., VTSAX)")
    parser.add_argument("--start", default="1970-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument("--update", action="store_true", help="Update existing stock CSV")
    parser.add_argument("--update_all", action="store_true", help="Update existing stock CSVs under data/STOCK")
    # 解析命令行参数
    args = parser.parse_args()
    return args

def download_data(args=None):
    if args is None:
        args = cli_parser()

    if hasattr(args, "update_all") and args.update_all:
        update_all()
    elif hasattr(args, "update") and args.update:
        update_code(args.code)
    else:
        _download_data(args)

def _download_data(args):
    # 使用命令行参数
    code = args.code
    if not code:
        print("Error: No stock code specified for download.")
        return
    start = args.start if hasattr(args, 'start') else "2025-10-01"
    end = args.end if hasattr(args, 'end') else datetime.date.today().isoformat()

    # 下载历史日线数据 - 使用未调整价格 (auto_adjust=False)
    # 因为分红被单独计算，使用调整价格会导致分红被重复计算
    df = yf.download(code, start=start, end=end, progress=False, auto_adjust=False)
    # df.loc[:, ('Close','BND')]
    # Create a new dataframe with just the Close price
    the_df = df[['Close']].copy()
    the_df.columns = ['Value']
    the_df.index.name = 'Date'

    stock_dir = data_path("STOCK")
    out = stock_dir / f"{code}.csv"
    if out.exists():
        existing_df = pd.read_csv(out, index_col='Date')
        existing_df.index = pd.to_datetime(existing_df.index)
        the_df.index = pd.to_datetime(the_df.index)
        # fill the_df by the missing dates from existing_df
        # so that the new data takes precedence
        merged_df = the_df.combine_first(existing_df)
        merged_df = merged_df.sort_index()
        merged_df.to_csv(out)
        print(f"Merged {len(the_df)} rows with existing data and saved to {out}")

    else:
        the_df.to_csv(out)
        print(f"Saved {len(the_df)} rows to {out}")

def update_all():
    """Update all existing stock CSVs under data/STOCK using download_data.

    Strategy:
    - For each CSV in data/STOCK, treat its filename stem as the ticker code.
    - Read the CSV to find the last available date; use the next day as the download start
      to avoid re-downloading the entire history and to prevent duplicate rows.
    - If parsing fails, fall back to downloading from 1970-01-01.
    - End date is today.
    - Delegate actual fetching/merge to download_data().
    """
    from tqdm import tqdm

    stock_dir = data_path("STOCK")
    stock_dir.mkdir(parents=True, exist_ok=True)


    updated = 0
    for csv_path in tqdm(sorted(stock_dir.glob("*.csv"))):
        if csv_path.name.startswith("."):
            continue  # skip hidden files
        
        update_code(csv_path.stem)
        updated += 1
    print(f"Updated {updated} tickers under {stock_dir}")


def update_code(code):
    start = "1970-01-01"
    today = datetime.date.today().isoformat()
    try:
        df = pd.read_csv(data_path("STOCK", code + ".csv"), index_col='Date')
        if not df.empty:
            # ensure datetime index
            df.index = pd.to_datetime(df.index, errors='coerce')
            last = df.index.max()
            if pd.notna(last):
                # start = (last + pd.Timedelta(days=1)).date().isoformat()
                start = last.date().isoformat()
    except Exception as e:
        # keep default start
        print(e)
        pass

    # Construct args-like object
    class Args:
        pass
    args = Args()
    args.code = code
    args.start = start
    args.end = today

    print(f"Updating {code}: {start} → {today}")
    try:
        _download_data(args)
        # Automatically sync dividends whenever we update price data
        download_dividends(code)
    except Exception as e:
        print(f"Failed to update {code}: {e}")


def download_dividends(code: str):
    """Download dividend data for a stock and save to CSV."""
    try:
        ticker = yf.Ticker(code)
        div = ticker.dividends
        
        if div.empty:
            print(f"No dividend data for {code}")
            return
        
        # Convert to DataFrame
        div_df = div.reset_index()
        div_df.columns = ['Date', 'Dividend']
        
        # Remove timezone info from dates
        div_df['Date'] = pd.to_datetime(div_df['Date']).dt.tz_localize(None)
        div_df['Date'] = div_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        stock_dir = data_path("STOCK")
        out = stock_dir / f"{code}_dividends.csv"
        div_df.to_csv(out, index=False)
        print(f"Saved {len(div_df)} dividend records to {out}")
    except Exception as e:
        print(f"Failed to download dividends for {code}: {e}")

def update_all_dividends():
    """Update dividend CSVs for all stocks found in data/STOCK."""
    stock_dir = data_path("STOCK")
    for csv_path in sorted(stock_dir.glob("*.csv")):
        if csv_path.name.startswith(".") or "_dividends" in csv_path.name:
            continue
        code = csv_path.stem
        download_dividends(code)


def download_rates():
    """Download interest rate data (FEDFUNDS) from FRED."""
    try:
        import pandas_datareader.data as web
        start = datetime.datetime(1954, 7, 1)
        end = datetime.datetime.now()
        
        # FEDFUNDS: Effective Federal Funds Rate (Monthly, Percent, NSA)
        df = web.DataReader('FEDFUNDS', 'fred', start, end)
        df = df.reset_index()
        df.columns = ['Date', 'Value']
        
        # Create 'interest.csv' format: Date,Value (Percent)
        out = data_path("interest.csv")
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} interest rate records to {out}")
        
    except Exception as e:
        print(f"Failed to download interest rates: {e}")


def download_inflation():
    """Download inflation data (CPIAUCNS) from FRED."""
    try:
        import pandas_datareader.data as web
        start = datetime.datetime(1913, 1, 1)
        end = datetime.datetime.now()
        
        # CPIAUCNS: Consumer Price Index for All Urban Consumers: All Items
        # This is an INDEX (Level), e.g. 250.0
        df = web.DataReader('CPIAUCNS', 'fred', start, end)
        df = df.reset_index()
        df.columns = ['Date', 'Value']
        
        # Calculate Year-over-Year Inflation Rate (%)
        # (CPI_t / CPI_{t-12} - 1) * 100
        df['Value'] = df['Value'].pct_change(12) * 100
        
        # Drop NaN values (first 12 months)
        df = df.dropna()
        
        # Create 'inflation.csv' format: Date,Value (Percent)
        out = data_path("inflation.csv")
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} inflation records to {out}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to download inflation data: {e}")


if __name__ == "__main__":
    download_data()
