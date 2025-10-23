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
    if hasattr(args, "update") and args.update:
        update_code(args.code)
    else:
        _download_data(args)

def _download_data(args):
    # 使用命令行参数
    code = args.code
    start = args.start if hasattr(args, 'start') else "2025-10-01"
    end = args.end if hasattr(args, 'end') else datetime.date.today().isoformat()

    # 下载 VBTLX 的历史日线数据
    df = yf.download(code, start=start, end=end, progress=False)
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
        # Merge the existing data with the new data, preferring the new data
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
    except Exception as e:
        print(f"Failed to update {code}: {e}")


if __name__ == "__main__":
    download_data()
    
