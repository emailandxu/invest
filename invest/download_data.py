# save_vbtlx.py
import argparse
import datetime

import yfinance as yf

from ._paths import data_path

def cli_parser():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance and save to CSV.")

    # 添加参数
    parser.add_argument("CODE", help="Stock code to download (e.g., VTSAX)")
    parser.add_argument("--start", default="1970-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.date.today().isoformat(), help="End date (YYYY-MM-DD)")

    # 解析命令行参数
    args = parser.parse_args()
    return args

def download_data(args=None):
    if args is None:
        args = cli_parser()

    # 使用命令行参数
    CODE = args.CODE
    start = args.start
    end = args.end

    # 下载 VBTLX 的历史日线数据
    df = yf.download(CODE, start=start, end=end, progress=False)
    # df.loc[:, ('Close','BND')]
    # Create a new dataframe with just the Close price
    the_df = df[['Close']].copy()
    the_df.columns = ['Value']
    the_df.index.name = 'Date'

    stock_dir = data_path("STOCK")
    out = stock_dir / f"{CODE}.csv"
    the_df.to_csv(out)
    print(f"Saved {len(the_df)} rows to {out}")

if __name__ == "__main__":
    download_data()
    