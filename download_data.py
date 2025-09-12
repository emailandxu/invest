# save_vbtlx.py
import yfinance as yf
import datetime

# 设置时间区间（例：从 2000-01-01 到 今天）
start = "1970-01-01"
end = datetime.date.today().isoformat()

CODE = "GLD"
# 下载 VBTLX 的历史日线数据
df = yf.download(CODE, start=start, end=end, progress=False)
# df.loc[:, ('Close','BND')]
# Create a new dataframe with just the Close price
the_df = df[['Close']].copy()
the_df.columns = ['Value']
the_df.index.name = 'Date'

# 保存为 CSV
out ="data/STOCK/{}.csv".format(CODE)
the_df.to_csv(out)
print(f"Saved {len(the_df)} rows to {out}")