import random

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

from j1stools import parquet_db

import mplfinance as mpf
import numpy as np

import pandas as pd
import numpy as np
import mplfinance as mpf
from scipy.signal import argrelextrema


def extract_triangle_shapes(df, order=5, ratio=1.5):
    features = []
    for stock_id, group in df.groupby("stock_id"):
        close = group["close"].values
        dates = group["date"].values

        high_idx = argrelextrema(close, np.greater, order=order)[0]
        low_idx = argrelextrema(close, np.less, order=order)[0]

        # 至少要有 2 個高點和 2 個低點
        if len(high_idx) >= 2 and len(low_idx) >= 2:
            # 取最近的 2 個高點和 2 個低點
            h1, h2 = high_idx[-2], high_idx[-1]
            l1, l2 = low_idx[-2], low_idx[-1]

            # 計算斜率
            slope_high = (close[h2] - close[h1]) / (h2 - h1)
            slope_low = (close[l2] - close[l1]) / (l2 - l1)

            # 收斂條件：上斜率 < 0，下斜率 > 0
            if slope_high < 0 and slope_low > 0:
                init_diff = close[h1] - close[l1]
                end_diff = close[h2] - close[l2]

                if init_diff > ratio * end_diff:
                    features.append(
                        {
                            "stock_id": stock_id,
                            "points_high": [(dates[h1], close[h1]), (dates[h2], close[h2])],
                            "points_low": [(dates[l1], close[l1]), (dates[l2], close[l2])],
                            "slope_high": slope_high,
                            "slope_low": slope_low,
                            "init_diff": init_diff,
                            "end_diff": end_diff,
                        }
                    )
    return pd.DataFrame(features)


import matplotlib.pyplot as plt
import mplfinance as mpf

import matplotlib.pyplot as plt
import mplfinance as mpf


def batch_plot_triangles(df, df_triangle, n=5):
    # 隨機挑 n 個樣本
    n = min(n, len(df_triangle))
    sample = df_triangle.sample(n)

    # 建立一張大圖，裡面有 n 個 subplot
    fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))

    if n == 1:
        axes = [axes]  # 保證 axes 是 list

    for ax, (_, row) in zip(axes, sample.iterrows()):
        stock_id = row["stock_id"]
        data = df[df["stock_id"] == stock_id].set_index("date")

        # 畫 K 線
        mpf.plot(data, type="candle", style="charles", ax=ax)

        # 高點線段 (用日期)
        h_points = row["points_high"]
        ax.plot([h_points[0][0], h_points[1][0]], [h_points[0][1], h_points[1][1]], color="red")

        # 低點線段 (用日期)
        l_points = row["points_low"]
        ax.plot([l_points[0][0], l_points[1][0]], [l_points[0][1], l_points[1][1]], color="green")

        ax.set_title(f"Stock {stock_id} 收斂三角形")

    plt.tight_layout()
    plt.show()


def validate_breakout(df, stock_id, window=200, threshold=0.1):
    data = df[df["stock_id"] == stock_id].tail(window).set_index("date")
    close = data["close"]
    volume = data["volume"]

    # 假設最後一天是突破日
    breakout_day = close.index[-1]
    breakout_price = close.iloc[-1]

    # 計算未來 20 天漲跌幅
    future = close.tail(20)
    pct_change = (future.max() - breakout_price) / breakout_price

    # 成交量驗證
    avg_vol = volume.iloc[:-1].mean()
    breakout_vol = volume.iloc[-1]

    valid = (pct_change > threshold) and (breakout_vol > avg_vol)

    return {
        "stock_id": stock_id,
        "breakout_day": breakout_day,
        "pct_change": pct_change,
        "breakout_vol": breakout_vol,
        "avg_vol": avg_vol,
        "valid_breakout": valid,
    }


import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import mplfinance as mpf


def test():
    stocks = parquet_db.query_stocks_ids_list()
    stocks = random.sample(parquet_db.query_stocks_no_etf(), 400)
    df = parquet_db.query_price(stocks, "2021-01-01", "2099-01-01")
    df_triangle = extract_triangle_shapes(df, order=5, ratio=1.5)
    print("df_triangle.head()")
    print(df_triangle.head())
    # row = df_triangle.iloc[0]  # 例如第一筆
    # stock_id = row["stock_id"]
    # slope_high = row["slope_high"]
    # slope_low = row["slope_low"]

    # 畫圖驗證
    batch_plot_triangles(df, df_triangle, n=5)


# test()
