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

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go


def detect_strict_triangle(group, order=7):
    df = group.sort_values("date").reset_index(drop=True)
    for col in ["h1_idx", "h2_idx", "l1_idx", "l2_idx"]:
        df[col] = np.nan
    df["is_triangle"] = False

    high_idx = argrelextrema(df["high"].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df["low"].values, np.less, order=order)[0]

    if len(high_idx) < 2 or len(low_idx) < 2:
        return df

    for i in range(len(df)):
        curr_highs = high_idx[high_idx < i]
        curr_lows = low_idx[low_idx < i]

        if len(curr_highs) < 2 or len(curr_lows) < 2:
            continue

        h1_idx, h2_idx = curr_highs[-2], curr_highs[-1]
        l1_idx, l2_idx = curr_lows[-2], curr_lows[-1]

        # --- 新增：時間重疊判斷 ---
        # 確保高點區間與低點區間不是前後分開，而是交錯的
        if not (max(h1_idx, l1_idx) < min(h2_idx, l2_idx)):
            continue

        h1, h2 = df.loc[h1_idx, "high"], df.loc[h2_idx, "high"]
        l1, l2 = df.loc[l1_idx, "low"], df.loc[l2_idx, "low"]

        # 基本收斂判定
        if h1 > h2 and l1 < l2 and h2 > l2:
            # 趨勢線斜率
            slope_h = (h2 - h1) / (h2_idx - h1_idx)
            slope_l = (l2 - l1) / (l2_idx - l1_idx)

            # 檢查區間內無突破 (檢查範圍改為四點的最早到最晚)
            start_check = min(h1_idx, l1_idx)
            check_h = all(df.loc[idx, "high"] <= h1 + slope_h * (idx - h1_idx) for idx in range(h1_idx + 1, h2_idx))
            check_l = all(df.loc[idx, "low"] >= l1 + slope_l * (idx - l1_idx) for idx in range(l1_idx + 1, l2_idx))

            if check_h and check_l:
                df.at[i, "is_triangle"] = True
                df.at[i, "h1_idx"], df.at[i, "h2_idx"] = h1_idx, h2_idx
                df.at[i, "l1_idx"], df.at[i, "l2_idx"] = l1_idx, l2_idx

    return df


def find_convergence(df):
    # 依照 stock_id 分組處理
    result = df.groupby("stock_id", group_keys=False).apply(detect_strict_triangle)
    return result


import plotly.graph_objects as go
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw_multiple_triangles_clean(df_triangle, n_plots=9):
    all_matches = df_triangle[df_triangle["is_triangle"] == True].sort_values("date", ascending=False)
    if all_matches.empty:
        return print("No triangle found.")

    n_plots = min(len(all_matches), n_plots)
    rows = (n_plots + 2) // 3
    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=[f"{r['stock_id']} | {r['date'].date()}" for _, r in all_matches.head(n_plots).iterrows()],
    )

    for idx, (original_idx, target) in enumerate(all_matches.head(n_plots).iterrows()):
        r, c = (idx // 3) + 1, (idx % 3) + 1
        stock_data = (
            df_triangle[df_triangle["stock_id"] == target["stock_id"]].sort_values("date").reset_index(drop=True)
        )

        # 定義繪圖區間：取四個點的最寬範圍再加一點緩衝
        p_min = int(min(target["h1_idx"], target["l1_idx"]))
        p_max = int(max(target["h2_idx"], target["l2_idx"]))
        df_crop = stock_data.iloc[max(0, p_min - 10) : min(len(stock_data), p_max + 20)]

        # K線圖
        fig.add_trace(
            go.Candlestick(
                x=df_crop.index,
                open=df_crop["open"],
                high=df_crop["high"],
                low=df_crop["low"],
                close=df_crop["close"],
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # 壓力線與支撐線
        for pts, color in [
            ([target["h1_idx"], target["h2_idx"]], "red"),
            ([target["l1_idx"], target["l2_idx"]], "green"),
        ]:
            p1, p2 = int(pts[0]), int(pts[1])
            y1 = stock_data.loc[p1, "high" if color == "red" else "low"]
            y2 = stock_data.loc[p2, "high" if color == "red" else "low"]
            fig.add_trace(
                go.Scatter(
                    x=[p1, p2], y=[y1, y2], mode="lines+markers", line=dict(color=color, width=3), showlegend=False
                ),
                row=r,
                col=c,
            )

    fig.update_layout(height=250 * rows, template="plotly_dark", showlegend=False)
    # 關鍵：強制關閉所有子圖的 rangeslider
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


# 呼叫方式
# draw_multiple_triangles(df_triangle, n_plots=9)
def test():
    stocks = parquet_db.query_stocks_ids_list()
    # stocks = random.sample(parquet_db.query_stocks_no_etf(), 100)
    df = parquet_db.query_price(stocks, "2015-01-01", "2099-01-01")
    df_triangle = find_convergence(df)
    print("找到標記的三角形數量：", len(df_triangle))
    print(df_triangle.head())
    draw_multiple_triangles_clean(df_triangle)


test()
