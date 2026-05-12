import random

import pandas as pd
import numpy as np
import mplfinance as mpf
import plotly.graph_objects as go
from scipy.signal import argrelextrema

from j1stools import parquet_db


def detect_strict_triangle(
    df, order=7, min_reduction=0.4, max_slope_ratio=3.0, min_overlap_ratio=0.5
):  # 新增：重疊比例參數

    def process_group(group):
        group = group.sort_values("date").reset_index(drop=True)
        group["is_triangle"] = False
        group["is_refined_triangle"] = False
        group["width_reduction"] = np.nan
        group["overlap_ratio"] = np.nan  # 存下重疊率方便觀察
        for col in ["h1_idx", "h2_idx", "l1_idx", "l2_idx"]:
            group[col] = np.nan

        high_idx = argrelextrema(group["high"].values, np.greater, order=order)[0]
        low_idx = argrelextrema(group["low"].values, np.less, order=order)[0]

        if len(high_idx) < 2 or len(low_idx) < 2:
            return group

        for i in range(len(group)):
            curr_highs = high_idx[high_idx < i]
            curr_lows = low_idx[low_idx < i]
            if len(curr_highs) < 2 or len(curr_lows) < 2:
                continue

            h1_idx, h2_idx = curr_highs[-2], curr_highs[-1]
            l1_idx, l2_idx = curr_lows[-2], curr_lows[-1]

            # --- 計算重疊程度 ---
            start_point = min(h1_idx, l1_idx)
            end_point = max(h2_idx, l2_idx)
            total_span = end_point - start_point

            overlap_start = max(h1_idx, l1_idx)
            overlap_end = min(h2_idx, l2_idx)
            overlap_len = overlap_end - overlap_start

            # 只要 overlap_len <= 0 代表完全沒交集，直接過濾
            if overlap_len <= 0:
                continue

            actual_overlap_ratio = overlap_len / total_span

            # 檢查交錯順序 (HLHL 或 LHLH)
            combined = sorted([(h1_idx, "H"), (h2_idx, "H"), (l1_idx, "L"), (l2_idx, "L")])
            pattern = "".join([x[1] for x in combined])
            if pattern not in ["HLHL", "LHLH"]:
                continue

            h1, h2 = group.loc[h1_idx, "high"], group.loc[h2_idx, "high"]
            l1, l2 = group.loc[l1_idx, "low"], group.loc[l2_idx, "low"]

            # 基礎收斂與無突破判定
            if h1 > h2 and l1 < l2 and h2 > l2:
                slope_h = (h2 - h1) / (h2_idx - h1_idx)
                slope_l = (l2 - l1) / (l2_idx - l1_idx)

                check_h = all(
                    group.loc[idx, "high"] <= h1 + slope_h * (idx - h1_idx) for idx in range(h1_idx + 1, h2_idx)
                )
                check_l = all(
                    group.loc[idx, "low"] >= l1 + slope_l * (idx - l1_idx) for idx in range(l1_idx + 1, l2_idx)
                )

                if check_h and check_l:
                    group.at[i, "is_triangle"] = True  # 原始標記 (只要有交集就標記)
                    group.at[i, "overlap_ratio"] = round(actual_overlap_ratio, 2)
                    group.at[i, "h1_idx"], group.at[i, "h2_idx"] = h1_idx, h2_idx
                    group.at[i, "l1_idx"], group.at[i, "l2_idx"] = l1_idx, l2_idx

                    # 精選標記：必須符合重疊比例與其他幾何參數
                    reduction = 1 - ((h2 - l2) / (h1 - l1))
                    group.at[i, "width_reduction"] = round(reduction, 2)

                    s_ratio = (
                        max(abs(slope_h), abs(slope_l)) / min(abs(slope_h), abs(slope_l))
                        if min(abs(slope_h), abs(slope_l)) > 0
                        else 99
                    )

                    if (
                        actual_overlap_ratio >= min_overlap_ratio
                        and reduction >= min_reduction
                        and s_ratio <= max_slope_ratio
                    ):
                        group.at[i, "is_refined_triangle"] = True

        return group

    return df.groupby("stock_id", group_keys=False).apply(process_group)


import plotly.graph_objects as go
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw_multiple_triangles_safe(df_source, df_signals, n_plots=9):
    """
    df_source: 原始完整大表 (包含所有歷史資料)
    df_signals: 過濾後只有 True 的標籤表
    """
    # 按照日期由新到舊排序
    all_matches = df_signals.sort_values("date", ascending=False)

    if all_matches.empty:
        print("沒有符合條件的訊號。")
        return

    n_plots = min(len(all_matches), n_plots)
    rows = (n_plots + 2) // 3
    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=[f"{r['stock_id']} | {r['date'].date()}" for _, r in all_matches.head(n_plots).iterrows()],
    )

    for idx, (original_idx, target) in enumerate(all_matches.head(n_plots).iterrows()):
        r, c = (idx // 3) + 1, (idx % 3) + 1

        # --- 關鍵修正：去 source 大表抓該股票的連續資料 ---
        stock_data = df_source[df_source["stock_id"] == target["stock_id"]].sort_values("date")

        # 設定繪圖範圍 (取前後緩衝)
        p_min = int(min(target["h1_idx"], target["l1_idx"]))
        p_max = int(max(target["h2_idx"], target["l2_idx"]))

        # 使用 iloc 前後切片，確保 K 線連續
        # 這裡假設你的 index 是重置過的 0,1,2...
        df_crop = stock_data.loc[max(0, p_min - 10) : min(len(stock_data), p_max + 20)]

        # 繪製 K 線
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

        # 繪製紅綠線 (使用 target 存下的座標)
        for pts, color in [
            ([target["h1_idx"], target["h2_idx"]], "red"),
            ([target["l1_idx"], target["l2_idx"]], "green"),
        ]:
            p1, p2 = int(pts[0]), int(pts[1])
            # 從 stock_data (大表) 抓取精確價格
            y1 = stock_data.loc[p1, "high" if color == "red" else "low"]
            y2 = stock_data.loc[p2, "high" if color == "red" else "low"]
            fig.add_trace(
                go.Scatter(
                    x=[p1, p2], y=[y1, y2], mode="lines+markers", line=dict(color=color, width=3), showlegend=False
                ),
                row=r,
                col=c,
            )

    fig.update_layout(height=250 * rows, template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


# 呼叫方式
# draw_multiple_triangles(df_triangle, n_plots=9)
def test():
    stocks = parquet_db.query_stocks_ids_list()
    # stocks = random.sample(parquet_db.query_stocks_no_etf(), 100)
    df = parquet_db.query_price(stocks, "2026-01-01", "2026-04-01")
    df_triangle = detect_strict_triangle(df, order=5, max_slope_ratio=99)
    print(
        "找到標記的三角形數量：",
        len(df_triangle),
        len(df_triangle[df_triangle["is_triangle"] == True]),
        len(df_triangle[df_triangle["is_refined_triangle"] == True]),
    )
    # print(df_triangle.head().T)
    refined_triangle = df_triangle[df_triangle["is_refined_triangle"] == True]
    draw_multiple_triangles_safe(df_triangle, refined_triangle)


test()
