import pandas as pd
import numpy as np
import mplfinance as mpf
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots
from j1stools import parquet_db


def detect_falling_wedge(df, order=7, min_reduction=0.3, max_slope_ratio=5.0, min_overlap_ratio=0.5):
    def process_group(group):
        # --- 核心修正：確保 index 是連續整數，並保留原始 index ---
        group = group.sort_values("date").copy()
        group["original_index"] = group.index
        group = group.reset_index(drop=True)

        group["is_wedge"] = False
        group["is_refined_wedge"] = False
        group["width_reduction"] = np.nan
        group["overlap_ratio"] = np.nan
        group["h1_date"] = pd.NaT
        group["h2_date"] = pd.NaT
        group["l1_date"] = pd.NaT
        group["l2_date"] = pd.NaT
        for col in ["h1_idx", "h2_idx", "l1_idx", "l2_idx"]:
            group[col] = np.nan

        high_idx = argrelextrema(group["high"].values, np.greater, order=order)[0]
        low_idx = argrelextrema(group["low"].values, np.less, order=order)[0]

        if len(high_idx) < 2 or len(low_idx) < 2:
            return group

        marked_structures = set()
        for i in range(len(group)):
            curr_highs = high_idx[high_idx < i]
            curr_lows = low_idx[low_idx < i]
            if len(curr_highs) < 2 or len(curr_lows) < 2:
                continue

            h1_idx, h2_idx = curr_highs[-2], curr_highs[-1]
            l1_idx, l2_idx = curr_lows[-2], curr_lows[-1]

            overlap_start = max(h1_idx, l1_idx)
            overlap_end = min(h2_idx, l2_idx)
            overlap_len = overlap_end - overlap_start
            if overlap_len <= 1:
                continue

            union_len = max(h2_idx, l2_idx) - min(h1_idx, l1_idx)
            actual_overlap_ratio = overlap_len / union_len

            combined = sorted([(h1_idx, "H"), (h2_idx, "H"), (l1_idx, "L"), (l2_idx, "L")])
            pattern = "".join([x[1] for x in combined])
            if pattern not in ["HLHL", "LHLH"]:
                continue

            h1, h2 = group.loc[h1_idx, "high"], group.loc[h2_idx, "high"]
            l1, l2 = group.loc[l1_idx, "low"], group.loc[l2_idx, "low"]

            # 下降楔形邏輯：高低點皆下降，且高點下降更快
            if h1 > h2 and l1 > l2 and h2 > l2 and (h1 - l1) > (h2 - l2):
                slope_h = (h2 - h1) / (h2_idx - h1_idx)
                slope_l = (l2 - l1) / (l2_idx - l1_idx)

                check_h = all(
                    group.loc[idx, "high"] <= h1 + slope_h * (idx - h1_idx) * 1.01 for idx in range(h1_idx + 1, h2_idx)
                )
                check_l = all(
                    group.loc[idx, "low"] >= l1 + slope_l * (idx - l1_idx) * 0.99 for idx in range(l1_idx + 1, l2_idx)
                )

                structure_id = (h1_idx, h2_idx, l1_idx, l2_idx)
                if structure_id in marked_structures:
                    continue

                if check_h and check_l:
                    group.at[i, "is_wedge"] = True
                    group.at[i, "overlap_ratio"] = round(actual_overlap_ratio, 2)
                    # 存入當前連續 index
                    group.at[i, "h1_idx"], group.at[i, "h2_idx"] = h1_idx, h2_idx
                    group.at[i, "l1_idx"], group.at[i, "l2_idx"] = l1_idx, l2_idx
                    group.at[i, "h1_date"] = group.loc[h1_idx, "date"]
                    group.at[i, "h2_date"] = group.loc[h2_idx, "date"]
                    group.at[i, "l1_date"] = group.loc[l1_idx, "date"]
                    group.at[i, "l2_date"] = group.loc[l2_idx, "date"]

                    reduction = 1 - ((h2 - l2) / (h1 - l1))
                    group.at[i, "width_reduction"] = round(reduction, 2)
                    s_ratio = abs(slope_h) / abs(slope_l) if abs(slope_l) > 0 else 99

                    if actual_overlap_ratio >= min_overlap_ratio and reduction >= min_reduction and s_ratio > 1.1:
                        group.at[i, "is_refined_wedge"] = True

                    marked_structures.add(structure_id)
        return group

    return df.groupby("stock_id", group_keys=False).apply(process_group)


def draw_wedge_safe(df_source, df_signals, n_plots=9):
    # 這裡調整欄位名稱以符合 wedge 版本
    all_matches = df_signals[df_signals["is_wedge"] == True].sort_values("date", ascending=False)

    if all_matches.empty:
        print("沒有符合條件的楔形訊號。")
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

        # 抓取該股票資料
        stock_data = df_source[df_source["stock_id"] == target["stock_id"]].sort_values("date").reset_index(drop=True)

        p_min = int(min(target["h1_idx"], target["l1_idx"]))
        p_max = int(max(target["h2_idx"], target["l2_idx"]))
        df_crop = stock_data.iloc[max(0, p_min - 10) : min(len(stock_data), p_max + 20)]

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

        # 畫壓力與支撐線
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
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


def find_wedge(df):
    df_source = detect_falling_wedge(
        df, order=5, min_reduction=0.2, min_overlap_ratio=0.4  # 楔形收斂有時較慢，可調低一點
    )

    return df_source[df_source["is_refined_wedge"] == True]


def test_wedge():
    stocks = parquet_db.query_stocks_ids_list()
    stocks = ["6209"]
    df = parquet_db.query_price(stocks, "2025-01-01", "2099-01-01")
    # 呼叫新函數
    df_source = detect_falling_wedge(
        df, order=5, min_reduction=0.2, min_overlap_ratio=0.4  # 楔形收斂有時較慢，可調低一點
    )

    df_refined_wedge = df_source[df_source["is_refined_wedge"] == True]
    df_wedge = df_source[df_source["is_wedge"] == True]
    print(df_refined_wedge.head().T)
    print(
        "找到標記的楔形數量：",
        len(df_source),
        len(df_wedge),
        len(df_refined_wedge),
    )
    draw_wedge_safe(
        df_source,
        df_refined_wedge,
        # df_refined_triangle,
    )


# test_wedge()
