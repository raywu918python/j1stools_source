import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

from j1stools import parquet_db


def detect_descending(
    df,
    window=7,  # 前後各 window//2 根做比較
    max_slope_ratio=3.0,
    min_overlap_ratio=0.5,
):
    """
    下降通道偵測（H1>H2, L1>L2）
    高點定義：這根K線的 high 是前後各 window//2 根裡最高（對稱局部極大值）
    低點定義：這根K線的 low  是前後各 window//2 根裡最低（對稱局部極小值）
    """
    order = window // 2  # window=7 → order=3

    def process_group(group):
        group = group.sort_values("date").reset_index(drop=True)

        group["is_descending"] = False
        group["is_refined_descending"] = False
        group["overlap_ratio"] = np.nan
        group["slope_ratio"] = np.nan
        group["h1_idx"] = np.nan
        group["h2_idx"] = np.nan
        group["l1_idx"] = np.nan
        group["l2_idx"] = np.nan
        group["h1_date"] = pd.NaT
        group["h2_date"] = pd.NaT
        group["l1_date"] = pd.NaT
        group["l2_date"] = pd.NaT

        # 對稱式局部極值（前後各 order 根）
        high_peaks = argrelextrema(group["high"].values, np.greater, order=order)[0].tolist()
        low_troughs = argrelextrema(group["low"].values, np.less, order=order)[0].tolist()

        marked_structures = set()

        for i in range(len(group)):
            curr_highs = [x for x in high_peaks if x < i]
            curr_lows = [x for x in low_troughs if x < i]
            if len(curr_highs) < 2 or len(curr_lows) < 2:
                continue

            h1_idx, h2_idx = curr_highs[-2], curr_highs[-1]
            l1_idx, l2_idx = curr_lows[-2], curr_lows[-1]

            # 重疊比例
            overlap_len = min(h2_idx, l2_idx) - max(h1_idx, l1_idx)
            if overlap_len <= 1:
                continue

            union_len = max(h2_idx, l2_idx) - min(h1_idx, l1_idx)
            actual_overlap_ratio = overlap_len / union_len

            h1 = group.loc[h1_idx, "high"]
            h2 = group.loc[h2_idx, "high"]
            l1 = group.loc[l1_idx, "low"]
            l2 = group.loc[l2_idx, "low"]

            # 下降條件：高點下降、低點也下降、通道不交叉
            if not (h1 > h2 and l1 > l2 and h1 > l1 and h2 > l2):
                continue

            structure_id = (h1_idx, h2_idx, l1_idx, l2_idx)
            if structure_id in marked_structures:
                continue

            slope_h = (h2 - h1) / (h2_idx - h1_idx)
            slope_l = (l2 - l1) / (l2_idx - l1_idx)

            s_ratio = (
                max(abs(slope_h), abs(slope_l)) / min(abs(slope_h), abs(slope_l))
                if min(abs(slope_h), abs(slope_l)) > 0
                else 99
            )

            group.at[i, "is_descending"] = True
            group.at[i, "overlap_ratio"] = round(actual_overlap_ratio, 2)
            group.at[i, "slope_ratio"] = round(s_ratio, 2)
            group.at[i, "h1_idx"] = h1_idx
            group.at[i, "h2_idx"] = h2_idx
            group.at[i, "l1_idx"] = l1_idx
            group.at[i, "l2_idx"] = l2_idx
            group.at[i, "h1_date"] = group.loc[h1_idx, "date"]
            group.at[i, "h2_date"] = group.loc[h2_idx, "date"]
            group.at[i, "l1_date"] = group.loc[l1_idx, "date"]
            group.at[i, "l2_date"] = group.loc[l2_idx, "date"]

            if actual_overlap_ratio >= min_overlap_ratio and s_ratio <= max_slope_ratio:
                group.at[i, "is_refined_descending"] = True

            marked_structures.add(structure_id)

        return group

    return df.groupby("stock_id", group_keys=False).apply(process_group)


def draw_descending(df_source, df_signals, n_plots=9):
    all_matches = df_signals.sort_values("date", ascending=False)

    if all_matches.empty:
        print("沒有符合條件的訊號。")
        return

    n_plots = min(len(all_matches), n_plots)
    rows = (n_plots + 2) // 3

    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=[
            f"{r['stock_id']} | {r['date'].date()} | overlap={r['overlap_ratio']}"
            for _, r in all_matches.head(n_plots).iterrows()
        ],
    )

    for idx, (_, target) in enumerate(all_matches.head(n_plots).iterrows()):
        r, c = (idx // 3) + 1, (idx % 3) + 1

        stock_data = df_source[df_source["stock_id"] == target["stock_id"]].sort_values("date").reset_index(drop=True)

        p_min = int(min(target["h1_idx"], target["l1_idx"]))
        p_max = int(max(target["h2_idx"], target["l2_idx"]))
        df_crop = stock_data.loc[max(0, p_min - 10) : min(len(stock_data) - 1, p_max + 20)]

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

        for (i1, i2), col_name, color in [
            ((target["h1_idx"], target["h2_idx"]), "high", "red"),
            ((target["l1_idx"], target["l2_idx"]), "low", "orange"),
        ]:
            p1, p2 = int(i1), int(i2)
            y1 = stock_data.loc[p1, col_name]
            y2 = stock_data.loc[p2, col_name]
            fig.add_trace(
                go.Scatter(
                    x=[p1, p2],
                    y=[y1, y2],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(height=280 * rows, template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


def test():
    stocks = parquet_db.query_stocks_ids_list()
    # stocks = random.sample(parquet_db.query_stocks_no_etf(), 100)
    df = parquet_db.query_price(stocks, "2026-01-01", "2026-03-01")

    df_source = detect_descending(
        df,
        window=7,  # 前後各 3 根比較
        max_slope_ratio=5,
        min_overlap_ratio=0.4,
    )

    df_descending = df_source[df_source["is_descending"] == True]
    df_refined = df_source[df_source["is_refined_descending"] == True]

    print("找到下降通道數量：", len(df_descending), "  精選：", len(df_refined))

    draw_descending(df_source, df_descending)
    # draw_descending(df_source, df_refined)


# test()
