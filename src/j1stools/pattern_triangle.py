import random

import pandas as pd
import numpy as np
import mplfinance as mpf
import plotly.graph_objects as go
from scipy.signal import argrelextrema

from j1stools import parquet_db


def _line_fit(points):
    if len(points) < 2:
        return None

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    if len(np.unique(x)) < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def _line_y(line, x):
    slope, intercept = line
    return slope * x + intercept


def _calc_atr(group, window=14):
    prev_close = group["close"].shift(1)
    true_range = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    fallback = group["close"] * 0.01
    return true_range.rolling(window, min_periods=1).mean().fillna(fallback).values


def _iter_fuzzy_pivot_windows(curr_highs, curr_lows, pivot_window=8, max_end_skip=1):
    recent = sorted(
        [(int(idx), "H") for idx in curr_highs[-pivot_window:]] + [(int(idx), "L") for idx in curr_lows[-pivot_window:]]
    )[-pivot_window:]

    if len(recent) < 4:
        return

    last_pos = len(recent) - 1
    for end_pos in range(last_pos, max(-1, last_pos - max_end_skip - 1), -1):
        for length in range(4, 7):
            start_pos = end_pos - length + 1
            if start_pos < 0:
                continue

            window = recent[start_pos : end_pos + 1]
            high_count = sum(kind == "H" for _, kind in window)
            low_count = len(window) - high_count
            if 2 <= high_count <= 3 and 2 <= low_count <= 3:
                yield window


def _evaluate_fuzzy_triangle(
    group,
    window,
    atr_values,
    min_reduction,
    min_overlap_ratio,
    price_tolerance,
    atr_multiplier,
    touch_multiplier,
    max_break_ratio,
    max_break_magnitude_ratio,
    min_close_inside_ratio,
    min_score,
    min_span,
):
    highs = [(idx, group.loc[idx, "high"]) for idx, kind in window if kind == "H"]
    lows = [(idx, group.loc[idx, "low"]) for idx, kind in window if kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    upper_line = _line_fit(highs)
    lower_line = _line_fit(lows)
    if upper_line is None or lower_line is None:
        return None

    upper_slope, upper_intercept = upper_line
    lower_slope, lower_intercept = lower_line
    start_idx = min(idx for idx, _ in highs + lows)
    end_idx = max(idx for idx, _ in highs + lows)
    if end_idx - start_idx < min_span:
        return None

    overlap_start = max(highs[0][0], lows[0][0])
    overlap_end = min(highs[-1][0], lows[-1][0])
    overlap_len = overlap_end - overlap_start
    union_len = end_idx - start_idx
    if union_len <= 0 or overlap_len <= 1:
        return None

    overlap_ratio = overlap_len / union_len
    if overlap_ratio < min_overlap_ratio:
        return None

    upper_start = _line_y(upper_line, start_idx)
    lower_start = _line_y(lower_line, start_idx)
    upper_end = _line_y(upper_line, end_idx)
    lower_end = _line_y(lower_line, end_idx)
    width_start = upper_start - lower_start
    width_end = upper_end - lower_end
    if width_start <= 0 or width_end <= 0:
        return None

    reduction = 1 - (width_end / width_start)
    if reduction < min_reduction or upper_slope >= lower_slope:
        return None

    price_ref = max(float(group.loc[end_idx, "close"]), 1e-9)
    span_atr = float(np.nanmax(atr_values[start_idx : end_idx + 1]))
    edge_tol = max(price_ref * price_tolerance, span_atr * atr_multiplier)
    if upper_end > upper_start + edge_tol:
        return None
    if lower_end < lower_start - edge_tol:
        return None

    touch_scores = []
    touch_ok = 0
    for idx, price in highs:
        tolerance = max(float(group.loc[idx, "close"]) * price_tolerance, atr_values[idx] * atr_multiplier, 1e-9)
        residual = abs(price - _line_y(upper_line, idx))
        touch_scores.append(residual / tolerance)
        touch_ok += residual <= tolerance * touch_multiplier

    for idx, price in lows:
        tolerance = max(float(group.loc[idx, "close"]) * price_tolerance, atr_values[idx] * atr_multiplier, 1e-9)
        residual = abs(price - _line_y(lower_line, idx))
        touch_scores.append(residual / tolerance)
        touch_ok += residual <= tolerance * touch_multiplier

    min_touch_count = max(4, int(np.ceil((len(highs) + len(lows)) * 0.8)))
    if touch_ok < min_touch_count:
        return None

    span = group.iloc[start_idx : end_idx + 1]
    x = np.arange(start_idx, end_idx + 1, dtype=float)
    upper_values = upper_slope * x + upper_intercept
    lower_values = lower_slope * x + lower_intercept
    close_values = span["close"].values.astype(float)
    tolerance_values = np.maximum(close_values * price_tolerance, atr_values[start_idx : end_idx + 1] * atr_multiplier)

    high_values = span["high"].values.astype(float)
    low_values = span["low"].values.astype(float)
    upper_break = high_values > upper_values + tolerance_values
    lower_break = low_values < lower_values - tolerance_values
    upper_break_ratio = float(upper_break.mean())
    lower_break_ratio = float(lower_break.mean())
    if upper_break_ratio > max_break_ratio or lower_break_ratio > max_break_ratio:
        return None

    upper_break_mag = np.maximum(high_values - (upper_values + tolerance_values), 0) / np.maximum(close_values, 1e-9)
    lower_break_mag = np.maximum((lower_values - tolerance_values) - low_values, 0) / np.maximum(close_values, 1e-9)
    if max(float(upper_break_mag.max()), float(lower_break_mag.max())) > max_break_magnitude_ratio:
        return None

    close_inside = (close_values <= upper_values + tolerance_values) & (close_values >= lower_values - tolerance_values)
    close_inside_ratio = float(close_inside.mean())
    if close_inside_ratio < min_close_inside_ratio:
        return None

    fit_score = max(0.0, 1.0 - (float(np.mean(touch_scores)) / max(touch_multiplier, 1e-9)))
    inside_score = (1 - upper_break_ratio + 1 - lower_break_ratio + close_inside_ratio) / 3
    reduction_score = min(1.0, reduction / max(min_reduction * 2, 1e-9))
    overlap_score = min(1.0, overlap_ratio / max(min_overlap_ratio * 2, 1e-9))
    score = 0.30 * fit_score + 0.35 * inside_score + 0.25 * reduction_score + 0.10 * overlap_score
    if score < min_score:
        return None

    high_indices = tuple(idx for idx, _ in highs)
    low_indices = tuple(idx for idx, _ in lows)
    return {
        "structure_id": (high_indices, low_indices),
        "match_type": f"fuzzy_{len(highs)}h{len(lows)}l",
        "score": score,
        "reduction": reduction,
        "overlap_ratio": overlap_ratio,
        "upper_break_ratio": upper_break_ratio,
        "lower_break_ratio": lower_break_ratio,
        "close_inside_ratio": close_inside_ratio,
        "touch_high_count": len(highs),
        "touch_low_count": len(lows),
        "high_indices": high_indices,
        "low_indices": low_indices,
        "upper_slope": upper_slope,
        "upper_intercept": upper_intercept,
        "lower_slope": lower_slope,
        "lower_intercept": lower_intercept,
    }


def _ensure_human_triangle_columns(group):
    group["is_fuzzy_triangle"] = False
    group["is_human_triangle"] = group["is_refined_triangle"].astype(bool)
    group["triangle_match_type"] = np.where(group["is_refined_triangle"], "strict_2h2l", None)
    group["triangle_score"] = np.where(group["is_refined_triangle"], 1.0, np.nan)
    group["upper_break_ratio"] = np.nan
    group["lower_break_ratio"] = np.nan
    group["close_inside_ratio"] = np.nan
    group["touch_high_count"] = np.where(group["is_triangle"], 2, np.nan)
    group["touch_low_count"] = np.where(group["is_triangle"], 2, np.nan)
    group["upper_slope"] = np.nan
    group["upper_intercept"] = np.nan
    group["lower_slope"] = np.nan
    group["lower_intercept"] = np.nan
    group["h3_idx"] = np.nan
    group["l3_idx"] = np.nan
    group["h3_date"] = pd.NaT
    group["l3_date"] = pd.NaT
    return group


def _mark_fuzzy_triangle(group, row_idx, result):
    high_indices = result["high_indices"]
    low_indices = result["low_indices"]
    h1_idx, h2_idx = high_indices[0], high_indices[-1]
    l1_idx, l2_idx = low_indices[0], low_indices[-1]

    group.at[row_idx, "is_fuzzy_triangle"] = True
    group.at[row_idx, "is_human_triangle"] = True
    previous_type = group.at[row_idx, "triangle_match_type"]
    group.at[row_idx, "triangle_match_type"] = (
        f"{previous_type}+{result['match_type']}" if previous_type else result["match_type"]
    )
    group.at[row_idx, "triangle_score"] = round(result["score"], 3)
    group.at[row_idx, "width_reduction"] = round(result["reduction"], 2)
    group.at[row_idx, "overlap_ratio"] = round(result["overlap_ratio"], 2)
    group.at[row_idx, "upper_break_ratio"] = round(result["upper_break_ratio"], 2)
    group.at[row_idx, "lower_break_ratio"] = round(result["lower_break_ratio"], 2)
    group.at[row_idx, "close_inside_ratio"] = round(result["close_inside_ratio"], 2)
    group.at[row_idx, "touch_high_count"] = result["touch_high_count"]
    group.at[row_idx, "touch_low_count"] = result["touch_low_count"]
    group.at[row_idx, "upper_slope"] = result["upper_slope"]
    group.at[row_idx, "upper_intercept"] = result["upper_intercept"]
    group.at[row_idx, "lower_slope"] = result["lower_slope"]
    group.at[row_idx, "lower_intercept"] = result["lower_intercept"]

    group.at[row_idx, "h1_idx"], group.at[row_idx, "h2_idx"] = h1_idx, h2_idx
    group.at[row_idx, "l1_idx"], group.at[row_idx, "l2_idx"] = l1_idx, l2_idx
    group.at[row_idx, "h1_date"] = group.loc[h1_idx, "date"]
    group.at[row_idx, "h2_date"] = group.loc[h2_idx, "date"]
    group.at[row_idx, "l1_date"] = group.loc[l1_idx, "date"]
    group.at[row_idx, "l2_date"] = group.loc[l2_idx, "date"]

    if len(high_indices) >= 3:
        group.at[row_idx, "h3_idx"] = high_indices[1]
        group.at[row_idx, "h3_date"] = group.loc[high_indices[1], "date"]
    if len(low_indices) >= 3:
        group.at[row_idx, "l3_idx"] = low_indices[1]
        group.at[row_idx, "l3_date"] = group.loc[low_indices[1], "date"]


def detect_strict_triangle(
    df, order=7, min_reduction=0.4, max_slope_ratio=3.0, min_overlap_ratio=0.5
):  # 新增：重疊比例參數
    """
    min_reduction 收鍊比例
    max_slope_ratio 平均斜率
    min_overlap_ratio 重疊比例
    """

    def process_group(group):
        group = group.sort_values("date").reset_index(drop=True)
        group["is_triangle"] = False
        group["is_refined_triangle"] = False
        group["width_reduction"] = np.nan
        group["overlap_ratio"] = np.nan  # 存下重疊率方便觀察
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

            # 1. 計算交集與聯集
            overlap_start = max(h1_idx, l1_idx)
            overlap_end = min(h2_idx, l2_idx)
            overlap_len = overlap_end - overlap_start

            # 2. 提早攔截：如果沒交集或交集太短，直接跳過（節省運算資源）
            # 這裡比 0 更嚴格一點，因為重疊不到 1 天根本連線都畫不出來
            if overlap_len <= 1:
                continue

            # 3. 計算聯集長度與比例
            union_len = max(h2_idx, l2_idx) - min(h1_idx, l1_idx)
            actual_overlap_ratio = overlap_len / union_len

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

                # 如果這組座標已經被標記過了，就跳過這一天
                structure_id = (h1_idx, h2_idx, l1_idx, l2_idx)
                if structure_id in marked_structures:
                    continue

                # --- 通過所有基礎與優化檢查後 ---
                if check_h and check_l:
                    group.at[i, "is_triangle"] = True  # 原始標記 (只要有交集就標記)
                    group.at[i, "overlap_ratio"] = round(actual_overlap_ratio, 2)
                    group.at[i, "h1_idx"], group.at[i, "h2_idx"] = h1_idx, h2_idx
                    group.at[i, "l1_idx"], group.at[i, "l2_idx"] = l1_idx, l2_idx
                    # 當判定成功時存入：
                    group.at[i, "h1_date"] = group.loc[h1_idx, "date"]
                    group.at[i, "h2_date"] = group.loc[h2_idx, "date"]
                    group.at[i, "l1_date"] = group.loc[l1_idx, "date"]
                    group.at[i, "l2_date"] = group.loc[l2_idx, "date"]

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

                    # 標記完成後，把這組結構加入 set，之後的日子就不會再重複標記它
                    marked_structures.add(structure_id)

        return group

    return df.groupby("stock_id", group_keys=False).apply(process_group)


def detect_human_triangle(
    df,
    order=7,
    min_reduction=0.4,
    max_slope_ratio=3.0,
    min_overlap_ratio=0.5,
    fuzzy_pivot_window=8,
    fuzzy_max_end_skip=1,
    fuzzy_min_reduction=0.18,
    fuzzy_min_overlap_ratio=0.25,
    fuzzy_price_tolerance=0.015,
    fuzzy_atr_multiplier=0.35,
    fuzzy_touch_multiplier=1.4,
    fuzzy_max_break_ratio=0.30,
    fuzzy_max_break_magnitude_ratio=0.04,
    fuzzy_min_close_inside_ratio=0.65,
    fuzzy_min_score=0.58,
    fuzzy_min_span=12,
):
    """
    Strict + fuzzy triangle detector.

    fuzzy 支援 3high/2low、2high/3low、3high/3low，允許少量高低點超出或不足。
    """
    df_source = detect_strict_triangle(
        df,
        order=order,
        min_reduction=min_reduction,
        max_slope_ratio=max_slope_ratio,
        min_overlap_ratio=min_overlap_ratio,
    )

    def process_group(group):
        group = group.sort_values("date").reset_index(drop=True)
        group = _ensure_human_triangle_columns(group)

        high_idx = argrelextrema(group["high"].values, np.greater, order=order)[0]
        low_idx = argrelextrema(group["low"].values, np.less, order=order)[0]
        if len(high_idx) < 2 or len(low_idx) < 2:
            return group

        atr_values = _calc_atr(group)
        marked_fuzzy_structures = set()

        for i in range(len(group)):
            curr_highs = high_idx[high_idx < i]
            curr_lows = low_idx[low_idx < i]
            if len(curr_highs) < 2 or len(curr_lows) < 2:
                continue

            best_fuzzy = None
            for window in _iter_fuzzy_pivot_windows(
                curr_highs,
                curr_lows,
                pivot_window=fuzzy_pivot_window,
                max_end_skip=fuzzy_max_end_skip,
            ):
                fuzzy_result = _evaluate_fuzzy_triangle(
                    group,
                    window,
                    atr_values,
                    min_reduction=fuzzy_min_reduction,
                    min_overlap_ratio=fuzzy_min_overlap_ratio,
                    price_tolerance=fuzzy_price_tolerance,
                    atr_multiplier=fuzzy_atr_multiplier,
                    touch_multiplier=fuzzy_touch_multiplier,
                    max_break_ratio=fuzzy_max_break_ratio,
                    max_break_magnitude_ratio=fuzzy_max_break_magnitude_ratio,
                    min_close_inside_ratio=fuzzy_min_close_inside_ratio,
                    min_score=fuzzy_min_score,
                    min_span=fuzzy_min_span,
                )
                if fuzzy_result is None or fuzzy_result["structure_id"] in marked_fuzzy_structures:
                    continue
                if best_fuzzy is None or fuzzy_result["score"] > best_fuzzy["score"]:
                    best_fuzzy = fuzzy_result

            if best_fuzzy is not None:
                _mark_fuzzy_triangle(group, i, best_fuzzy)
                marked_fuzzy_structures.add(best_fuzzy["structure_id"])

        return group

    return df_source.groupby("stock_id", group_keys=False).apply(process_group)


import plotly.graph_objects as go
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw_multiple_triangles_safe(df_source, df_signals, n_plots=9):
    """
    df_source: 原始完整大表 (包含所有歷史資料)
    df_signals: 過濾後只有 True 的標籤表
    """
    # 如果日期是字串，轉成 datetime（避免 .date() 呼叫失敗）
    if "date" in df_source.columns and not pd.api.types.is_datetime64_any_dtype(df_source["date"]):
        df_source = df_source.copy()
        df_source["date"] = pd.to_datetime(df_source["date"])
    if "date" in df_signals.columns and not pd.api.types.is_datetime64_any_dtype(df_signals["date"]):
        df_signals = df_signals.copy()
        df_signals["date"] = pd.to_datetime(df_signals["date"])

    # 按照日期由新到舊排序
    all_matches = df_signals.sort_values("date", ascending=False)

    if all_matches.empty:
        print("沒有符合條件的訊號。")
        return

    n_plots = min(len(all_matches), n_plots)
    rows = (n_plots + 2) // 3

    # 建立子圖標題（使用已轉換的 datetime）
    subplot_titles = []
    for _, r in all_matches.head(n_plots).iterrows():
        try:
            title_date = r["date"].date()
        except Exception:
            title_date = pd.to_datetime(r["date"]).date()
        match_type = r.get("triangle_match_type", "")
        score = r.get("triangle_score", np.nan)
        score_text = f" | score={score:.2f}" if pd.notna(score) else ""
        type_text = f" | {match_type}" if pd.notna(match_type) and match_type else ""
        subplot_titles.append(f"{r['stock_id']} | {title_date}{type_text}{score_text}")

    fig = make_subplots(rows=rows, cols=3, subplot_titles=subplot_titles)

    for idx, (original_idx, target) in enumerate(all_matches.head(n_plots).iterrows()):
        r, c = (idx // 3) + 1, (idx % 3) + 1

        # 取出該股票的完整時間序列，並重置 index 以使用位置索引
        stock_data = df_source[df_source["stock_id"] == target["stock_id"]].sort_values("date").reset_index(drop=True)

        # 設定繪圖範圍 (取前後緩衝)
        point_cols = ["h1_idx", "h2_idx", "h3_idx", "l1_idx", "l2_idx", "l3_idx"]
        valid_points = [
            int(target[col])
            for col in point_cols
            if col in target.index and pd.notna(target[col]) and 0 <= int(target[col]) < len(stock_data)
        ]
        if not valid_points:
            continue

        p_min = min(valid_points)
        p_max = max(valid_points)

        # 使用 iloc 前後切片，確保 K 線連續；加 1 包含 p_max
        df_crop = stock_data.iloc[max(0, p_min - 10) : min(len(stock_data) - 1, p_max + 20) + 1]

        # 繪製 K 線（以日期為 x 軸）
        fig.add_trace(
            go.Candlestick(
                x=df_crop["date"],
                open=df_crop["open"],
                high=df_crop["high"],
                low=df_crop["low"],
                close=df_crop["close"],
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # 繪製紅綠線。fuzzy 型態用擬合線，strict 型態則等同原本兩點連線。
        for line_name, point_names, color, price_col in [
            ("upper", ["h1_idx", "h2_idx", "h3_idx"], "red", "high"),
            ("lower", ["l1_idx", "l2_idx", "l3_idx"], "green", "low"),
        ]:
            line_points = [
                int(target[col])
                for col in point_names
                if col in target.index and pd.notna(target[col]) and 0 <= int(target[col]) < len(stock_data)
            ]
            if len(line_points) < 2:
                continue

            x_positions = [min(line_points), max(line_points)]
            if (
                f"{line_name}_slope" in target.index
                and f"{line_name}_intercept" in target.index
                and pd.notna(target[f"{line_name}_slope"])
                and pd.notna(target[f"{line_name}_intercept"])
            ):
                slope = target[f"{line_name}_slope"]
                intercept = target[f"{line_name}_intercept"]
                y_values = [slope * p + intercept for p in x_positions]
            else:
                y_values = [stock_data.iloc[p][price_col] for p in x_positions]

            fig.add_trace(
                go.Scatter(
                    x=[stock_data.iloc[p]["date"] for p in x_positions],
                    y=y_values,
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=[stock_data.iloc[p]["date"] for p in line_points],
                    y=[stock_data.iloc[p][price_col] for p in line_points],
                    mode="markers",
                    marker=dict(color=color, size=7),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(height=250 * rows, template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


def find_triangle(df):
    # stocks = parquet_db.query_stocks_ids_list()
    df_triangle = detect_human_triangle(
        df,
        order=7,
        max_slope_ratio=5,
        min_overlap_ratio=0.4,
        min_reduction=0.3,
    )
    return df_triangle


def find_good_triangle(df):
    return detect_human_triangle(
        df,
        order=7,
        max_slope_ratio=5,
        min_overlap_ratio=0.4,
        min_reduction=0.3,
    )


def find_human_triangle(df):
    '''
    重新偵測三角形，並只回傳 is_human_triangle=True 的訊號列。

    human_triangle = 原本 refined 三角形 + fuzzy 模糊三角形合併後的結果。
    注意：這個函式會呼叫 find_triangle(df)，所以會重新跑偵測，資料大時會比較慢。
    如果已經有 df_source 或 cache，請改用 add_triangle_compare_columns(df_source) 後自行篩選。
    '''
    df_source = find_triangle(df)
    return df_source[df_source["is_human_triangle"] == True]


def find_human_only(df, overlap_filter="all", min_overlap_ratio=0.8):
    '''
    重新偵測三角形，並只回傳 human_only 的訊號列。

    human_only = is_human_triangle=True 且 is_refined_triangle=False，
    也就是 fuzzy 模糊比對新增出來、不是原本 refined 就抓到的三角形。

    overlap_filter 可選：
        "all"          : 回傳全部 human_only
        "overlap_only" : 只回傳與其他 human_only 高度重疊的三角形
        "non_overlap"  : 只回傳沒有高度重疊的三角形
        "keep_best"    : 高度重疊群組中只回傳 score 較高、日期較新的代表

    min_overlap_ratio 是重疊門檻，例如 0.8 代表 x 區間重疊 >= 較短三角形的 80%。
    注意：這個函式會重新跑偵測；如果已經有快取，使用 load_or_create_triangle_cache() 會更快。
    '''
    df_source = find_triangle(df)
    df_source = add_triangle_compare_columns(df_source)

    if overlap_filter == "all":
        return df_source[df_source["human_only"] == True]

    _, df_human_only, _ = filter_triangle_by_overlap(
        df_source,
        signal_col="human_only",
        overlap_filter=overlap_filter,
        min_overlap_ratio=min_overlap_ratio,
    )
    return df_human_only


def add_triangle_compare_columns(df_source):
    '''
    在已偵測完成的 df_source 上補比較用欄位，不會重新偵測。

    輸入 df_source 需已包含：
        is_triangle
        is_refined_triangle
        is_fuzzy_triangle
        is_human_triangle

    新增欄位：
        fuzzy_only = is_fuzzy_triangle 且不是 is_refined_triangle
        human_only = is_human_triangle 且不是 is_refined_triangle

    這個函式會保留完整 K 線資料列，只是加欄位。
    適合用在 cache / detect_human_triangle() 跑完後，避免重複跑慢速偵測。
    '''
    work = df_source.copy()
    for col in ["is_triangle", "is_refined_triangle", "is_fuzzy_triangle", "is_human_triangle"]:
        work[col] = work[col].fillna(False).astype(bool)

    work["fuzzy_only"] = work["is_fuzzy_triangle"] & ~work["is_refined_triangle"]
    work["human_only"] = work["is_human_triangle"] & ~work["is_refined_triangle"]
    return work


def load_or_create_triangle_cache(
    cache_path="triangle_human_cache.pkl",
    refresh=False,
    sample_size=500,
    start_date="2025-01-01",
    end_date="2025-12-01",
):
    if not refresh:
        try:
            df_source = pd.read_pickle(cache_path)
            print(f"讀取快取：{cache_path}")
            return add_triangle_compare_columns(df_source)
        except FileNotFoundError:
            pass

    stocks = random.sample(parquet_db.query_stocks_no_etf(), sample_size)
    df = parquet_db.query_price(stocks, start_date, end_date)
    df_source = detect_human_triangle(
        df,
        order=7,
        max_slope_ratio=5,
        min_overlap_ratio=0.4,
        min_reduction=0.3,
    )
    df_source = add_triangle_compare_columns(df_source)
    df_source.to_pickle(cache_path)
    print(f"已存快取：{cache_path}")
    return df_source


def draw_random_human_only(
    df_source,
    n_plots=18,
    random_state=None,
    overlap_filter="all",
    overlap_threshold=0.8,
):
    df_source, df_human_only, _ = filter_triangle_by_overlap(
        df_source,
        signal_col="human_only",
        overlap_filter=overlap_filter,
        min_overlap_ratio=overlap_threshold,
    )
    if df_human_only.empty:
        print(f"沒有 human_only 訊號。overlap_filter={overlap_filter}")
        return

    n = min(n_plots, len(df_human_only))
    df_random = df_human_only.sample(n=n, random_state=random_state)
    print(
        f"draw human_only: overlap_filter={overlap_filter}, overlap_threshold={overlap_threshold}, count={len(df_human_only)}"
    )
    print(df_random["triangle_match_type"].value_counts().head(10))
    draw_multiple_triangles_safe(df_source, df_random, n_plots=n)


def add_triangle_span_columns(df_source):
    work = add_triangle_compare_columns(df_source)
    idx_cols = [col for col in ["h1_idx", "h2_idx", "h3_idx", "l1_idx", "l2_idx", "l3_idx"] if col in work.columns]
    date_cols = [
        col for col in ["h1_date", "h2_date", "h3_date", "l1_date", "l2_date", "l3_date"] if col in work.columns
    ]

    work["triangle_start_idx"] = work[idx_cols].min(axis=1)
    work["triangle_end_idx"] = work[idx_cols].max(axis=1)
    work["triangle_span_bars"] = work["triangle_end_idx"] - work["triangle_start_idx"] + 1

    for col in date_cols:
        work[col] = pd.to_datetime(work[col], errors="coerce")
    work["triangle_start_date"] = work[date_cols].min(axis=1)
    work["triangle_end_date"] = work[date_cols].max(axis=1)
    return work


def _build_triangle_overlap_pairs(signals, min_overlap_ratio):
    overlap_rows = []
    for stock_id, group in signals.groupby("stock_id"):
        group = group.sort_values(["triangle_start_idx", "triangle_end_idx", "date"]).reset_index()
        active = []

        for _, row in group.iterrows():
            start = float(row["triangle_start_idx"])
            end = float(row["triangle_end_idx"])
            active = [prev for prev in active if float(prev["triangle_end_idx"]) >= start]

            for prev in active:
                prev_start = float(prev["triangle_start_idx"])
                prev_end = float(prev["triangle_end_idx"])
                overlap_start = max(prev_start, start)
                overlap_end = min(prev_end, end)
                overlap_bars = max(0.0, overlap_end - overlap_start + 1)
                if overlap_bars <= 0:
                    continue

                span_a = prev_end - prev_start + 1
                span_b = end - start + 1
                overlap_ratio_short = overlap_bars / min(span_a, span_b)
                overlap_ratio_union = overlap_bars / (max(prev_end, end) - min(prev_start, start) + 1)
                if overlap_ratio_short < min_overlap_ratio:
                    continue

                overlap_rows.append(
                    {
                        "stock_id": stock_id,
                        "row_id_a": prev["index"],
                        "row_id_b": row["index"],
                        "date_a": prev["date"],
                        "date_b": row["date"],
                        "start_date_a": prev.get("triangle_start_date", pd.NaT),
                        "end_date_a": prev.get("triangle_end_date", pd.NaT),
                        "start_date_b": row.get("triangle_start_date", pd.NaT),
                        "end_date_b": row.get("triangle_end_date", pd.NaT),
                        "type_a": prev.get("triangle_match_type", ""),
                        "type_b": row.get("triangle_match_type", ""),
                        "score_a": prev.get("triangle_score", np.nan),
                        "score_b": row.get("triangle_score", np.nan),
                        "start_idx_a": prev_start,
                        "end_idx_a": prev_end,
                        "start_idx_b": start,
                        "end_idx_b": end,
                        "overlap_bars": overlap_bars,
                        "overlap_ratio_short": overlap_ratio_short,
                        "overlap_ratio_union": overlap_ratio_union,
                    }
                )

            active.append(row)

    return pd.DataFrame(overlap_rows)


def add_triangle_overlap_info(df_source, signal_col="human_only", min_overlap_ratio=0.8):
    work = add_triangle_span_columns(df_source).reset_index(drop=True)
    signals = work[
        work[signal_col].fillna(False) & work["triangle_start_idx"].notna() & work["triangle_end_idx"].notna()
    ].copy()
    overlaps = _build_triangle_overlap_pairs(signals, min_overlap_ratio)

    work["overlap_threshold"] = min_overlap_ratio
    work["overlap_pair_count"] = 0
    work["max_overlap_ratio_short"] = 0.0
    work["max_overlap_ratio_union"] = 0.0
    work["overlap_is_duplicate"] = False
    work["overlap_group_id"] = None
    work["overlap_group_size"] = 0
    work["overlap_rank"] = np.nan
    work["overlap_keep_best"] = False

    signal_idx = signals.index.tolist()
    if not signal_idx:
        return work, overlaps

    parent = {idx: idx for idx in signal_idx}

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    if not overlaps.empty:
        pair_counts = pd.concat([overlaps["row_id_a"], overlaps["row_id_b"]], ignore_index=True).value_counts()
        ratio_short = pd.concat(
            [
                overlaps[["row_id_a", "overlap_ratio_short"]].rename(columns={"row_id_a": "row_id"}),
                overlaps[["row_id_b", "overlap_ratio_short"]].rename(columns={"row_id_b": "row_id"}),
            ],
            ignore_index=True,
        )
        ratio_union = pd.concat(
            [
                overlaps[["row_id_a", "overlap_ratio_union"]].rename(columns={"row_id_a": "row_id"}),
                overlaps[["row_id_b", "overlap_ratio_union"]].rename(columns={"row_id_b": "row_id"}),
            ],
            ignore_index=True,
        )

        for _, row in overlaps.iterrows():
            union(int(row["row_id_a"]), int(row["row_id_b"]))

        work.loc[pair_counts.index, "overlap_pair_count"] = pair_counts.values
        work.loc[pair_counts.index, "overlap_is_duplicate"] = True
        work.loc[ratio_short.groupby("row_id")["overlap_ratio_short"].max().index, "max_overlap_ratio_short"] = (
            ratio_short.groupby("row_id")["overlap_ratio_short"].max().values
        )
        work.loc[ratio_union.groupby("row_id")["overlap_ratio_union"].max().index, "max_overlap_ratio_union"] = (
            ratio_union.groupby("row_id")["overlap_ratio_union"].max().values
        )

    roots = pd.Series({idx: find(idx) for idx in signal_idx}, name="root")
    root_to_group = {root: f"overlap_{n + 1}" for n, root in enumerate(sorted(roots.unique()))}
    work.loc[signal_idx, "overlap_group_id"] = [root_to_group[roots.loc[idx]] for idx in signal_idx]
    group_sizes = work.loc[signal_idx].groupby("overlap_group_id").size()
    work.loc[signal_idx, "overlap_group_size"] = work.loc[signal_idx, "overlap_group_id"].map(group_sizes)

    rank_source = work.loc[signal_idx].copy()
    rank_source["_score_sort"] = rank_source["triangle_score"].fillna(-1)
    rank_source["_date_sort"] = pd.to_datetime(rank_source["date"], errors="coerce")
    rank_source = rank_source.sort_values(
        ["overlap_group_id", "_score_sort", "_date_sort", "triangle_span_bars"],
        ascending=[True, False, False, False],
    )
    ranks = rank_source.groupby("overlap_group_id").cumcount() + 1
    work.loc[rank_source.index, "overlap_rank"] = ranks.values
    work.loc[rank_source.index, "overlap_keep_best"] = ranks.values == 1
    return work, overlaps


def filter_triangle_by_overlap(df_source, signal_col="human_only", overlap_filter="all", min_overlap_ratio=0.8):
    work, overlaps = add_triangle_overlap_info(
        df_source,
        signal_col=signal_col,
        min_overlap_ratio=min_overlap_ratio,
    )
    signals = work[work[signal_col].fillna(False)]

    if overlap_filter == "all":
        filtered = signals
    elif overlap_filter == "overlap_only":
        filtered = signals[signals["overlap_is_duplicate"]]
    elif overlap_filter == "non_overlap":
        filtered = signals[~signals["overlap_is_duplicate"]]
    elif overlap_filter == "keep_best":
        filtered = signals[signals["overlap_keep_best"]]
    else:
        raise ValueError("overlap_filter must be one of: all, overlap_only, non_overlap, keep_best")

    return work, filtered, overlaps


def check_triangle_overlap(
    df_source,
    signal_col="human_only",
    min_overlap_ratio=0.5,
    save_csv=True,
    output_prefix="triangle_overlap",
):
    work = add_triangle_span_columns(df_source).reset_index(drop=True)
    signals = work[
        work[signal_col].fillna(False) & work["triangle_start_idx"].notna() & work["triangle_end_idx"].notna()
    ].copy()
    overlaps = _build_triangle_overlap_pairs(signals, min_overlap_ratio)
    if overlaps.empty:
        summary = pd.DataFrame(
            [
                {
                    "signal_col": signal_col,
                    "signals": len(signals),
                    "stocks": signals["stock_id"].nunique(),
                    "overlap_pairs": 0,
                    "overlapped_signals": 0,
                    "overlap_stocks": 0,
                    "overlap_pair_ratio": 0.0,
                    "overlapped_signal_ratio": 0.0,
                }
            ]
        )
        by_stock = pd.DataFrame(
            columns=["stock_id", "signals", "overlap_pairs", "overlapped_signals", "max_overlap_ratio_short"]
        )
    else:
        overlap_signal_ids = pd.concat(
            [
                overlaps[["stock_id", "row_id_a"]].rename(columns={"row_id_a": "row_id"}),
                overlaps[["stock_id", "row_id_b"]].rename(columns={"row_id_b": "row_id"}),
            ],
            ignore_index=True,
        ).drop_duplicates()
        overlapped_signal_count = len(overlap_signal_ids)
        overlap_signal_counts = overlap_signal_ids.groupby("stock_id").size().rename("overlapped_signals").reset_index()
        by_stock = (
            overlaps.groupby("stock_id")
            .agg(
                overlap_pairs=("stock_id", "size"),
                max_overlap_ratio_short=("overlap_ratio_short", "max"),
                avg_overlap_ratio_short=("overlap_ratio_short", "mean"),
            )
            .reset_index()
        )
        signal_counts = signals.groupby("stock_id").size().rename("signals").reset_index()
        by_stock = (
            signal_counts.merge(by_stock, on="stock_id", how="left")
            .merge(overlap_signal_counts, on="stock_id", how="left")
            .fillna({"overlap_pairs": 0, "max_overlap_ratio_short": 0, "avg_overlap_ratio_short": 0})
            .fillna({"overlapped_signals": 0})
            .sort_values(["overlap_pairs", "max_overlap_ratio_short", "signals"], ascending=False)
        )
        summary = pd.DataFrame(
            [
                {
                    "signal_col": signal_col,
                    "signals": len(signals),
                    "stocks": signals["stock_id"].nunique(),
                    "overlap_pairs": len(overlaps),
                    "overlapped_signals": overlapped_signal_count,
                    "overlap_stocks": int((by_stock["overlap_pairs"] > 0).sum()),
                    "overlap_pair_ratio": len(overlaps) / max(len(signals), 1),
                    "overlapped_signal_ratio": overlapped_signal_count / max(len(signals), 1),
                }
            ]
        )

    if save_csv:
        summary.to_csv(f"{output_prefix}_summary.csv", index=False)
        by_stock.to_csv(f"{output_prefix}_by_stock.csv", index=False)
        overlaps.to_csv(f"{output_prefix}_pairs.csv", index=False)

    print(summary.to_string(index=False))
    print(by_stock.head(20).to_string(index=False))
    return overlaps, summary, by_stock


def compare_triangle_methods(df, save_csv=True, output_prefix="triangle_method"):
    df_source = find_triangle(df)
    work = add_triangle_compare_columns(df_source)

    summary = pd.DataFrame(
        [
            {
                "total_rows": len(work),
                "strict_hits": int(work["is_triangle"].sum()),
                "refined_hits": int(work["is_refined_triangle"].sum()),
                "fuzzy_hits": int(work["is_fuzzy_triangle"].sum()),
                "fuzzy_only_hits": int(work["fuzzy_only"].sum()),
                "human_hits": int(work["is_human_triangle"].sum()),
                "human_only_hits": int(work["human_only"].sum()),
            }
        ]
    )

    by_stock = (
        work.groupby("stock_id")
        .agg(
            strict_hits=("is_triangle", "sum"),
            refined_hits=("is_refined_triangle", "sum"),
            fuzzy_hits=("is_fuzzy_triangle", "sum"),
            fuzzy_only_hits=("fuzzy_only", "sum"),
            human_hits=("is_human_triangle", "sum"),
            human_only_hits=("human_only", "sum"),
        )
        .reset_index()
        .sort_values(["human_hits", "fuzzy_only_hits", "strict_hits"], ascending=False)
    )

    by_type = (
        work.loc[work["is_human_triangle"], "triangle_match_type"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("triangle_match_type")
        .reset_index(name="hits")
    )

    if save_csv:
        summary.to_csv(f"{output_prefix}_summary.csv", index=False)
        by_stock.to_csv(f"{output_prefix}_comparison_by_stock.csv", index=False)
        by_type.to_csv(f"{output_prefix}_comparison_by_type.csv", index=False)
        work[work["is_triangle"]].to_csv("triangle_strict_results.csv", index=False)
        work[work["is_fuzzy_triangle"]].to_csv("triangle_fuzzy_results.csv", index=False)
        work[work["is_human_triangle"]].to_csv("triangle_human_results.csv", index=False)
        work[work["human_only"]].to_csv("triangle_human_only_results.csv", index=False)

    print(summary.to_string(index=False))
    print(by_type.head(20).to_string(index=False))
    return work, summary, by_stock, by_type


# 呼叫方式
# draw_multiple_triangles(df_triangle, n_plots=9)
def test():
    overlap_threshold = 0.8
    df_source = load_or_create_triangle_cache(
        cache_path="triangle_human_cache.pkl",
        refresh=False,
        sample_size=500,
        start_date="2025-01-01",
        end_date="2025-12-01",
    )

    df_refined_triangle = df_source[df_source["is_refined_triangle"] == True]
    df_fuzzy_triangle = df_source[df_source["is_fuzzy_triangle"] == True]
    df_human_triangle = df_source[df_source["is_human_triangle"] == True]
    df_human_only = df_source[df_source["human_only"]]
    df_triangle = df_source[df_source["is_triangle"] == True]
    print(
        "找到標記的三角形數量 total / strict / refined / fuzzy / human / human_only：",
        len(df_source),
        len(df_triangle),
        len(df_refined_triangle),
        len(df_fuzzy_triangle),
        len(df_human_triangle),
        len(df_human_only),
    )
    print(df_human_only["triangle_match_type"].value_counts().head(10))
    # print(df_refined_triangle.head().T)

    check_triangle_overlap(
        df_source,
        signal_col="human_only",
        min_overlap_ratio=overlap_threshold,
        output_prefix=f"triangle_overlap_{int(overlap_threshold * 100)}",
    )
    draw_random_human_only(
        df_source,
        n_plots=18,
        overlap_filter="all",
        overlap_threshold=overlap_threshold,
    )


def find_refined_triangle(df):
    df_source = detect_strict_triangle(
        df,
        order=4,
        max_slope_ratio=99,
        min_overlap_ratio=0.35,
        min_reduction=0.25,
    )

    return df_source[df_source["is_triangle"] == True]
    # return df_source[df_source["is_refined_triangle"] == True]


if __name__ == "__main__":
    test()
