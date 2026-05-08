import pandas as pd
import numpy as np
import talib
import pandas as pd
import pandas_ta as ta
import numpy as np

import pandas as pd
import pandas_ta as ta
import numpy as np


def f_macd_hist_divergences(df: pd.DataFrame):
    # 1. 確保排序
    df = df.sort_values(by=["date", "stock_id"])

    # 2. 轉為寬表格
    close_wide = df.pivot(index="date", columns="stock_id", values="close")

    # 3. 計算 MACD 組件 (Fast, Slow, Hist)
    def get_macd_components(series):
        if len(series) < 35:
            return pd.DataFrame(np.nan, index=series.index, columns=["macd", "signal", "hist"])
        m = ta.macd(series, fast=12, slow=26, signal=9)
        return pd.DataFrame({"macd": m.iloc[:, 0], "signal": m.iloc[:, 2], "hist": m.iloc[:, 1]})

    macd_results = {sid: get_macd_components(close_wide[sid]) for sid in close_wide.columns}
    macd_wide = pd.DataFrame({sid: macd_results[sid]["macd"] for sid in close_wide.columns})
    signal_wide = pd.DataFrame({sid: macd_results[sid]["signal"] for sid in close_wide.columns})
    hist_wide = pd.DataFrame({sid: macd_results[sid]["hist"] for sid in close_wide.columns})

    # --- 特徵 1：Hist 底背離 ---
    n = 5
    is_price_low = close_wide == close_wide.rolling(2 * n + 1, center=True).min()

    def find_prev_val(val_df, mask_df):
        points = val_df.where(mask_df)
        return points.shift(1).ffill(limit=30)

    prev_close_at_low = find_prev_val(close_wide, is_price_low)
    prev_hist_at_low = find_prev_val(hist_wide, is_price_low)
    f1_hist_div = (is_price_low & (close_wide < prev_close_at_low) & (hist_wide > prev_hist_at_low)).astype(int)

    # --- 特徵 2：MACD & Signal > 0 ---
    f2_trend_above_zero = ((macd_wide > 0) & (signal_wide > 0)).astype(int)

    # --- 特徵 3：快線 > 慢線 (DIF > DEA) ---
    f3_fast_above_slow = (macd_wide > signal_wide).astype(int)

    # --- 特徵 4：回看 3 天內 Hist 最低時的價格守穩 ---
    # 1. 取得 3 天內的 Hist 最小值矩陣
    hist_min_3d = hist_wide.rolling(window=3).min()

    # 2. 找出「當下就是 3 天內最低 Hist」的時刻，並記下當時價格
    # 其餘時刻透過 ffill(limit=2) 向後遞延，確保我們拿到的永遠是「3天內最低Hist發生時」的價格
    price_at_min_hist = close_wide.where(hist_wide == hist_min_3d).ffill(limit=2)

    # 3. 檢查「這 3 天的收盤價」是否都沒跌破該價格
    # 我們比對這 3 天的最低收盤價是否 >= 該基準價
    f4_price_support = (close_wide.rolling(window=3).min() >= price_at_min_hist).astype(int)

    # ── 整合回長表格 ──────────────────────────────
    df_output = df.copy().set_index(["date", "stock_id"])

    df_output["f_macd_hist_div"] = f1_hist_div.stack(future_stack=True)
    df_output["f_macd_trend_up"] = f2_trend_above_zero.stack(future_stack=True)
    df_output["f_macd_bull_regime"] = f3_fast_above_slow.stack(future_stack=True)
    df_output["f_macd_price_support"] = f4_price_support.stack(future_stack=True)

    # 填補滾動計算產生的空值
    feat_cols = [col for col in df_output.columns if col.startswith("f_")]
    df_output[feat_cols] = df_output[feat_cols].fillna(0).astype(int)

    return df_output.reset_index()
