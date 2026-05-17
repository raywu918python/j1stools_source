import pandas as pd
import numpy as np
import talib
import pandas as pd
import pandas_ta as ta
import numpy as np

import pandas as pd
import pandas_ta as ta
import numpy as np


def f_macd_hist_divergences_with_atr(df: pd.DataFrame):
    # 1. 確保排序
    df = df.sort_values(by=["date", "stock_id"])

    # 2. 轉為寬表格 (需要 high, low, close 來算 ATR)
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")

    # 3. 計算 MACD 與 ATR 組件
    def get_indicators(group):
        # ATR (取 14 天)
        atr = ta.atr(group["high"], group["low"], group["close"], length=14)
        # MACD
        macd_df = ta.macd(group["close"], fast=12, slow=26, signal=9)

        res = pd.DataFrame(index=group.index)
        res["atr"] = atr
        if macd_df is not None:
            res["macd"] = macd_df.iloc[:, 0]
            res["signal"] = macd_df.iloc[:, 2]
            res["hist"] = macd_df.iloc[:, 1]
        return res

    # 批次處理指標 (使用長表格 groupby 再轉回寬表格，這在多指標計算時較穩健)
    indicators = df.groupby("stock_id", group_keys=False).apply(get_indicators, include_groups=False)

    # 轉回寬表格供矩陣運算
    df_temp = df[["date", "stock_id"]].copy()
    df_temp = pd.concat([df_temp, indicators], axis=1)

    atr_wide = df_temp.pivot(index="date", columns="stock_id", values="atr")
    macd_wide = df_temp.pivot(index="date", columns="stock_id", values="macd")
    signal_wide = df_temp.pivot(index="date", columns="stock_id", values="signal")
    hist_wide = df_temp.pivot(index="date", columns="stock_id", values="hist")

    # --- 特徵 1-3: (與先前相同，略過邏輯描述直接實作) ---
    n = 5
    is_price_low = close_wide == close_wide.rolling(2 * n + 1, center=True).min()

    def find_prev_val(val_df, mask_df):
        return val_df.where(mask_df).shift(1).ffill(limit=30)

    prev_close_at_low = find_prev_val(close_wide, is_price_low)
    prev_hist_at_low = find_prev_val(hist_wide, is_price_low)

    f1_hist_div = (is_price_low & (close_wide < prev_close_at_low) & (hist_wide > prev_hist_at_low)).astype(int)
    f2_trend_above_zero = ((macd_wide > 0) & (signal_wide > 0)).astype(int)
    f3_fast_above_slow = (macd_wide > signal_wide).astype(int)

    # --- 特徵 4: Hist 最低點後的價格守穩 ---
    hist_min_3d = hist_wide.rolling(window=3).min()
    price_at_min_hist = close_wide.where(hist_wide == hist_min_3d).ffill(limit=2)
    f4_price_support = (close_wide.rolling(window=3).min() >= price_at_min_hist).astype(int)

    # --- 特徵 5: ATR 波動率過濾 (核心新增) ---
    # 邏輯：當前 ATR 必須大於其 20 日平均（代表波動率正在放大，非死魚盤）
    atr_ma = atr_wide.rolling(20).mean()
    f5_atr_filter = (atr_wide > atr_ma).astype(int)

    # ── 整合回長表格 ──────────────────────────────
    df_output = df.copy().set_index(["date", "stock_id"])
    df_output["f_macd_hist_div"] = f1_hist_div.stack(future_stack=True)
    df_output["f_macd_trend_up"] = f2_trend_above_zero.stack(future_stack=True)
    df_output["f_macd_bull_regime"] = f3_fast_above_slow.stack(future_stack=True)
    df_output["f_macd_price_support"] = f4_price_support.stack(future_stack=True)
    df_output["f_atr_active"] = f5_atr_filter.stack(future_stack=True)

    # 處理 NaN
    feat_cols = [col for col in df_output.columns if col.startswith("f_")]
    df_output[feat_cols] = df_output[feat_cols].fillna(0).astype(int)

    return df_output.reset_index()


def f_macd_continuous_features(df: pd.DataFrame):
    df = df.sort_values(by=["date", "stock_id"])

    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")

    def get_indicators(group):
        atr = ta.atr(group["high"], group["low"], group["close"], length=14)
        macd_df = ta.macd(group["close"], fast=12, slow=26, signal=9)
        res = pd.DataFrame(index=group.index)
        res["atr"] = atr
        if macd_df is not None:
            res["macd"] = macd_df.iloc[:, 0]
            res["signal"] = macd_df.iloc[:, 2]
            res["hist"] = macd_df.iloc[:, 1]
        return res

    indicators = df.groupby("stock_id", group_keys=False).apply(get_indicators, include_groups=False)
    df_temp = pd.concat([df[["date", "stock_id"]], indicators], axis=1)

    atr_wide = df_temp.pivot(index="date", columns="stock_id", values="atr")
    macd_wide = df_temp.pivot(index="date", columns="stock_id", values="macd")
    signal_wide = df_temp.pivot(index="date", columns="stock_id", values="signal")
    hist_wide = df_temp.pivot(index="date", columns="stock_id", values="hist")

    # 避免除以零
    atr_safe = atr_wide.replace(0, np.nan)

    # ── 特徵 1: 背離強度 (hist 改善幅度 / ATR) ──────────────────
    n = 5
    is_price_low = close_wide == close_wide.rolling(2 * n + 1, center=True).min()

    def find_prev_val(val_df, mask_df):
        return val_df.where(mask_df).shift(1).ffill(limit=30)

    prev_hist_at_low = find_prev_val(hist_wide, is_price_low)
    prev_close_at_low = find_prev_val(close_wide, is_price_low)

    # 只在實際背離點計算，其他時間填 0
    div_raw = (hist_wide - prev_hist_at_low) / atr_safe
    at_low_with_div = is_price_low & (close_wide < prev_close_at_low) & (hist_wide > prev_hist_at_low)
    f1_div_strength = div_raw.where(at_low_with_div, 0.0)

    # ── 特徵 2: 趨勢強度 (macd + signal 之和 / ATR) ─────────────
    # 正值 = 多頭區域且越強；負值 = 空頭
    f2_trend_strength = (macd_wide + signal_wide) / atr_safe

    # ── 特徵 3: 動量（hist / ATR，已內含方向） ──────────────────
    # hist = macd - signal，正值代表快線在慢線上方且擴張
    f3_momentum = hist_wide / atr_safe

    # ── 特徵 4: 支撐邊際（rolling_min 距離支撐點 / ATR）──────────
    hist_min_3d = hist_wide.rolling(window=3).min()
    price_at_min_hist = close_wide.where(hist_wide == hist_min_3d).ffill(limit=2)
    rolling_min_close = close_wide.rolling(window=3).min()
    # 正值 = 守住支撐；負值 = 跌破
    f4_support_margin = (rolling_min_close - price_at_min_hist) / atr_safe

    # ── 特徵 5: ATR 動能比（log scale，以 0 為中心）─────────────
    atr_ma = atr_wide.rolling(20).mean()
    atr_ma_safe = atr_ma.replace(0, np.nan)
    # > 0 表示波動擴張；< 0 表示萎縮
    f5_atr_ratio = np.log(atr_safe / atr_ma_safe)

    # ── 限制極端值（tanh 或 clip 擇一）─────────────────────────
    def soft_clip(df_feat, scale=3.0):
        """tanh 壓縮：保留方向與相對大小，抑制極端值"""
        return np.tanh(df_feat / scale)

    f1 = soft_clip(f1_div_strength, scale=2.0)  # 背離訊號通常較小
    f2 = soft_clip(f2_trend_strength)
    f3 = soft_clip(f3_momentum)
    f4 = soft_clip(f4_support_margin)
    f5 = soft_clip(f5_atr_ratio, scale=1.0)  # log ratio 本身已較緊縮

    # ── 整合回長表格 ──────────────────────────────────────────
    df_output = df.copy().set_index(["date", "stock_id"])
    df_output["f_macd_hist_div_strength"] = f1.stack(future_stack=True)
    df_output["f_macd_trend_strength"] = f2.stack(future_stack=True)
    df_output["f_macd_momentum"] = f3.stack(future_stack=True)
    df_output["f_price_support_margin"] = f4.stack(future_stack=True)
    df_output["f_atr_ratio"] = f5.stack(future_stack=True)

    feat_cols = [c for c in df_output.columns if c.startswith("f_")]
    df_output[feat_cols] = df_output[feat_cols].fillna(0.0)

    return df_output.reset_index()
