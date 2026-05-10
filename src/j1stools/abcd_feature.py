import pandas as pd
import numpy as np


def detect_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    趨勢相關特徵，用來輔助 N 字型訊號過濾。
    N 字在上升趨勢裡成功率更高，這些特徵幫助模型判斷趨勢強弱。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, high, low, volume

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_trend_ 開頭特徵欄位
    """
    close_wide = df.pivot(index="date", columns="stock_id", values="close")
    high_wide = df.pivot(index="date", columns="stock_id", values="high")
    low_wide = df.pivot(index="date", columns="stock_id", values="low")
    volume_wide = df.pivot(index="date", columns="stock_id", values="volume")

    # ── Feature 1: f_trend_ma20_slope ────────────────────────────────── #
    # MA20 斜率：(今日MA20 - N日前MA20) / N日前MA20
    # 正值代表均線向上，趨勢健康
    ma20 = close_wide.rolling(20).mean()
    f_trend_ma20_slope = (ma20 - ma20.shift(5)) / (ma20.shift(5) + 1e-9)

    # ── Feature 2: f_trend_close_vs_ma20 ─────────────────────────────── #
    # 收盤相對 MA20 的位置
    # > 0 代表站上均線，N 字在均線上方成功率更高
    f_trend_close_vs_ma20 = (close_wide - ma20) / (ma20 + 1e-9)

    # ── Feature 3: f_trend_close_vs_ma60 ─────────────────────────────── #
    # 收盤相對 MA60 的位置（中期趨勢）
    ma60 = close_wide.rolling(60).mean()
    f_trend_close_vs_ma60 = (close_wide - ma60) / (ma60 + 1e-9)

    # ── Feature 4: f_trend_ma20_vs_ma60 ──────────────────────────────── #
    # 短均線 vs 長均線（黃金交叉/死亡交叉）
    # > 0 代表 MA20 在 MA60 上方，多頭排列
    f_trend_ma20_vs_ma60 = (ma20 - ma60) / (ma60 + 1e-9)

    # ── Feature 5: f_trend_high_breakout ─────────────────────────────── #
    # 近期是否創 N 日新高（突破前高是上升趨勢的特徵）
    rolling_high_60 = high_wide.rolling(60).max().shift(1)
    f_trend_high_breakout = (close_wide - rolling_high_60) / (rolling_high_60 + 1e-9)

    # ── Feature 6: f_trend_volume_trend ──────────────────────────────── #
    # 成交量趨勢：近期均量 vs 前期均量
    # > 1 代表量能放大，趨勢有支撐
    vol_ma10 = volume_wide.rolling(10).mean()
    vol_ma30 = volume_wide.rolling(30).mean()
    f_trend_volume_trend = vol_ma10 / (vol_ma30 + 1e-9)

    # ── Feature 7: f_trend_atr_ratio ─────────────────────────────────── #
    # ATR 比率：近期波動 vs 長期波動
    # N 字啟動時通常波動放大
    daily_range = high_wide - low_wide
    atr10 = daily_range.rolling(10).mean()
    atr30 = daily_range.rolling(30).mean()
    f_trend_atr_ratio = atr10 / (atr30 + 1e-9)

    # ── Feature 8: f_trend_momentum_20 ───────────────────────────────── #
    # 20日動能（價格動量）
    f_trend_momentum_20 = close_wide.pct_change(20)

    # ── Feature 9: f_trend_rsi_14 ────────────────────────────────────── #
    # RSI(14)：衡量超買超賣，N 字啟動時 RSI 通常從 40~60 向上
    delta = close_wide.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    f_trend_rsi_14 = 100 - (100 / (1 + rs))

    # ── Feature 10: f_trend_above_ma_days ────────────────────────────── #
    # 近 20 根中有幾根收盤在 MA20 以上（趨勢穩定度）
    above_ma20 = (close_wide > ma20).astype(float)
    f_trend_above_ma_days = above_ma20.rolling(20).sum() / 20

    # ── 整合回長表格 ──────────────────────────────────────────────────── #
    feature_wides = {
        "f_trend_ma20_slope": f_trend_ma20_slope,
        "f_trend_close_vs_ma20": f_trend_close_vs_ma20,
        "f_trend_close_vs_ma60": f_trend_close_vs_ma60,
        "f_trend_ma20_vs_ma60": f_trend_ma20_vs_ma60,
        "f_trend_high_breakout": f_trend_high_breakout,
        "f_trend_volume_trend": f_trend_volume_trend,
        "f_trend_atr_ratio": f_trend_atr_ratio,
        "f_trend_momentum_20": f_trend_momentum_20,
        "f_trend_rsi_14": f_trend_rsi_14,
        "f_trend_above_ma_days": f_trend_above_ma_days,
    }

    feature_longs = []
    for feat_name, feat_wide in feature_wides.items():
        feat_long = feat_wide.stack().reset_index()
        feat_long.columns = ["date", "stock_id", feat_name]
        feature_longs.append(feat_long.set_index(["date", "stock_id"]))

    features_df = pd.concat(feature_longs, axis=1).reset_index()
    result = df.merge(features_df, on=["date", "stock_id"], how="left")
    return result
