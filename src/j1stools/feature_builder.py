from calendar import c
from time import time

import pandas as pd

from j1stools import abcd_feature, margin_feature, margin_ibbuysell_feature, parquet_db, power_feature
from j1stools.CONFIG import BaseDataBuilderConfig
from j1stools.TYPE import FEATURE_TYPE
from j1stools.lgbm_test_feature import lgbm_all_f_features, lgbm_feature
from j1stools.lite_db import margin
from j1stools.macdh_divergences import f_macd_hist_divergences_with_atr
from j1stools.obj_hv_feature import HvFeature
from j1stools.obj_ma_feature import MaFeature
from j1stools.obj_macd_feature import MacdFeature
from j1stools.obj_market_feature import MarketFeature
from j1stools.obj_random_feature import RandomFeature
from j1stools.obj_vwap_pvt_feature import VolumePriceFeature
from j1stools.train_flow import print_target_counts


def gen_feature(
    df: pd.DataFrame,
    feature_type: FEATURE_TYPE = FEATURE_TYPE.normal,
    dfs: list = None,
) -> pd.DataFrame:
    t1 = time()
    if df is not None:
        st = df["date"].min()
        end = df["date"].max()
    # df = VolumeFeature.init(df)
    # df = PriceFeature.add_feature(df)
    #

    # df = AtrFeature.add_feature(df)
    #############################################################
    if FEATURE_TYPE.margin_ibbuysell in feature_type:
        df = margin_ibbuysell_feature.add_feature(dfs[0], dfs[1], dfs[2])
    if FEATURE_TYPE.today in feature_type:
        # return generate_features_t1(df)
        df = generate_features_today(df)
    if FEATURE_TYPE.macd_is_continuous in feature_type:
        df = f_macd_continuous_features(df)
    if FEATURE_TYPE.macd in feature_type:
        df = f_macd_hist_divergences_with_atr(df)
    if FEATURE_TYPE.abcd in feature_type:
        # df = abcd_feature.detect_n_shape_features(df, seg=9)
        # df = abcd_feature.detect_trend_features(df)
        df = abcd_feature.detect_n_shape_features(df, seg=9)
    if FEATURE_TYPE.test_lgbm_feature in feature_type:
        market_df = parquet_db.query_price(["0050"], st, end)
        # df = lgbm_feature(df, market_df)
        df = lgbm_all_f_features(df, market_df)
    if FEATURE_TYPE.margin in feature_type:
        # df = abcd_feature.detect_trend_features(df)
        df = margin_feature.detect_short_squeeze_features(df)

        feature_cols = [
            #
            "close",
            "volume",
            "high",
            "low",
            "open",
            "stock_id",
            "date",
            # N 字特徵
            "f_n_ab_gain",
            "f_n_bc_retracement",
            "f_n_c_above_a",
            "f_n_d_breakout_strength",
            "f_n_cd_vs_ab_momentum",
            "f_n_close_vs_b",
            "f_n_structure_score",
            "f_n_volume_confirm",
            # 軋空特徵
            "f_sq_short_ratio",
            "f_sq_cover_days",
            "f_sq_short_growth",
            "f_sq_short_utilization",
            "f_sq_net_short",
            "f_sq_price_resilience",
            "f_sq_short_ma5_growth",
            "f_sq_offset_ratio",
            "f_sq_squeeze_score",
            "f_n_confirmed",
        ]

    if FEATURE_TYPE.power in feature_type:
        markget_df = parquet_db.query_price(["0050"], st, end)
        df = power_feature.detect_relative_strength_features(df, markget_df)
        # df = df[feature_cols]
    if FEATURE_TYPE.normal in feature_type:
        df = HvFeature.add_feature(df)
        df = MaFeature.add_feature(df)
        df = MacdFeature.add_feature(df)
        df = MarketFeature.add_feature(df)
        df = VolumePriceFeature.add_feature(df)
        # df = RandomFeature.add_feature(df)
        # df = f_macd_hist_divergences_with_atr(df)
    print(f"gen_feature: {time() - t1:.2f} 秒")

    f = df.select_dtypes(include="number").describe().T.round(2)
    print(f)
    f.to_csv("describe.csv")

    return df


def pick_feature(x, is_using_rfc=False):
    ft = [
        "f_HV_squeeze",
        "f_div_refined",
        "f_div_score",
        "f_bottom_slope",
        "f_hist_pos_direction",
        "f_hist_rise_days",
        "f_macdh_slope12",
        "f_macdh_slope23",
        "f_market_rs_10d",
        "f_market_rs_3d",
        "f_market_rs_1d",
        "f_market_stock_rsi",
        "f_market_ret3",
        "f_market_close_change",
        "f_vwap_gap",
        "f_vwap_roc",
        "f_pvt_gap",
        "f_pvt_roc",
    ]
    if is_using_rfc:
        ft.append("f_rfc")
    return x[ft]


import numpy as np
import pandas as pd


def generate_features_t1(df: pd.DataFrame):
    """計算短線特徵

    參數:
    df_dict (dict): 包含 'Open', 'High', 'Low', 'Close', 'Volume' 鍵值的字典，
                    每個值都是寬表格 (DataFrame: index=日期, columns=股票代碼)
    """

    df.sort_values(by=["date", "stock_id"], inplace=True)
    df_dict = df.pivot(index="date", columns="stock_id", values=["high", "low", "close", "volume"])

    close = df_dict["close"]
    high = df_dict["high"]
    low = df_dict["low"]
    volume = df_dict["volume"]

    ret_1d = close.pct_change(1)
    ret_3d = close.pct_change(3)

    # 2. v_ratio_5d : 今日成交量 / 過去 5 日均量
    ma5_volume = volume.rolling(5).mean()
    v_ratio_5d = volume / ma5_volume

    # 3. upper_shadow_pct : (最高價 - 收盤價) / 股價 (賣壓過濾)
    upper_shadow_pct = (high - close) / close

    # 4. dist_to_ma5 : (收盤價 - MA5) / MA5
    ma5_price = close.rolling(5).mean()
    dist_to_ma5 = (close - ma5_price) / ma5_price

    # 5. atr_ratio : 當前波動幅度 / ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(14).mean()
    current_vol = (high - low).abs()
    atr_ratio = current_vol / atr

    MA60 = close.rolling(60).mean()
    dist_to_ma60 = (close - MA60) / MA60

    # ----------------- 統一 Z-score 標準化 (視窗設為 60 天) -----------------
    def zscore_transform(df_item, window=60):
        mean = df_item.rolling(window).mean()
        std = df_item.rolling(window).std().replace(0, np.nan).fillna(1e-5)
        return ((df_item - mean) / std).clip(-3, 3)

    # 標準化會改變絕對數值的特徵（注意：本身就是比例的不用過度轉換，這裡對數值特徵標準化）
    z_dist_to_ma60 = zscore_transform(dist_to_ma60)
    z_dist_to_ma5 = zscore_transform(dist_to_ma5)
    z_v_ratio = zscore_transform(v_ratio_5d)
    z_atr_ratio = zscore_transform(atr_ratio)

    # ----------------- 堆疊為長表格 -----------------
    df = df.set_index(["date", "stock_id"])
    df["f_dist_to_ma60"] = z_dist_to_ma60.stack(future_stack=True)
    df["f_ret_1d"] = ret_1d.stack(future_stack=True)
    df["f_ret_3d"] = ret_3d.stack(future_stack=True)
    df["f_v_ratio_5d"] = z_v_ratio.stack(future_stack=True)
    df["f_upper_shadow_pct"] = upper_shadow_pct.stack(future_stack=True)
    df["f_dist_to_ma5"] = z_dist_to_ma5.stack(future_stack=True)
    df["f_atr_ratio"] = z_atr_ratio.stack(future_stack=True)
    df = df.reset_index()

    return df


def time_series_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """對寬表格進行時間序列的滾動 Z-score 標準化

    參數:
    - df: 寬表格 DataFrame，Index 為日期，Columns 為股票代碼
    - window: 計算平均與標準差的天數（例如 60 天）
    """
    # 計算滾動平均值與滾動標準差
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()

    # # 避免標準差為 0 造成除以零的錯誤（填補微小值）
    rolling_std = rolling_std.replace(0, np.nan).fillna(1e-5)

    # # 計算 Z-score
    zscore_df = (df - rolling_mean) / rolling_std

    zscore_df = zscore_df.clip(lower=-3, upper=3)

    return df


import pandas as pd
import numpy as np


def generate_features_today(df: pd.DataFrame):
    """
    價格動量（5個）
    f_ret_1d
    1日報酬率
    f_ret_3d
    3日報酬率
    f_ret_5d
    5日報酬率
    f_pos_10d
    現價在10日高低點的相對位置 (0~1)
    f_gap
    開盤缺口大小 (open/prev_close - 1)
    技術指標（6個）
    f_ema5_slope
    EMA5 斜率（5日變化率）
    f_ema_cross
    EMA5 / EMA20 比值（金叉強度）
    f_rsi6
    RSI(6) — 短線超買超賣
    f_rsi14
    RSI(14)
    f_macd_hist
    MACD histogram (12/26/9)
    f_bb_pct
    布林帶 %B 位置
    成交量（3個，你已有1個）
    f_v_ratio_5d
    ✓ 已有 — 量比5日均量
    f_v_ratio_10d
    量比10日均量
    f_obv_slope
    OBV 5日斜率（資金持續流入/流出）
    波動率（3個）
    f_atr5
    ATR(5) 正規化（/close）
    f_atr_ratio
    當日ATR / 10日均ATR（爆量突破偵測）
    f_hv5
    5日歷史波動率（收盤標準差）
    K線型態（3個）
    f_body_ratio
    實體大小 |close-open| / (high-low)
    f_upper_shadow
    上影線比例
    f_lower_shadow
    下影線比例（長下影 = 支撐強）
    """

    df.sort_values(by=["date", "stock_id"], inplace=True)
    df_dict = df.pivot(index="date", columns="stock_id", values=["open", "high", "low", "close", "volume"])

    open_ = df_dict["open"]
    close = df_dict["close"]
    high = df_dict["high"]
    low = df_dict["low"]
    volume = df_dict["volume"]

    # ── 價格動量 ──────────────────────────────────
    ret_1d = close.pct_change(1)
    ret_3d = close.pct_change(3)
    ret_5d = close.pct_change(5)

    roll10_high = high.rolling(10).max()
    roll10_low = low.rolling(10).min()
    pos_10d = (close - roll10_low) / (roll10_high - roll10_low + 1e-9)

    gap = open_ / close.shift(1) - 1

    # ── 技術指標 ──────────────────────────────────
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema5_slope = ema5.pct_change(5)
    ema_cross = ema5 / ema20

    def rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    rsi6 = rsi(close, 6)
    rsi14 = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── 成交量 ────────────────────────────────────
    ma5_vol = volume.rolling(5).mean()
    ma10_vol = volume.rolling(10).mean()
    v_ratio_5d = volume / (ma5_vol + 1e-9)
    v_ratio_10d = volume / (ma10_vol + 1e-9)

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_slope = obv.pct_change(5)

    # ── 波動率 ────────────────────────────────────
    tr = (
        pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        if False
        else None
    )  # 寬表格要逐欄計算

    def atr_wide(h, l, c, period):
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=0)
        # 寬表格用 apply
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        true_range = (
            pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            if False
            else tr1.where(tr1 >= tr2, tr2).where(lambda x: x >= tr3, tr3)
        )
        # 正確寬表格寫法：
        tr_w = pd.DataFrame(
            np.maximum(
                np.maximum((h.values - l.values), np.abs(h.values - c.shift(1).values)),
                np.abs(l.values - c.shift(1).values),
            ),
            index=h.index,
            columns=h.columns,
        )
        return tr_w.rolling(period).mean(), tr_w

    atr5_raw, tr_w = atr_wide(high, low, close, 5)
    atr5 = atr5_raw / (close + 1e-9)
    atr10_raw, _ = atr_wide(high, low, close, 10)
    atr_ratio = atr5_raw / (atr10_raw + 1e-9)
    hv5 = close.pct_change().rolling(5).std()

    # ── K線型態 ───────────────────────────────────
    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    body_ratio = body / candle_range
    body_top = np.maximum(close.values, open_.values)
    body_bottom = np.minimum(close.values, open_.values)

    upper_shadow = pd.DataFrame(
        (high.values - body_top) / (candle_range.values + 1e-9), index=high.index, columns=high.columns
    )

    lower_shadow = pd.DataFrame(
        (body_bottom - low.values) / (candle_range.values + 1e-9), index=low.index, columns=low.columns
    )

    # ── 堆疊回長表格 ──────────────────────────────
    df = df.set_index(["date", "stock_id"])

    feature_map = {
        "f_ret_1d": ret_1d,
        "f_ret_3d": ret_3d,
        "f_ret_5d": ret_5d,
        "f_pos_10d": pos_10d,
        "f_gap": gap,
        "f_ema5_slope": ema5_slope,
        "f_ema_cross": ema_cross,
        "f_rsi6": rsi6,
        "f_rsi14": rsi14,
        "f_macd_hist": macd_hist,
        "f_bb_pct": bb_pct,
        "f_v_ratio_5d": v_ratio_5d,
        "f_v_ratio_10d": v_ratio_10d,
        "f_obv_slope": obv_slope,
        "f_atr5": atr5,
        "f_atr_ratio": atr_ratio,
        "f_hv5": hv5,
        "f_body_ratio": body_ratio,
        "f_upper_shadow": upper_shadow,
        "f_lower_shadow": lower_shadow,
        # 在進場前先判斷現在是恐慌性錯殺還是趨勢性崩跌：
        "f_index_ret_5d": close.pct_change(5),  # 大盤5日報酬
        "f_index_ret_20d": close.pct_change(20),  # 大盤20日報酬
        "f_index_ma200": close / close.rolling(200).mean() - 1,  # 離年線距離
    }

    for fname, wide_df in feature_map.items():
        df[fname] = wide_df.stack(future_stack=True)

    df = df.reset_index()
    return df
