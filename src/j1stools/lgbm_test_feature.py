import pandas as pd
import numpy as np


def lgbm_feature(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    # 1. 確保格式與時間排序 (對長表格 groupby 運算至關重要)
    df["date"] = pd.to_datetime(df["date"])
    market_df["date"] = pd.to_datetime(market_df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    market_df = market_df.sort_values("date")

    # 2. 準備大盤資料 (0050)
    market_df = market_df[["date", "open", "high", "low", "close"]].rename(
        columns={"open": "m_open", "high": "m_high", "low": "m_low", "close": "m_close"}
    )
    # 計算大盤 5 日報酬率
    market_df["m_ret_5d"] = market_df["m_close"].pct_change(5)

    # 合併大盤資料
    df = pd.merge(df, market_df, on="date", how="left")

    # 3. 建立 Groupby 對象以進行個股隔離計算
    group_obj = df.groupby("stock_id")

    # --- [f_ 籌碼維度] ---
    # 券資比：軋空核心能量
    df["f_short_margin_ratio"] = df["short_sale_today_balance"] / (df["margin_purchase_today_balance"] + 1)

    # 融券使用率：剩餘燃料空間
    df["f_short_utilization"] = df["short_sale_today_balance"] / (df["short_sale_limit"] + 1)

    # 融券變動率 (3日)：捕捉空頭急劇增加的訊號
    df["f_short_chg_3d"] = group_obj["short_sale_today_balance"].pct_change(3)

    # --- [f_ 相對強度與大盤維度] ---
    # 個股 5 日報酬率
    df["stock_ret_5d"] = group_obj["close"].pct_change(5)
    # f_alpha_5d: 個股是否強於大盤 (10日交易的關鍵選股因子)
    df["f_alpha_5d"] = df["stock_ret_5d"] - df["m_ret_5d"]

    # f_mkt_corr_10d: 與大盤的 10 日相關性 (觀察是否具備獨立飆漲股特性)
    df["f_mkt_corr_10d"] = group_obj.apply(lambda x: x["close"].rolling(10).corr(x["m_close"])).reset_index(
        0, drop=True
    )

    # --- [f_ 技術面與波動度] ---
    # f_bias_10: 10日乖離率 (判斷短線回檔或過熱)
    ma10 = group_obj["close"].transform(lambda x: x.rolling(10).mean())
    df["f_bias_10"] = (df["close"] - ma10) / ma10

    # f_atr_rel: 波動度 (10日高低價差/股價)，判斷當前是否有足夠的漲幅空間
    df["f_volatility_10d"] = group_obj.apply(
        lambda x: ((x["high"] - x["low"]) / x["close"]).rolling(10).mean()
    ).reset_index(0, drop=True)

    # --- [f_ 類別處理] ---
    # 將 group 重新命名並轉換為 category 類型
    if "group" in df.columns:
        df["f_group"] = df["group"].astype("category")

    # 4. 整理輸出
    # 保留你需要的原始報價 + 所有 f_ 開頭的特徵
    # 移除計算過程中產生的暫存欄位 (不以 f_ 開頭的自定義計算)
    temp_cols = ["stock_ret_5d", "m_ret_5d", "m_open", "m_high", "m_low", "m_close"]
    df = df.drop(columns=[c for c in temp_cols if c in df.columns])

    # 確保所有 NaN 數值 (因 rolling 或 pct_change 產生) 處理妥當
    # LGBM 可處理 NaN，但若要做信心評分，建議至少保留有 10 日資料後的樣本
    df = df.dropna(subset=["f_bias_10"]).reset_index(drop=True)

    return clean_f_features(df)


def lgbm_all_f_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    # 確保排序
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    group_obj = df.groupby("stock_id")

    # --- [A. 信用交易轉化為比例 (解決規模問題)] ---
    # 融資/融券使用率 (相對於限額)
    df["f_margin_utilization"] = df["margin_purchase_today_balance"] / (df["margin_purchase_limit"] + 1)
    df["f_short_utilization"] = df["short_sale_today_balance"] / (df["short_sale_limit"] + 1)

    # 券資比
    df["f_short_to_margin_ratio"] = df["short_sale_today_balance"] / (df["margin_purchase_today_balance"] + 1)

    # 籌碼集中度 (資券互抵佔成交量比)
    df["f_offset_ratio"] = df["offset_loan_and_short"] / (df["volume"] + 1)

    # --- [B. 信用交易轉化為動能 (解決趨勢問題)] ---
    # 融資 3 日增加率
    df["f_margin_buy_slope"] = group_obj["margin_purchase_today_balance"].pct_change(3)
    # 融券 3 日增加率
    df["f_short_sell_slope"] = group_obj["short_sale_today_balance"].pct_change(3)

    # --- [C. 量價特徵 (f_ 化)] ---
    # 價格位置：10日乖離
    ma10 = group_obj["close"].transform(lambda x: x.rolling(10).mean())
    df["f_price_bias_10"] = (df["close"] - ma10) / ma10

    # 成交量能：今日 vs 5日均量
    vma5 = group_obj["volume"].transform(lambda x: x.rolling(5).mean())
    df["f_volume_ratio"] = df["volume"] / (vma5 + 1)

    # --- [D. 類別特徵] ---
    df["f_group"] = df["group"].astype("category")

    # --- [E. 整理輸出] ---
    # 只保留 f_ 開頭以及必要的識別欄位
    f_cols = [c for c in df.columns if c.startswith("f_")]
    essential_cols = ["date", "stock_id", "close", "high", "low", "open"]  # 保留 high/close 用來算 label 或對答案

    return clean_f_features(df[essential_cols + f_cols])


import numpy as np


def clean_f_features(df):
    f_cols = [c for c in df.columns if c.startswith("f_")]

    for col in f_cols:
        # 1. 檢查是否為數值型，如果不是就跳過 (排除 f_group 等類別欄位)
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"跳過類別欄位: {col}")
            continue

        # 2. 處理 inf
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # 3. 處理 NaN
        df[col] = df[col].fillna(0)

        # 4. 處理離群值 (只針對數值進行 clip)
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower_bound, upper_bound)

    return df
