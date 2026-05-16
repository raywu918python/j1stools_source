import pandas as pd
import numpy as np


def add_market_filter(df, df_market, vol_threshold=0.008, trend_days=20):
    """
    加入市場狀態過濾條件，只在適合的市場環境交易

    過濾條件：
    1. 大盤波動率 > vol_threshold（排除過於平靜的市場）
    2. 大盤中期趨勢向上（排除熊市）

    Parameters
    ----------
    df            : 已有特徵的個股 df
    df_market     : 大盤資料，需包含 date, close
    vol_threshold : 波動率門檻，預設 0.008
    trend_days    : 趨勢判斷天數，預設 20 天

    回傳欄位
    --------
    market_vol      : 大盤20日波動率
    market_trend    : 大盤N日報酬（正=上漲趨勢）
    can_trade       : True = 可以交易，False = 不交易
    """

    df = df.copy()

    # 大盤計算
    mkt = df_market.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)

    mkt["market_vol"] = mkt["close"].pct_change(1).rolling(20).std()
    mkt["market_trend"] = mkt["close"].pct_change(trend_days)

    # merge 回個股 df
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(mkt[["date", "market_vol", "market_trend"]], on="date", how="left")

    # 過濾條件
    df["can_trade"] = (df["market_vol"] > vol_threshold) & (df["market_trend"] > 0)  # 波動率夠高  # 大盤趨勢向上

    return df


def apply_filter(df):
    """
    只回傳可以交易的資料
    """
    return df[df["can_trade"]].reset_index(drop=True)
