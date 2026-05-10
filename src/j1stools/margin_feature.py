import pandas as pd
import numpy as np

from j1stools import lite_db, parquet_db

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def detect_short_squeeze_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    軋空相關特徵，用來找「融券大增但價格撐住」的潛在軋空股。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位:
        date, stock_id, close, volume,
        short_sale_today_balance, short_sale_yesterday_balance,
        short_sale_sell, short_sale_buy, short_sale_limit,
        margin_purchase_today_balance, offset_loan_and_short

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_sq_ 開頭特徵欄位
    """
    g = df.groupby("stock_id")

    # ── Feature 1: f_sq_short_ratio ──────────────────────────────────── #
    # 券資比：融券餘額 / 融資餘額
    # 越高代表看空籌碼越重，軋空燃料越多
    df["f_sq_short_ratio"] = df["short_sale_today_balance"] / (df["margin_purchase_today_balance"] + 1e-9)

    # ── Feature 2: f_sq_cover_days ───────────────────────────────────── #
    # 融券回補天數：空單要幾天才能全部回補
    # 越高代表軋起來越持久、越猛
    # volume 單位是股，short_sale 單位是張（1張=1000股），需轉換
    df["f_sq_cover_days"] = df["short_sale_today_balance"] / (df["volume"] / 1000 + 1e-9)

    # ── Feature 3: f_sq_short_change ─────────────────────────────────── #
    # 融券餘額變化量（今日 - 昨日）
    # 正值代表空單增加
    df["f_sq_short_change"] = df["short_sale_today_balance"] - df["short_sale_yesterday_balance"]

    # ── Feature 4: f_sq_short_growth ─────────────────────────────────── #
    # 融券增加速度（比例）
    # 正值且大 → 空方積極加碼，潛在軋空燃料快速累積
    # 避免昨日餘額為 0 時產生極大值，設下限為 1
    df["f_sq_short_growth"] = (df["f_sq_short_change"] / (df["short_sale_yesterday_balance"].clip(lower=1))).clip(
        -1, 10
    )  # 限制在合理範圍內

    # ── Feature 5: f_sq_short_utilization ────────────────────────────── #
    # 融券利用率：融券餘額 / 融券限額
    # 接近 1 代表快借不到券，軋空更容易發生
    df["f_sq_short_utilization"] = df["short_sale_today_balance"] / (df["short_sale_limit"] + 1e-9)

    # ── Feature 6: f_sq_net_short ────────────────────────────────────── #
    # 融券淨空單：融券賣出 - 融券買進（當日新增空單）
    # 正值代表當日淨增空單
    df["f_sq_net_short"] = df["short_sale_sell"] - df["short_sale_buy"]

    # ── Feature 7: f_sq_price_resilience ─────────────────────────────── #
    # 價格韌性：融券增加但價格沒跌
    # 空方加碼 × 當日漲幅，正值代表軋空訊號出現
    price_chg = g["close"].pct_change(1)
    short_increasing = df["f_sq_short_change"].clip(lower=0)
    df["f_sq_price_resilience"] = price_chg * short_increasing

    # ── Feature 8: f_sq_short_ma5_growth ─────────────────────────────── #
    # 融券餘額 5 日均線成長率（平滑化，避免單日異常）
    short_ma5 = g["short_sale_today_balance"].transform(lambda x: x.rolling(5).mean())
    short_ma5_prev = g["short_sale_today_balance"].transform(lambda x: x.rolling(5).mean().shift(5))
    df["f_sq_short_ma5_growth"] = (short_ma5 - short_ma5_prev) / (short_ma5_prev + 1e-9)

    # ── Feature 9: f_sq_offset_ratio ─────────────────────────────────── #
    # 資券互抵比例：軋空時互抵會增加
    df["f_sq_offset_ratio"] = df["offset_loan_and_short"] / (df["volume"] + 1e-9)

    # ── Feature 10: f_sq_squeeze_score ───────────────────────────────── #
    # 綜合軋空分數（規則合成）
    # 券資比高 + 回補天數長 + 空單還在增加 + 價格撐住
    score = (
        df["f_sq_short_ratio"].clip(0, 1) * 0.3
        + df["f_sq_cover_days"].clip(0, 10) / 10 * 0.3
        + df["f_sq_short_growth"].clip(0, 1) * 0.2
        + price_chg.clip(0, 0.1) / 0.1 * 0.2
    )
    df["f_sq_squeeze_score"] = score

    return df


def filter_short_squeeze(
    df: pd.DataFrame,
    min_cover_days: float = 3.0,  # 回補天數下限
    min_short_growth: float = 0.1,  # 融券增加比例下限（10%）
    min_short_ratio: float = 0.05,  # 券資比下限（5%）
    price_must_up: bool = True,  # 當日價格是否必須不跌
) -> pd.DataFrame:
    """
    規則篩選軋空候選股。

    Parameters
    ----------
    df : pd.DataFrame
        需已執行 detect_short_squeeze_features，包含 f_sq_ 特徵
    min_cover_days : 融券回補天數下限
    min_short_growth : 融券增加比例下限
    min_short_ratio : 券資比下限
    price_must_up : 是否要求當日價格不跌

    Returns
    -------
    pd.DataFrame
        符合條件的候選股
    """
    price_chg = df.groupby("stock_id")["close"].pct_change(1)

    mask = (
        (df["f_sq_cover_days"] >= min_cover_days)
        & (df["f_sq_short_growth"] >= min_short_growth)
        & (df["f_sq_short_ratio"] >= min_short_ratio)
    )

    if price_must_up:
        mask = mask & (price_chg > 0)

    return df[mask].copy()


def evaluate_squeeze_signals(
    df: pd.DataFrame,
    hold_days_list: list = [3, 5, 10],  # 驗證持有幾天的報酬
) -> pd.DataFrame:
    """
    驗證軋空訊號的效果：
    計算訊號後 N 日的報酬率分布。

    Parameters
    ----------
    df : pd.DataFrame
        完整資料（含價格），需有 stock_id, date, close
    hold_days_list : 驗證的持有天數列表

    Returns
    -------
    pd.DataFrame
        每個訊號的未來報酬率
    """
    # 取未來報酬率
    result = df[
        ["date", "stock_id", "close", "f_sq_cover_days", "f_sq_short_growth", "f_sq_short_ratio", "f_sq_squeeze_score"]
    ].copy()

    for h in hold_days_list:
        future_close = df.groupby("stock_id")["close"].shift(-h)
        result[f"return_{h}d"] = (future_close - df["close"]) / (df["close"] + 1e-9)

    return result


def print_squeeze_report(signals: pd.DataFrame, hold_days_list: list = [3, 5, 10]):
    """
    印出軋空訊號的統計報告。
    """
    print(f"{'='*50}")
    print(f"軋空訊號數量：{len(signals)} 筆")
    print(f"涵蓋股票數：{signals['stock_id'].nunique()} 檔")
    print(f"時間範圍：{signals['date'].min()} ~ {signals['date'].max()}")
    print(f"{'='*50}")

    for h in hold_days_list:
        col = f"return_{h}d"
        if col not in signals.columns:
            continue
        ret = signals[col].dropna()
        win_rate = (ret > 0).mean()
        print(f"\n持有 {h} 日：")
        print(f"  勝率（> 0）  ：{win_rate:.2%}")
        print(f"  平均報酬    ：{ret.mean():.2%}")
        print(f"  中位數報酬  ：{ret.median():.2%}")
        print(f"  最大獲利    ：{ret.max():.2%}")
        print(f"  最大虧損    ：{ret.min():.2%}")
        print(f"  標準差      ：{ret.std():.2%}")

    print(f"\n{'='*50}")
    print("squeeze_score 分布：")
    print(signals["f_sq_squeeze_score"].describe())


def test():
    stocks = parquet_db.query_stocks_ids_list()
    df = lite_db.margin(stocks, "2021-05-01", "2024-01-01")
    # 1. 計算特徵
    df = detect_short_squeeze_features(df)

    # 2. 規則篩選
    signals = filter_short_squeeze(
        df,
        min_cover_days=3.0,
        min_short_growth=0.1,
        min_short_ratio=0.05,
    )

    # 3. 加上未來報酬
    signals = evaluate_squeeze_signals(
        df.merge(signals[["date", "stock_id"]], on=["date", "stock_id"]), hold_days_list=[3, 5, 10]
    )

    # 4. 看結果
    print_squeeze_report(signals)

    # 先看特徵有沒有值
    print(df[["f_sq_cover_days", "f_sq_short_growth", "f_sq_short_ratio"]].describe())

    # 看各條件各自篩出多少
    print("cover_days >= 3:", (df["f_sq_cover_days"] >= 3).sum())
    print("short_growth >= 0.1:", (df["f_sq_short_growth"] >= 0.1).sum())
    print("short_ratio >= 0.05:", (df["f_sq_short_ratio"] >= 0.05).sum())
    print("price_up:", (df.groupby("stock_id")["close"].pct_change(1) > 0).sum())


test()
