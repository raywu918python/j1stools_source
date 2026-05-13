import random

import pandas as pd

from j1stools import parquet_db
from j1stools.triangle_google import draw_multiple_triangles_safe, find_triangle


def find_break_price(df_price: pd.DataFrame, df_triangle: pd.DataFrame) -> pd.DataFrame:
    """
    找出每個三角形收斂型態在 h2_date 後 20 個交易日內
    收盤價或最高價突破 h2 當天 high 價位的股票。

    Parameters
    ----------
    df_price     : 含 [date, stock_id, open, high, low, close] 的 DataFrame
    df_triangle  : 含三角形型態欄位的 DataFrame（見 columns below）

    Returns
    -------
    DataFrame with columns:
        stock_id, h2_date, h2_high (突破價位),
        break_date (突破當天), break_high, break_close,
        days_to_break (從 h2_date 起第幾個交易日突破)
    """

    # 確保 date 欄位為 datetime
    df_price = df_price.copy()
    df_triangle = df_triangle.copy()
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_triangle["h2_date"] = pd.to_datetime(df_triangle["h2_date"])

    results = []

    for _, row in df_triangle.iterrows():
        sid = row["stock_id"]
        h2_date = row["h2_date"]

        # 取得 h2_date 當天的 high 作為突破價位
        h2_bar = df_price[(df_price["stock_id"] == sid) & (df_price["date"] == h2_date)]
        if h2_bar.empty:
            continue
        h2_high = h2_bar.iloc[0]["high"]

        # 取 h2_date 之後的價格資料（嚴格大於，不含當天）
        future = (
            df_price[(df_price["stock_id"] == sid) & (df_price["date"] > h2_date)]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # 只看前 20 個交易日
        future = future.head(20)

        # 找第一個 high > h2_high 的 bar（向上突破）
        broke = future[future["high"] > h2_high]

        if broke.empty:
            results.append(
                {
                    "stock_id": sid,
                    "h2_date": h2_date,
                    "h2_high": h2_high,
                    "break_date": None,
                    "break_high": None,
                    "break_close": None,
                    "days_to_break": None,
                    "is_break": False,
                }
            )
        else:
            first = broke.iloc[0]
            results.append(
                {
                    "stock_id": sid,
                    "h2_date": h2_date,
                    "h2_high": h2_high,
                    "break_date": first["date"],
                    "break_high": first["high"],
                    "break_close": first["close"],
                    "days_to_break": broke.index[0] + 1,  # 第幾個交易日
                    "is_break": True,
                }
            )

    df_result = pd.DataFrame(results)
    return df_result


# ── 統計摘要 ──────────────────────────────────────────────


def summarize_break(df_result: pd.DataFrame) -> None:
    """印出突破統計。"""
    total = len(df_result)
    broke = df_result["is_break"].sum()
    no_break = total - broke

    print(f"=== 突破統計 ===")
    print(f"總型態數       : {total}")
    print(f"20日內突破     : {broke}  ({broke/total*100:.1f}%)")
    print(f"20日內未突破   : {no_break}  ({no_break/total*100:.1f}%)")

    if broke > 0:
        sub = df_result[df_result["is_break"]]
        print(f"\n突破後平均天數 : {sub['days_to_break'].mean():.1f} 交易日")
        print(f"最快突破天數   : {sub['days_to_break'].min()} 交易日")
        print(f"最慢突破天數   : {sub['days_to_break'].max()} 交易日")
        print(f"\n突破股票名單：")
        print(
            sub[["stock_id", "h2_date", "h2_high", "break_date", "break_high", "days_to_break"]].to_string(index=False)
        )


# ── 使用範例 ──────────────────────────────────────────────
# df_result = find_break_price(df_price, df_triangle)
# summarize_break(df_result)
# df_result.to_csv('break_result.csv', index=False)
def test():
    # stocks = random.sample(parquet_db.query_stocks_no_etf(), 500)
    stocks = parquet_db.query_stocks_ids_list()
    df_price = parquet_db.query_price(stocks, "2026-03-01", "2099-01-12")
    df_triangle_base = find_triangle(df_price)
    df_triangle = df_triangle_base[df_triangle_base["is_refined_triangle"] == True]
    draw_multiple_triangles_safe(df_triangle_base, df_triangle)
    print("找到標記的三角形數量：", len(df_triangle))
    df_triangle.to_csv("triangle.csv", index=False)
    print(df_triangle.head().T)
    df_break_price = find_break_price(df_price, df_triangle)
    df_break_price = df_break_price[df_break_price["is_break"] == True]
    print("找到突破的三角形數量：", len(df_break_price))
    print(df_break_price.head(20).T)
    summarize_break(df_break_price)


# test()
