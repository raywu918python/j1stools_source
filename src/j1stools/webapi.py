import json

import pandas as pd
from regex import F

from j1stools import j1s_main, lgbm_main, parquet_db


def query_stock_group_unique_list():
    return parquet_db.query_stock_group_unique_list()


def update_pf():
    signal = lgbm_main.query(parquet_db.query_stocks_no_etf(), "2024-01", "2099-01")
    # signal = lgbm_main.query(parquet_db.query_stocks_no_etf(), "2024-01", "2099-01")
    signal.to_csv("gold_signal.csv", index=False)
    portfolio_value, trades_df, positions = j1s_main.query(signal, proba_threshold=0.9)
    print("9999")
    portfolio_value.to_csv("portfolio_value.csv")
    trades_df.to_csv("trades_df.csv")
    # print(type(positions))
    # print(positions)
    pd.DataFrame(positions).T.to_csv("positions.csv")


def query_pf():
    df = pd.read_csv("portfolio_value.csv")
    # print(df.iloc[-1].to_dict()["0"])
    return df.iloc[-1].to_dict()["0"]


def query_water():
    df = pd.read_csv("positions.csv")
    df["calculated_value"] = df["shares"] * df["entry_price"]

    return df["calculated_value"].sum()


def query_positions():
    df: pd.DataFrame = pd.read_csv("positions.csv")
    df.reset_index(drop=False, inplace=True)
    # print(df.iloc[0])
    # print(df.iloc[0, 1])
    # stock_id = df.iloc[0, 1]
    # print(df.loc[0, "entry_price"])
    # entry_price = df.loc[0, "entry_price"]
    # tp = entry_price * 1.15
    # sl = entry_price * 0.9
    # print(df.loc[0, "entry_price"])
    # # print(df[0].head())
    # print(df.iloc[:, 1].values)
    stocks = df.iloc[:, 1].values
    # print(type(stocks))
    str_list = [str(x) for x in stocks]
    rs = []
    for i in range(len(str_list)):
        print(str_list[i])
        stock_id = str_list[i]
        price = parquet_db.query_last_price([stock_id])
        entry_price = df.loc[i, "entry_price"]
        tp = entry_price * 1.15
        sl = entry_price * 0.9
        now = price["close"].values[0]
        # print("stock_id", stock_id)
        # print(f"{entry_price:.2f}, {tp:.2f}, {sl:.2f}, {now:.2f}")
        win = (now - entry_price) / entry_price
        # print(entry_price:2f, tp:2f, sl:2f, now:2f)
        data = parquet_db.query_group_by_ids([stock_id])
        name = next(iter(data.values()))["name"]
        rs.append(
            {
                "stock_id": stock_id,
                "entry_price": f"{entry_price:.2f}",
                "tp": f"{tp:.2f}",
                "sl": f"{sl:.2f}",
                "now": f"{now:.2f}",
                "win": f"{win:.2f}",
                "name": name,
            }
        )
    return rs


# print(query_positions())
# query_pf()
# update_pf()
# print(query_water())
