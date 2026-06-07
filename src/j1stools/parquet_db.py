from datetime import datetime
from re import S

# from j1stools.django_orm import *
import pyarrow as pa
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import j1stools
import j1stools.utils as utils
import os

# 將 Schema 移到最外層作為全域變數，確保所有邏輯共用，絕對不會再「找不到」
MY_SCHEMA = pa.schema(
    [
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("volume", pa.int64()),
        ("stock_id", pa.string()),
        ("date", pa.timestamp("ns")),
    ]
)


# def create_info():
#     df: pd.DataFrame = pd.read_csv(utils.get_base_url() + f"stock_list.csv", dtype={"代號": str})
#     df = df.iloc[:, 1:]
#     df.reset_index(drop=True, inplace=True)
#     df.to_parquet("db/info/info.parquet")


# def check_dup_key():
#     # 假設 old_df 是從 history.parquet 讀回來的，new_df 是 4 月新資料
#     combined_df = pd.concat([old_df, new_df], axis=0)

#     # 根據你的鍵值（日期與股票代碼）來檢查重複
#     # keep='last' 表示如果重複，保留最新（4月）的那一筆
#     combined_df = combined_df.drop_duplicates(subset=["date", "stock_id"], keep="last")

#     # 排序後存檔
#     combined_df = combined_df.sort_values(["date", "stock_id"])
#     table = pa.Table.from_pandas(combined_df)
#     pq.write_table(table, "history.parquet")


# def query_last_price(stocks: list):
#     dataset = ds.dataset("db/price/", format="parquet")

#     table = dataset.to_table(
#         filter=ds.field("stock_id").isin(stocks), columns=["date", "stock_id", "close", "volume", "high", "low", "open"]
#     ).sort_by([("date", "ascending")])

#     return table.slice(table.num_rows - 1, 1).to_pandas()


def query_price(stocks: list, st="2015-01-01", end="2099-01-01", is_include_end=False):
    dataset = ds.dataset("db/price/", format="parquet")
    st_str = pd.Timestamp(st).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    if is_include_end:
        condition = (ds.field("date") >= st_str) & (ds.field("date") <= end_str) & (ds.field("stock_id").isin(stocks))
    else:
        condition = (ds.field("date") >= st_str) & (ds.field("date") < end_str) & (ds.field("stock_id").isin(stocks))

    table = dataset.to_table(
        filter=condition, columns=["date", "stock_id", "close", "volume", "high", "low", "open"]
    ).sort_by([("date", "ascending")])

    df: pd.DataFrame = table.to_pandas()
    df["close"] = df["close"].astype("float32")
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df.drop_duplicates(subset=["date", "stock_id"], keep="last", inplace=True)
    return df


def stock0056():
    return [
        "2891",  # 中信金
        "2303",  # 聯電
        "2454",  # 聯發科
        "3711",  # 日月光投控
        "2382",  # 廣達
        "2357",  # 華碩
        "2603",  # 長榮
        "2880",  # 華南金
        "2890",  # 永豐金
        "2885",  # 元大金
        "3045",  # 台灣大
        "3231",  # 緯創
        "2449",  # 京元電子
        "2886",  # 兆豐金
        "1216",  # 統一
        "2301",  # 光寶科
        "3036",  # 文曄
        "2317",  # 鴻海
        "2379",  # 瑞昱
        "2884",  # 玉山金
        "2883",  # 凱基金
        "3034",  # 聯詠
        "2887",  # 台新新光金
        "4938",  # 和碩
        "2618",  # 長榮航
        "5876",  # 上海商銀
        "3044",  # 健鼎
        "2376",  # 技嘉
        "6239",  # 力成
        "2356",  # 英業達
        "3702",  # 大聯大
        "2347",  # 聯強
        "2324",  # 仁寶
        "1102",  # 亞泥
        "2474",  # 可成
        "2105",  # 正新
        "2385",  # 群光
        "2027",  # 大成鋼
        "6285",  # 啟碁
        "2353",  # 宏碁
        "2377",  # 微星
        "9904",  # 寶成
        "3005",  # 神基
        "3023",  # 信邦
        "1477",  # 聚陽
        "2006",  # 東和鋼鐵
        "1319",  # 東陽
        "2645",  # 長榮航太
        "6176",  # 瑞儀
    ]


def stock0050():
    return [
        "0050",  # etf
        "0052",  # etf
        "0056",  # etf
        "2330",
        "2308",
        "2317",
        "2454",
        "3711",
        "2891",
        "2345",
        "2383",
        "2382",
        "2881",
        "2882",
        "2303",
        "3017",
        "2360",
        "2887",
        "2412",
        "2884",
        "2885",
        "2886",
        "2890",
        "2357",
        "3231",
        "2327",
        "1303",
        "1216",
        "6669",
        "3653",
        "2880",
        "2892",
        "2883",
        "2368",
        "2449",
        "2344",
        "2301",
        "5880",
        "2408",
        "2603",
        "2002",
        "3008",
        "3661",
        "7769",
        "1301",
        "2059",
        "4904",
        "3045",
        "2395",
        "2207",
        "6919",
        "6505",
    ]


def query_stocks_no_etf():
    df = pd.read_parquet("db/active_stocks/")
    return [s for s in df["stock_id"] if not s.startswith("00")]


#############################################################
# build_stock_parquet("2015-01", "2026-05")
# date_str = datetime.now().strftime("%Y-%m")
# update_price(date_str)


# def init_margin():
#     qs = StocksMargin.objects.filter(date__gte="2015-01-01", date__lt="2026-05-01")
#     df = pd.DataFrame(list(qs.values()))
#     df = df.drop(columns=["id"], errors="ignore")
#     df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
#     df.to_parquet("db/margin/201501_202605.parquet")


# def init_ib():
#     qs = StocksIbBuySell.objects.filter(date__gte="2015-01-01", date__lt="2026-05-01")
#     df = pd.DataFrame(list(qs.values()))
#     df = df.drop(columns=["id"], errors="ignore")
#     df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
#     df.to_parquet("db/ib/201501_202605.parquet")


def activate_stocks():
    return list(pd.read_parquet("db/active_stocks/")["stock_id"])


def query_ib(stocks: list, st="2015-01-01", end="2099-01-01"):
    dataset = ds.dataset("db/ib/", format="parquet")
    st_str = pd.Timestamp(st).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
    condition = (ds.field("date") >= st_str) & (ds.field("date") < end_str) & (ds.field("stock_id").isin(stocks))
    df = dataset.to_table(filter=condition).to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.sort_values(["date", "stock_id", "name"], inplace=True)
    df.drop_duplicates(subset=["date", "stock_id", "name"], inplace=True)
    return df


def query_day_trade(stocks: list, st="2015-01-01", end="2099-01-01"):
    dataset = ds.dataset("db/day_trade/", format="parquet")
    st_str = pd.Timestamp(st).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
    condition = (ds.field("date") >= st_str) & (ds.field("date") < end_str) & (ds.field("stock_id").isin(stocks))
    df = dataset.to_table(filter=condition).to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["date", "stock_id"], inplace=True)
    return df


def query_margin(stocks: list, st="2015-01-01", end="2099-01-01"):
    dataset = ds.dataset("db/margin/", format="parquet")
    st_str = pd.Timestamp(st).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
    condition = (ds.field("date") >= st_str) & (ds.field("date") < end_str) & (ds.field("stock_id").isin(stocks))
    df = dataset.to_table(filter=condition).to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.sort_values(["date", "stock_id"], inplace=True)
    df.drop_duplicates(subset=["date", "stock_id"], inplace=True)
    return df


_FEATURE_COLS_PATH = "db/feature_cols/feature_cols.parquet"


def delete_feature_cols(model_name: str):
    if not os.path.exists(_FEATURE_COLS_PATH):
        return
    df = pd.read_parquet(_FEATURE_COLS_PATH)
    df[df["model_name"] != model_name].to_parquet(_FEATURE_COLS_PATH, index=False)


def save_feature_cols(model_name: str, feature_cols: list):
    os.makedirs("db/feature_cols", exist_ok=True)
    new_df = pd.DataFrame({"model_name": model_name, "feature_name": feature_cols})
    if os.path.exists(_FEATURE_COLS_PATH):
        old_df = pd.read_parquet(_FEATURE_COLS_PATH)
        old_df = old_df[old_df["model_name"] != model_name]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_parquet(_FEATURE_COLS_PATH, index=False)


def load_feature_cols(model_name: str) -> list:
    df = pd.read_parquet(_FEATURE_COLS_PATH)
    return list(df[df["model_name"] == model_name]["feature_name"])


def query_stock_info() -> pd.DataFrame:
    """stock_id, name, market_type, group"""
    return pd.read_parquet("db/info/")


if __name__ == "__main__":
    pass
    # init_margin()
    # init_ib()
    # df = query_ib(
    #     stocks=["2003", "2330"],
    #     st="2025-01-01",
    #     end="2026-05-01",
    # )
    # df = query_margin(
    #     stocks=["2003", "2330"],
    #     st="2025-01-01",
    #     end="2026-05-01",
    # )
    # print(df.head())
