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
import j1stools.django_orm
import j1stools.utils as utils
import os

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


# def merge_folder():
#     # 只要指定資料夾路徑，PyArrow 會自動把裡面所有的 .parquet 視為同一個大表
#     dataset = ds.dataset("my_data/", format="parquet")

#     # 查詢時，它會同時掃描 part-1 和 part-2
#     table = dataset.to_table(filter=ds.field("stock_id") == "2330")


# def merge():

#     # 1. 讀取舊的歷史資料 (建議用 pyarrow table 讀取，省記憶體)
#     old_table = pq.read_table("history.parquet")

#     # 2. 準備 4 月的新資料 (DataFrame)
#     new_df = pd.get_4_month_data()  # 假設這是你獲取資料的 function
#     new_df = new_df.reset_index()  # 確保與歷史資料結構一致
#     new_table = pa.Table.from_pandas(new_df)

#     # 3. 合併 (Concatenate)
#     combined_table = pa.concat_tables([old_table, new_table])

#     # 4. 重新排序 (重要！因為新資料在最下面，為了維持查詢效能需要重排)
#     import pyarrow.compute as pc

#     indices = pc.sort_indices(combined_table, sort_keys=[("date", "ascending"), ("stock_id", "ascending")])
#     sorted_table = combined_table.take(indices)

#     # 5. 寫回檔案 (覆蓋舊檔)
#     pq.write_table(sorted_table, "history.parquet", row_group_size=100000)


# def read():
#     dataset = ds.dataset("db/price/2026_3.parquet", format="parquet")
#     table = dataset.to_table()
#     # table = dataset.to_table(
#     #     filter=ds.field("stock_id") == "2330", columns=["date", "stock_id", "close"]  # 篩選行  # 選取特定欄位
#     # )
#     df: pd.DataFrame = table.to_pandas()
#     print(df.head())
#     print(df.tail())


# def query_info():
#     dataset = ds.dataset("db/info/", format="parquet")

#     # 組合條件：時間區間 AND 股票清單
#     condition = (
#         (ds.field("date") >= start_date) & (ds.field("date") < end_date) & (ds.field("stock_id").isin(target_stocks))
#     )

#     # 執行查詢並取出特定欄位
#     table = dataset.to_table(filter=condition, columns=["date", "stock_id", "close", "volume"]).sort_by(
#         [("date", "ascending")]
#     )

#     df: pd.DataFrame = table.to_pandas()
#     print(df.head())
#     print(df.tail())


#     # print(df.head())
#     # print(df.tail())


# def query_stocks_ids_list():
# df: pd.DataFrame = pd.read_parquet("db/info/")
# return df.iloc[:, 0].values.tolist()


# print(query_info())


# def reset_price(group):
#     if group.name == "0050":
#         group["close"] = group["close"] + 1000
#     return group


# def create_features(df: pd.DataFrame):
#     df.reset_index(drop=True, inplace=True)
#     df.to_parquet("db/feature/history.parquet")


# def read_features():
#     dataset = ds.dataset("db/feature/history.parquet", format="parquet")
#     table = dataset.to_table()
#     # table = dataset.to_table(
#     #     filter=ds.field("stock_id") == "2330", columns=["date", "stock_id", "close"]  # 篩選行  # 選取特定欄位
#     # )
#     df: pd.DataFrame = table.to_pandas()
#     # print(df.head().T)
#     # print(df.tail().T)
#     print(df.describe())


def create_stock_group():
    """不含上櫃，要把stock_list.csv 中的其他業分類成其他業"""
    df_group: pd.DataFrame = pd.read_csv(
        "stock_group.csv",
        encoding="utf-8",
        header=None,
        dtype={0: str},
    )
    df_stock: pd.DataFrame = pd.read_csv(
        "data/stock_list.csv",
        encoding="utf-8",
        dtype={"代號": str},
    )
    df_group.rename(columns={0: "stock_id", 5: "group"}, inplace=True)
    df_stock.rename(columns={"代號": "stock_id", "名稱": "name", "市場": "market"}, inplace=True)
    df = pd.merge(
        df_stock[["stock_id", "name", "market"]],
        df_group[["stock_id", "group"]],
        on="stock_id",
        how="left",
    )
    df.loc[df["group"] == "其他業", "group"] = df["stock_id"].str[:2] + "XX"
    df.loc[df["group"].isna(), "group"] = df["stock_id"].str[:2] + "XX"

    df.to_parquet("db/stock_group/all/stock_group.parquet", index=False)

    df.to_csv(
        "tmp.csv",
        index=False,
    )


def query_stock_group_all():
    """
            0        1             2           3   4       5       6
    0     1101       台泥  TW0001101004  1962/02/09  上市    水泥工業  ESVUFR
    stock_id,name,market,group
    3481,群創,市,光電業
    """

    dataset = ds.dataset("db/stock_group/all/", format="parquet")
    loaded_df = dataset.to_table().to_pandas()
    return loaded_df


# print(query_stock_group_all())

# print(query_stock_group_all())
#          0        1             2           3   4       5       6
# 0     1101       台泥  TW0001101004  1962/02/09  上市    水泥工業  ESVUFR
# 1     1102       亞泥  TW0001102002  1962/06/08  上市    水泥工業  ESVUFR
# 2     1103       嘉泥  TW0001103000  1969/11/14  上市    水泥工業  ESVUFR
# 3     1104       環泥  TW0001104008  1971/02/01  上市    水泥工業  ESVUFR
# 4     1108       幸福  TW0001108009  1990/06/06  上市    水泥工業  ESVUFR

# df = query_stock_group().iloc[:, 5].unique()
# df = df.iloc[:, 5].unique()
# print(df)


def query_stock_group_unique_list():
    groups = query_stock_group_all()["group"].unique()
    return groups


def query_stock2group_dict():
    # 讀回來的df把head()刪掉
    result_dict = query_stock_group_all().set_index("stock_id")["group"].to_dict()
    # print(result_dict)
    # df = query_stock_group()
    return result_dict


# print(query_stock2group_dict())


def query_group_by_name(name):
    """回傳dict > stock_id,name,group的資料"""
    df = query_stock_group_all()
    df = df.loc[df["group"] == name, ["stock_id", "name", "group"]]
    result_dict = df.set_index("stock_id").to_dict(orient="index")
    return result_dict


def query_group_by_ids(ids: list):
    """回傳sid,name,group的資料"""
    dataset = ds.dataset("db/stock_group/all/", format="parquet")
    condition = ds.field("stock_id").isin(ids)
    table = dataset.to_table(
        filter=condition, columns=["stock_id", "name", "group"]
    )  # .sort_by([("date", "ascending")])
    df: pd.DataFrame = table.to_pandas()
    # print(df.head())

    df.columns = ["stock_id", "name", "group"]
    result_dict = df.set_index("stock_id").to_dict(orient="index")
    return result_dict


# print(query_group_by_ids(["2330", "2308", "3105"]))
# print(query_stock_by_group("半導體業"))

# print(query_stock_group_name())
# start = time.time()
# print(query_stock2group_dict())
# end = time.time()
# print(end - start)


# def create_stock_groupXX():
#     dict = query_stock_group_name()
#     stocks: list = list()
#     all_stocks = query_stocks()
#     # all_sid_group = {}
#     for sid in all_stocks:
#         # if sid not in stocks:
#         dict[sid] = sid[:2] + "XX"

#     df = pd.DataFrame(
#         list(dict.items()),
#         columns=["stock_id", "group"],
#     )
#     # df = df.set_index("stock_id")
#     print(df.head())
#     # df.to_parquet("db/stock_group/stock_group.parquet", engine="pyarrow")


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
    condition = (ds.field("date") >= st) & (ds.field("date") < end) & (ds.field("stock_id").isin(stocks))
    df = dataset.to_table(filter=condition).to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.sort_values(["date", "stock_id", "name"], inplace=True)
    return df


def query_margin(stocks: list, st="2015-01-01", end="2099-01-01"):
    dataset = ds.dataset("db/margin/", format="parquet")
    condition = (ds.field("date") >= st) & (ds.field("date") < end) & (ds.field("stock_id").isin(stocks))
    df = dataset.to_table(filter=condition).to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.sort_values(["date", "stock_id"], inplace=True)
    return df


_FEATURE_COLS_PATH = "db/feature_cols/feature_cols.parquet"


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


MY_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("stock_id", pa.string()),
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("volume", pa.int64()),
    ]
)


def init_price(csv_dir="data", out_dir="db/price"):
    """把 data/*_1d.csv 全部存成分月 parquet"""
    import glob

    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(csv_dir, "*_1d.csv"))
    print(f"找到 {len(files)} 支股票 CSV")

    dfs = []
    for f in files:
        stock_id = os.path.basename(f).replace("_1d.csv", "")
        df = pd.read_csv(f, dtype={"Volume": "Int64"})
        df["stock_id"] = stock_id
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        inplace=True,
    )
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close"]:
        all_df[col] = all_df[col].astype("float32")
    all_df["volume"] = all_df["volume"].fillna(0).astype("int64")

    columns = ["date", "stock_id", "open", "high", "low", "close", "volume"]
    all_df = all_df[columns].drop_duplicates(subset=["date", "stock_id"], keep="last")
    all_df = all_df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    cutoff = "2026-05-01"

    # 2026-05 前：存成一份 history.parquet
    hist = all_df[all_df["date"] < cutoff].reset_index(drop=True)
    hist_path = os.path.join(out_dir, "history.parquet")
    hist.to_parquet(hist_path, engine="pyarrow", schema=MY_SCHEMA, compression="zstd", index=False)
    print(f"  {hist_path}  {len(hist):,} 筆")

    # 2026-05 起：按月分檔
    recent = all_df[all_df["date"] >= cutoff]
    for month, group in recent.groupby(recent["date"].str[:7]):
        out_path = os.path.join(out_dir, f"{month.replace('-', '_')}.parquet")
        group.reset_index(drop=True).to_parquet(
            out_path, engine="pyarrow", schema=MY_SCHEMA, compression="zstd", index=False
        )
        print(f"  {out_path}  {len(group):,} 筆")

    print("init_price 完成")


from j1stools.django_orm import *


def init_info():
    qs = StocksInfo.objects.all().values("stock_id", "name", "market_type", "group")
    df = pd.DataFrame(list(qs))
    os.makedirs("db/info", exist_ok=True)
    df.to_parquet("db/info/info.parquet", index=False)


def check_data_loss(st="2024-01-01", end="2026-01-01"):
    """检查 price / ib / margin 資料完整性"""
    checks = [
        ("price", "db/price/"),
        ("ib", "db/ib/"),
        ("margin", "db/margin/"),
    ]

    stock_sets = {}

    for name, path in checks:
        try:
            dataset = ds.dataset(path, format="parquet")
            date_type = dataset.schema.field("date").type
            if pa.types.is_timestamp(date_type):
                condition = (ds.field("date") >= pd.Timestamp(st)) & (ds.field("date") < pd.Timestamp(end))
            else:
                condition = (ds.field("date") >= st) & (ds.field("date") < end)
            df = dataset.to_table(filter=condition, columns=["date", "stock_id"]).to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            df["ym"] = df["date"].dt.to_period("M")

            stock_sets[name] = set(df["stock_id"].unique())
            months = df["ym"].nunique()
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            monthly = df.groupby("ym")["stock_id"].nunique()
            low_months = monthly[monthly < monthly.median() * 0.8]

            print(f"\n[{name}]  {date_min} ~ {date_max}  月數:{months}  股票數:{len(stock_sets[name])}")
            if len(low_months):
                print(f"  ⚠ 資料偏少的月份：{list(low_months.index)}")
            else:
                print(f"  ✓ 無明顯缺漏月份")
        except Exception as e:
            print(f"\n[{name}]  ✗ 讀取失敗：{e}")

    # 跨資料集比對
    if len(stock_sets) > 1:
        print("\n=== 股票數量差異 ===")
        names = list(stock_sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                only_a = sorted(stock_sets[a] - stock_sets[b])
                only_b = sorted(stock_sets[b] - stock_sets[a])
                if only_a or only_b:
                    print(f"\n  [{a}] 有但 [{b}] 沒有（{len(only_a)} 支）：{only_a}")
                    print(f"  [{b}] 有但 [{a}] 沒有（{len(only_b)} 支）：{only_b}")
                else:
                    print(f"\n  [{a}] vs [{b}]：完全相同")


def check_price():
    dataset = ds.dataset("db/price/", format="parquet")

    # 5月之前的歷史股票（不含5月）
    hist_condition = ds.field("date") < "2026-05-01"
    all_stocks = set(dataset.to_table(filter=hist_condition, columns=["stock_id"]).to_pandas()["stock_id"].unique())

    # 5月的股票
    condition = (ds.field("date") >= "2026-05-01") & (ds.field("date") < "2026-06-01")
    may_stocks = set(dataset.to_table(filter=condition, columns=["stock_id"]).to_pandas()["stock_id"].unique())

    only_hist = sorted(all_stocks - may_stocks)
    only_may = sorted(may_stocks - all_stocks)

    print(f"歷史全部股票數：{len(all_stocks)}")
    print(f"2026-05 股票數：{len(may_stocks)}")
    if only_hist:
        print(f"\n歷史有但5月沒有（{len(only_hist)} 支）：{only_hist}")
    if only_may:
        print(f"\n5月有但歷史沒有（{len(only_may)} 支）：{only_may}")
    if not only_hist and not only_may:
        print("\n兩者股票完全相同")

    # 2024後下市的股票
    condition_2024 = (ds.field("date") >= "2024-01-01") & (ds.field("date") < "2026-01-01")
    active_2024 = set(dataset.to_table(filter=condition_2024, columns=["stock_id"]).to_pandas()["stock_id"].unique())
    delisted = sorted(all_stocks - active_2024)
    print(f"\n2024前下市（歷史有但2024後無資料）（{len(delisted)} 支）：{delisted}")


def _check_dataset(path: str, label: str, cutoff="2026-05-01"):
    dataset = ds.dataset(path, format="parquet")
    before = set(
        dataset.to_table(filter=ds.field("date") < cutoff, columns=["stock_id"]).to_pandas()["stock_id"].unique()
    )
    after = set(
        dataset.to_table(filter=ds.field("date") >= cutoff, columns=["stock_id"]).to_pandas()["stock_id"].unique()
    )
    only_before = sorted(before - after)
    only_after = sorted(after - before)

    print(f"\n[{label}]  {cutoff} 前：{len(before)} 支  /  後：{len(after)} 支")
    if only_before:
        print(f"  前有後無（{len(only_before)} 支）：{only_before}")
    if only_after:
        print(f"  後有前無（{len(only_after)} 支）：{only_after}")
    if not only_before and not only_after:
        print("  兩段股票完全相同")


def check_ib_margin():
    _check_dataset("db/ib/", "ib")
    _check_dataset("db/margin/", "margin")


if __name__ == "__main__":
    # init_price()
    # check_price()
    # check_data_loss()
    check_ib_margin()
