from datetime import datetime
from re import S
from j1stools.django_orm import *
import pyarrow as pa
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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


def update_price(start_date: str, end_date: str = None):
    """
    通用型股票資料整合函式 (支援單月更新與歷史區間合併)

    用法 1: build_stock_parquet("2015-01", "2026-05") -> 產出 201501_202605.parquet
    用法 2: build_stock_parquet("2026-05")            -> 產出 2026_05.parquet
    """
    # 1. 解析日期邊界 (使用 Pandas Timestamp 處理跨月/跨年邏輯最無腦且精準)
    st_dt = pd.to_datetime(f"{start_date}-01")

    if end_date:
        # 用法 1: 取到結束月份的下個月 1 號 (左閉右開)
        end_dt = pd.to_datetime(f"{end_date}-01") + pd.DateOffset(months=1)
        file_name = (
            f"{start_date}.parquet"
            if start_date == end_date
            else f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        )
    else:
        # 用法 2: 沒傳 end_date，代表只做單月。結束時間就是下個月 1 號
        end_dt = st_dt + pd.DateOffset(months=1)
        file_name = f"{st_dt.year}_{st_dt.month:02d}.parquet"

    file_path = f"db/price/{file_name}"
    print(f"開始處理區間: {st_dt.strftime('%Y-%m-%d')} 至 {end_dt.strftime('%Y-%m-%d')} -> 目標檔案: {file_path}")

    # 2. 抓取今日/新撈到的資料
    stocks = utils.get_file_list()
    new_tables = []

    for stock_id in stocks:
        try:
            df = utils.stock(stock_id)
            if df.empty:
                continue

            # 轉換為 datetime 以利精準比較
            df["date"] = pd.to_datetime(df["date"])

            # 過濾時間區間 [start, end)
            df = df[(df["date"] >= st_dt) & (df["date"] < end_dt)]

            if not df.empty:
                # 轉 PyArrow 前先去除該股票內部的重複值
                df = df.drop_duplicates(subset=["date"])

                # 確保欄位順序與 Schema 一致，並剔除可能多出來的 index 欄位
                df = df[MY_SCHEMA.names]

                new_tables.append(pa.Table.from_pandas(df, schema=MY_SCHEMA))
        except Exception as e:
            print(f"stock_id {stock_id} 處理失敗: {e}")

    if not new_tables:
        print("❌ 該區間內找不到任何新資料")
        return

    # 合併新撈出來的所有股票資料
    combined_table = pa.concat_tables(new_tables)

    # 3. 如果是「每日更新單月」的模式，需要讀取舊檔進行增量合併
    if end_date is None and os.path.exists(file_path):
        try:
            existing_table = pq.read_table(file_path).cast(MY_SCHEMA)
            combined_table = pa.concat_tables([existing_table, combined_table])
            print("讀取既有月檔成功，進行增量合併...")
        except Exception as e:
            print(f"讀取舊檔失敗 (可能 Schema 衝突)，將直接覆蓋。錯誤: {e}")

    # 4. 全局去重 (轉回 Pandas 做最保險，避免跨股票或增量更新造成的重複)
    df_all = combined_table.to_pandas()
    df_all.drop_duplicates(subset=["date", "stock_id"], keep="last", inplace=True)

    # 5. 重新包回 Table 並強制套用 Schema 與排序
    final_table = pa.Table.from_pandas(df_all, schema=MY_SCHEMA)
    indices = pc.sort_indices(final_table, sort_keys=[("date", "ascending"), ("stock_id", "ascending")])
    final_table = final_table.take(indices)

    # 6. 寫入 Parquet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pq.write_table(final_table, file_path, compression="snappy")
    print(f"🎉 處理完成！總筆數: {len(final_table)}\n")


def create_info():
    df: pd.DataFrame = pd.read_csv(utils.get_base_url() + f"stock_list.csv", dtype={"代號": str})
    df = df.iloc[:, 1:]
    df.reset_index(drop=True, inplace=True)
    df.to_parquet("db/info/info.parquet")


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


def query_last_price(stocks: list):
    dataset = ds.dataset("db/price/", format="parquet")

    table = dataset.to_table(
        filter=ds.field("stock_id").isin(stocks), columns=["date", "stock_id", "close", "volume", "high", "low", "open"]
    ).sort_by([("date", "ascending")])

    return table.slice(table.num_rows - 1, 1).to_pandas()


def query_price(stocks: list, st="2015-01-01", end="2099-01-01", is_include_end=False):
    dataset = ds.dataset("db/price/", format="parquet")
    # 定義你的查詢條件
    # stocks = ["0050"]  # 這裡放你的 300 支股票代號
    st = pd.Timestamp(st)
    end = pd.Timestamp(end)

    # 組合條件：時間區間 AND 股票清單
    if is_include_end:
        condition = (ds.field("date") >= st) & (ds.field("date") <= end) & (ds.field("stock_id").isin(stocks))
    else:
        condition = (ds.field("date") >= st) & (ds.field("date") < end) & (ds.field("stock_id").isin(stocks))

    # 執行查詢並取出特定欄位
    table = dataset.to_table(
        filter=condition, columns=["date", "stock_id", "close", "volume", "high", "low", "open"]
    ).sort_by([("date", "ascending")])

    df: pd.DataFrame = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype("float32")
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")

    df.drop_duplicates(subset=["date", "stock_id"], keep="last", inplace=True)
    return df


def merge_folder():
    # 只要指定資料夾路徑，PyArrow 會自動把裡面所有的 .parquet 視為同一個大表
    dataset = ds.dataset("my_data/", format="parquet")

    # 查詢時，它會同時掃描 part-1 和 part-2
    table = dataset.to_table(filter=ds.field("stock_id") == "2330")


def merge():

    # 1. 讀取舊的歷史資料 (建議用 pyarrow table 讀取，省記憶體)
    old_table = pq.read_table("history.parquet")

    # 2. 準備 4 月的新資料 (DataFrame)
    new_df = pd.get_4_month_data()  # 假設這是你獲取資料的 function
    new_df = new_df.reset_index()  # 確保與歷史資料結構一致
    new_table = pa.Table.from_pandas(new_df)

    # 3. 合併 (Concatenate)
    combined_table = pa.concat_tables([old_table, new_table])

    # 4. 重新排序 (重要！因為新資料在最下面，為了維持查詢效能需要重排)
    import pyarrow.compute as pc

    indices = pc.sort_indices(combined_table, sort_keys=[("date", "ascending"), ("stock_id", "ascending")])
    sorted_table = combined_table.take(indices)

    # 5. 寫回檔案 (覆蓋舊檔)
    pq.write_table(sorted_table, "history.parquet", row_group_size=100000)


def read():
    dataset = ds.dataset("db/price/2026_3.parquet", format="parquet")
    table = dataset.to_table()
    # table = dataset.to_table(
    #     filter=ds.field("stock_id") == "2330", columns=["date", "stock_id", "close"]  # 篩選行  # 選取特定欄位
    # )
    df: pd.DataFrame = table.to_pandas()
    print(df.head())
    print(df.tail())


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


def query_stock_info():
    """index,代號,名稱,市場"""
    df = pd.read_parquet("db/info/")
    # df = df[["代號"]]
    return df


#     # print(df.head())
#     # print(df.tail())


# def query_stocks_ids_list():
# df: pd.DataFrame = pd.read_parquet("db/info/")
# return df.iloc[:, 0].values.tolist()


# print(query_info())


def reset_price(group):
    if group.name == "0050":
        group["close"] = group["close"] + 1000
    return group


def create_features(df: pd.DataFrame):
    df.reset_index(drop=True, inplace=True)
    df.to_parquet("db/feature/history.parquet")


def read_features():
    dataset = ds.dataset("db/feature/history.parquet", format="parquet")
    table = dataset.to_table()
    # table = dataset.to_table(
    #     filter=ds.field("stock_id") == "2330", columns=["date", "stock_id", "close"]  # 篩選行  # 選取特定欄位
    # )
    df: pd.DataFrame = table.to_pandas()
    # print(df.head().T)
    # print(df.tail().T)
    print(df.describe())


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
    pass
    # all_stocks = set(s.stock_id for s in MyappActivestocks.select())
    # excluded_stocks = set(stock0050()) | set(stock0056())  # 使用 | 進行聯集
    # return list(all_stocks - excluded_stocks)


#############################################################
# build_stock_parquet("2015-01", "2026-05")
# date_str = datetime.now().strftime("%Y-%m")
# update_price(date_str)


def init_margin():
    qs = StocksMargin.objects.filter(date__gte="2015-01-01", date__lt="2026-05-01")
    df = pd.DataFrame(list(qs.values()))
    df.to_parquet("db/margin/201501_202605.parquet")


if __name__ == "__main__":
    init_margin()
