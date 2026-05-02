from inspect import getabsfile
import pathlib

# from dotenv import load_dotenv

# def is_has(stock):
#     return os.path.exists(f'{stock}.csv')

# def load_csv(stock):
#     return pd.read_csv(f'{stock}.csv')

# def save_csv(df, stock):
#     df.to_csv(f'{stock}.csv')


# def get_base_url():
#     load_dotenv()
#     return os.getenv("BASE_URL")


# def stock(stock_id: str, is_to_lower=True, is_add_noise=False):
#     df = pd.read_csv(
#         get_base_url() + f"{stock_id}_d1.csv",
#         dtype={"代號": str},
#         parse_dates=["Date"],  # 這裡請填入你 CSV 裡的日期欄位名
#         index_col="Date",
#     )
#     if is_to_lower:
#         df.rename(columns=str.lower, inplace=True)
#     # df.index.name = df.index.name.lower()

#     df["stock_id"] = stock_id
#     df["date"] = df.index
#     df = df[df["volume"] != 0]
#     df = df.loc["2015":]

#     if is_add_noise:
#         noise_level = 0.05
#         sigma = df["close"].std() * noise_level
#         # 生成與資料長度相同的隨機雜訊
#         noise = np.random.normal(0, sigma, len(df))
#         # 產生改動後的股價
#         df["close"] = df["close"] + noise

#     return df


# def get_file_list():
#     df: pd.DataFrame = pd.read_csv(get_base_url() + f"stock_list.csv", dtype={"代號": str})
#     # df = df[~df["代號"].astype(str).str.startswith("00")]
#     list_file = list(df["代號"])
#     return list_file


def keep_recent_files(folder_path, count=10):
    # 1. 建立路徑物件
    path = pathlib.Path(folder_path)
    # 2. 取得所有檔案（排除資料夾），並按修改時間排序（由新到舊）
    # .stat().st_mtime 代表檔案的最後修改時間
    files = sorted(
        [f for f in path.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    # 3. 取得需要刪除的檔案清單（第 10 筆之後的所有檔案）
    files_to_delete = files[count:]
    # 4. 執行刪除
    for file in files_to_delete:
        try:
            file.unlink()
            print(f"已刪除舊檔案: {file.name}")
        except Exception as e:
            print(f"無法刪除 {file.name}: {e}")
    print(f"處理完成，保留了最新的 {len(files[:count])} 筆資料。")
