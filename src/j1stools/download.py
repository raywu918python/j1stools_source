from patsy.mgcv_cubic_splines import te
import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from j1stools.utils import get_base_url


def download_from_yf(stock, min, market):
    min_dic = {1: "1m", 5: "5m", 0: "d"}
    market_dic = {"市": "TW", "櫃": "TWO"}
    template = None

    if min == 0:
        load_dotenv()
        s = os.getenv("START")
        e = os.getenv("END")

        template = yf.download(f"{stock}.{market_dic[market]}", start=s, end=e, repair=True)
    else:
        template = yf.download(
            tickers=f"{stock}.{market_dic[market]}",
            period="5d",
            interval=min_dic[min],
            repair=True,
        )

        template.index = template.index.tz_convert("Asia/Taipei")

    # 如果你希望移除時區符號 (+08:00)，只保留時間數值
    template.index = template.index.tz_localize(None)

    # 1. 先只抓取特定的欄位 (通常 yf 下載回來會有 'Adj Close')
    template = template[["Open", "High", "Low", "Close", "Volume"]]

    # 2. 再進行重新命名 (如果你需要自定義名稱或大小寫)
    template.columns = ["Open", "High", "Low", "Close", "Volume"]

    return template


def getDataFM(stock):
    # if(utils.isFileExit(stock)):
    # return utils.getCsvFile(stock)

    load_dotenv()
    s = os.getenv("START")
    e = os.getenv("END")

    token = os.getenv("FINMIND_TOKEN")
    fm = DataLoader(token)
    print(f"fm.api_usage_limit = {fm.api_usage_limit}")
    print(f"fm.api_usage = {fm.api_usage}")
    # 先抓取原始資料物件
    df = fm.taiwan_stock_daily(stock_id=stock, start_date=s, end_date=e)

    # df = fm.taiwan_stock_kbar(
    #     stock_id='2330',
    #     date='2026-03-04'
    # )

    # utils.save(df,'2330_20260309_1.csv')
    return df


# data = getDataFM('2330')
# data = getDataYF('2330')
# data.columns = [i.lower() for i in data.columns]
# print(data.sample)


#
# 產生檔案
#
def format_stock_file(file_name):
    df = pd.read_csv(f"{get_base_url() + file_name}")
    # print(df.head(5)["代號"].re('=',''))

    # code = df.head(5).to_string.strip('="')
    df["代號"] = df["代號"].str.replace("=", "", regex=False).str.replace('"', "", regex=False)
    # df['代號'] = df['代號'].str.replace('=', '', regex=False)
    df = df[df["代號"].str.len() == 4]

    new_df = df[["代號", "名稱", "市場"]]
    # code = df['代號']
    # print(new_df.head)
    return new_df


def save_download_stock_file():
    # https://goodinfo.tw/tw/StockList.asp?MARKET_CAT=%E7%86%B1%E9%96%80%E6%8E%92%E8%A1%8C&INDUSTRY_CAT=%E6%88%90%E4%BA%A4%E5%BC%B5%E6%95%B8+%28%E9%AB%98%E2%86%92%E4%BD%8E%29%40%40%E6%88%90%E4%BA%A4%E5%BC%B5%E6%95%B8%40%40%E7%94%B1%E9%AB%98%E2%86%92%E4%BD%8E&REINIT=46113.9056944444
    df1 = format_stock_file("StockList1.csv")
    df2 = format_stock_file("StockList2.csv")
    df3 = format_stock_file("StockList3.csv")
    df4 = format_stock_file("StockList4.csv")
    df5 = format_stock_file("StockList5.csv")
    df6 = format_stock_file("StockList6.csv")
    df7 = format_stock_file("StockList7.csv")
    df7 = format_stock_file("StockList8.csv")
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7]).drop_duplicates(subset=["代號"], ignore_index=True)
    # )  # .reset_index(drop=True)
    # df = df.drop_duplicates(subset=["代號"], keep="first").reset_index()

    # df.to_csv(get_base_url() + "stock_list.csv", index=False, header=False)
    df.to_csv(get_base_url() + "stock_list.csv")


def save_price_file(is_min, filter_stock=None):
    df_from = pd.read_csv(get_base_url() + "stock_list.csv", dtype={"代號": str})

    test = False
    if test:
        # 5871
        df_from = df_from[df_from["代號"] == "0050"]

        for stock, market in zip(df_from["代號"], df_from["市場"]):
            pass
            # download_m(stock, 5, market)
            # download_from_yf(stock, 0, market)

            # df_now = download_from_yf(stock, 0, market)
            # print(f'{df_now.head}')
            # df_old = pd.read_csv(f'{stock}_m1.csv', dtype={'代號': str}
            #                     ,parse_dates=['Datetime'] # 這裡請填入你 CSV 裡的日期欄位名
            #                     ,index_col='Datetime')

            # utils.save_csv(df_now, f'{stock}_m1')

    elif is_min:
        for stock, market in zip(df_from["代號"], df_from["市場"]):
            print(f"代號: {stock}, 市場: {market}")
            download_m(stock, 1, market)
            download_m(stock, 5, market)

    else:
        for stock, market in zip(df_from["代號"], df_from["市場"]):

            if filter_stock != None:
                if stock != filter_stock:
                    continue

            # for stock, market in zip(["0050"], ["市"]):
            df_new = download_from_yf(stock, 0, market)
            if not df_new.empty:
                filename = f"{get_base_url()+stock}_d1.csv"
                try:
                    df_old = pd.read_csv(
                        filename,
                        dtype={"代號": str},
                        parse_dates=["Date"],  # 這裡請填入你 CSV 裡的日期欄位名
                        index_col="Date",
                    )
                except FileNotFoundError:
                    print(f"找不到檔案 {filename}，將建立新的 DataFrame")
                    df_old = pd.DataFrame()
                df_combined = (
                    pd.concat([df_old, df_new])
                    .reset_index()
                    .drop_duplicates(subset=["Date"], keep="last")
                    .set_index("Date")
                    .sort_index()
                )
                df_combined.to_csv(f"{get_base_url()+stock}_d1.csv")


def download_m(stock, min, market):
    df_now = download_from_yf(stock, min, market)
    if not df_now.empty:
        filename = f"{get_base_url()+stock}_m{min}.csv"
        try:
            df_old = pd.read_csv(
                filename,
                dtype={"代號": str},
                parse_dates=["Datetime"],
                index_col="Datetime",
            )
        except FileNotFoundError:
            print(f"找不到檔案 {filename}，將建立新的 DataFrame")
            df_old = pd.DataFrame()

        df_combined = (
            pd.concat([df_old, df_now])
            .reset_index()
            .drop_duplicates(subset=["Datetime"], keep="last")
            .set_index("Datetime")
            .sort_index()
        )
        df_combined.to_csv(f"{get_base_url()+stock}_m{min}.csv")


def load_m30():
    # 1. 讀取資料 (確保 Datetime 是 index 並轉換為時間格式)
    path = get_base_url() + "0050_m5.csv"

    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)

    # 2. 過濾出每天 09:00 到 09:30 的資料
    # between_time 會檢查索引中的「時間」部分，忽略日期
    df_first_30m = df.between_time("09:00", "09:30")

    # 3. 查看結果
    print(df_first_30m.head(35))  # 可以看到跨日期的前 30 分鐘


def clear_format(file_name):
    # ,Date,Unnamed: 0.1,Unnamed: 0,Open,High,Low,Close,Volume
    df = pd.read_csv(f"{get_base_url() + file_name}_d1.csv")
    d = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    d.set_index("Date").to_csv(get_base_url() + file_name + "_d1.csv")
    # print(d.head())


if __name__ == "__main__":

    # save_download_stock_file()
    save_price_file(0)
    save_price_file(1)

    # df = download_from_yf("0050", 0, "市")
    # print(df.columns)
    # print(df.tail())

    # Price           Close       High        Low       Open Repaired?     Volume
    # Ticker        0050.TW    0050.TW    0050.TW    0050.TW   0050.TW    0050.TW
    # Date
    # 2026-04-17  84.150002  84.500000  84.000000  84.199997     False   93304532
    # 2026-04-20  84.550003  85.099998  84.449997  84.550003     False   95994589
    # 2026-04-21  86.000000  86.099998  85.000000  85.500000     False   71987659
    # 2026-04-22  86.349998  86.599998  85.550003  85.750000      True   58297374
    # 2026-04-23  86.150002  88.800003  85.150002  87.650002     False  139715388

    # load_dotenv()
    # s = os.getenv("START")
    # e = os.getenv("END")
    # template = yf.download(f"6494.TWO", start=s, end=e, repair=True)
    # # df = pd.DataFrame(template)

    # # 如果你希望移除時區符號 (+08:00)，只保留時間數值
    # template.index = template.index.tz_localize(None)

    # # 1. 先只抓取特定的欄位 (通常 yf 下載回來會有 'Adj Close')
    # template = template[["Open", "High", "Low", "Close", "Volume"]]

    # # 2. 再進行重新命名 (如果你需要自定義名稱或大小寫)
    # template.columns = ["Open", "High", "Low", "Close", "Volume"]
    # # print(df.tail())

    pass
