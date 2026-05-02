import numpy as np


class HvFeature:
    def add_feature(df, windows=[5, 10, 20, 60]):
        # 1. 計算對數收益率 (這裡你寫對了，必須 groupby)
        df["log_ret"] = df.groupby("stock_id")["close"].transform(lambda x: np.log(x / x.shift(1)))

        # 建立 groupby 物件，避免在迴圈內重複計算分組，提高速度
        grouped_ret = df.groupby("stock_id")["log_ret"]

        for w in windows:
            col_name = f"f_HV_{w}"
            # 2. 這裡也要 groupby！使用 transform 確保長度跟原表一致
            df[col_name] = grouped_ret.transform(lambda x: x.rolling(window=w).std()) * np.sqrt(252)

        # 3. 擠壓指標 (因為 f_HV 已經是根據各股算的，這裡直接除即可)
        df["f_HV_squeeze"] = df["f_HV_5"] / df["f_HV_20"]

        return df

    def test():
        # pd.read_csv()
        # df = stock("2330")

        # df = HvFeature.add_feature(df)
        # print(df.tail().T)
        pass


# HvFeature.test()
