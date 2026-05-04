import pandas as pd


def help(df: pd.DataFrame):
    """寬表格用法"""
    ma20 = df["close"].ewm(span=5, adjust=False).mean()

    ma20a = ma20.stack(future_stack=True).rename("close_ma20")  # 單一欄位
    ma20a = ma20a.reset_index()  # 單一欄位
    print(ma20a.head())
    ma20a.columns = [f"{col}_ma20" for col in ma20a.columns]  # 多欄位
    print(ma20a.head())
    # 4. 併回原表
    df = df.merge(ma20a.reset_index(), on=["date", "stock_id"], how="left")


class MaFeature:
    def add_feature(df: pd.DataFrame):
        dfclose = df.pivot(index="date", columns="stock_id", values="close")
        dfclose.to_csv("dfclose.csv")

        # ma = adjust=False, ema = adjust=True
        ma5 = dfclose.ewm(span=5, adjust=False).mean()
        ma10 = dfclose.ewm(span=10, adjust=False).mean()
        ma20 = dfclose.ewm(span=20, adjust=False).mean()
        f_ma5_d = dfclose / ma5
        f_ma10_d = dfclose / ma10
        f_ma20_d = dfclose / ma20
        f_ma20_change = f_ma20_d.pct_change(periods=3, fill_method=None)

        df = df.set_index(["date", "stock_id"])
        df["f_ma5_d"] = f_ma5_d.stack(future_stack=True)
        df["f_ma10_d"] = f_ma10_d.stack(future_stack=True)
        df["f_ma20_d"] = f_ma20_d.stack(future_stack=True)
        df["f_ma20_change"] = f_ma20_change.stack(future_stack=True)

        df = df.reset_index()
        # print(df.head().T)

        return df

    def test():
        pass
        # # pd.read_csv()
        # df = stock("2330")

        # df = MaFeature.add_feature(df)
        # print(df.describe().T)
        # # print(df.head().T)


# MaFeature.test()
