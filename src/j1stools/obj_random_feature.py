from random import sample

import pandas as pd
import numpy as np


class RandomFeature:
    def add_feature(df: pd.DataFrame):
        df["f_random"] = np.random.rand(len(df))
        return df

    def test():
        pass
        # # pd.read_csv()
        # df = stock("2330")

        # df = RandomFeature.add_feature(df)
        # # d = df.describe().T
        # print(df.head().T)


# JS_random_feature.test()
