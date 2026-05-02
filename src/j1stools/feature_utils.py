from logging import addLevelName

import pandas as pd


def add_lag(column: str, df: pd.DataFrame, lag_count=5, step=1):
    dic = {}
    for i in range(1, lag_count + 1, 5):
        dic[f"{column}_lag{i}"] = df[column].shift(i)
    rs = pd.DataFrame(dic)
    return pd.concat([df, rs], axis=1)
