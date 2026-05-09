import numpy as np


def drop_na_inf(x, y):
    # 1. 把 inf 換成 NaN
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 2. 找出哪些列是乾淨的（沒有 NaN）
    # 注意：xtrain 和 ytrain 的列必須同步刪除，否則 index 會對不起來
    clean_mask = x.isnull().any(axis=1) == False
    x = x[clean_mask]
    y = y[clean_mask]
    return x, y
