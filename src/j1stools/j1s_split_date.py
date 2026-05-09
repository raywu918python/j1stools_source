import pandas as pd
from regex import D
from sklearn.model_selection import train_test_split

from j1stools.CONFIG import BaseDataBuilderConfig


def rfc_split_date(df: pd.DataFrame, is_gen_train=True, is_gen_test=True, cfg: BaseDataBuilderConfig = None):
    # group
    df = df.sort_values(by=cfg.index_cols)
    x = df[[col for col in df.columns if col.startswith("f_")]]
    y = df["target"]
    if is_gen_test and is_gen_train:
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=cfg.trainging_idx, shuffle=False)
        return xtrain, xtest, ytrain, ytest
    elif is_gen_train:
        return x, None, y, None
    elif is_gen_test:
        return None, x, None, y
    else:
        raise Exception("參數錯誤")
