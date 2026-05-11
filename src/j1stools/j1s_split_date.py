import pandas as pd
from regex import D
from sklearn.model_selection import train_test_split

from j1stools.CONFIG import BaseDataBuilderConfig


def rfc_split_date(df: pd.DataFrame, is_gen_train=True, is_gen_test=True, cfg: BaseDataBuilderConfig = None):
    # group
    df = df.sort_values(by=cfg.index_cols)
    # x = df[[col for col in df.columns if col.startswith("f_")]]
    x = df
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


def lgbm_split_date(
    df: pd.DataFrame,
    is_gen_train_data=True,
    is_gen_test_data=True,
    cfg: BaseDataBuilderConfig = None,
):
    # group
    df.sort_values(by=cfg.index_cols, inplace=True)

    # x = df[[col for col in df.columns if col.startswith("f_")]]
    x = df
    y = df["target"]
    if is_gen_train_data and is_gen_test_data:
        xremain, xtest, yremain, ytest = train_test_split(x, y, test_size=0.15, shuffle=False)
        xtrain, xval, ytrain, yval = train_test_split(xremain, yremain, test_size=0.176, shuffle=False)

        print(f"訓練集大小: {len(xtrain)}")
        print(f"驗證集大小: {len(xval)}")
        print(f"測試集大小: {len(xtest)}")
        return xtrain, xval, xtest, ytrain, yval, ytest
    elif is_gen_train_data:
        xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.15, shuffle=False)
        return xtrain, xval, None, ytrain, yval, None
    elif is_gen_test_data:
        return None, None, x, None, None, y
