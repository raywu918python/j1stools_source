import lightgbm as lgb

from j1stools.RESULT import DataBuilderResult


def rfc_train_function(d: DataBuilderResult, model):
    sample_weights = d.ytrain.map({0: 1, 1: 1, 2: 3})
    # m = xtrain["f_atr_just"]
    # xtrain = xtrain.drop(["f_atr_just"], axis=1)
    # print(m.head(10))

    model.fit(
        d.xtrain,
        d.ytrain,
        # sample_weight=m.values**2,
        # sample_weight=sample_weights,
    )


def lgbm_function_train(d: DataBuilderResult, model):
    # sample_weights = ytrain.map({0: 1, 1: 1, 2: 3})

    model.fit(
        d.xtrain,
        d.ytrain,
        eval_set=[(d.xval, d.yval)],
        eval_metric="multi_logloss",
        callbacks=[
            # 如果 50 輪內沒進步就停止
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
