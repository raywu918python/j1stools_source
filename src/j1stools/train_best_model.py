import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier

    _has_xgb = True
except ImportError:
    _has_xgb = False

try:
    from catboost import CatBoostClassifier

    _has_cat = True
except ImportError:
    _has_cat = False

import j1stools.parquet_db as parquet_db
from j1stools.TYPE import FILTER_TYPE
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools.train_flow import batter_predict, get_full_name
from j1stools.rfc_main import prepare_data


def train_best_model(
    stocks=None,
    st="2015-01-01",
    end="2099-01-01",
    trainging_idx=0.8,
    cv=3,
):
    if not _has_xgb:
        print("xgboost 未安裝，跳過 XGB")
    if not _has_cat:
        print("catboost 未安裝，跳過 CatBoost")

    if stocks is None:
        stocks = parquet_db.query_stocks_no_etf()

    print("=" * 60, "prepare data")
    df = prepare_data(stocks, st, end, atr_filter_type=FILTER_TYPE.add_and_del)
    xtrain, xtest, ytrain, ytest = rfc_split_date(df, True, True, trainging_idx)
    xtrain, ytrain = drop_na_inf(xtrain, ytrain)
    xtest, ytest = drop_na_inf(xtest, ytest)

    f_cols = [col for col in xtrain.columns if col.startswith("f_")]
    xtrain, xtest = xtrain[f_cols], xtest[f_cols]
    print(f"features: {len(f_cols)}, train: {len(xtrain)}, test: {len(xtest)}")

    # LGBM / CatBoost 專用：加入股票類別欄位（來源：db/info/info.parquet）
    # 找不到 group 的股票用前 2 碼 + "XX" 作為 fallback
    stock2group = parquet_db.query_stock_info().set_index("stock_id")["group"].to_dict()

    def _get_group(sid):
        return stock2group.get(sid, sid[:2] + "XX")

    train_ids = xtrain.index.get_level_values("stock_id")
    test_ids = xtest.index.get_level_values("stock_id")
    train_groups = [_get_group(sid) for sid in train_ids]
    test_groups = [_get_group(sid) for sid in test_ids]
    all_groups = list(set(stock2group.values()) | set(train_groups) | set(test_groups))

    xtrain_cat = xtrain.copy()
    xtest_cat = xtest.copy()
    xtrain_cat["f_group"] = pd.Categorical(train_groups, categories=all_groups)
    xtest_cat["f_group"] = pd.Categorical(test_groups, categories=all_groups)

    pipelines = {
        "rfc": Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "etc": Pipeline([("clf", ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))]),
        "lgbm": Pipeline([("clf", LGBMClassifier(random_state=42, n_jobs=1, class_weight="balanced", verbose=-1))]),
        "gbm": Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
    }
    if _has_xgb:
        pipelines["xgb"] = Pipeline(
            [
                ("clf", XGBClassifier(random_state=42, n_jobs=1, eval_metric="mlogloss")),
            ]
        )
    if _has_cat:
        pipelines["cat"] = Pipeline(
            [
                (
                    "clf",
                    CatBoostClassifier(
                        random_state=42,
                        thread_count=1,
                        verbose=0,
                        auto_class_weights="Balanced",
                        cat_features=["f_group"],
                    ),
                ),
            ]
        )

    param_grids = {
        "rfc": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [8, 12],
            "clf__min_samples_leaf": [10, 20],
            "clf__max_features": [0.2, 0.4],
        },
        "etc": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [8, 12, None],
            "clf__min_samples_leaf": [10, 20],
            "clf__max_features": [0.2, 0.4],
        },
        "lgbm": {
            "clf__n_estimators": [300, 600],
            "clf__num_leaves": [31, 63],
            "clf__max_depth": [6, 10],
            "clf__learning_rate": [0.05, 0.1],
        },
        "gbm": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__min_samples_leaf": [10, 20],
        },
        "xgb": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__min_child_weight": [5, 10],
        },
        "cat": {
            "clf__iterations": [300, 600],
            "clf__depth": [4, 6, 8],
            "clf__learning_rate": [0.05, 0.1],
            "clf__l2_leaf_reg": [1, 3],
        },
    }

    best_score = -np.inf
    best_name = None
    best_pipeline = None
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*60} GridSearch: {name}")
        gs = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=1,
            verbose=1,
            refit=True,
        )
        x_fit = xtrain_cat if name in ("cat", "lgbm") else xtrain
        fit_params = {"clf__categorical_feature": ["f_group"]} if name == "lgbm" else {}
        try:
            with joblib.parallel_backend("threading"):
                gs.fit(x_fit, ytrain, **fit_params)
        except Exception as e:
            import traceback
            print(f"  [SKIP] {name} 訓練失敗: {e}")
            traceback.print_exc()
            continue
        results[name] = {"best_params": gs.best_params_, "cv_score": gs.best_score_}
        print(f"  best_params: {gs.best_params_}")
        print(f"  cv_score:    {gs.best_score_:.4f}")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_name = name
            best_pipeline = gs.best_estimator_

    print(f"\n{'='*60} 結果比較")
    for name, r in results.items():
        mark = " <-- winner" if name == best_name else ""
        print(f"  {name}: {r['cv_score']:.4f}  {r['best_params']}{mark}")

    save_path = get_full_name(f"best_{best_name}")
    joblib.dump(best_pipeline, save_path)
    print(f"\nsaved: {save_path}")

    print("=" * 60, "test on holdout")
    x_eval = xtest_cat if best_name in ("cat", "lgbm") else xtest
    dfyproba = batter_predict(best_pipeline, x_eval, ytest)
    print(dfyproba.head())

    return best_pipeline, best_name, results


if __name__ == "__main__":
    pipeline, best_name, results = train_best_model(
        st="2021-01-01",
        end="2024-01-01",
        cv=3,
    )
