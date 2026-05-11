#
# training
#
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


def gen_rfc_model():
    return RandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=15,  # 稍微下修，對 1000 筆數據較友善
        max_depth=10,  # 數據量小，建議深度再淺一點 (12 -> 10)，防過擬合
        max_features=0.3,  # 40個特徵，抽 8 個比較能抓到關鍵特徵 (0.1太少)
        # 手動指定權重，避免自動權重在小樣本下的極端波動
        # class_weight={0: 2.5, 1: 2.5, 2: 1},
        # class_weight={0: 1, 1: 2.5, 2: 6},
        # class_weight={0: 1, 1: 1, 2: 10},
        # 增加 OOB 評估，讓你在訓練完可以直接看 OOB Score 準不準
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        criterion="entropy",
        n_jobs=-1,
    )


def gen_lgbm_r_model():
    from lightgbm import LGBMRegressor  # 不是 LGBMClassifier

    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="regression",  # 明確指定回歸
        metric="rmse",  # 回歸用 rmse
    )


def gen_lgbm_c_model():
    model = LGBMClassifier(
        class_weight="balanced",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return model
