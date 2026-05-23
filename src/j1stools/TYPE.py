from enum import Enum, Flag, auto


class FEATURE_TYPE(Flag):
    normal = auto()
    macd = auto()
    macd_is_continuous = auto()
    today = auto()
    abcd = auto()
    margin = auto()
    power = auto()
    test_lgbm_feature = auto()
    margin_ibbuysell = auto()


class MODEL_TYPE(Enum):
    rfc = "rfc"
    lgbm_c = "lgbm_c"
    lgbm_r = "lgbm_r"
    lgbm_orgin = "lgbm_orgin"


class TRAIN_TYPE(Enum):
    train = 1
    create_model = 2
    predict = 3


class FILTER_TYPE(Enum):
    add_ = 1
    del_ = 2
    add_and_del = 3
    none_ = 4


class MARGIN_RUN(Flag):
    train = auto()       # walk_forward_train：驗證模型，取 val_model
    evaluate = auto()    # evaluate_selection：用 val_model 評估 OOS 選股品質
    build = auto()       # train_final_model：全量訓練，產生部署用模型
    predict = auto()     # select_stocks：用 final_model 選股

    # 常用組合
    full = train | evaluate | build | predict   # 完整流程（每月底重訓）
    deploy = build | predict                    # 跳過驗證，直接建模部署
    eval_only = train | evaluate                # 只評估，不部署
