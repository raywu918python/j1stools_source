from enum import Enum, Flag, auto


class FEATURE_TYPE(Flag):
    normal = auto()
    macd = auto()
    today = auto()
    abcd = auto()
    margin = auto()
    power = auto()
    test_lgbm_feature = auto()


class MODEL_TYPE(Enum):
    rfc = "rfc"
    lgbm_c = "lgbm_c"
    lgbm_r = "lgbm_r"


class TRAIN_TYPE(Enum):
    train = 1
    create_model = 2
    predict = 3


class FILTER_TYPE(Enum):
    add_ = 1
    del_ = 2
    add_and_del = 3
    none_ = 4
