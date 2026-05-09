from enum import Enum


class FEATURE_TYPE(Enum):
    normal = 1
    macd = 2
    today = 3
    abcd = 4


class MODEL_TYPE(Enum):
    rfc = "rfc"
    lgbm = "lgbm"


class TRAIN_TYPE(Enum):
    train = 1
    create_model = 2
    predict = 3
