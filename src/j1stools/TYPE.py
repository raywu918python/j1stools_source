from enum import Enum, Flag, auto


class FEATURE_TYPE(Flag):
    normal = auto()
    macd = auto()
    today = auto()
    abcd = auto()
    margin = auto()


class MODEL_TYPE(Enum):
    rfc = "rfc"
    lgbm = "lgbm"


class TRAIN_TYPE(Enum):
    train = 1
    create_model = 2
    predict = 3
