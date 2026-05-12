from datetime import datetime

from sympy import rf


from j1stools import parquet_db
from j1stools.TYPE import FEATURE_TYPE, FILTER_TYPE, MODEL_TYPE
from j1stools.TYPE import TRAIN_TYPE
from j1stools.model_builder import gen_lgbm_c_model, gen_lgbm_r_model, gen_rfc_model


import random

from j1stools.train_function import lgbm_function_train, lgbm_r_function_train, rfc_train_function


class BaseDataBuilderConfig:
    def __init__(self):
        self.stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
        self.model_run_type = TRAIN_TYPE.train
        self.st = "2024-01-01"
        self.end = "2099-01-01"
        self.trainging_idx = 0.8
        self.pick_import_feature = True
        self.ichcfg = FILTER_TYPE.none_
        self.atrcfg = FILTER_TYPE.add_
        self.pick_import_feature = True
        self.is_using_rfc = False
        self.index_cols = ["date", "stock_id"]
        self.feature_type = FEATURE_TYPE.normal
        self.is_continuous = False
        self.label_cfg = BaseLabelConfig()
        self.train_config: BaseTrainConfig = None


class BaseLabelConfig:
    def __init__(self):
        self.hold_days = 10
        self.profit_target = 0.10
        self.stop_loss = -0.10


class MACDLabelConfig(BaseLabelConfig):
    def __init__(self):
        self.hold_days = 10
        self.profit_target = 0.10
        self.stop_loss = -0.10


class NormalDataBuilderConfig(BaseDataBuilderConfig):
    def __init__(self):
        self.stocks = parquet_db.query_stocks_ids_list()
        self.st = "2024-01-01"
        self.end = "2099-01-01"
        self.trainging_idx = 0.8
        self.pick_import_feature = True
        self.ichcfg = FILTER_TYPE.none_
        self.atrcfg = FILTER_TYPE.add_
        self.pick_import_feature = True
        self.is_using_rfc = False
        self.index_cols = ["date", "stock_id"]
        self.feature_type = FEATURE_TYPE.normal
        self.is_continuous = False


class MACDDataBuilterConfig(BaseDataBuilderConfig):
    def __init__(self):
        super().__init__()
        self.stocks = parquet_db.query_stocks_ids_list()
        self.st = "2015-01-01"
        self.end = "2018-01-01"
        self.trainging_idx = 0.8
        self.ichcfg = FILTER_TYPE.none_
        self.atrcfg = FILTER_TYPE.add_
        self.pick_import_feature = False
        self.feature_type: FEATURE_TYPE = FEATURE_TYPE.macd
        self.is_continuous = False
        self.label_cfg = MACDLabelConfig()


class TodayDataBuilterConfig(BaseDataBuilderConfig):
    def __init__(self):
        super().__init__()
        self.stocks = parquet_db.query_stocks_ids_list()
        self.st = "2015-01-01"
        self.end = "2018-01-01"
        self.trainging_idx = 0.8
        self.ichcfg = FILTER_TYPE.none_
        self.atrcfg = FILTER_TYPE.add_
        self.model_run_type = TRAIN_TYPE.train
        self.pick_import_feature = False
        self.feature_type: FEATURE_TYPE = FEATURE_TYPE.today
        self.is_continuous = False


class BaseTrainConfig:

    def __init__(self):
        self.DEBUG = True
        self.threshold = 0.5
        self.n = 0
        now = datetime.now()
        self.t = now.strftime("%Y%m%d_%H%M%S")
        self.train_type = TRAIN_TYPE.train
        self.model_type = MODEL_TYPE.rfc
        self.model = None


class RfcTrainConfig(BaseTrainConfig):
    def __init__(self):
        super().__init__()
        self.model_type = MODEL_TYPE.rfc
        self.model_run_type = TRAIN_TYPE.train
        self.is_print_import_ft = True
        self.function_train = rfc_train_function
        self.model = gen_rfc_model()


class LgbmTrainConfig(BaseTrainConfig):
    def __init__(self):
        super().__init__()
        self.model_type = MODEL_TYPE.lgbm_c
        self.model_run_type = TRAIN_TYPE.train
        self.is_print_import_ft = True
        self.model = gen_lgbm_c_model()
        self.function_train = lgbm_r_function_train
        self.is_use_rfc = False
