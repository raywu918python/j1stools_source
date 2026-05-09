from j1stools import parquet_db
from j1stools.TYPE import FEATURE_TYPE
from j1stools.obj_base_model import MODEL_RUN_TYPE
from j1stools.obj_filter_data import FILTER_CONFIG


import random


class BaseDataBuilderConfig:
    def __init__(self):
        self.stocks = random.sample(parquet_db.query_stocks_ids_list(), 100)
        self.model_run_type = MODEL_RUN_TYPE.train
        self.st = "2024-01-01"
        self.end = "2099-01-01"
        self.trainging_idx = 0.8
        self.pick_import_feature = True
        self.ichcfg = FILTER_CONFIG.none_
        self.atrcfg = FILTER_CONFIG.add_
        self.pick_import_feature = True
        self.is_using_rfc = False
        self.index_cols = ["date", "stock_id"]
        self.feature_type = FEATURE_TYPE.normal
        self.is_continuous = False
        self.label_cfg = BaseLabelConfig()


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
        self.model_run_type = MODEL_RUN_TYPE.train
        self.st = "2024-01-01"
        self.end = "2099-01-01"
        self.trainging_idx = 0.8
        self.pick_import_feature = True
        self.ichcfg = FILTER_CONFIG.none_
        self.atrcfg = FILTER_CONFIG.add_
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
        self.ichcfg = FILTER_CONFIG.none_
        self.atrcfg = FILTER_CONFIG.add_
        self.model_run_type = MODEL_RUN_TYPE.train
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
        self.ichcfg = FILTER_CONFIG.none_
        self.atrcfg = FILTER_CONFIG.add_
        self.model_run_type = MODEL_RUN_TYPE.train
        self.pick_import_feature = False
        self.feature_type: FEATURE_TYPE = FEATURE_TYPE.today
        self.is_continuous = False
