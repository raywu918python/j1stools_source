from j1stools import feature_builder, parquet_db
from j1stools.CONFIG import BaseDataBuilderConfig
from j1stools.j1s_split_date import rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools.obj_base_model import MODEL_RUN_TYPE
from j1stools.obj_filter_data import FilterData
from j1stools.obj_label import Label


class DataBuilderResult:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest


class DataBuilder:
    def __init__(self, cfg: BaseDataBuilderConfig):
        self.cfg = cfg

    def build(self):
        cfg = self.cfg
        if cfg.model_run_type == MODEL_RUN_TYPE.train:
            is_gen_train_data = True
            is_gen_test_data = True
        elif cfg.model_run_type == MODEL_RUN_TYPE.create_model:
            is_gen_train_data = True
            is_gen_test_data = False
        elif cfg.model_run_type == MODEL_RUN_TYPE.predict:
            is_gen_train_data = False
            is_gen_test_data = True

        df = parquet_db.query_price(cfg.stocks, cfg.st, cfg.end)
        print("delete.before:", df.shape)
        df = feature_builder.gen_feature(df, cfg)
        df = Label.add_label(df, cfg.label_cfg.hold_days, cfg.label_cfg.profit_target, cfg.label_cfg.stop_loss)
        df = FilterData.get_data(
            df,
            log=True,
            ichcfg=cfg.ichcfg,
            atrcfg=cfg.atrcfg,
        )
        # parquet_db.create_features(df)
        # 資料在這裡刪
        df.set_index(cfg.index_cols, inplace=True)
        xtrain, xtest, ytrain, ytest = rfc_split_date(df, is_gen_train_data, is_gen_test_data, cfg)
        if is_gen_train_data:
            xtrain, ytrain = drop_na_inf(xtrain, ytrain)
            if cfg.pick_import_feature:
                xtrain = feature_builder.pick_feature(xtrain, cfg.is_using_rfc)
        if is_gen_test_data:
            xtest, ytest = drop_na_inf(xtest, ytest)
            if cfg.pick_import_feature:
                xtest = feature_builder.pick_feature(xtest)

        return DataBuilderResult(xtrain, xtest, ytrain, ytest)

    def test():
        pass
