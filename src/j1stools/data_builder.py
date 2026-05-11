import scipy as sp

from j1stools import feature_builder, lite_db, parquet_db
from j1stools.CONFIG import BaseDataBuilderConfig, LgbmTrainConfig
from j1stools.RESULT import DataBuilderResult
from j1stools.j1s_split_date import lgbm_split_date, rfc_split_date
from j1stools.model_utils import drop_na_inf
from j1stools.TYPE import FEATURE_TYPE, MODEL_TYPE, TRAIN_TYPE
from j1stools.obj_filter_data import FilterData
from j1stools.label_builder import power_label, profit_label


class DataBuilder:
    def __init__(self, cfg: BaseDataBuilderConfig):
        self.cfg = cfg

    def build(self):
        data = self.cfg
        if data.model_run_type == TRAIN_TYPE.train:
            is_gen_train_data = True
            is_gen_test_data = True
        elif data.model_run_type == TRAIN_TYPE.create_model:
            is_gen_train_data = True
            is_gen_test_data = False
        elif data.model_run_type == TRAIN_TYPE.predict:
            is_gen_train_data = False
            is_gen_test_data = True
        #############################################################gen df
        if FEATURE_TYPE.test_lgbm_feature in data.feature_type:
            df = lite_db.margin_group(data.stocks, data.st, data.end)
        elif FEATURE_TYPE.power in data.feature_type:
            df = lite_db.group(data.stocks, data.st, data.end)
        elif FEATURE_TYPE.margin in data.feature_type:
            df = lite_db.margin(data.stocks, data.st, data.end)
        else:
            df = parquet_db.query_price(data.stocks, data.st, data.end)
        #############################################################gen feature
        df = feature_builder.gen_feature(df, data)
        if data.train_config.model_type == MODEL_TYPE.lgbm_c and data.train_config.is_use_rfc:
            df = feature_builder.add_rfc_feature(df, data)
        #############################################################gel label
        if self.cfg.train_config.model_type == MODEL_TYPE.lgbm_r:
            df = power_label(df)
        else:
            df = profit_label(df, data.label_cfg.hold_days, data.label_cfg.profit_target, data.label_cfg.stop_loss)
        #############################################################filter data
        print("delete.before:", df.shape)
        df = FilterData.get_data(
            df,
            log=True,
            ichcfg=data.ichcfg,
            atrcfg=data.atrcfg,
        )
        if FEATURE_TYPE.abcd in data.feature_type:
            mask = df["f_n_confirmed"] == 1
            df = df[mask]
            df.drop(columns=["f_n_confirmed"], inplace=True)
        print("delete.after:", df.shape)
        # 資料在這裡刪
        df.set_index(data.index_cols, inplace=True)
        #############################################################split_date data
        if data.train_config.model_type == MODEL_TYPE.rfc:
            xtrain, xtest, ytrain, ytest = rfc_prepare_data(df, is_gen_train_data, is_gen_test_data, data)
            return DataBuilderResult(xtrain, xtest, ytrain, ytest)
        elif data.train_config.model_type == MODEL_TYPE.lgbm_c or data.train_config.model_type == MODEL_TYPE.lgbm_r:
            xtest_future_return = df["future_return"]  # 實際報酬率
            xtrain, xval, xtest, ytrain, yval, ytest = lgbm_prepare_data(df, is_gen_train_data, is_gen_test_data, data)

            return DataBuilderResult(
                xtrain,
                xtest,
                ytrain,
                ytest,
                xval=xval,
                yval=yval,
                xtest_future_return=xtest_future_return,
            )
        else:
            raise Exception("未知的模型類型")

    def test():
        pass


def rfc_prepare_data(df, is_gen_train_data, is_gen_test_data, cfg):
    xtrain, xtest, ytrain, ytest = rfc_split_date(df, is_gen_train_data, is_gen_test_data, cfg)
    if is_gen_train_data:
        xtrain, ytrain = drop_na_inf(xtrain, ytrain)
        if cfg.pick_import_feature:
            xtrain = feature_builder.pick_feature(xtrain, cfg.is_using_rfc)
    if is_gen_test_data:
        xtest, ytest = drop_na_inf(xtest, ytest)
        if cfg.pick_import_feature:
            xtest = feature_builder.pick_feature(xtest)

    return xtrain, xtest, ytrain, ytest


def lgbm_prepare_data(df, is_gen_train_data, is_gen_test_data, cfg):
    train_cfg: LgbmTrainConfig = cfg.train_config
    is_use_rfc = train_cfg.is_use_rfc

    xtrain, xval, xtest, ytrain, yval, ytest = lgbm_split_date(df, is_gen_train_data, is_gen_test_data, cfg)

    if is_gen_train_data:
        xtrain, ytrain = drop_na_inf(xtrain, ytrain)
        xval, yval = drop_na_inf(xval, yval)
        if cfg.pick_import_feature:
            xtrain = feature_builder.pick_feature(xtrain, is_use_rfc)
            xval = feature_builder.pick_feature(xval, is_use_rfc)
    if is_gen_test_data:
        xtest, ytest = drop_na_inf(xtest, ytest)
        if cfg.pick_import_feature:
            xtest = feature_builder.pick_feature(xtest, is_use_rfc)

    return xtrain, xval, xtest, ytrain, yval, ytest
