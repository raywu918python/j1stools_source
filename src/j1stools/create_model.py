from j1stools import parquet_db
from j1stools.obj_base_model import MODEL_RUN_TYPE


def create_model(step=1):
    """
    訓練模型並儲存
    "2015-01-01", "2020-12-31"
    """
    if step == 1:
        import j1stools.rfc_main as rfc

        rfc.exec(
            parquet_db.query_stocks_ids_list(),
            st="2015-01-01",
            end="2021-01-01",
            is_del_atr=False,
            pick_import_feature=False,
            run_type=MODEL_RUN_TYPE.create_model,
        )
    elif step == 2:
        """
        訓練模型並儲存
        "2021-01-01", "2024-01-01"
        """
        import j1stools.lgbm_main as lgbm

        lgbm.exec(
            parquet_db.query_stocks_ids_list(),
            st="2021-01-01",
            end="2024-01-01",
            is_del_atr=False,
            pick_import_feature=False,
            is_using_rfc=True,
            run_type=MODEL_RUN_TYPE.create_model,
        )


# create_model(2)
