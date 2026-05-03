from j1stools import parquet_db


def create_model():
    """
    訓練模型並儲存
    "2015-01-01", "2020-12-31"
    """
    step = 1
    if step == 1:
        import j1stools.rfc_main as rfc

        rfc.exec(
            parquet_db.query_stocks_ids_list(),
            "2015-01-01",
            "2020-12-31",
            trainging_idx=0.8,
        )
    elif step == 2:
        """
        訓練模型並儲存
        "2021-01-01", "2024-01-01"
        """
        # import j1stools.lgbm_main as lgbm

        # lgbm.exec(parquet_db.query_stocks_ids_list(), "2021-01-01", "2024-01-01", trainging_idx=0.8, is_using_rfc=True)


# create_model()
