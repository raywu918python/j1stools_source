import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import parquet_db
from j1stools.gmm_classify import (
    load_data,
    run,
    evaluate_clustering,
    analyze_cluster_returns,
    plot_clusters,
    CLASSIFY_FEATURES_RAW,
)

MIN_ATR = 0.02
TRAIN_ST = "2020-01-01"
TRAIN_END = "2023-12-31"
EVAL_ST = "2024-01-01"
MIN_CLUSTER_SIZE = 1000

if __name__ == "__main__":
    clf, _ = run(
        st=TRAIN_ST,
        end=TRAIN_END,
        model="hdbscan",
        n_clusters=MIN_CLUSTER_SIZE,
        min_atr_pct=MIN_ATR,
        retrain=True,
        features=CLASSIFY_FEATURES_RAW,
    )

    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]
    df_test = load_data(stocks, st=EVAL_ST, min_atr_pct=MIN_ATR)
    df_test["cluster"] = clf.predict(df_test)

    evaluate_clustering(clf=clf, df=df_test)
    analyze_cluster_returns(df_test, hold_days=5)
    analyze_cluster_returns(df_test, hold_days=10)
    analyze_cluster_returns(df_test, hold_days=20)
    plot_clusters(clf, df_test)
