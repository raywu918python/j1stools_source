import pandas as pd
import numpy as np


def detect_relative_strength_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    相對強弱特徵，適合 LGBM 排名模型。

    計算個股相對大盤和同類股的強弱。

    Parameters
    ----------
    df : pd.DataFrame
        長表格，需包含欄位: date, stock_id, close, volume, group
    market_df : pd.DataFrame
        大盤資料，需包含欄位: date, close（例如 0050 的日收盤價）

    Returns
    -------
    pd.DataFrame
        原始長表格 + f_rs_ / f_mom_ 開頭特徵欄位
    """
    df = df.copy()
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # ── 計算各股報酬率 ─────────────────────────────────────────────── #
    for d in [5, 10, 20]:
        df[f"_ret_{d}d"] = df.groupby("stock_id")["close"].pct_change(d)

    # ── 大盤報酬率 ────────────────────────────────────────────────── #
    market_df = market_df[["date", "close"]].copy().sort_values("date")
    for d in [5, 10, 20]:
        market_df[f"_market_ret_{d}d"] = market_df["close"].pct_change(d)
    market_df = market_df.drop(columns=["close"])

    df = df.merge(market_df, on="date", how="left")

    # ── 類股平均報酬率 ────────────────────────────────────────────── #
    for d in [5, 10, 20]:
        sector_avg = df.groupby(["date", "group"])[f"_ret_{d}d"].transform("mean")
        df[f"_sector_ret_{d}d"] = sector_avg

    # ── Feature 1~2: f_mom_ 動能 ─────────────────────────────────── #
    for d in [5, 10, 20]:
        df[f"f_mom_{d}d"] = df[f"_ret_{d}d"]

    # ── Feature 3~4: f_rs_vs_market_ 個股 vs 大盤 ────────────────── #
    for d in [5, 10, 20]:
        df[f"f_rs_vs_market_{d}d"] = df[f"_ret_{d}d"] - df[f"_market_ret_{d}d"]

    # ── Feature 5~6: f_rs_vs_sector_ 個股 vs 同類股 ──────────────── #
    for d in [5, 10, 20]:
        df[f"f_rs_vs_sector_{d}d"] = df[f"_ret_{d}d"] - df[f"_sector_ret_{d}d"]

    # ── Feature 7: f_rs_sector_vs_market 類股 vs 大盤 ────────────── #
    # 強勢類股內的強勢股，雙重加分
    for d in [5, 20]:
        df[f"f_rs_sector_vs_market_{d}d"] = df[f"_sector_ret_{d}d"] - df[f"_market_ret_{d}d"]

    # ── Feature 8: f_rs_rank_in_sector 類股內排名 ────────────────── #
    # 每天在同類股內的報酬率排名（0~1）
    for d in [5, 20]:
        df[f"f_rs_rank_in_sector_{d}d"] = df.groupby(["date", "group"])[f"_ret_{d}d"].rank(pct=True)

    # ── Feature 9: f_rs_rank_in_market 全市場排名 ────────────────── #
    for d in [5, 20]:
        df[f"f_rs_rank_in_market_{d}d"] = df.groupby("date")[f"_ret_{d}d"].rank(pct=True)

    # ── Feature 10: f_rs_momentum_consistency 動能一致性 ─────────── #
    # 5日、10日、20日動能都是正的才算真正強勢
    df["f_rs_momentum_consistency"] = (
        (df["_ret_5d"] > 0).astype(int) + (df["_ret_10d"] > 0).astype(int) + (df["_ret_20d"] > 0).astype(int)
    ) / 3

    # ── Feature 11: f_rs_volume_momentum 量價配合度 ──────────────── #
    # 漲的時候量大、跌的時候量小 → 強勢股特徵
    vol_ma20 = df.groupby("stock_id")["volume"].transform(lambda x: x.rolling(20).mean())
    rel_vol = df["volume"] / (vol_ma20 + 1e-9)
    df["f_rs_volume_momentum"] = df["_ret_5d"] * rel_vol

    # ── 移除暫存欄位 ──────────────────────────────────────────────── #
    tmp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=tmp_cols)

    return df
