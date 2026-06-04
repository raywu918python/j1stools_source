from enum import IntFlag
import random
from time import time

import joblib
import numpy as np
import pandas as pd
from regex import P

pd.set_option("future.no_silent_downcasting", True)

from pandas_ta import ma
import vectorbt as vbt

from j1stools import j1s_chart, parquet_db, rfc_main

# ============================================================
# 技術指標
# ============================================================

import os

from j1stools.backtest_engine import backtest_engine

IS_USE_CACHE = False


class LgbmPrepareDate:
    def __init__(self, signal, top_n, threshold):
        self.signal = signal
        self.top_n = top_n
        self.threshold = threshold
        t1 = time()
        #
        stocks = signal["stock_id"].unique().tolist()
        date = signal["date"]
        st = date.min()
        end = date.max()
        # st = "2026-03-01"
        # end = "2029-01-01"
        full_df = parquet_db.query_price(stocks, st, end)
        full_df["date"] = pd.to_datetime(full_df["date"])
        input = pd.merge(
            full_df,
            signal[["date", "stock_id", "pred_score"]],
            on=["date", "stock_id"],
            how="left",  # 只取索引部分  # 以全時段為準
        ).fillna(
            0
        )  # 沒預測到的（ATR太小的）補 0

        self.proba = input.pivot(index="date", columns="stock_id", values="pred_score").fillna(0)
        self.close = input.pivot(index="date", columns="stock_id", values="close").ffill()
        self.high = input.pivot(index="date", columns="stock_id", values="high").ffill()
        self.low = input.pivot(index="date", columns="stock_id", values="low").ffill()
        self.stock_group = parquet_db.query_stock_info().set_index("stock_id")["group"].to_dict()

        # 前處理（向量化）
        self.market_danger = check_market(self.close)
        self.my_filter = gen_filter(self.close, self.high, self.low)
        self.entries = gen_entries(self.my_filter, self.proba, top_n, threshold)
        self.exits = gen_exits(self.market_danger, self.proba)

        print(f"前處理時間: {time() - t1:.2f} 秒")


class PrepareDate:
    def __init__(self, signal, top_n, threshold):
        self.signal = signal
        self.top_n = top_n
        self.threshold = threshold
        t1 = time()
        #
        stocks = signal["stock_id"].unique().tolist()
        date = signal["date"]
        st = date.min()
        end = date.max()
        # st = "2026-03-01"
        # end = "2029-01-01"
        full_df = parquet_db.query_price(stocks, st, end)
        full_df["date"] = pd.to_datetime(full_df["date"])
        input = pd.merge(
            full_df,
            signal[["date", "stock_id", "2"]],
            on=["date", "stock_id"],
            how="left",  # 只取索引部分  # 以全時段為準
        ).fillna(
            0
        )  # 沒預測到的（ATR太小的）補 0

        self.proba = input.pivot(index="date", columns="stock_id", values="2").fillna(0)
        self.close = input.pivot(index="date", columns="stock_id", values="close").ffill()
        self.high = input.pivot(index="date", columns="stock_id", values="high").ffill()
        self.low = input.pivot(index="date", columns="stock_id", values="low").ffill()
        self.stock_group = parquet_db.query_stock_info().set_index("stock_id")["group"].to_dict()

        # 前處理（向量化）
        self.market_danger = check_market(self.close)
        self.my_filter = gen_filter(self.close, self.high, self.low)
        self.entries = gen_entries(self.my_filter, self.proba, top_n, threshold)
        self.exits = gen_exits(self.market_danger, self.proba)

        print(f"前處理時間: {time() - t1:.2f} 秒")


def calc_adx(high, low, close, cache_path="db/adx/adx.parquet", is_using_cache=False):
    os.makedirs("db/adx", exist_ok=True)

    # ✅ 用資料的最後日期當作快取 key
    last_date = str(close.index[-1].date())
    cache_path = f"db/adx/adx_{last_date}.parquet"

    if is_using_cache and os.path.exists(cache_path):
        print("載入 ADX 快取...")
        return pd.read_parquet(cache_path)

    print("計算 ADX...")
    adx = vbt.pandas_ta("ADX").run(high, low, close, length=14).adx
    if isinstance(adx.columns, pd.MultiIndex):
        adx.columns = adx.columns.get_level_values(-1)
    adx.columns = adx.columns.astype(str)
    adx.to_parquet(cache_path)

    # ✅ 清掉舊快取
    for f in os.listdir("db/adx"):
        if f.startswith("adx_") and f != f"adx_{last_date}.parquet":
            os.remove(f"db/adx/{f}")
            print(f"刪除舊快取：{f}")

    return adx


# ============================================================
# 前處理
# ============================================================


def check_market(close_df):
    ma60_df = close_df.rolling(60).mean()
    under_ma60_ratio = (close_df < ma60_df).sum(axis=1) / close_df.shape[1]
    daily_ret = close_df.pct_change()
    crash_8pct_ratio = (daily_ret < -0.08).sum(axis=1) / close_df.shape[1]
    market_danger = (under_ma60_ratio > 0.9) | (crash_8pct_ratio > 0.9)
    return market_danger


def gen_filter(close, high, low):
    daily_return = close.pct_change()
    filter_limit_up = daily_return < 0.09

    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.DataFrame(
        np.maximum(np.maximum(hl.values, hc.values), lc.values),
        index=close.index,
        columns=close.columns,
    )
    atr = tr.rolling(14).mean()

    adx = calc_adx(high, low, close, is_using_cache=IS_USE_CACHE)

    # ✅ 把 MultiIndex 拿掉，只保留 stock_id
    if isinstance(adx.columns, pd.MultiIndex):
        adx.columns = adx.columns.get_level_values("stock_id")

    filter_atr = (((atr / close) > 0.00) & (adx > 0)).fillna(False).infer_objects(copy=False)
    return filter_atr & filter_limit_up


def gen_entries(my_filter, df_proba, top_n=3, proba_threshold=0.6):

    filtered_proba = df_proba.where(my_filter)
    rank_after_filter = filtered_proba.rank(axis=1, ascending=False)
    entries = (rank_after_filter <= top_n) & (df_proba >= proba_threshold)

    return entries


def gen_exits(market_danger, df_proba):
    return market_danger


# ============================================================
# 主流程
# ============================================================


def prepare_data_backtest(
    signal,
    top_n=3,
    threshold=0.6,
    max_positions=10,
    sl_trail=0.15,
    hold_days=5,
    sl_stop=0.10,
    tp_stop=0.15,
    use_sl_trail=True,
    group_limit=3,
    use_hold_days=False,
    use_fixed_sl=False,
    use_fixed_tp=False,
):

    p = PrepareDate(signal, top_n=top_n, threshold=threshold)

    # 回測
    st = time()
    portfolio_value, trades_df, positions = backtest_engine(
        close=p.close,
        entries=p.entries,
        exits=p.exits,
        df_proba=p.proba,
        max_positions=max_positions,
        hold_days=hold_days,
        init_cash=1_000_000,
        fee=0.001,
        use_hold_days=use_hold_days,
        use_sl_trail=use_sl_trail,
        use_fixed_sl=use_fixed_sl,
        use_fixed_tp=use_fixed_tp,
        sl_trail=sl_trail,
        sl_stop=sl_stop,
        group_limit=group_limit,
        tp_stop=tp_stop,
        stock_group=p.stock_group,
    )
    # j1s_chart.chart_allocation(portfolio_value, trades_df, close=p.close)
    # j1s_chart.chart_gantt(trades_df)

    print(f"回測時間: {time() - st:.2f} 秒")

    # 結果
    final_value = portfolio_value.iloc[-1]
    total_return = (final_value / 1_000_000 - 1) * 100
    print(f"最終資產：{final_value:,.0f}")
    print(f"總報酬率：{total_return:.2f}%")
    if not trades_df.empty:
        print(f"總交易次數：{len(trades_df)}")
        print(f"勝率：{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"平均報酬：{trades_df['return_pct'].mean():.2f}%")

    return portfolio_value, trades_df, positions, p.close


def local_signals(file_name):
    signal = pd.read_csv(file_name, dtype={"stock_id": str})
    signal["date"] = pd.to_datetime(signal["date"])
    return signal


def optimize(signal):

    import itertools
    from time import time

    # ── 定義參數範圍 ──────────────────────────────────────
    param_grid = {
        "top_n": [5, 10],
        "proba_threshold": [0.7, 0.9],
        "max_positions": [5, 10],
        "group_limit": [2, 5],
        "use_sl_trail": [True],
        "sl_trail": [0.10, 0.20],
        "use_fixed_sl": [True],
        "sl_stop": [0.1, 0.20],
        "use_fixed_tp": [True],
        "tp_stop": [0.1, 0.20],
        "use_hold_days": [True],
        "hold_days": [5, 22],
        # "use_fixed_sl_tp": [True, False],
        # "use_proba_sizing": [True, False],  # 未實作
    }

    # ── 預先算好 entries（避免重複計算）──────────────────
    p = PrepareDate(signal, top_n=3, threshold=0.9)
    my_filter = gen_filter(p.close, p.high, p.low)

    # ── 產生所有組合 ──────────────────────────────────────
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    print(f"總共 {len(combinations)} 組參數")

    results = []
    t0 = time()

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # 每組 top_n / proba_threshold 都要重新算 entries
        entries = gen_entries(
            my_filter,
            p.proba,
            top_n=params["top_n"],
            proba_threshold=params["proba_threshold"],
        )
        exits = gen_exits(check_market(p.close), p.proba)
        portfolio_value, trades_df, _ = backtest_engine(
            close=p.close,
            entries=entries,
            exits=exits,
            df_proba=p.proba,
            max_positions=params["max_positions"],
            use_sl_trail=params["use_sl_trail"],
            sl_trail=params["sl_trail"],
            use_fixed_sl=params["use_fixed_sl"],
            sl_stop=params["sl_stop"],
            use_fixed_tp=params["use_fixed_tp"],
            tp_stop=params["tp_stop"],
            use_hold_days=params["use_hold_days"],
            hold_days=params["hold_days"],
            stock_group=p.stock_group,
            group_limit=params["group_limit"],
            # use_fixed_sl_tp=params["use_fixed_sl_tp"],
            init_cash=1_000_000,
            fee=0.001,
        )

        if trades_df.empty:
            continue

        final_value = portfolio_value.iloc[-1]
        total_return = (final_value / 1_000_000 - 1) * 100
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        avg_return = trades_df["return_pct"].mean()
        trade_count = len(trades_df)

        # 最大回撤
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        results.append(
            {
                **params,
                "total_return": round(total_return, 2),
                "max_drawdown": round(max_dd, 2),
                "win_rate": round(win_rate, 2),
                "avg_return": round(avg_return, 2),
                "trade_count": trade_count,
                "final_value": round(final_value, 0),
            }
        )

        if (i + 1) % 100 == 0:
            print(f"進度: {i+1}/{len(combinations)}，耗時 {time()-t0:.1f} 秒")

    results_df = pd.DataFrame(results)

    # ── 排序：報酬率高、回撤小 ────────────────────────────
    results_df = results_df.sort_values("total_return", ascending=False)

    print(f"\n總耗時: {time()-t0:.1f} 秒")
    print(f"\n前 10 名：")
    print(results_df.head(10).to_string())

    results_df.to_csv("grid_search_results.csv", index=False)
    return results_df

    # portfolio_value.to_csv("portfolio_value.csv")
    # signal.to_csv("query_last.csv", index=False)


class Chart(IntFlag):
    NONE = 0
    PF = 1  # 2^0
    FLOW = 2  # 2^1
    INFO = 4  # 2^2
    # DELETE = 8    # 2^3


def win6XX(signal):
    p = PrepareDate(signal, top_n=5, threshold=0.7)
    return backtest_engine(
        use_sl_trail=False,
        sl_trail=0.2,
        use_fixed_sl=True,
        sl_stop=0.10,
        use_fixed_tp=True,
        tp_stop=0.10,
        use_hold_days=True,
        hold_days=10,
        #############################################################
        use_fixed_sl_tp=True,
        use_proba_sizing=False,
        #############################################################
        max_positions=5,
        group_limit=2,
        stock_group=p.stock_group,
        #############################################################
        init_cash=1_000_000,
        fee=0.001,
        close=p.close,
        entries=p.entries,
        exits=p.exits,
        df_proba=p.proba,
    )


def main(
    st="2024-01-01",
    end="2026-01-01",
):
    print("======main======")
    if False:
        signal = rfc_main.predict(parquet_db.query_stocks_ids_list(), st, end)
        signal.to_csv("rfc_macd_6xx.csv", index=False)
        p = PrepareDate(signal, top_n=5, threshold=0.7)
    else:
        signal = lgbm_main.predict(parquet_db.query_stocks_ids_list(), st, end)
        signal.to_csv("lgbm_main_signal.csv", index=False)
        p = LgbmPrepareDate(signal, top_n=10, threshold=0.2)
        # signal = local_signals("lgbm_main_signal.csv")

    # signal = local_signals()
    show_chart = Chart.NONE
    # show_chart |= Chart.PF
    # show_chart |= Chart.FLOW
    # show_chart |= Chart.INFO

    #############################################################
    print("======lgbm prepare======")
    print(p.proba.head())
    t1 = time()
    portfolio_value, trades_df, positions = backtest_engine(
        use_sl_trail=True,
        sl_trail=0.1,
        use_fixed_sl=True,
        sl_stop=0.2,
        use_fixed_tp=True,
        tp_stop=0.1,
        use_hold_days=True,
        hold_days=20,
        #############################################################
        use_fixed_sl_tp=True,
        use_proba_sizing=False,
        #############################################################
        max_positions=5,
        group_limit=5,
        stock_group=p.stock_group,
        #############################################################
        init_cash=1_000_000,
        fee=0.001,
        close=p.close,
        entries=p.entries,
        exits=p.exits,
        df_proba=p.proba,
    )
    print(f"回測時間: {time() - t1:.2f} 秒")

    #############################################################
    final_value = portfolio_value.iloc[-1]
    total_return = (final_value / 1_000_000 - 1) * 100
    print(f"最終資產：{final_value:,.0f}")
    print(f"總報酬率：{total_return:.2f}%")
    if not trades_df.empty:
        print(f"總交易次數：{len(trades_df)}")
        print(f"勝率：{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"平均報酬：{trades_df['return_pct'].mean():.2f}%")
    #############################################################

    if show_chart & Chart.PF:
        j1s_chart.plot_performance(
            portfolio_value=portfolio_value,
            trades_df=trades_df,
            is_web=False,
        )
    if show_chart & Chart.FLOW:
        j1s_chart.chart_gantt(trades_df)
    if show_chart & Chart.INFO:
        j1s_chart.chart_allocation(
            portfolio_value,
            trades_df,
            close=p.close,
        )


#############################################################
# optimize(local_signals())
# main()
# raise Exception("未完成")
# 2024-03-18沒資料，之後再檢查
# x, y, z = query_last()
# j1s_chart.plot_performance(
#     portfolio_value=x,
#     trades_df=y,
#     is_web=True,
# )

# query()
# query(good_search=True)
