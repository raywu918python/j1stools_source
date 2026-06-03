from enum import IntFlag
from time import time

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

import vectorbt as vbt

from j1stools import j1s_chart, parquet_db
from j1stools.backtest_engine import backtest_engine
from release.rfc_macd import rfc_main

IS_USE_CACHE = False


class PrepareDate:
    def __init__(self, signal, top_n, threshold):
        self.signal = signal
        self.top_n = top_n
        self.threshold = threshold
        t1 = time()

        stocks = signal["stock_id"].unique().tolist()
        date = signal["date"]
        st = pd.to_datetime(date.min()).strftime("%Y-%m-%d")
        end = pd.to_datetime(date.max()).strftime("%Y-%m-%d")

        full_df = parquet_db.query_price(stocks, st, end)
        full_df["date"] = pd.to_datetime(full_df["date"])
        signal = signal.copy()
        signal["date"] = pd.to_datetime(signal["date"])
        input = pd.merge(
            full_df,
            signal[["date", "stock_id", "2"]],
            on=["date", "stock_id"],
            how="left",
        ).fillna(0)

        self.proba = input.pivot(index="date", columns="stock_id", values="2").fillna(0)
        self.close = input.pivot(index="date", columns="stock_id", values="close").ffill()
        self.high = input.pivot(index="date", columns="stock_id", values="high").ffill()
        self.low = input.pivot(index="date", columns="stock_id", values="low").ffill()
        info = parquet_db.query_stock_info()
        self.stock_group = info.set_index("stock_id")["group"].to_dict()

        self.market_danger = check_market(self.close)
        self.my_filter = gen_filter(self.close, self.high, self.low)
        self.entries = gen_entries(self.my_filter, self.proba, top_n, threshold)
        self.exits = gen_exits(self.market_danger, self.proba)

        print(f"前處理時間: {time() - t1:.2f} 秒")


def calc_adx(high, low, close, cache_path="db/adx/adx.parquet", is_using_cache=False):
    import os
    os.makedirs("db/adx", exist_ok=True)

    last_date = str(close.index[-1].date())
    cache_path = f"db/adx/adx_{last_date}.parquet"

    if is_using_cache and os.path.exists(cache_path):
        print("載入 ADX 快取...")
        return pd.read_parquet(cache_path)

    print("計算 ADX...")
    adx = vbt.pandas_ta("ADX").run(high, low, close, length=14).adx
    if isinstance(adx.columns, pd.MultiIndex):
        adx.columns = adx.columns.get_level_values(-1).astype(str)
    else:
        adx.columns = adx.columns.astype(str)
    adx.to_parquet(cache_path)

    for f in os.listdir("db/adx"):
        if f.startswith("adx_") and f != f"adx_{last_date}.parquet":
            os.remove(f"db/adx/{f}")
            print(f"刪除舊快取：{f}")

    return adx


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

    print(f"回測時間: {time() - st:.2f} 秒")

    final_value = portfolio_value["total"].iloc[-1]
    total_return = (final_value / 1_000_000 - 1) * 100
    print(f"最終資產：{final_value:,.0f}")
    print(f"總報酬率：{total_return:.2f}%")
    if not trades_df.empty:
        print(f"總交易次數：{len(trades_df)}")
        print(f"勝率：{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"平均報酬：{trades_df['return_pct'].mean():.2f}%")

    return portfolio_value, trades_df, positions, p.close


def main(
    st="2024-01-01",
    end="2099-01-01",
):
    stocks = parquet_db.query_stocks_no_etf()
    signal = rfc_main.predict(stocks, st, end)
    portfolio_value, trades_df, positions, close_df = prepare_data_backtest(
        signal,
        top_n=5,
        threshold=0.6,
        max_positions=5,
        use_sl_trail=False,
        use_fixed_sl=True,
        sl_stop=0.10,
        use_fixed_tp=True,
        tp_stop=0.10,
        use_hold_days=True,
        hold_days=10,
        group_limit=2,
    )

    # 大盤對比（0050）
    eq_dates = pd.to_datetime(portfolio_value["date"]).dt.normalize()
    mkt = parquet_db.query_price(["0050"], st, end)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
    mkt = mkt[mkt["date"].isin(eq_dates)].reset_index(drop=True)
    if mkt.empty:
        mkt = parquet_db.query_price(["0050"], st, end)
        mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
        mkt = mkt[mkt["date"] >= eq_dates.iloc[0]].reset_index(drop=True)
    start_val = portfolio_value["total"].iloc[0]
    mkt["total"] = (mkt["close"] / mkt["close"].iloc[0] * start_val).round(2)
    market_df = mkt[["date", "total"]]

    return portfolio_value, trades_df, positions, close_df, market_df


if __name__ == "__main__":
    main()
