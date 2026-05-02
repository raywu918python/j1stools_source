import time

import numpy as np
import pandas as pd
from pandas_ta import ma
import vectorbt as vbt

import lgbm_main
import parquet_db

# ============================================================
# 技術指標
# ============================================================

import os


def calc_adx(high, low, close, cache_path="cache/adx.pkl"):
    os.makedirs("cache", exist_ok=True)

    # ✅ 用資料的最後日期當作快取 key
    last_date = str(close.index[-1].date())
    cache_path = f"cache/adx_{last_date}.pkl"

    if os.path.exists(cache_path):
        print("載入 ADX 快取...")
        return pd.read_pickle(cache_path)

    print("計算 ADX...")
    adx = vbt.pandas_ta("ADX").run(high, low, close, length=14).adx
    adx.to_pickle(cache_path)

    # ✅ 清掉舊快取
    for f in os.listdir("cache"):
        if f.startswith("adx_") and f != f"adx_{last_date}.pkl":
            os.remove(f"cache/{f}")
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

    adx = calc_adx(high, low, close)

    # ✅ 把 MultiIndex 拿掉，只保留 stock_id
    if isinstance(adx.columns, pd.MultiIndex):
        adx.columns = adx.columns.get_level_values("stock_id")

    filter_atr = (((atr / close) > 0.05) & (adx > 20)).fillna(False)
    return filter_atr & filter_limit_up


def gen_entries(my_filter, df_proba, top_n=3, proba_threshold=0.6):

    filtered_proba = df_proba.where(my_filter)
    rank_after_filter = filtered_proba.rank(axis=1, ascending=False)
    entries = (rank_after_filter <= top_n) & (df_proba >= proba_threshold)

    return entries


def gen_exits(market_danger, df_proba):
    return market_danger


# ============================================================
# 輕量回測引擎
# ============================================================


def simple_backtest(
    close,
    entries,
    exits,
    df_proba,
    stock_group,
    max_positions=10,
    sl_trail=0.1,
    hold_days=5,
    init_cash=1_000_000,
    fee=0.001,
    group_limit=3,  # ✅ 每族群最多 3 支
):
    dates = close.index
    cash = float(init_cash)
    positions = {}
    portfolio_value = []
    trades = []

    # ✅ 按機率排序
    entries_dict = {}
    for dt, row in entries.iterrows():
        if row.any():
            true_cols = row[row].index.tolist()
            true_cols = sorted(true_cols, key=lambda x: df_proba.loc[dt, x], reverse=True)
            entries_dict[dt] = true_cols

    exits_arr = exits.values

    for i, dt in enumerate(dates):
        is_danger = bool(exits_arr[i]) if i < len(exits_arr) else False

        # ── 出場 ──────────────────────────────────────────
        for sid in list(positions.keys()):
            if sid not in close.columns:
                continue
            price = close.loc[dt, sid]
            pos = positions[sid]
            pos["highest"] = max(pos["highest"], price)

            should_exit = is_danger or price <= pos["highest"] * (1 - sl_trail)
            # or (i - pos["entry_bar"]) >= hold_days

            if should_exit:
                sell_value = price * pos["shares"] * (1 - fee)
                pnl = sell_value - pos["cost"]
                cash += sell_value
                trades.append(
                    {
                        "stock_id": sid,
                        "entry_date": pos["entry_date"],
                        "exit_date": dt,
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "pnl": pnl,
                        "size": pos["shares"],
                        "return_pct": pnl / pos["cost"] * 100,
                    }
                )
                del positions[sid]

        # ── 進場（市場危險時不進場）──────────────────────
        if not is_danger:
            slots = max_positions - len(positions)
            if slots > 0:
                candidates = [sid for sid in entries_dict.get(dt, []) if sid not in positions and sid in close.columns][
                    :slots
                ]

                if candidates:
                    per_slot = cash / slots
                    for sid in candidates:
                        # ✅ 族群限制
                        if stock_group is not None:
                            group = stock_group.get(sid, "未知")
                            current_group_count = sum(1 for s in positions if stock_group.get(s, "未知") == group)
                            if current_group_count >= group_limit:
                                print(f"🚫 {sid} 族群 {group} 已滿 {group_limit} 支")
                                continue
                        price = close.loc[dt, sid]
                        if price <= 0:
                            continue
                        # ✅ 手續費只算一次
                        cost = per_slot
                        shares = cost * (1 - fee) / price
                        cash -= cost
                        positions[sid] = {
                            "shares": shares,
                            "entry_price": price,
                            "entry_date": dt,
                            "entry_bar": i,
                            "highest": price,
                            "cost": cost,
                        }

        # ── 每日資產價值 ──────────────────────────────────
        pos_value = sum(close.loc[dt, sid] * pos["shares"] for sid, pos in positions.items() if sid in close.columns)
        portfolio_value.append(cash + pos_value)

    portfolio_value = pd.Series(portfolio_value, index=dates)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return portfolio_value, trades_df, positions


# ============================================================
# 主流程
# ============================================================


def main(
    close,
    high,
    low,
    df_proba,
    st,
    end,
    top_n=3,
    proba_threshold=0.6,
    max_positions=10,
    sl_trail=0.1,
    hold_days=5,
):

    from time import time

    t0 = time()

    # 前處理（向量化）
    market_danger = check_market(close)
    my_filter = gen_filter(close, high, low)
    entries = gen_entries(my_filter, df_proba, top_n, proba_threshold)
    exits = gen_exits(market_danger, df_proba)
    stock_group = parquet_db.query_stock2group_dict()

    print(f"前處理時間: {time() - t0:.2f} 秒")
    t1 = time()

    # 回測
    portfolio_value, trades_df, positions = simple_backtest(
        close=close,
        entries=entries,
        exits=exits,
        df_proba=df_proba,
        max_positions=max_positions,
        sl_trail=sl_trail,
        hold_days=hold_days,
        init_cash=1_000_000,
        fee=0.001,
        stock_group=stock_group,
    )
    # j1s_chart.chart_allocation(portfolio_value, trades_df, close=close)
    # j1s_chart.chart_gantt(trades_df)
    # j1s_chart.plot_performance(
    #     portfolio_value=portfolio_value,
    #     trades_df=trades_df,
    #     st=st,
    #     end=end,
    # )

    print("9999")
    print(len(positions))

    print(f"回測時間: {time() - t1:.2f} 秒")

    # 結果
    final_value = portfolio_value.iloc[-1]
    total_return = (final_value / 1_000_000 - 1) * 100
    print(f"最終資產：{final_value:,.0f}")
    print(f"總報酬率：{total_return:.2f}%")
    if not trades_df.empty:
        print(f"總交易次數：{len(trades_df)}")
        print(f"勝率：{(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"平均報酬：{trades_df['return_pct'].mean():.2f}%")

    return portfolio_value, trades_df


def local_signals():
    try:
        signal = pd.read_csv("gold_signal.csv", dtype={"stock_id": str})
        signal["date"] = pd.to_datetime(signal["date"])
        return signal
    except Exception as e:
        print(e)
        return None


def query(signal=local_signals(), good_search=False):
    #
    stocks = signal["stock_id"].unique().tolist()
    date = signal["date"]
    st = date.min()
    end = date.max()
    # st = "2026-03-01"
    # end = "2029-01-01"
    full_df = parquet_db.query_price(stocks, st, end)
    full_df["date"] = pd.to_datetime(full_df["date"])
    vbt_input = pd.merge(
        full_df,
        signal[["date", "stock_id", "y_proba"]],
        on=["date", "stock_id"],
        how="left",  # 只取索引部分  # 以全時段為準
    ).fillna(
        0
    )  # 沒預測到的（ATR太小的）補 0

    df_proba = vbt_input.pivot(index="date", columns="stock_id", values="y_proba").fillna(0)
    df_close = vbt_input.pivot(index="date", columns="stock_id", values="close")
    df_high = vbt_input.pivot(index="date", columns="stock_id", values="high")
    df_low = vbt_input.pivot(index="date", columns="stock_id", values="low")
    # df_close = df_close.ffill()
    # df_high = df_high.ffill()
    # df_low = df_low.ffill()
    # check_df_value(df_close)
    # check_df_value(df_high)
    # check_df_value(df_low)

    if good_search:
        return grid_search(
            close=df_close,
            high=df_high,
            low=df_low,
            df_proba=df_proba,
            stock_group=parquet_db.query_stock2group_dict(),
            exits=gen_exits(check_market(df_close), df_proba),
        )
    else:
        return main(
            close=df_close,
            high=df_high,
            low=df_low,
            df_proba=df_proba,
            st=st,
            end=end,
            max_positions=10,
            proba_threshold=0.6,
        )


def save_signal(signal):
    signal.rename(columns={2: "y_proba"}, inplace=True)
    signal["date"] = pd.to_datetime(signal["date"])
    signal.to_csv("gold_signal.csv", index=False)


def grid_search(close, high, low, df_proba, stock_group, exits):

    import itertools
    from time import time

    # ── 定義參數範圍 ──────────────────────────────────────
    param_grid = {
        "top_n": [3, 5, 10],
        "proba_threshold": [0.5, 0.6, 0.7],
        "max_positions": [5, 10, 15],
        "sl_trail": [0.05, 0.1, 0.15],
        "hold_days": [3, 5, 10],
        "group_limit": [2, 3, 5],
    }

    # ── 預先算好 entries（避免重複計算）──────────────────
    my_filter = gen_filter(close, high, low)

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
            df_proba,
            top_n=params["top_n"],
            proba_threshold=params["proba_threshold"],
        )

        portfolio_value, trades_df = simple_backtest(
            close=close,
            entries=entries,
            exits=exits,
            df_proba=df_proba,
            max_positions=params["max_positions"],
            sl_trail=params["sl_trail"],
            hold_days=params["hold_days"],
            stock_group=stock_group,
            group_limit=params["group_limit"],
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


def query_new():
    signal = lgbm_main.query(parquet_db.query_stocks_no_etf(), "2024-01", "2029-01")
    print("signal.shape")
    print(signal.shape)
    save_signal(signal)
    query()


# query_new()
# main()
# query()
# query(good_search=True)
