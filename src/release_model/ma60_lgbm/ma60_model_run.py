from os import times
import os
from j1stools.django_orm import *
import pandas as pd

from j1stools import momentum_backtest, momentum_stats, parquet_db, triangle_stats
from release_model.ma60_lgbm import margin_lgbm_main


def _etf_rows(equity_df, df_market, init_capital=1_000_000, min_pct=0.02, max_positions=3):
    """從 equity_df["cash"] 計算 0050 停泊記錄。
    每次現金餘額變動（買/賣個股）就結清舊倉、用新餘額重開。
    回傳 (trades_df, open_row_dict|None)
    """
    mkt = df_market.copy()
    mkt.columns = mkt.columns.str.strip().str.lower()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)

    def _price(date):
        row = mkt[mkt["date"] <= date]
        return float(row["close"].iloc[-1]) if len(row) else None

    def _hold(d1, d2):
        return int(mkt[mkt["date"] <= d2].shape[0] - mkt[mkt["date"] <= d1].shape[0])

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    threshold = init_capital * min_pct

    trade_rows = []
    entry_date = entry_price = cost = entry_weight = None

    prev_cash = None
    for _, r in eq.iterrows():
        date = r["date"]
        cash = r["cash"]
        n_pos = int(r.get("n_positions", 0))
        has_cash = cash > threshold

        cash_changed = prev_cash is not None and abs(cash - prev_cash) > threshold
        prev_cash = cash

        if entry_date is not None and (cash_changed or not has_cash):
            ep = _price(date)
            if entry_price and ep:
                ret = (ep / entry_price - 1) * 100
                trade_rows.append(
                    {
                        "stock_id": "0050",
                        "entry_date": entry_date,
                        "entry_price": round(entry_price, 4),
                        "exit_date": date,
                        "exit_price": round(ep, 4),
                        "exit_reason": "etf_exit",
                        "highest": round(max(entry_price, ep), 4),
                        "cost": round(cost, 2),
                        "pnl": round((ep - entry_price) / entry_price * cost, 2),
                        "return_pct": round(ret, 3),
                        "hold_days": _hold(entry_date, date),
                        "weight": entry_weight,
                    }
                )
            entry_date = entry_price = cost = entry_weight = None

        if has_cash and entry_date is None:
            ep = _price(date)
            if ep:
                entry_date = date
                entry_price = ep
                cost = cash
                entry_weight = round((max_positions - n_pos) / max_positions, 4)

    open_row = None
    if entry_date is not None:
        last_date = eq["date"].iloc[-1]
        last_price = _price(last_date)
        if entry_price and last_price:
            open_row = {
                "stock_id": "0050",
                "entry_date": entry_date,
                "entry_price": round(entry_price, 4),
                "stop": None,
                "target": None,
                "highest": round(last_price, 4),
                "cost": round(cost, 2),
                "last_date": last_date,
                "pnl_pct": round((last_price / entry_price - 1) * 100, 2),
                "weight": entry_weight,
            }

    return pd.DataFrame(trade_rows), open_row


def _print_etf_adjusted(equity_df, etf_trades, etf_open, df_market, init_capital=1_000_000):
    etf_pnl = float(etf_trades["pnl"].sum()) if len(etf_trades) else 0.0
    if etf_open:
        etf_pnl += etf_open["pnl_pct"] / 100 * etf_open["cost"]
    if etf_pnl == 0.0:
        return
    base_final = equity_df["total"].iloc[-1]
    adj_final = base_final + etf_pnl
    years = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days / 365
    adj_ret = (adj_final / init_capital - 1) * 100
    adj_cagr = ((adj_final / init_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    mkt = df_market.copy()
    mkt.columns = mkt.columns.str.strip().str.lower()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date")
    start_date, end_date = equity_df["date"].iloc[0], equity_df["date"].iloc[-1]
    mkt = mkt[(mkt["date"] >= start_date) & (mkt["date"] <= end_date)]
    mkt_ret = (mkt["close"].iloc[-1] / mkt["close"].iloc[0] - 1) * 100 if len(mkt) >= 2 else 0.0
    mkt_cagr = (
        ((mkt["close"].iloc[-1] / mkt["close"].iloc[0]) ** (1 / years) - 1) * 100
        if len(mkt) >= 2 and years > 0
        else 0.0
    )

    print(f"{'含0050停泊':20}  {adj_ret:>+9.2f}%  {mkt_ret:>+9.2f}%  {adj_ret-mkt_ret:>+9.2f}%")
    print(f"{'年化報酬(含0050)':20}  {adj_cagr:>+9.2f}%  {mkt_cagr:>+9.2f}%  {adj_cagr-mkt_cagr:>+9.2f}%")


def marge_signals(signals: pd.DataFrame, pred) -> pd.DataFrame:
    import pandas as pd

    # 確保格式一致
    signals["date"] = pd.to_datetime(signals["date"])
    signals["stock_id"] = signals["stock_id"].astype(str).str.strip()

    pred["date"] = pd.to_datetime(pred["date"])
    pred["stock_id"] = pred["stock_id"].astype(str).str.strip()

    # merge：只保留兩者都有的 (date, stock_id)
    selected = signals.merge(pred[["date", "stock_id", "pred_score"]], on=["date", "stock_id"], how="inner")

    # 每天按 pred_score 排序，取前 N 名
    selected = selected.sort_values(["date", "pred_score"], ascending=[True, False]).reset_index(drop=True)

    print(f"原始訊號：{len(signals):,} 筆")
    print(f"有排序的：{len(selected):,} 筆")
    return selected


def get_sord_margin(stocks, st, end, mode="top20"):
    """
    mode="score" : pred_score > 0.2
    mode="top20" : 每日前20名
    """
    df_select_stocks = margin_lgbm_main.predict(stocks, st, end)
    if mode == "score":
        df_select_stocks = df_select_stocks[df_select_stocks["pred_score"] > 0.2]
    else:
        df_select_stocks = (
            df_select_stocks.groupby("date", group_keys=False)
            .apply(lambda g: g.nlargest(20, "pred_score"))
            .reset_index(drop=True)
        )
    return df_select_stocks


def good_search(signals, df_price, df_market):
    """
    momentum_stats.USE_COND5 = False
    momentum_stats.USE_COND3 = False
    --- 大盤比較 ---
                                  策略          大盤          超額
    總報酬                     +156.03%     +71.34%     +84.69%
    年化報酬(CAGR)               +76.83%     +38.61%     +38.22%
    最大回撤                     -14.55%     -27.48%     +12.94%
    """
    momentum_backtest.run_backtest(
        signals,
        df_price,
        df_market=df_market,
        stop_pct=0.07,
        target_pct=0.15,
        # exit_mode="trailing",
        trail_pct=0.10,
        max_positions=3,
        position_pct=0.33,
        max_hold=30,
        sort_by=[
            # momentum_backtest.SortBy.PRED_SCORE.desc(),  # 分數大→小
            momentum_backtest.SortBy.MA60_TURN_DAYS.asc(),  # 天數小→大
            momentum_backtest.SortBy.DEV_MA60.asc(),  # 距離小→大
        ],
    )


def main(
    st="2024-01-01",
    end="2029-01-01",
):
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
    # stocks = parquet_db.activate_stocks()
    stocks = parquet_db.query_stocks_no_etf()
    # 多往前抓 120 天，確保 st 第一天就能算出 MA60
    warmup_st = (pd.Timestamp(st) - pd.DateOffset(days=120)).strftime("%Y-%m-%d")
    # query_price 的 end 是 exclusive，+1 天確保 end 當天資料包含進來
    end_excl = (pd.Timestamp(end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    df_price = parquet_db.query_price(stocks, warmup_st, end_excl)
    df_market = parquet_db.query_price(["0050"], warmup_st, end_excl)

    #############################################################
    # momentum_stats.USE_DEDUP = False
    momentum_stats.USE_COND5 = False
    momentum_stats.USE_COND3 = False
    # 調整門檻（例如 30%）
    momentum_stats.MAX_DEV_MA60 = 0.2
    momentum_stats.USE_MAX_DEV_MA60 = True
    signals = momentum_stats.run(df_price, df_market, signal_date_from=st)
    signals = marge_signals(signals, get_sord_margin(stocks, st, end, mode="score"))
    #############################################################
    if "pred_score" in signals.columns:
        equity_df, trades_df, open_df = momentum_backtest.run_backtest(
            signals,
            df_price,
            df_market=df_market,
            stop_pct=0.07,
            target_pct=0.15,
            # exit_mode="trailing",
            trail_pct=0.10,
            max_positions=3,
            position_pct=0.33,
            max_hold=30,
            sort_by=[
                # momentum_backtest.SortBy.PRED_SCORE.desc(),  # 分數大→小
                momentum_backtest.SortBy.MA60_TURN_DAYS.asc(),  # 天數小→大
                momentum_backtest.SortBy.DEV_MA60.asc(),  # 距離小→大
            ],
        )
    else:
        equity_df, trades_df, open_df = momentum_backtest.run_backtest(
            signals,
            df_price,
            df_market=df_market,
        )

    # 把 0050 停泊記錄 append 進 trades_df / open_df（不影響統計）
    etf_trades, etf_open = _etf_rows(equity_df, df_market, max_positions=3)
    if len(etf_trades):
        trades_df = (
            pd.concat([trades_df, etf_trades], ignore_index=True).sort_values("exit_date").reset_index(drop=True)
        )
    if etf_open:
        stock_weight_sum = open_df["weight"].sum() if len(open_df) else 0.0
        etf_open["weight"] = round(max(0.0, 1.0 - stock_weight_sum), 4)
        open_df = pd.concat([open_df, pd.DataFrame([etf_open])], ignore_index=True)

    # 補印含 0050 的調整後報酬（格式同大盤比較表格）
    _print_etf_adjusted(equity_df, etf_trades, etf_open, df_market)

    # print("\n=== 現金水位 ===")
    # print(equity_df.tail().T)
    # print("\n=== 交易記錄 ===")
    # print(trades_df.shape)
    # print(trades_df.tail().T)
    print("\n=== 未平倉 ===")
    print(open_df.to_string() if len(open_df) else "（無）")
    # draw_chart(equity_df, trades_df, df_market=df_market)

    # 0050 轉成與 equity_df 相同格式（date + total）
    start_val = equity_df["total"].iloc[0]
    eq_dates = pd.to_datetime(equity_df["date"]).dt.normalize()
    mkt = df_market[["date", "close"]].copy()
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
    mkt = mkt[mkt["date"].isin(eq_dates)].reset_index(drop=True)
    if mkt.empty:
        mkt = df_market[["date", "close"]].copy()
        mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()
        mkt = mkt[mkt["date"] >= eq_dates.iloc[0]].reset_index(drop=True)
    mkt["total"] = (mkt["close"] / mkt["close"].iloc[0] * start_val).round(2)
    market_df = mkt[["date", "total"]]

    return equity_df, trades_df, open_df, market_df


def draw_chart(equity_df, trades_df, df_market=None, out="backtest_chart.html"):
    import plotly.graph_objects as go

    BG = "#0f1117"
    ORANGE = "#ff8c00"
    GRAY = "#888888"

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["cum_ret"] = (eq["total"] / eq["total"].iloc[0] - 1) * 100

    traces = [
        go.Scatter(
            x=eq["date"],
            y=eq["cum_ret"],
            name="策略模型",
            line=dict(color=ORANGE, width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        )
    ]

    if df_market is not None:
        mkt = df_market.copy()
        mkt.columns = mkt.columns.str.strip().str.lower()
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt = mkt.sort_values("date")
        mkt = mkt[(mkt["date"] >= eq["date"].iloc[0]) & (mkt["date"] <= eq["date"].iloc[-1])]
        if len(mkt) >= 2:
            mkt["cum_ret"] = (mkt["close"] / mkt["close"].iloc[0] - 1) * 100
            traces.append(
                go.Scatter(
                    x=mkt["date"],
                    y=mkt["cum_ret"],
                    name="大盤對比",
                    line=dict(color=GRAY, width=1.8, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
                )
            )

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="📊 近 2 年累計報酬率對比", font=dict(size=16, color="white"), x=0),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=GRAY),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", font=dict(color="white")),
        xaxis=dict(
            tickformat="%Y-%m",
            gridcolor="#1e2030",
            linecolor="#1e2030",
            tickcolor=GRAY,
        ),
        yaxis=dict(
            ticksuffix="%",
            gridcolor="#1e2030",
            linecolor="#1e2030",
            tickcolor=GRAY,
            zeroline=True,
            zerolinecolor="#333344",
        ),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig

    # with open("chartxxx.json", "w") as f:
    # f.write(fig.to_json())


if __name__ == "__main__":
    main()
    # df = parquet_db.query_price(["6291"], "2026-01-01", "2099-01-01")
    # print(df.tail())
    pass
