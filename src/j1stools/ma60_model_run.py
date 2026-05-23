from os import times
import os
from j1stools.django_orm import *
import pandas as pd

from j1stools import margin_lgbm_main, momentum_backtest, momentum_stats, parquet_db, triangle_stats


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


def get_sord_margin(stocks, st, end):
    from datetime import date

    today = date.today()
    find_name = "margin" + today.strftime("%Y%m%d") + st + end + ".csv"

    if os.path.exists(find_name):
        df_margin = pd.read_csv(find_name)
    else:
        df_margin = margin_lgbm_main.predict(stocks, st, end)
        df_margin.to_csv(find_name, index=False)

    # df_margin = df_margin[df_margin["pred_score"] > 0.2]
    return df_margin


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


def main():
    # stocks = random.sample(parquet_db.query_stocks_ids_list(), 500)
    st = "2024-01-01"
    end = "2029-01-01"
    # stocks = parquet_db.query_stocks_ids_list()
    stocks = parquet_db.query_stocks_no_etf()
    df_price = parquet_db.query_price(stocks, st, end)
    df_market = parquet_db.query_price(["0050"], st, end)

    #############################################################
    # momentum_stats.USE_DEDUP = False
    momentum_stats.USE_COND5 = False
    momentum_stats.USE_COND3 = False
    # 調整門檻（例如 30%）
    momentum_stats.MAX_DEV_MA60 = 0.2
    momentum_stats.USE_MAX_DEV_MA60 = False
    signals = momentum_stats.run(df_price, df_market)
    signals = marge_signals(signals, get_sord_margin(stocks, st, end))
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

    print("\n=== 現金水位 ===")
    print(equity_df.tail().T)
    print("\n=== 交易記錄 ===")
    print(trades_df.shape)
    print(trades_df.tail().T)
    print("\n=== 未平倉 ===")
    print(open_df.to_string() if len(open_df) else "（無）")
    return equity_df, trades_df, open_df
    # draw_chart(equity_df, trades_df, df_market=df_market)


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
    fig.show()

    with open("chartxxx.json", "w") as f:
        f.write(fig.to_json())


if __name__ == "__main__":
    main()
    # df = parquet_db.query_price(["6291"], "2026-01-01", "2099-01-01")
    # print(df.tail())
    pass
