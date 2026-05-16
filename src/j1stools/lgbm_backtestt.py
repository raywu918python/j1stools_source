import numpy as np
import pandas as pd


def run_backtest_engin(
    df,
    models,
    df_market,
    top_n=20,
    forward_days=20,
    start_date=None,
    use_filter=False,
    vol_threshold=0.008,
    trend_threshold=0,
):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    # 大盤狀態計算
    mkt = df_market.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt["market_vol"] = mkt["close"].pct_change(1).rolling(20).std()
    mkt["market_trend"] = mkt["close"].pct_change(20)
    mkt["market_return"] = mkt["close"].pct_change(forward_days)

    all_dates = sorted(df["date"].unique())

    if start_date:
        trade_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
    else:
        start_idx = int(len(all_dates) * 0.8)
        trade_dates = all_dates[start_idx:]

    rebalance_dates = trade_dates[::forward_days]

    results = []

    for date in rebalance_dates:
        today = df[df["date"] == date]
        if len(today) == 0:
            continue

        mkt_today = mkt[mkt["date"] == date]
        if len(mkt_today) == 0:
            continue

        market_vol = mkt_today["market_vol"].iloc[0]
        market_trend = mkt_today["market_trend"].iloc[0]
        market_return = mkt_today["market_return"].iloc[0]

        # 過濾器開關
        if use_filter:
            can_trade = (market_vol > vol_threshold) and (market_trend > trend_threshold)
        else:
            can_trade = True

        if not can_trade:
            results.append(
                {
                    "date": date,
                    "actual_return": 0,
                    "excess_return": 0 - market_return,  # 空手相對大盤的超額
                    "market_return": market_return,
                    "avg_pred_score": 0,
                    "n_stocks": 0,
                    "can_trade": False,
                    "stocks": [],
                }
            )
            continue

        preds = np.mean([m.predict(today[feature_cols]) for m in models], axis=0)
        today = today.copy()
        today["pred_score"] = preds

        top = today.nlargest(top_n, "pred_score")[["stock_id", "pred_score", "target"]]

        excess_return = top["target"].mean()
        actual_return = excess_return + market_return

        results.append(
            {
                "date": date,
                "actual_return": actual_return,
                "excess_return": excess_return,
                "market_return": market_return,
                "avg_pred_score": top["pred_score"].mean(),
                "n_stocks": len(top),
                "can_trade": True,
                "stocks": top["stock_id"].tolist(),
            }
        )

    result_df = pd.DataFrame(results)
    result_df["cumulative_actual"] = (1 + result_df["actual_return"]).cumprod() - 1
    result_df["cumulative_excess"] = (1 + result_df["excess_return"]).cumprod() - 1
    result_df["cumulative_market"] = (1 + result_df["market_return"]).cumprod() - 1

    trade_only = result_df[result_df["can_trade"]]

    print("=" * 50)
    print(f"過濾器：{'開啟' if use_filter else '關閉'}")
    print(f"回測期間：{result_df['date'].min().date()} ~ {result_df['date'].max().date()}")
    print(f"總期數：{len(result_df)} 期")
    print(f"實際交易期數：{len(trade_only)} 期（空手 {len(result_df)-len(trade_only)} 期）")
    print(f"")
    print(f"【策略實際報酬】")
    print(f"平均每期報酬：  {trade_only['actual_return'].mean():.2%}")
    print(f"累積報酬：      {result_df['cumulative_actual'].iloc[-1]:.2%}")
    print(f"")
    print(f"【超額報酬（相對大盤）】")
    print(f"平均每期超額：  {trade_only['excess_return'].mean():.2%}")
    print(f"累積超額報酬：  {result_df['cumulative_excess'].iloc[-1]:.2%}")
    print(f"")
    print(f"【大盤報酬】")
    print(f"平均每期大盤：  {result_df['market_return'].mean():.2%}")
    print(f"累積大盤報酬：  {result_df['cumulative_market'].iloc[-1]:.2%}")
    print(f"")
    print(f"【直覺比較（100萬本金）】")
    strategy_val = 100 * (1 + result_df["cumulative_actual"].iloc[-1])
    market_val = 100 * (1 + result_df["cumulative_market"].iloc[-1])
    print(f"策略終值：      {strategy_val:.1f}萬")
    print(f"大盤終值：      {market_val:.1f}萬")
    print(f"實際多賺：      {strategy_val - market_val:.1f}萬")
    print(f"")
    print(f"勝率：          {(trade_only['excess_return'] > 0).mean():.1%}")
    print(f"最大單期虧損：  {trade_only['actual_return'].min():.2%}")
    print(f"最大單期獲利：  {trade_only['actual_return'].max():.2%}")

    # 看異常值分布
    print(f"target > 50%  的筆數：{(df['target'] >  0.5).sum()}")
    print(f"target < -50% 的筆數：{(df['target'] < -0.5).sum()}")
    print(f"總筆數：{len(df)}")
    print(f"異常比例：{((df['target'].abs() > 0.5).sum() / len(df)):.2%}")

    # 檢查 bug 過 100%的報酬
    check_bug = False
    if check_bug:
        print(df[df["target"].abs() > 0.5]["stock_id"].value_counts().head(20))

        date = "2025-05-08"
        today = df[df["date"] == date].copy()

        feature_cols = [c for c in df.columns if c.startswith("f_")]
        preds = np.mean([m.predict(today[feature_cols]) for m in models[-2:]], axis=0)
        today["pred_score"] = preds

        top = today.nlargest(20, "pred_score")[["stock_id", "pred_score", "target"]]
        print(top.to_string())

    print("=" * 50)

    return result_df
