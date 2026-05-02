import json
from os import replace
import random

from cycler import V
import joblib
from narwhals import group_by
import pandas as pd
import plotly
from regex import T
from sqlalchemy import column
import lgbm_main
from rfc_main import RFCModel
from . import rfc_main as rfc_main
from . import parquet_db as parquet_db
import vectorbt as vbt
import quantstats as qs
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta

import vectorbt as vbt
import pandas as pd

# import streamlit as st


class VbtValues:
    # 最安全
    # ATR_FILTER = 0.05
    # ADR_FILTER = 20
    # THRESHOLD = 0.7
    # TOP_N = 5
    # 第2安全
    # ATR_FILTER = 0.02
    # ADR_FILTER = 10
    # THRESHOLD = 0.7
    # TOP_N = 5
    # 5賺最多
    # ATR_FILTER = 0.01
    # ADR_FILTER = 30
    # THRESHOLD = 0.5
    # TOP_N = 5
    # 5.5賺最多
    ATR_FILTER = 0.05
    ADR_FILTER = 30
    THRESHOLD = 0.5
    TOP_N = 5


def get_exits(entries, exits, df_prices):
    # A. 手動抓取進場價格：只在 entries 為 True 的那天保留價格，其餘填充
    # entries 是你的布林矩陣，df_prices 是收盤價矩陣
    entry_prices = df_prices[entries].ffill()

    # B. 計算持有天數：從進場那天開始累加，出場後歸零
    # 這裡用一個簡單的 cumsum 邏輯
    group = entries.cumsum()
    holding_days = group.copy()
    for col in group.columns:
        holding_days[col] = group[col].groupby(group[col]).cumcount()
    # 只在持倉期間保留天數
    holding_days = holding_days.where(entries.ffill() & ~exits.ffill(), 0)

    # C. 計算目前漲幅
    current_return = (df_prices - entry_prices) / entry_prices

    # D. 【核心改善】實施 3 天 0.5% 處決邏輯
    # 當天數來到第 3 天 (或根據你的 freq 調整)，若漲幅未達標，產生出場訊號
    stagnant_exit = (holding_days == 3) & (current_return < 0.005)

    # E. 合併到最終出場矩陣
    return exits | stagnant_exit


def gen_filter(close, high, low):
    # 1. 檢查原始資料有沒有問題
    if close.isnull().all().all():
        print("警告：輸入的 close 資料全是空值！")

    # 2. 計算 ATR
    atr_obj = vbt.ATR.run(high, low, close, window=14)
    atr = atr_obj.atr
    atr_ratio = atr / close

    # 3. 計算 ADX (改用 vbt 原生方式比較穩)
    adx = vbt.pandas_ta("ADX").run(high, low, close, length=14).adx

    # --- 關鍵診斷：看看誰是 NaN ---
    print(f"ATR 有效數值數量: {atr.notnull().sum().sum()}")
    print(f"ADX 有效數值數量: {adx.notnull().sum().sum()}")

    # 4. 設定門檻
    filter_atr = atr_ratio > VbtValues.ATR_FILTER  # 1%
    filter_adx = adx > VbtValues.ADR_FILTER

    # 檢查門檻後的 True 數量
    print(f"ATR 門檻通過數量: {filter_atr.sum().sum()}")
    print(f"ADX 門檻通過數量: {filter_adx.sum().sum()}")

    # 5. 結合
    # 先暫時拿掉漲停限制測試，看看訊號會不會出來
    final_filter = filter_atr & filter_adx

    return final_filter.fillna(False)


def gen_entries(my_filter, df_proba, market_danger, is_limit_up, top_n=10, proba_threshold=0.5):
    # 1. 強制日期格式統一
    df_proba.index = pd.to_datetime(df_proba.index).tz_localize(None)
    my_filter.index = pd.to_datetime(my_filter.index).tz_localize(None)
    is_limit_up.index = pd.to_datetime(is_limit_up.index).tz_localize(None)
    market_danger.index = pd.to_datetime(market_danger.index).tz_localize(None)

    # 2. 強制股票代號統一 (轉為字串並移除 .TW)
    def force_clean_cols(df):
        cols = df.columns.get_level_values(-1) if hasattr(df.columns, "levels") else df.columns
        return cols.astype(str).str.replace(r"\.TW.*", "", regex=True).str.strip()

    df_proba.columns = force_clean_cols(df_proba)
    my_filter.columns = force_clean_cols(my_filter)
    is_limit_up.columns = force_clean_cols(is_limit_up)

    # 3. 再次執行對齊
    my_filter = my_filter.reindex_like(df_proba).fillna(False)
    is_limit_up = is_limit_up.reindex_like(df_proba).fillna(False)

    # 核心修正：除了原本的濾網，還要加上「不是漲停板」
    # 我們要把「漲停的股票」從候選名單中踢除
    final_filter = my_filter & (~is_limit_up)

    # 先過濾 (剔除技術面不合 & 漲停的)，後排名
    filtered_proba = df_proba.where(final_filter, -1.0)
    rank = filtered_proba.rank(axis=1, ascending=False, method="first")

    # 判定條件
    is_top = rank <= top_n
    is_confident = df_proba >= proba_threshold

    # 大盤安全邏輯
    market_danger_aligned = market_danger.reindex(df_proba.index).fillna(True)
    is_market_safe = (~market_danger_aligned.values).reshape(-1, 1)

    # 最終 Entry：排名在前、信心夠、通過技術濾網、且「當天沒漲停」、大盤安全
    entries = is_top & is_confident & final_filter & is_market_safe

    print(f"DEBUG: 整個資料集中 is_limit_up 為 True 的總數: {is_limit_up.sum().sum()}")
    print(f"DEBUG: 因漲停被剔除的潛在訊號數: {(is_top & is_confident & my_filter & is_limit_up).sum().sum()}")

    return entries


# ==========================================
# 1. 回測引擎模組 (Backtest Engine)
# ==========================================
def backtest(close, high, low, df_proba, top_n=3, proba_threshold=0.6):
    init_cash = 1000000  # 初始資金
    max_slots = 10  # 總倉位上限

    market_danger = gen_market_danger(close)
    my_filter = gen_filter(close, high, low)
    is_limit_up = close >= close.shift(1) * 1.095
    my_entries = gen_entries(my_filter, df_proba, market_danger, is_limit_up, top_n, proba_threshold)
    my_exits = market_danger

    size_df = gen_size_by_group(my_entries, total_capital=init_cash)
    rank = df_proba.rank(axis=1, ascending=False, method="first")
    # 執行回測
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=my_entries,
        exits=my_exits,
        # 核心：使用固定金額，且避開百分比陷阱
        # size=size_df,  # 假設 100 萬分給 10 支，每支買 10 萬
        size=100000,  # 假設 100 萬分給 10 支，每支買 10 萬
        size_type="value",  # 用 value 最安全
        init_cash=init_cash,
        fees=0.001,
        cash_sharing=True,  # 共用資金
        group_by=True,
        sl_stop=0.1,  # 停損
        sl_trail=True,  # 移動停損
    )
    return pf


# 還沒有沒其它的回測套件好用的，我用模型信心度，把上百支股都


def pk0050_vbt_1chat_ai(portfolio, st, end):
    """
    績效
    """

    df0050 = parquet_db.query_price(["0050"], st, end)
    df0050.set_index("date", inplace=True)
    benchmark_close = df0050["close"]

    # 2. 準備資產價值數據 (Equity)
    initial_cash = 1000000
    strategy_value = portfolio.value()  # 你的策略總價值
    print(strategy_value)

    # 計算 0050 的總價值 (從 100 萬開始買入持有)
    benchmark_returns = benchmark_close.pct_change().fillna(0)
    benchmark_value = initial_cash * (1 + benchmark_returns).cumprod()

    import plotly.graph_objects as go
    import pandas as pd

    # === 1. 資料對齊（關鍵）===
    strategy_value, benchmark_value = strategy_value.align(benchmark_value)

    # === 2. trades 處理 ===
    trades = portfolio.trades.records_readable.copy()
    trades["Entry Date"] = pd.to_datetime(trades["Entry Timestamp"])
    trades["Exit Date"] = pd.to_datetime(trades["Exit Timestamp"])
    # Index(['Exit Trade Id', 'Column', 'Size', 'Entry Timestamp', 'Avg Entry Price',
    #    'Entry Fees', 'Exit Timestamp', 'Avg Exit Price', 'Exit Fees', 'PnL',
    #    'Return', 'Direction', 'Status', 'Position Id'],
    #   dtype='object')

    trades["Entry Price"] = trades["Avg Entry Price"]
    trades["Exit Price"] = trades["Avg Exit Price"]

    # 👉 用 asof 對齊價格（避免時間不完全一致）
    trades["Entry Value"] = trades["Entry Date"].apply(lambda x: strategy_value.asof(x))
    trades["Exit Value"] = trades["Exit Date"].apply(lambda x: strategy_value.asof(x))

    # print(trades[["Column", "Entry Timestamp"]])
    # === 3. 建立 subplot ===
    fig = vbt.make_subplots(
        rows=1,
        cols=1,
        # shared_xaxes=True,
        # vertical_spacing=0.05,
    )

    # === 4. 主圖 ===
    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value,
            name="策略價值",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value,
            name="0050 基準",
            line=dict(color="blue", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # === 5. 買入點（一次畫）===
    fig.add_trace(
        go.Scatter(
            x=trades["Entry Date"],
            y=trades["Entry Value"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="blue"),
            name="買入",
            text=(
                "標的: "
                + trades["Column"].astype(str)
                + "<br>買入價: "
                + trades["Entry Price"].round(2).astype(str)
                + "<br>方向: "
                + trades["Direction"].astype(str)
            ),
            hoverinfo="text+name",
        ),
        row=1,
        col=1,
    )

    # === 6. 賣出點（一次畫）===
    fig.add_trace(
        go.Scatter(
            x=trades["Exit Date"],
            y=trades["Exit Value"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="賣出",
            text=(
                "標的: "
                + trades["Column"].astype(str)
                + "<br>賣出價: "
                + trades["Exit Price"].round(2).astype(str)
                + "<br>獲利: "
                + trades["PnL"].round(2).astype(str)
                + "<br>報酬率: "
                + (trades["Return"] * 100).round(2).astype(str)
                + "%"
            ),
            hoverinfo="text+name",
        ),
        row=1,
        col=1,
    )

    # === 8. layout（簡化版，避免衝突）===
    # fig.update_layout(
    #     paper_bgcolor="rgba(0,0,0,0)",  # 外部背景透明
    #     plot_bgcolor="rgba(0,0,0,0)",  # 繪圖區背景透明
    #     font=dict(color="#e0e0e0"),  # 文字改成淺灰色
    #     # margin=dict(l=20, r=20, t=50, b=20),
    #     xaxis=dict(gridcolor="#333333", zerolinecolor="#444444"),  # 網格線顏色調暗
    #     yaxis=dict(gridcolor="#333333", zerolinecolor="#444444"),
    #     hovermode="x unified",  # 增強懸停體驗
    #     template="plotly_dark",  # 直接套用深色模板作為基礎
    # )

    fig.update_layout(
        hovermode="x unified",
        hoverdistance=1000,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="#e0e0e0"),  # 文字改成淺灰色
        xaxis=dict(gridcolor="#333333", zerolinecolor="#444444"),  # 網格線顏色調暗
        yaxis=dict(gridcolor="#333333", zerolinecolor="#444444"),
        paper_bgcolor="rgba(0,0,0,0)",  # 外部背景透明
        plot_bgcolor="rgba(0,0,0,0)",  # 繪圖區背景透明
        width=None,  # for html
        height=None,  # for html
        hoverlabel=dict(
            bgcolor="#444444",  # 資訊窗背景顏色
            font_size=14,  # 文字大小
            font_family="Rockwell",  # 字體
            font_color="#e0e0e0",  # 文字顏色
            bordercolor="#333333",  # 邊框顏色
        ),
    )
    # 針對買入(Buy)與賣出(Sell)的標記進行美化
    fig.update_traces(
        marker=dict(size=10),
        selector=dict(mode="markers"),
    )

    fig.update_yaxes(title_text="資產價值", row=1, col=1)

    fig.show()
    with open("chart/chart_data.json", "w", encoding="utf-8") as f:
        f.write(fig.to_json())
        # f.write(fig.to_json(pretty=False))
    # return fig.to_json()


def chart_11(pf, df_prices):
    """針對百支股票的程式碼範例：相關性與回撤分布"""

    import vectorbt as vbt
    import pandas as pd

    # 假設你已經有一個包含百支股票收盤價的 DataFrame: df_prices
    # 1. 計算相關性矩陣 (確保籃子裡的股票夠分散)
    print(df_prices.head())
    returns = df_prices.pct_change()
    corr_matrix = returns.corr()

    # 繪製相關性熱力圖
    corr_matrix.vbt.heatmap(title="Asset Correlation Matrix").show()

    # 2. 執行一籃子股票的回測 (假設對所有股票套用同一組簡單策略)
    fast_ma = vbt.MA.run(df_prices, 20)
    slow_ma = vbt.MA.run(df_prices, 60)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    # pf = vbt.Portfolio.from_signals(df_prices, entries, exits)

    # 3. 繪製所有資產的夏普比率分布 (Bar Chart)
    # 當股票很多時，用柱狀圖看分布最快
    # pf.sharpe_ratio().vbt.barplot(title="Sharpe Ratio Distribution Across Assets").show()
    #
    # 假設是日線資料，指定年度化週期為 252 天
    # pf.sharpe_ratio(annualize=True).vbt.barplot(title="Sharpe Ratio").show()
    # 或者明確指定頻率（如果 vbt 抓不到）
    pf.sharpe_ratio(freq="1D").vbt.barplot(title="Sharpe Ratio").show()


def local_signals():
    signal = pd.read_csv("gold_signal.csv", dtype={"stock_id": str})
    signal["date"] = pd.to_datetime(signal["date"])
    return signal


def chart_drawdown(portfolio):
    """1. 水平回撤圖 (Drawdown Plot)
    這張圖可以幫你識別策略在哪段時間發生了資產縮水，以及縮水的深度。
    """
    portfolio.plot_drawdowns().show()


def chart_assct_bar(portfolio):
    """3. 資產權重分配圖 (Asset Allocation Stacked Bar)
    如果你有做動態調倉（例如每月重新分配權重），這張圖能顯示隨著時間推移，資金是如何在百支股票間流動的。"""

    portfolio.assets_proportions().vbt.plot(kind="bar", stacked=True).show()


def chart_multi_line_returns(portfolio):
    """1. 累積報酬率對比圖 (Stacked or Multi-line Returns)
    當標的多達百支時，直接畫線會像亂麻。vbt 可以讓你快速對比「整體組合」與「基準指數」或是「各個產業」的表現。
    """
    portfolio.returns().vbt.plot().show()


def chart1(pf):
    #
    #
    # 1. 繪製「整體組合」的累積收益 (因為你設了 group_by=True，這會是一條線)
    pf.cumulative_returns().vbt.plot(title="Total Portfolio Cumulative Returns").show()


def chart2(pf):
    #
    # 2. 如果想看這 100 支股票「各自」的表現 (不套用 group_by)
    # 這樣可以抓出誰是害群之馬
    # pf.returns(group_by=False).vbt.plot(title="Individual Stock Returns").show()

    # 1. 先計算總報酬率並排序
    total_returns = pf.returns(group_by=False).sum().sort_values(ascending=False)

    # 2. 取出前 5 名與後 5 名的名稱 (Index)
    top_5 = total_returns.head(5).index
    bottom_5 = total_returns.tail(5).index
    selected_assets = top_5.union(bottom_5)

    # 3. 篩選資料並繪圖
    # pf.returns(group_by=False)[selected_assets].vbt.plot(title="Top 5 and Bottom 5 Returns").show()

    fig = pf.returns(group_by=False)[selected_assets].vbt.plot(title="Top 5 and Bottom 5 Returns")

    # 優化佈局
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),  # 垂直排列  # 靠上對齊  # 放在右側
        margin=dict(r=150),  # 給右側留點空間放圖例
    )

    fig.show()


def chart3(pf):
    #
    #
    # 抓取每支股票的回撤數據並畫成熱力圖
    drawdowns = (
        pf.drawdowns.drawdown.to_pd()
    )  # 預設不 group # drawdowns = pf.regroup(False).drawdowns.drawdown caching=true,不能改 group
    drawdowns.vbt.heatmap(trace_kwargs=dict(colorscale="Reds"), title="Portfolio Risk Map: Drawdown by Asset").show()


def chart4(pf, df_close):
    #
    #
    # 繪製隨時間變化的資產權重
    # 這會顯示一個堆疊區域圖，顯示每一刻資金在各標的的佔比
    # pf.asset_value(group_by=False)        # 每個資產的市值（絕對值）
    # pf.gross_exposure(group_by=False)     # 曝險比例
    # pf.value(group_by=False)             # 總資產價值
    asset_value = pf.asset_value(group_by=False)
    total_value = pf.value()
    # 計算比例：各個資產價值 / 總價值
    proportions = asset_value.div(total_value, axis=0)
    # proportions.vbt.plot(
    #     trace_kwargs=dict(stackgroup="one", fill="tonexty"),  # 這行等同於 stacked=True  # 這行等同於 kind='area'
    #     title="Asset Allocation Over Time",
    # ).show()
    import plotly.express as px

    fig = px.area(proportions, title="Asset Allocation Over Time", labels={"value": "Proportion", "date": "Date"})
    fig.show()
    #
    #
    # 參數熱力圖 (Parameter Heatmap)
    # 1. 計算每日報酬率 (Returns)
    returns = df_close.pct_change()

    # 2. 計算相關性矩陣
    corr_matrix = returns.corr()

    # 3. 使用 vectorbt 的 heatmap 繪製
    corr_matrix.vbt.heatmap(
        title="Asset Correlation Matrix (Based on Daily Returns)",
        trace_kwargs=dict(
            colorscale="RdBu_r",  # 紅藍配色，_r 代表反轉，讓 1 是紅，-1 是藍
            zmin=-1,  # 設定顏色軸的最小值
            zmax=1,  # 設定顏色軸的最大值
        ),
    ).show()


def chart5(pf):
    #
    #
    # 獲取每支股票的年化報酬與波動率
    returns = pf.annualized_return(group_by=False, freq="D")  # 日線
    volatility = pf.annualized_volatility(group_by=False, freq="D")  # 日線
    # 使用 Plotly 畫散佈圖
    import plotly.express as px

    fig = px.scatter(
        x=volatility,
        y=returns,
        text=returns.index,
        labels={"x": "Annualized Volatility", "y": "Annualized Return"},
        title="Risk vs Reward Scatter Plot (Individual Assets)",
    )
    fig.update_traces(textposition="top center")
    fig.show()


def chart6(pf):
    #
    #
    # 夏普比率分布 (Sharpe Ratio Barplot)：檢查「運氣還是實力」怎麼看
    import plotly.express as px

    sr = pf.sharpe_ratio(freq="1D", group_by=False)

    # 確保 index 是股票名稱
    fig = px.bar(
        x=sr.index.astype(str),  # 強制轉成字串
        y=sr.values,
        title="Sharpe Ratio",
        labels={"x": "Asset", "y": "Sharpe Ratio"},
    )
    fig.show()


def chart7(pf, df_proba, df_close):
    #
    #
    # 參數熱力圖 (Parameter Heatmap)
    # 定義你想測試的閾值 (例如 0.5 到 0.9)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    # 2. 手動迴圈跑參數 (基礎版不支援 vbt.Param 廣播時的替代方案)
    for t in thresholds:
        # 產生進場訊號：當機率大於閾值
        entries = df_proba > t

        # 執行回測 (group_by=True 來看整體組合表現)
        pf = vbt.Portfolio.from_signals(df_close, entries, exits=None, group_by=True)

        # 紀錄你想看的指標，例如夏普值
        results[t] = pf.sharpe_ratio(freq="1D", group_by=False)

    # 3. 轉換為 Series 並繪圖
    # res_series = pd.Series(results)
    # res_series.index.name = "entry_threshold"
    # res_series.vbt.heatmap(title="Sharpe Ratio by Entry Threshold").show()
    # 將迴圈結果轉為 Pandas Series
    res_series = pd.Series(results)
    res_series.index.name = "entry_threshold"

    # 使用 barplot 代替 heatmap
    res_series.vbt.barplot(
        title="Sharpe Ratio by Entry Threshold", xaxis_title="Probability Threshold", yaxis_title="Sharpe Ratio"
    ).show()


def chart8(pf):
    #
    #
    # 假設你的初始資金 (init_cash) 是 1,000,000
    pf.value().vbt.plot(title="Equity Curve (Real Money)").show()


def gen_market_danger(close_df):
    # 假設 close_df 是 (date, symbol) 的股價矩陣
    ma60_df = close_df.rolling(60).mean()

    # 1. 計算全市場有多少比例在 MA60 以下
    under_ma60_ratio = (close_df < ma60_df).sum(axis=1) / close_df.shape[1]

    # 2. 計算全市場有多少比例跌超過 8%
    # 這裡用 (今日收盤 / 昨日收盤) - 1
    daily_ret = close_df.pct_change()
    crash_8pct_ratio = (daily_ret < -0.08).sum(axis=1) / close_df.shape[1]

    # 3. 定義市場危險訊號 (Market Danger)
    market_danger = (under_ma60_ratio > 0.9) | (crash_8pct_ratio > 0.9)
    return market_danger


def gen_size_by_group(entries, total_capital=1000000):
    """
    將原本的比例邏輯轉換為絕對金額
    total_capital: 你的起始資金 (例如 1,000,000)
    """
    sector_series = parquet_db.query_stock_group()
    unique_sectors = sector_series.iloc[:, 0].unique()
    final_size_df = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)

    # 這裡定義金額上限（根據你的比例需求）
    # 產業上限 30% = 300,000；單檔上限 10% = 100,000
    sector_value_limit = total_capital * 0.3
    single_stock_value_limit = total_capital * 0.1

    for sector in unique_sectors:
        sector_stocks = sector_series[sector_series == sector].index
        active_stocks = [s for s in sector_stocks if s in entries.columns]

        if not active_stocks:
            continue

        # 每天該產業總共有幾個買入訊號
        n_signals = entries[active_stocks].sum(axis=1)

        # 計算該產業當天「每一份」應分配的金額
        # 公式：產業總額 / 訊號數 (但不能超過單檔上限)
        sector_unit_value = np.where(n_signals > 0, sector_value_limit / n_signals, 0.0)

        # 強制限制每筆買入不能超過單檔上限 (10%)
        sector_unit_value = np.clip(sector_unit_value, 0, single_stock_value_limit)

        for stock in active_stocks:
            # 只有在 entries 為 True 的那天，填入該筆要買的金額
            final_size_df[stock] = np.where(entries[stock], sector_unit_value, 0.0)

    return final_size_df


def chart9(pf):
    import plotly.express as px
    import pandas as pd

    # 1. 提取紀錄
    df_plot = pf.trades.records_readable

    # 2. 格式化日期數據 (這是關鍵：將其轉為字串格式存入新欄位，專供 Hover 使用)
    # 這樣不會影響 timeline 繪圖用的原始 Timestamp 物件
    df_plot["Entry_Day"] = pd.to_datetime(df_plot["Entry Timestamp"]).dt.strftime("%Y-%m-%d")
    df_plot["Exit_Day"] = pd.to_datetime(df_plot["Exit Timestamp"]).dt.strftime("%Y-%m-%d")

    pnl_col = "PnL" if "PnL" in df_plot.columns else "Return"
    df_plot["Stock"] = df_plot["Column"]
    df_plot["Status"] = df_plot[pnl_col].apply(lambda x: "Profit" if x > 0 else "Loss")

    # 3. 繪圖
    fig = px.timeline(
        df_plot,
        x_start="Entry Timestamp",
        x_end="Exit Timestamp",
        y="Stock",
        color="Status",
        color_discrete_map={"Profit": "#26a69a", "Loss": "#ef5350"},
        # 這裡設定 Hover 顯示的內容
        hover_data={
            "Entry Timestamp": False,  # 隱藏原本帶有時分秒的欄位
            "Exit Timestamp": False,  # 隱藏原本帶有時分秒的欄位
            "Entry_Day": True,  # 顯示我們格式化好的日期
            "Exit_Day": True,  # 顯示我們格式化好的日期
            "PnL": ":.0f",
            "Return": ":.2%",
        },
        title="Trade Duration & Timeline",
    )

    # 4. 排序 Y 軸即可，完全不碰 update_xaxes
    fig.update_yaxes(categoryorder="min ascending", title="Stock ID")

    fig.show()


def check_df_value(df):
    # 1. 取得各項統計
    null_s = df.isnull().sum()
    inf_s = np.isinf(df.select_dtypes(include=[np.number])).sum()
    zero_s = (df == 0).sum()

    # 2. 合併統計表
    stats = pd.concat([null_s, inf_s, zero_s], axis=1)
    stats.columns = ["Null", "Inf", "Zero"]

    # 3. 【關鍵】篩選出任一項大於 0 的欄位
    problem_fields = stats[(stats > 0).any(axis=1)]

    if problem_fields.empty:
        print("恭喜！所有欄位皆無 Null, Inf 或 0。")
    else:
        print("--- 僅顯示有問題的欄位 ---")
        print(problem_fields)


def main(signal=local_signals()):
    #
    stocks = signal["stock_id"].unique().tolist()
    date = signal["date"]
    st = date.min()
    end = date.max()
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
    pf = backtest(
        df_close,
        df_high,
        df_low,
        df_proba,
        top_n=VbtValues.TOP_N,
        proba_threshold=VbtValues.THRESHOLD,
    )
    chart1(pf)
    chart2(pf)
    chart3(pf)
    chart4(pf, df_close)
    chart5(pf)
    chart6(pf)
    chart7(pf, df_proba, df_close)
    chart8(pf)
    chart9(pf)
    #
    return pk0050_vbt_1chat_ai(pf, st, end)


def query(stocks, st, end):

    # signal = rfc_main.query(stocks, st, end)  # win 0050
    # signal = lgbm_main.query_no_rfc(stocks, st, end)
    signal = lgbm_main.query(stocks, st, end)

    save_signal(signal)

    return main(signal)


def save_signal(signal):
    signal.rename(columns={2: "y_proba"}, inplace=True)
    signal["date"] = pd.to_datetime(signal["date"])
    signal.to_csv("gold_signal.csv", index=False)


# main()
# query(parquet_db.stocks(), "2024-01", "2099-01")
# query(parquet_db.stocks_non_0050(), "2024-01", "2099-01")
# query(["2308", "3105"], "2024-01", "2029-01")
# query(random.sample(parquet_db.stocks(), 100), "2024-01", "2099-01")


# 1. 相關性矩陣 (Correlation Heatmap)：檢查「雞蛋是否放在同一個籃子」這張圖是百支股票回測的起點。怎麼看： 找深色（高相關）的區塊。如果你的矩陣看起來「整片通紅」，代表你的選股池高度同質化。分析要點：風險： 當市場下挫時，高相關性的組合會讓你遭遇「集體重挫」，失去分散風險的意義。優化方案： 如果發現某幾支股票相關性高達 $0.9$ 以上，建議在 Just 1 Stock 的過濾邏輯中，只保留其中一支表現最穩定的，或加入不同產業的標的。
# 2. 回撤圖 (Drawdown Plot)：檢查「心臟受不受得了」回撤圖比獲利圖重要十倍。怎麼看： 觀察「水坑」的深度與寬度。分析要點：深度 (Magnitude)： 最大回撤（MDD）是否超過你的承受範圍（例如 $-20\%$）？寬度 (Duration)： 策略被「套牢」的時間有多長？如果一個策略賺錢，但要讓你等待兩年才解套，這在實務上很難執行。同步性： 觀察回撤是否發生在特定的市場黑天鵝時期。
# 3. 夏普比率分布 (Sharpe Ratio Barplot)：檢查「運氣還是實力」怎麼看： 看這百支股票的夏普值分布。分析要點：一致性： 是只有一兩支「妖股」暴賺帶動整體獲利，還是大部分股票都能穩定提供正回報？異常值： 如果某支股票夏普值高得離譜（例如 $>3$），請檢查數據是否有誤（如漏掉除權息修正），或者是因為交易次數太少導致的樣本偏差。
# 4. 資產配置圖 (Asset Allocation)：檢查「資金流向」怎麼看： 觀察色塊的厚度變化。分析要點：空倉比例： 如果某段時間色塊變得很薄，代表你的策略在那段時期「空手」。這是否符合你的預期？過度集中： 是否有某段時間資金全部擠在少數兩三支股票上？這會讓你的整體組合變成「單押」，風險會瞬間飆升。
# 5. 參數熱力圖 (Parameter Heatmap)：檢查「穩定性」怎麼看： 找「平原」，避開「孤島」。分析要點：魯棒性 (Robustness)： 如果你的最佳參數（深色點）周圍都是慘澹的顏色，那這就是過度擬合 (Overfitting)。真正的強健策略應該是：就算參數稍微改一點（例如均線從 20 改成 22），表現依然差不多。總結你的分析流程你可以試著按照這個順序思考：先看相關性：我選的這籃子股票夠散嗎？看資產配置：我的策略真的有在不同時間抓到不同機會嗎？看回撤與夏普：如果現在開始執行，我能忍受這種幅度的虧損嗎？對於你的 Quant.AI 系統，如果你能自動化分析並給出一段文字評語（例如：「警告：您的組合相關性過高」或「策略穩定度高，無明顯過擬合」），那這個產品的商業價值就非常高了。目前的圖表中，哪一張最讓你意外？我們可以針對那張深入討論。
