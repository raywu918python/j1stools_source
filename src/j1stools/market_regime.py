"""
市場 Regime 偵測（HMM）

用 Hidden Markov Model 對大盤的籌碼 + 價格特徵做時序分群，
偵測市場目前處於哪個狀態（多頭 / 盤整 / 空頭）。

每個交易日輸出一個 regime label，可作為選股模型的大盤過濾器。

典型用法：
    hmm_model, df = run_hmm(st="2015-01-01", n_states=3)
    today = predict_today_regime()
    # → {"date": ..., "regime": 0, "prob": 0.92}
"""

import os
import pickle

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from j1stools import parquet_db

N_STATES = 3
MODEL_PATH = "db/models/market_regime_hmm.pkl"
MARKET_PROXY = "0050"


def build_market_features(st: str = "2015-01-01", end: str = "2099-01-01") -> pd.DataFrame:
    """
    建立市場層級的日頻特徵序列（每天一行）。

    特徵：
      - mkt_return       : 大盤日報酬
      - mkt_volatility_5d: 5日滾動波動率
      - mkt_volume_ratio : 當日成交量 / 5日均量
      - foreign_net_pct  : 外資總淨買超 / 大盤總成交量（張）
      - trust_net_pct    : 投信總淨買超 / 大盤總成交量（張）
      - margin_change    : 全市場融資餘額日變化率
      - short_change     : 全市場融券餘額日變化率

    Returns
    -------
    pd.DataFrame : index=date（已去 NaN），columns=特徵
    """
    stocks = [s for s in parquet_db.activate_stocks() if not s.startswith("00")]

    # ── 大盤價格特徵 ──────────────────────────────────────────────── #
    df_mkt = parquet_db.query_price([MARKET_PROXY], st, end)
    df_mkt["date"] = pd.to_datetime(df_mkt["date"])
    df_mkt = df_mkt.sort_values("date").set_index("date")

    mkt_ret = df_mkt["close"].pct_change(fill_method=None)
    mkt_vol = mkt_ret.rolling(5).std()
    mkt_vol_ratio = df_mkt["volume"] / df_mkt["volume"].rolling(5).mean()

    feat = pd.DataFrame({
        "mkt_return": mkt_ret,
        "mkt_volatility_5d": mkt_vol,
        "mkt_volume_ratio": mkt_vol_ratio,
    })

    # ── 法人總淨買超 ──────────────────────────────────────────────── #
    print("載入法人資料...")
    df_ib = parquet_db.query_ib(stocks, st, end)
    df_ib["date"] = pd.to_datetime(df_ib["date"])
    df_ib["net"] = df_ib["buy"] - df_ib["sell"]

    foreign = df_ib[df_ib["name"] == "Foreign_Investor"].groupby("date")["net"].sum()
    trust = df_ib[df_ib["name"] == "Investment_Trust"].groupby("date")["net"].sum()

    # 個股總成交量（張）作為分母
    print("載入個股成交量...")
    df_price_all = parquet_db.query_price(stocks, st, end)
    df_price_all["date"] = pd.to_datetime(df_price_all["date"])
    daily_volume_lots = df_price_all.groupby("date")["volume"].sum() / 1000

    feat["foreign_net_pct"] = (foreign / daily_volume_lots).reindex(feat.index)
    feat["trust_net_pct"] = (trust / daily_volume_lots).reindex(feat.index)

    # ── 融資融券總變化 ────────────────────────────────────────────── #
    print("載入融資融券資料...")
    df_margin = parquet_db.query_margin(stocks, st, end)
    df_margin["date"] = pd.to_datetime(df_margin["date"])

    daily_margin = df_margin.groupby("date")[
        ["margin_purchase_today_balance", "short_sale_today_balance"]
    ].sum()

    feat["margin_change"] = daily_margin["margin_purchase_today_balance"].pct_change().reindex(feat.index)
    feat["short_change"] = daily_margin["short_sale_today_balance"].pct_change().reindex(feat.index)

    # ── 截斷極端值（1%~99%）── ───────────────────────────────────── #
    for col in feat.columns:
        q01, q99 = feat[col].quantile([0.01, 0.99])
        feat[col] = feat[col].clip(q01, q99)

    # ── 5 日滾動平滑（關鍵）─────────────────────────────────────── #
    # 日頻籌碼資料噪聲大，直接餵 HMM 會導致 regime 每幾天就切換。
    # 平滑後 HMM 能抓到持續數週的真實市場狀態。
    feat = feat.rolling(5, min_periods=3).mean()

    feat = feat.dropna()
    print(f"市場特徵完成：{feat.index.min().date()} ~ {feat.index.max().date()}，{len(feat)} 個交易日")
    return feat


class MarketRegimeHMM:
    """
    Hidden Markov Model 市場 Regime 分類器。

    訓練後自動識別市場狀態（如多頭/盤整/空頭），
    需用 describe_regimes() 輸出後人工對應各狀態名稱。

    關鍵屬性：
      - transmat_  : 狀態轉移矩陣，看各 regime 之間的切換機率
      - regime_profiles : 各 regime 的特徵均值，幫助命名
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.model: GaussianHMM | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list = []
        self.regime_profiles: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "MarketRegimeHMM":
        """
        訓練 HMM。

        Parameters
        ----------
        df : build_market_features() 的輸出（index=date）
        """
        self.feature_names = list(df.columns)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df.values)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=500,
            random_state=42,
            verbose=False,
        )
        self.model.fit(X, lengths=[len(X)])

        labels = self.model.predict(X, lengths=[len(X)])
        self.regime_profiles = (
            df.assign(regime=labels).groupby("regime")[self.feature_names].mean().round(4)
        )

        print(f"收斂：{self.model.monitor_.converged}")
        print(f"Log-likelihood：{self.model.score(X, lengths=[len(X)]):.2f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """對時序資料推斷每日 regime（Viterbi 解碼）。"""
        if self.model is None:
            raise RuntimeError("請先呼叫 fit()")
        X = self.scaler.transform(df[self.feature_names].values)
        labels = self.model.predict(X, lengths=[len(X)])
        return pd.Series(labels, index=df.index, name="regime")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """回傳各 regime 的後驗機率（前向-後向演算法）。"""
        if self.model is None:
            raise RuntimeError("請先呼叫 fit()")
        X = self.scaler.transform(df[self.feature_names].values)
        proba = self.model.predict_proba(X)
        cols = [f"prob_{k}" for k in range(self.n_states)]
        return pd.DataFrame(proba, index=df.index, columns=cols)

    def describe_regimes(self):
        """印出各 regime 特徵均值 + 轉移矩陣，幫助人工命名。"""
        if self.regime_profiles is None:
            raise RuntimeError("請先呼叫 fit()")

        print(f"\n{'='*60}")
        print(f"各 Regime 特徵均值（{self.n_states} 個狀態）")
        print(f"{'='*60}")
        print(self.regime_profiles.T.to_string())

        print(f"\n轉移矩陣（row=今天, col=明天）：")
        trans = pd.DataFrame(
            self.model.transmat_,
            index=[f"regime_{k}" for k in range(self.n_states)],
            columns=[f"→{k}" for k in range(self.n_states)],
        )
        print(trans.round(3).to_string())

        print(f"\n各 regime 平均持續天數：")
        for k in range(self.n_states):
            stay_prob = self.model.transmat_[k, k]
            avg_days = 1 / (1 - stay_prob) if stay_prob < 1 else float("inf")
            print(f"  Regime {k}: 平均持續 {avg_days:.1f} 天")

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"HMM 模型已儲存：{path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "MarketRegimeHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"HMM 模型已載入：{path}（n_states={obj.n_states}）")
        return obj


def plot_regimes(df: pd.DataFrame, df_price: pd.DataFrame | None = None):
    """
    視覺化 regime 時序：
      上：大盤走勢 + regime 背景色
      下：各 regime 後驗機率堆疊圖
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]

    n_k = df["regime"].nunique()
    palette = plt.cm.Set1.colors[:n_k]
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    n_rows = 2 if df_price is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # ── 上：大盤走勢 + regime 背景色 ───────────────────────────────── #
    if df_price is not None:
        ax1 = axes[0]
        price_aligned = df_price["close"].reindex(df.index)
        ax1.plot(df.index, price_aligned, color="black", linewidth=0.8, zorder=3)
        y_min = price_aligned.min() * 0.95
        y_max = price_aligned.max() * 1.05
        for k in range(n_k):
            mask = df["regime"] == k
            ax1.fill_between(df.index, y_min, y_max, where=mask,
                             alpha=0.25, color=palette[k], label=f"Regime {k}")
        ax1.set_title("大盤走勢 + Market Regime")
        ax1.set_ylabel("指數")
        ax1.legend(loc="upper left")

    # ── 下：機率堆疊圖 ─────────────────────────────────────────────── #
    ax2 = axes[-1]
    bottom = np.zeros(len(df))
    for k, col in enumerate(prob_cols):
        if col in df.columns:
            vals = df[col].values
            ax2.fill_between(df.index, bottom, bottom + vals,
                             color=palette[k], alpha=0.8, label=f"Regime {k}")
            bottom += vals
    ax2.set_title("各 Regime 後驗機率")
    ax2.set_ylabel("機率")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def run_hmm(
    st: str = "2015-01-01",
    end: str = "2099-01-01",
    n_states: int = N_STATES,
) -> tuple:
    """
    主流程：建立市場特徵 → 訓練 HMM → 描述 regime → 畫圖 → 儲存。

    Returns
    -------
    (model, df) : MarketRegimeHMM + 含 regime, prob_* 欄位的 DataFrame
    """
    df_feat = build_market_features(st, end)

    print(f"\n訓練 HMM（n_states={n_states}）...")
    model = MarketRegimeHMM(n_states=n_states)
    model.fit(df_feat)

    df_feat["regime"] = model.predict(df_feat)
    proba = model.predict_proba(df_feat)
    df_feat = pd.concat([df_feat, proba], axis=1)

    model.describe_regimes()

    df_price = parquet_db.query_price([MARKET_PROXY], st, end)
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.set_index("date")

    plot_regimes(df_feat, df_price)
    model.save()

    return model, df_feat


def predict_today_regime(st: str = "2024-01-01") -> dict:
    """
    載入訓練好的 HMM，推斷最新一日的市場 regime。

    Returns
    -------
    dict : date, regime, prob（最高機率的 regime 信心）
    """
    model = MarketRegimeHMM.load()

    df_feat = build_market_features(st)
    df_feat["regime"] = model.predict(df_feat)
    proba = model.predict_proba(df_feat)
    df_feat = pd.concat([df_feat, proba], axis=1)

    latest = df_feat.iloc[-1]
    regime = int(latest["regime"])
    prob = float(latest[f"prob_{regime}"])

    print(f"\n最新日期：{df_feat.index[-1].date()}")
    print(f"市場 Regime：{regime}  信心：{prob:.1%}")
    prob_cols = [c for c in df_feat.columns if c.startswith("prob_")]
    print(latest[prob_cols].round(3).to_string())

    return {"date": df_feat.index[-1], "regime": regime, "prob": prob}


if __name__ == "__main__":
    model, df = run_hmm(st="2015-01-01", n_states=N_STATES)
