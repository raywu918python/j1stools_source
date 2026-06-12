import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import trade_api


def execute(open_df: dict, total_capital: float = 1_000_000, simulation: bool = True):
    """
    open_df: backtest 回傳的目前應持倉 {stock_id: position_info}
    total_capital: 假設總資金（元）
    simulation: True 用模擬帳號，False 用正式帳號
    """
    try:
        api = trade_api.test() if simulation else trade_api.login()
    except Exception as e:
        print(f"[ERROR] 登入失敗，跳過下單：{e}")
        return

    try:
        target = set(open_df.keys()) if open_df else set()
        current = trade_api.get_positions(api)

        to_open = target - set(current)
        to_close = set(current) - target

        if not to_open and not to_close:
            print("無需下單")
            return

        # 計算每支股票張數：總資金 ÷ 持倉數 ÷ (股價 × 1000)
        budget_per_stock = total_capital / max(len(target), 1)
        contracts = [api.Contracts.Stocks[sid] for sid in to_open]
        if contracts:
            snapshots = {s.code: s.close for s in api.snapshots(contracts)}
        else:
            snapshots = {}

        for sid in to_open:
            try:
                price = snapshots.get(sid, 0)
                if price <= 0:
                    print(f"[SKIP] {sid} 無法取得報價")
                    continue
                lots = int(budget_per_stock / (price * 1000))
                if lots < 1:
                    print(f"[SKIP] {sid} 資金不足一張（股價 {price}，預算 {budget_per_stock:.0f}）")
                    continue
                trade = trade_api.open(api, sid, lots)
                print(f"[OPEN] {sid} {lots}張 → {trade.status.status}")
            except Exception as e:
                print(f"[ERROR] 開倉 {sid} 失敗：{e}")

        for sid in to_close:
            try:
                lots = current[sid]
                trade = trade_api.close(api, sid, lots)
                print(f"[CLOSE] {sid} {lots}張 → {trade.status.status}")
            except Exception as e:
                print(f"[ERROR] 平倉 {sid} 失敗：{e}")

    except Exception as e:
        print(f"[ERROR] execute 失敗：{e}")
    finally:
        try:
            api.logout()
        except Exception as e:
            print(f"[ERROR] execute 失敗：{e}")


if __name__ == "__main__":
    from run_rfc_backtest import backtest

    _, _, open_df, _, _ = backtest()
    execute(open_df, total_capital=1_000_000, simulation=True)
