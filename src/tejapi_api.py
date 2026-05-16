import tejapi

tejapi.ApiConfig.api_key = "mPy4yqWO3X3LtVYnPxKLAPhdsLJNhV"
info = tejapi.ApiConfig.info()
print(info)
# data = tejapi.get("TWN/ASHR1")

import requests
import pandas as pd

API_KEY = "mPy4yqWO3X3LtVYnPxKLAPhdsLJNhV"  # TODO: 換成自己的 key
BASE_URL = "https://api.tej.com.tw/api/datatables"


def get_chip_dist(coid="2330", start="2024-01-01", end="2024-12-31", limit=5000):
    """
    TWN/ASHR1 籌碼分佈資料
    coid: 證券代碼，例如 2330
    start, end: 日期(YYYY-MM-DD)
    """
    url = f"{BASE_URL}/TWN/ASHR1.json"
    params = {
        "api_key": API_KEY,
        "coid": coid,
        "mdate": f"{start}:{end}",
        "topn": limit,
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    # TEJ datatables 格式通常是 {"data":[{...}, {...}]}
    df = pd.DataFrame(data["data"])
    return df


if __name__ == "__main__":
    df = get_chip_dist(coid="2330", start="2025-01-01", end="2025-01-31")
    print(df.head())
    # 如果要存檔：
    df.to_csv("ASHR1_2330_202501.csv", index=False, encoding="utf-8-sig")
