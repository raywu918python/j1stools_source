import os
import json
import time
import urllib.parse
import feedparser
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def fetch_google_news(stock_id, company_name="", days=30, max_items=10):
    query = f'{stock_id} {company_name} 股票 OR 股價 OR 營收 OR 法說 OR 展望 when:{days}d'
    encoded = urllib.parse.quote(query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )

    feed = feedparser.parse(url)

    news = []
    for entry in feed.entries[:max_items]:
        news.append({
            "title": entry.get("title", ""),
            "published": entry.get("published", ""),
            "link": entry.get("link", ""),
            "summary": entry.get("summary", "")
        })

    return news


def rank_stocks_by_news(stocks):
    all_news = {}

    for stock in stocks:
        stock_id = stock["id"]
        company = stock.get("name", "")
        news = fetch_google_news(stock_id, company)
        all_news[stock_id] = {
            "company": company,
            "news": news
        }
        time.sleep(1)

    prompt = f"""
你是台股新聞分析助理。

請根據「近30天新聞」幫我對以下股票做投資吸引力排名。

限制：
1. 只能根據我提供的新聞資料判斷
2. 不要使用你自己的記憶
3. 不要分析營收、EPS、法人、股價動能
4. 分數 0~100
5. 風險越高，分數要扣越多
6. 只輸出 JSON，不要輸出 markdown

評分標準：
- 正面新聞強度：40%
- 負面新聞風險：30%
- 題材延續性：20%
- 新聞可信度/密集度：10%

輸出格式：
{{
  "ranking": [
    {{
      "rank": 1,
      "stock_id": "xxxx",
      "company": "公司名",
      "score": 85,
      "summary": "一句話總結",
      "positive_factors": ["..."],
      "negative_factors": ["..."],
      "reason": "為什麼這樣排名"
    }}
  ]
}}

股票新聞資料：
{json.dumps(all_news, ensure_ascii=False)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Gemini 有時候會包 ```json，這裡清掉
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


if __name__ == "__main__":
    stocks = [
        {"id": "3406", "name": "玉晶光"},
        {"id": "6806", "name": "森崴能源"},
        {"id": "2330", "name": "台積電"}
    ]

    result = rank_stocks_by_news(stocks)

    df = pd.DataFrame(result["ranking"])
    print(df[["rank", "stock_id", "company", "score", "summary"]])

    df.to_csv("stock_news_ranking.csv", index=False, encoding="utf-8-sig")