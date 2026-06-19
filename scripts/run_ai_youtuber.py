import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync, ai_youtuber

# 題材來自 HF 上的 db/chat/{date}/（AI 圓桌會議結果），但畫K線圖需要本地 db/price、db/info；
# 雲端跑（GitHub Actions）db/ 是空的，要先 pull；本機開發已經有 db/ 這步會直接跳過。
hf_sync.pull(["db/price", "db/info"])

date_str = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
ai_youtuber.main(date_str=date_str)
