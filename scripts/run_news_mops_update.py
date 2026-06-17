import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from j1stools import hf_sync
from news_mops import update_mops_news

hf_sync.pull(["db/news_mops", "db/news_mops_flags"])
update_mops_news()
hf_sync.push(["db/news_mops", "db/news_mops_flags"])
