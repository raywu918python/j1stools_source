import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync, price_updater

hf_sync.pull(["db/price", "db/active_stocks", "db/info"])
price_updater.update(price_updater.INTERVAL.day, period="5d")
hf_sync.push(["db/price"])
