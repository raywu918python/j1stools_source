import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync
from j1stools.day_trade_updater import update_day_trade

hf_sync.pull(["db/day_trade", "db/day_trade_flags"])
update_day_trade()
hf_sync.push(["db/day_trade", "db/day_trade_flags"])
