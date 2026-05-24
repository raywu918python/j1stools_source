import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync, margin_updater

hf_sync.pull(["db/margin", "db/margin_flags", "db/active_stocks"])
margin_updater.update_margin()
hf_sync.push(["db/margin", "db/margin_flags"])
