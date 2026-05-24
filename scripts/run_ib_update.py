import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from j1stools import hf_sync, ib_updater

hf_sync.pull(["db/ib", "db/ib_flags", "db/active_stocks"])
ib_updater.update_ib()
hf_sync.push(["db/ib", "db/ib_flags"])
