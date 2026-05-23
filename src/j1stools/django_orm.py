import os
import sys

import django

sys.path.insert(0, "/Users/wumingrui/Library/CloudStorage/Dropbox/just1stock_web")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "just1stock.settings")
django.setup()

from myapp.models import (
    Portfolio,
    Positions,
    StocksInfo,
    ActiveStocks,
    StocksMargin,
    StocksUpdateFlag,
    StocksIbBuySell,
    FeatureCols,
)
