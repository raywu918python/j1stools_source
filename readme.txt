pwiz -e sqlite just1stock.db > src/db_models/peewee_models.py 

pip install -e .

import sys
sys.path.insert(0, "src")

from j1stools.hf_sync import push
push(["models", "db/feature_cols"])


https://raywu918python-j1s-api.hf.space
/predictions — 預設用 lgbm_timeseries_ensemble
/predictions?model=lgbm_timeseries_ensemble — 同上，明確指定
/predictions/2026-05-24?model=xxx — 指定日期 + 模型

curl https://raywu918python-j1s-api.hf.space/predictions/

ollama serve

aider --model ollama/qwen2.5-coder:14b src/free_agent/free_agent.py