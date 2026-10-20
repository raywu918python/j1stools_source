"""把重訓模型換上 HF 的 models/rfc_macd_ib_6xx.joblib，先備份舊檔到 models/backup/。"""

import os, sys

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = os.environ.get("HF_REPO_ID", "raywu918python/j1s-data")
TARGET = "models/rfc_macd_ib_6xx.joblib"
BACKUP = "models/backup/rfc_macd_ib_6xx_before_nolook_2026-10-20.joblib"
NEW_MODEL = sys.argv[1]

api = HfApi(token=os.environ["HF_TOKEN"])
old = hf_hub_download(REPO_ID, TARGET, repo_type="dataset", token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=old, path_in_repo=BACKUP, repo_id=REPO_ID, repo_type="dataset", commit_message="backup rfc_macd_ib_6xx before nolook retrain")
print(f"backed up → {BACKUP} ({os.path.getsize(old)} bytes)")

import joblib

model = joblib.load(NEW_MODEL)
print("new model features:", list(model.feature_names_in_), "classes:", list(model.classes_))
api.upload_file(path_or_fileobj=NEW_MODEL, path_in_repo=TARGET, repo_id=REPO_ID, repo_type="dataset", commit_message="deploy rfc_macd_ib_6xx retrained without f_macd_hist_div lookahead")
print(f"deployed → {TARGET} ({os.path.getsize(NEW_MODEL)} bytes)")
