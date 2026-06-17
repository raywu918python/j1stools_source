import os
import time
from huggingface_hub import HfApi, snapshot_download

_REPO_ID = os.environ.get("HF_REPO_ID", "raywu918python/j1s-data")


def pull(folders: list, local_dir="."):
    token = os.environ.get("HF_TOKEN")
    for folder in folders:
        expected = os.path.join(local_dir, folder)
        if os.environ.get("GITHUB_ACTIONS") == "true" and os.path.isdir(expected) and any(os.scandir(expected)):
            print(f"[hf_sync] cache hit, skip pull: {folder}")
            continue
        for attempt in range(5):
            try:
                snapshot_download(
                    repo_id=_REPO_ID,
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=f"{folder}/**",
                    token=token,
                )
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"[hf_sync] pull failed after 5 attempts: {folder}") from e
                wait = 90 * (attempt + 1)
                print(f"[hf_sync] download error ({e}), retry in {wait}s...")
                time.sleep(wait)
                continue
            expected = os.path.join(local_dir, folder)
            if os.path.isdir(expected) and any(os.scandir(expected)):
                print(f"[hf_sync] pulled: {folder}")
                break
            if attempt == 4:
                raise RuntimeError(f"[hf_sync] pull failed after 5 attempts: {folder}")
            wait = 90
            print(f"[hf_sync] {folder} not found after pull, retry in {wait}s...")
            time.sleep(wait)


def push(folders: list, local_dir="."):
    """把本地資料夾推回 HF Hub（需要 HF_TOKEN 環境變數）"""
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    for folder in folders:
        local_path = os.path.join(local_dir, folder)
        api.upload_folder(
            repo_id=_REPO_ID,
            repo_type="dataset",
            folder_path=local_path,
            path_in_repo=folder,
        )
        print(f"[hf_sync] pushed: {folder}")
