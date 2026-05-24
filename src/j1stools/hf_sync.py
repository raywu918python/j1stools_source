import os
from huggingface_hub import HfApi, snapshot_download

_REPO_ID = os.environ.get("HF_REPO_ID", "raywu918python/j1s-data")


def pull(folders: list, local_dir="."):
    """從 HF Hub 下載指定資料夾到本地（公開 repo，不需要 token）"""
    for folder in folders:
        snapshot_download(
            repo_id=_REPO_ID,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=f"{folder}/**",
        )
        print(f"[hf_sync] pulled: {folder}")


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
