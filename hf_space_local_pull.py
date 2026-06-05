import sys


def push():
    from j1stools.hf_sync import push

    sys.path.insert(0, "src")

    push(["models", "db/feature_cols"])


# python test.py
def pull():

    from dotenv import load_dotenv

    load_dotenv()

    sys.path.insert(0, "src")
    from j1stools.hf_sync import pull

    pull(["db/price", "db/ib", "db/margin", "db/active_stocks", "db/feature_cols"])


if __name__ == "__main__":
    pull()
