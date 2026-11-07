"""Builds one clean rating file per dataset for the EV (data characteristics) study.

Output for each dataset: data/<name>/<name>_ratings.tsv with columns
user  item  rating  timestamp   (ints, no header)

Steps: read raw file -> keep only user, item, rating, time -> drop duplicates
(keep the last rating of a user for an item) -> k-core filter -> map ids to ints.

usage: python prepare_datasets.py yelp amazon goodreads
"""
import gzip
import json
import sys

import pandas as pd

K_CORE = {"yelp": 10, "amazon": 5, "goodreads": 10}


def read_yelp():
    rows = []
    with open("data/yelp/yelp_academic_dataset_review.json", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["user_id"], r["business_id"], r["stars"], r["date"]))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10 ** 9
    return df


def read_amazon(category="Musical_Instruments"):
    rows = []
    with open(f"data/amazon/{category}.jsonl", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["user_id"], r["parent_asin"], r["rating"], r["timestamp"] // 1000))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])


def read_goodreads():
    # rating 0 means the book is only on a shelf without a rating -> dropped
    rows = []
    with gzip.open("data/goodreads/goodreads_interactions_poetry.json.gz", "rt", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["rating"] > 0:
                rows.append((r["user_id"], r["book_id"], r["rating"], r["date_updated"]))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%a %b %d %H:%M:%S %z %Y", utc=True).astype("int64") // 10 ** 9
    return df


def k_core(df, k):
    while True:
        n = len(df)
        df = df[df.groupby("user")["item"].transform("count") >= k]
        df = df[df.groupby("item")["user"].transform("count") >= k]
        if len(df) == n:
            return df


if __name__ == "__main__":
    readers = {"yelp": read_yelp, "amazon": read_amazon, "goodreads": read_goodreads}
    for name in sys.argv[1:]:
        df = readers[name]()
        print(name, "raw:", len(df), "ratings,", df.user.nunique(), "users,", df.item.nunique(), "items")
        df = df.sort_values("timestamp").drop_duplicates(["user", "item"], keep="last")
        df = k_core(df, K_CORE[name])
        df["user"] = df["user"].astype("category").cat.codes
        df["item"] = df["item"].astype("category").cat.codes
        df["rating"] = df["rating"].astype(int)
        df.to_csv(f"data/{name}/{name}_ratings.tsv", sep="\t", header=False, index=False)
        n_u, n_i = df.user.nunique(), df.item.nunique()
        print(name, f"{K_CORE[name]}-core:", len(df), "ratings,", n_u, "users,", n_i, "items, density",
              round(len(df) / (n_u * n_i), 5))
