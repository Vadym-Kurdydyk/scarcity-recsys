"""Sampling (Algorithm 1 of the paper) and explanatory variables (Section 3.2)."""
import numpy as np
import pandas as pd
from scipy import stats

EV_NAMES = ["space", "shape", "density", "rpu", "rpi", "gini_item", "gini_user",
            "pop_avg", "pop_std", "pop_skew", "pop_kurt",
            "lt_avg", "lt_std", "lt_skew", "lt_kurt",
            "rating_avg", "rating_std", "rating_skew", "rating_kurt"]


def load_ratings(path):
    return pd.read_csv(path, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])


def gini(counts):
    # eq. (8)/(9): counts sorted ascending, 0 = everyone has the same number of ratings, 1 = total inequality
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    k = np.arange(1, n + 1)
    return 1 - 2 * np.sum((n + 1 - k) / (n + 1) * x / x.sum())


def compute_evs(df):
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()
    n_ratings = len(df)
    ev = {}

    # structure of the URM
    ev["space"] = n_users * n_items
    ev["shape"] = n_users / n_items
    ev["density"] = n_ratings / (n_users * n_items)
    ev["rpu"] = n_ratings / n_users
    ev["rpi"] = n_ratings / n_items

    # rating frequency
    item_counts = df.groupby("item").size()
    user_counts = df.groupby("user").size()
    ev["gini_item"] = gini(item_counts.values)
    ev["gini_user"] = gini(user_counts.values)

    # popularity bias: average popularity of the items in each user profile
    pop = item_counts / n_users
    user_pop = df["item"].map(pop).groupby(df["user"]).mean()
    ev["pop_avg"] = user_pop.mean()
    ev["pop_std"] = user_pop.std()
    ev["pop_skew"] = stats.skew(user_pop)
    ev["pop_kurt"] = stats.kurtosis(user_pop)

    # long tail: short head = most popular items that make up 80% of the ratings
    sorted_counts = item_counts.sort_values(ascending=False)
    cum = sorted_counts.cumsum() / n_ratings
    short_head = set(sorted_counts.index[cum <= 0.8])
    is_lt = ~df["item"].isin(short_head)
    user_lt = is_lt.groupby(df["user"]).mean()
    ev["lt_avg"] = user_lt.mean()
    ev["lt_std"] = user_lt.std()
    ev["lt_skew"] = stats.skew(user_lt)
    ev["lt_kurt"] = stats.kurtosis(user_lt)

    # rating values
    r = df["rating"].values
    ev["rating_avg"] = r.mean()
    ev["rating_std"] = r.std()
    ev["rating_skew"] = stats.skew(r)
    ev["rating_kurt"] = stats.kurtosis(r)
    return ev


def sample_urm(df, rng, max_users=3000, max_items=5000, min_rpu=3, min_size=100, max_tries=1000):
    """Algorithm 1: pick a random number of users and items, keep the sample if it
    has at least 100 users / items and on average min_rpu ratings per user."""
    users = df["user"].unique()
    for _ in range(max_tries):
        n_u = rng.integers(min_size, min(len(users), max_users) + 1)
        chosen_users = rng.choice(users, n_u, replace=False)
        sub = df[df["user"].isin(chosen_users)]

        cand_items = sub["item"].unique()
        if len(cand_items) < min_size:
            continue
        n_i = rng.integers(min_size, min(len(cand_items), max_items) + 1)
        chosen_items = rng.choice(cand_items, n_i, replace=False)
        sub = sub[sub["item"].isin(chosen_items)]

        # users need at least 2 ratings, otherwise they can't be in train and test
        sub = sub[sub.groupby("user")["item"].transform("count") >= 2]
        n_users, n_items = sub["user"].nunique(), sub["item"].nunique()
        if n_users < min_size or n_items < min_size:
            continue
        if len(sub) / n_users < min_rpu:
            continue
        return sub.reset_index(drop=True)
    return None
