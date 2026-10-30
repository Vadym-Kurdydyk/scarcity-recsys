"""Convert data/yelp/yelp.tsv into the formats Elliot expects.

Outputs (in data/yelp/):
  yelp_elliot.tsv      user  item  rating  timestamp   (ints, no header)
  item_attributes.tsv  item  cat1  cat2 ...            (business categories)
  user_ids.tsv, item_ids.tsv, category_ids.tsv         (int -> original id/name)
"""
import json
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "yelp")

ratings = pd.read_csv(os.path.join(DATA, "yelp.tsv"), sep="\t", index_col=0)

users = {u: i for i, u in enumerate(ratings["user_id"].unique())}
items = {b: i for i, b in enumerate(ratings["business_id"].unique())}

out = pd.DataFrame({
    "user": ratings["user_id"].map(users),
    "item": ratings["business_id"].map(items),
    "rating": ratings["stars"],
    "timestamp": pd.to_datetime(ratings["date"]).astype("int64") // 10 ** 9,
})
out.to_csv(os.path.join(DATA, "yelp_elliot.tsv"), sep="\t", header=False, index=False)

categories = {}
attribute_lines = []
with open(os.path.join(DATA, "yelp_academic_dataset_business.json"), encoding="utf8") as f:
    for line in f:
        business = json.loads(line)
        if business["business_id"] not in items or not business["categories"]:
            continue
        cats = [c.strip() for c in business["categories"].split(",") if c.strip()]
        cat_ids = sorted({categories.setdefault(c, len(categories)) for c in cats})
        attribute_lines.append("\t".join(map(str, [items[business["business_id"]], *cat_ids])))

with open(os.path.join(DATA, "item_attributes.tsv"), "w", encoding="utf8") as f:
    f.write("\n".join(attribute_lines) + "\n")

pd.Series(list(users)).to_csv(os.path.join(DATA, "user_ids.tsv"), sep="\t", header=False)
pd.Series(list(items)).to_csv(os.path.join(DATA, "item_ids.tsv"), sep="\t", header=False)
pd.Series(list(categories)).to_csv(os.path.join(DATA, "category_ids.tsv"), sep="\t", header=False)

print(f"{len(out)} ratings, {len(users)} users, {len(items)} items, "
      f"{len(attribute_lines)} items with categories, {len(categories)} categories")
