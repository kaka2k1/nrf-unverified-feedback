"""Prepare BFCL multiple + live_multiple as a function-selection task with data-driven contexts.

python scripts/prep_bfcl.py  ->  data/bfcl_items.jsonl
Each item: id, group (function-set group), query, candidates [{name, description}], gold (index), context (cluster id),
context_terms. Contexts = KMeans(k) over TF-IDF of the candidate function names+descriptions (deterministic).
"""
import json
import os
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "bfcl")
MAX_CANDS = 10
DESC_CHARS = 160
K = 8


def load(name):
    Q = [json.loads(line) for line in open(os.path.join(D, name + ".json")) if line.strip()]
    A = {}
    for line in open(os.path.join(D, "possible_answer", name + ".json")):
        if line.strip():
            r = json.loads(line)
            A[r["id"]] = r
    out = []
    for q in Q:
        if q["id"] not in A or len(q["function"]) > MAX_CANDS:
            continue
        msgs = q["question"][0]
        user = [m["content"] for m in msgs if m["role"] == "user"]
        if not user:
            continue
        gt = list(A[q["id"]]["ground_truth"][0].keys())[0]
        names = [f["name"] for f in q["function"]]
        if gt not in names or len(set(names)) != len(names):
            continue
        grp = ("L" + q["id"].split("-")[1]) if q["id"].startswith("live") else ("M" + str(hash(frozenset(names)) % 10**8))
        out.append(dict(id=q["id"], group=grp, query=user[-1].strip()[:600],
                        candidates=[dict(name=f["name"], description=re.sub(r"\s+", " ", f.get("description", ""))[:DESC_CHARS])
                                    for f in q["function"]],
                        gold=names.index(gt)))
    return out


def assign_contexts(items):
    docs = [" ".join(re.sub(r"[._]", " ", c["name"]) + " " + c["description"] for c in it["candidates"]) for it in items]
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.3, sublinear_tf=True)
    X = vec.fit_transform(docs)
    km = KMeans(K, n_init=20, random_state=0).fit(X)
    terms = np.array(vec.get_feature_names_out())
    top = {k: terms[np.argsort(-km.cluster_centers_[k])[:6]].tolist() for k in range(K)}
    for it, c in zip(items, km.labels_):
        it["context"] = int(c)
        it["context_terms"] = top[int(c)]
    return km.labels_, top


def main():
    items = load("BFCL_v3_multiple") + load("BFCL_v3_live_multiple")
    # groups for "multiple" via deterministic key (python hash is salted -> use sorted names instead)
    for it in items:
        if it["group"].startswith("M"):
            it["group"] = "M:" + "|".join(sorted(c["name"] for c in it["candidates"]))[:120]
    labels, top = assign_contexts(items)
    with open(os.path.join(ROOT, "data", "bfcl_items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    cnt = np.bincount(labels, minlength=K)
    print(len(items), "items;", len(set(i["group"] for i in items)), "groups")
    for k in range(K):
        print(k, cnt[k], top[k])
    print("mean #cands", np.mean([len(i["candidates"]) for i in items]))


if __name__ == "__main__":
    main()
