"""Break-even analysis: how many verified corrections is the UNVERIFIED feedback pool worth?

For each learner, noise condition and seed we trace the verified-only learning curve a_V(rho) over a fine grid of
verify rates (masks are nested across rho for the same seed, because make_benchmark draws the verify mask last from
the same RNG stream), and run the weighting methods at a few rho. The "exchange rate" of a method at rho is
rho_eq / rho, where a_V(rho_eq) = a_method(rho) (inverse of the seed-averaged verified-only curve): >1 means the
unverified pool is worth extra verifications, <1 means using it costs verifications.

Learners (increasing strength, all CPU):
  tfidf     TF-IDF(1-2 gram) + logistic regression (the paper's CPU learner)
  bge_small frozen BAAI/bge-small-en-v1.5 CLS embeddings + logistic regression
  bge_base  frozen BAAI/bge-base-en-v1.5  CLS embeddings + logistic regression

Extra baseline (reviewer question "do the feedback LABELS add anything beyond the unlabelled inputs?"):
  self_train  train on seed + verified, pseudo-label the unverified inputs with that model (feedback labels ignored),
              weight q**gamma * 1[q >= tau] with q the model's max probability.

python breakeven_cpu.py --learners tfidf bge_small --seeds 10 --out results/cpu_breakeven
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import issparse, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nrf import make_benchmark, method_weights, observable_view

RHO_CURVE = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
RHO_METHODS = [0.02, 0.05, 0.1, 0.25]
METHODS = ["naive", "verifier_corrected", "conf_filter", "source_prior", "ctx_eb", "ca_eb", "self_train",
           "oracle_clean"]
CONDS = [("context", 0.4), ("source", 0.4), ("uniform", 0.2), ("context", 0.2)]
EMB_MODELS = {"bge_small": "BAAI/bge-small-en-v1.5", "bge_base": "BAAI/bge-base-en-v1.5"}
EMB = {}


def embed_texts(texts, name, cache_dir, bs=64):
    """CLS-pooled, L2-normalised sentence embeddings (bge convention), cached on disk."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = os.path.join(cache_dir, f"emb_{name}.npz")
    have = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        have = dict(zip(z["texts"].tolist(), z["emb"]))
    todo = sorted(set(texts) - set(have))
    if todo:
        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(EMB_MODELS[name])
        mod = AutoModel.from_pretrained(EMB_MODELS[name]).eval()
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(todo), bs):
                b = tok(todo[i:i + bs], padding=True, truncation=True, max_length=96, return_tensors="pt")
                e = mod(**b).last_hidden_state[:, 0]
                e = torch.nn.functional.normalize(e, dim=-1).numpy()
                have.update(zip(todo[i:i + bs], e))
        print(f"embedded {len(todo)} texts with {name} in {time.time() - t0:.0f}s", flush=True)
        keys = list(have)
        np.savez(path, texts=np.array(keys, dtype=object), emb=np.stack([have[k] for k in keys]))
    return have


def featurize(kind, seed_df, adapt, test):
    if kind == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        vec.fit(pd.concat([seed_df.text, adapt.text]))
        return vec.transform(seed_df.text), vec.transform(adapt.text), vec.transform(test.text)
    E = EMB[kind]
    f = lambda s: np.stack([E[t] for t in s])  # noqa: E731
    return f(seed_df.text), f(adapt.text), f(test.text)


def fit(kind, X, y, w=None):
    C = 5.0 if kind == "tfidf" else 10.0
    return LogisticRegression(C=C, max_iter=3000).fit(X, y, sample_weight=w)


def stack(a, b):
    return vstack([a, b]) if issparse(a) else np.vstack([a, b])


def proba4(clf, X):
    p = np.zeros((X.shape[0], 4))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def self_train_weights(kind, Xs, ys, Xa, obs, tau, gamma):
    """Pseudo-label the unverified inputs with a model trained on seed + verified items; feedback labels ignored."""
    ver = obs.verified.to_numpy().astype(bool)
    yv = obs.y_ver.to_numpy()
    clf = fit(kind, stack(Xs, Xa[ver]), np.r_[ys, yv[ver]])
    p = proba4(clf, Xa)
    q = p.max(1)
    lab = np.where(ver, yv, p.argmax(1))
    w = np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0))
    return lab, w, q


def run_job(job):
    kind, nt, nz, s, tau, gamma = job
    rows = []
    for rho in RHO_CURVE:
        b = make_benchmark(seed=1000 + s, noise=nz, noise_type=nt, verify_rate=rho)
        seed_df, adapt, test = b["seed"], b["adapt"], b["test"]
        obs = observable_view(adapt)
        Xs, Xa, Xt = featurize(kind, seed_df, adapt, test)
        ys, yt = seed_df.y_true.to_numpy(), test.y_true.to_numpy()
        y_true_a = adapt.y_true.to_numpy()
        unv = ~obs.verified.to_numpy().astype(bool)
        seed_clf = fit(kind, Xs, ys)
        p_seed = proba4(seed_clf, Xa)
        meths = ["verified_only"] + (METHODS if rho in RHO_METHODS else [])
        for m in meths:
            if m == "self_train":
                lab, w, q = self_train_weights(kind, Xs, ys, Xa, obs, tau, gamma)
            else:
                lab, w, q = method_weights(m, obs, p_seed, tau=tau, gamma=gamma,
                                           y_true_for_oracle=y_true_a if m == "oracle_clean" else None)
            keep = w > 0
            clf = fit(kind, stack(Xs, Xa[keep]), np.r_[ys, lab[keep]], np.r_[np.ones(len(ys)), w[keep]])
            pred = clf.predict(Xt)
            ok = pred == yt
            wu = w[unv]
            cu = (lab == y_true_a)[unv]
            rows.append(dict(learner=kind, noise_type=nt, noise=nz, seed=s, rho=rho, method=m,
                             n_ver=int((~unv).sum()), acc=float(ok.mean()),
                             acc_seen=float(ok[test.split.to_numpy() == "seen"].mean()),
                             acc_held=float(ok[test.split.to_numpy() == "held"].mean()),
                             w_unv=float(wu.sum()), prec_unv=float((wu * cu).sum() / max(wu.sum(), 1e-9)),
                             realized_noise=b["meta"]["realized_noise"]))
        if rho == RHO_CURVE[0]:
            ok = seed_clf.predict(Xt) == yt
            rows.append(dict(learner=kind, noise_type=nt, noise=nz, seed=s, rho=0.0, method="seed_only", n_ver=0,
                             acc=float(ok.mean()), acc_seen=np.nan, acc_held=np.nan, w_unv=0.0, prec_unv=np.nan,
                             realized_noise=b["meta"]["realized_noise"]))
    return rows


def _init(emb):
    EMB.update(emb)
    os.environ["OMP_NUM_THREADS"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="+", default=["tfidf", "bge_small"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--out", default="results/cpu_breakeven")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    texts = set()
    for s in range(a.seeds):
        b = make_benchmark(seed=1000 + s, noise=0.4, noise_type="context", verify_rate=0.1)
        texts |= set(b["seed"].text) | set(b["adapt"].text) | set(b["test"].text)
    for nt, nz in CONDS:  # texts do not depend on the noise draw; check once
        b = make_benchmark(seed=1000, noise=nz, noise_type=nt, verify_rate=0.1)
        assert set(b["adapt"].text) <= texts
    emb = {}
    for k in a.learners:
        if k in EMB_MODELS:
            emb[k] = embed_texts(sorted(texts), k, a.out)
    jobs = [(k, nt, nz, s, a.tau, a.gamma) for k in a.learners for nt, nz in CONDS for s in range(a.seeds)]
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(emb,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            rows += r
            if i % 5 == 0:
                print(f"{i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    json.dump(dict(args=vars(a), rho_curve=RHO_CURVE, rho_methods=RHO_METHODS, conds=CONDS, methods=METHODS,
                   sec=time.time() - t0), open(f"{a.out}/run_meta.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
