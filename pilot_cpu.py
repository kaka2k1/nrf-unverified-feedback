"""CPU pilot: TF-IDF + logistic regression learner on the synthetic noisy-feedback benchmark.

python pilot_cpu.py --seeds 8 --noise 0 0.2 0.4 --noise-type context --verify-rate 0.25 --out results/cpu_context
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from nrf import METHODS, REFERENCE, make_benchmark, method_weights, observable_view, weight_diagnostics


def featurize(train_texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    vec.fit(train_texts)
    return vec


def fit_lr(X, y, w=None):
    clf = LogisticRegression(C=5.0, max_iter=2000)
    clf.fit(X, y, sample_weight=w)
    return clf


def _self_confidence(Xs, ys, Xa, obs, folds=2):
    """Out-of-fold p(y_fb | x) from a model trained on seed + (verifier-corrected) noisy feedback."""
    from scipy.sparse import vstack
    ver = obs.verified.to_numpy()
    lab = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
    f = np.random.default_rng(0).integers(folds, size=Xa.shape[0])
    q = np.zeros(Xa.shape[0])
    for k in range(folds):
        tr, te = f != k, f == k
        clf = fit_lr(vstack([Xs, Xa[tr]]), np.r_[ys, lab[tr]])
        pr = np.zeros((te.sum(), 4))
        pr[:, clf.classes_] = clf.predict_proba(Xa[te])
        q[te] = pr[np.arange(te.sum()), obs.y_fb.to_numpy()[te]]
    return q


def _acc_policy(pred, test):
    m = test.policy_item.to_numpy()
    return float((pred == test.y_true.to_numpy())[m].mean()) if m.any() else float("nan")


def run_one(bench, methods, tau, gamma):
    seed_df, adapt, test = bench["seed"], bench["adapt"], bench["test"]
    obs = observable_view(adapt)
    # vocabulary from all *inputs* available at training time (no labels involved)
    vec = featurize(pd.concat([seed_df.text, adapt.text]))
    Xs, Xa, Xt = vec.transform(seed_df.text), vec.transform(adapt.text), vec.transform(test.text)
    seed_clf = fit_lr(Xs, seed_df.y_true)
    proba = np.zeros((len(adapt), 4))
    proba[:, seed_clf.classes_] = seed_clf.predict_proba(Xa)
    p_seed_fb = proba[np.arange(len(adapt)), obs.y_fb.to_numpy()]
    rows = []
    ys = seed_df.y_true.to_numpy()
    for m in methods:
        t0 = time.time()
        if m == "self_filter":  # ROPO-style: filter by the (cross-fitted) noisy-trained model's own confidence
            q = _self_confidence(Xs, ys, Xa, obs)
            ver = obs.verified.to_numpy()
            labels = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
            w = np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0))
        else:
            labels, w, q = method_weights(m, obs, proba, tau=tau, gamma=gamma,
                                          y_true_for_oracle=adapt.y_true.to_numpy() if m == "oracle_clean" else None)
        keep = w > 0
        from scipy.sparse import vstack
        X = vstack([Xs, Xa[keep]])
        y = np.r_[ys, labels[keep]]
        sw = np.r_[np.ones(len(ys)), w[keep]]
        clf = fit_lr(X, y, sw)
        pred = clf.predict(Xt)
        yt = test.y_true.to_numpy()
        r = dict(method=m, **{k: bench["meta"][k] for k in ["seed", "noise", "noise_type", "verify_rate",
                                                             "realized_noise"]},
                 acc=float((pred == yt).mean()),
                 acc_seen=float((pred == yt)[test.split == "seen"].mean()),
                 acc_held=float((pred == yt)[test.split == "held"].mean()),
                 acc_policy=_acc_policy(pred, test),
                 macro_f1=float(f1_score(yt, pred, average="macro")),
                 sec=time.time() - t0)
        r.update(weight_diagnostics(obs, adapt.y_true.to_numpy(), labels, w, q))
        rows.append(r)
    # frozen = seed model only
    pred = seed_clf.predict(Xt)
    yt = test.y_true.to_numpy()
    rows.append(dict(method="seed_only", **{k: bench["meta"][k] for k in ["seed", "noise", "noise_type",
                                                                          "verify_rate", "realized_noise"]},
                     acc=float((pred == yt).mean()), acc_seen=float((pred == yt)[test.split == "seen"].mean()),
                     acc_held=float((pred == yt)[test.split == "held"].mean()),
                     acc_policy=_acc_policy(pred, test), macro_f1=float(f1_score(yt, pred, average="macro"))))
    return rows


def summarize(df, ref="source_prior"):
    g = df.groupby(["noise_type", "verify_rate", "noise", "method"])
    s = g.agg(acc=("acc", "mean"), acc_sd=("acc", "std"), acc_seen=("acc_seen", "mean"),
              acc_held=("acc_held", "mean"), acc_pol=("acc_policy", "mean"), f1=("macro_f1", "mean"),
              acc_noisy=("accepted_noisy_rate", "mean"), n=("acc", "size")).reset_index()
    # paired difference vs reference baseline (same seed)
    piv = df.pivot_table(index=["noise_type", "verify_rate", "noise", "seed"], columns="method", values="acc")
    diffs = []
    for m in piv.columns:
        if ref in piv.columns:
            d = (piv[m] - piv[ref]).groupby(level=[0, 1, 2])
            dd = d.agg(["mean", "std", "count"]).reset_index()
            dd["method"] = m
            diffs.append(dd)
    if diffs:
        dd = pd.concat(diffs)
        dd["d_vs_ref"] = dd["mean"]
        dd["d_ci95"] = 1.96 * dd["std"] / np.sqrt(dd["count"])
        s = s.merge(dd[["noise_type", "verify_rate", "noise", "method", "d_vs_ref", "d_ci95"]],
                    on=["noise_type", "verify_rate", "noise", "method"], how="left")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--noise-type", nargs="+", default=["context"])
    ap.add_argument("--verify-rate", type=float, nargs="+", default=[0.25])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--methods", nargs="+", default=METHODS + ["self_filter"] + REFERENCE)
    ap.add_argument("--out", default="results/cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, metas = [], []
    for nt in a.noise_type:
        for vr in a.verify_rate:
            for nz in a.noise:
                for s in range(a.seeds):
                    b = make_benchmark(seed=1000 + s, noise=nz, noise_type=nt, verify_rate=vr)
                    metas.append(b["meta"])
                    rows += run_one(b, a.methods, a.tau, a.gamma)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/raw_results.csv", index=False)
    s = summarize(df)
    s.to_csv(f"{a.out}/summary.csv", index=False)
    json.dump(dict(args=vars(a), metas=metas), open(f"{a.out}/run_meta.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
