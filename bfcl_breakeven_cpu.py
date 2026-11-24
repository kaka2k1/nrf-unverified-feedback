"""Break-even analysis on REAL LLM feedback (BFCL function selection), CPU learner.

Same split protocol as real_feedback.py (item-level split; one random source's choice per adapt item; verify mask
drawn last, hence nested across verify rates for a fixed seed). Learner: frozen bge sentence embeddings, pairwise
logistic scorer on phi(q, c) = [e_q * e_c, cos(e_q, e_c)] over (request, candidate) pairs; prediction = argmax over
the item's candidates; p(candidate | item) = softmax of the scores.

python bfcl_breakeven_cpu.py --emb bge_small --view names --seeds 20 --out results/cpu_bfcl_breakeven
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from breakeven_cpu import embed_texts
from nrf.reliability import method_weights

SOURCES = ["Qwen2.5-0.5B-Instruct", "SmolLM2-1.7B-Instruct", "SmolLM2-360M-Instruct", "Phi-3.5-mini-instruct",
           "Qwen2.5-1.5B-Instruct", "Qwen3-1.7B", "granite-3.1-2b-instruct"]
RHO_CURVE = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
RHO_METHODS = [0.02, 0.05, 0.1, 0.25]
METHODS = ["naive", "verifier_corrected", "conf_filter", "source_prior", "ctx_eb", "ca_eb", "self_train",
           "oracle_clean"]


def make_split(items, preds, seed, n_test, n_seed, verify_rate):
    """Copy of real_feedback.make_split (kept identical; importing real_feedback needs the GPU stack)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    te, sd, ad = perm[:n_test], perm[n_test:n_test + n_seed], perm[n_test + n_seed:]
    src = rng.integers(len(SOURCES), size=len(ad))
    y_fb = np.array([preds[SOURCES[s]][i] for s, i in zip(src, ad)])
    gold = np.array([items[i]["gold"] for i in ad])
    ver = rng.random(len(ad)) < verify_rate
    adapt = pd.DataFrame(dict(source=src, domain=[items[i]["context"] for i in ad], y_fb=y_fb, y_true=gold,
                              verified=ver, text=[items[i]["query"] for i in ad]))
    return te, sd, ad, adapt


class PairLearner:
    def __init__(self, items, E, view, C):
        self.C = C
        self.feat = []
        for it in items:
            q = E[it["query"]]
            cs = np.stack([E[self.ctext(c, view)] for c in it["candidates"]])
            self.feat.append(np.c_[cs * q, cs @ q])

    @staticmethod
    def ctext(c, view):
        return c["name"] if view == "names" else f"{c['name']}: {c['description']}"

    def fit(self, idx, labels, w):
        X, y, sw = [], [], []
        for i, l, ww in zip(idx, labels, w):
            if ww <= 0:
                continue
            F = self.feat[i]
            X.append(F)
            t = np.zeros(len(F))
            t[l] = 1
            y.append(t)
            sw.append(np.full(len(F), ww))
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(np.vstack(X), np.concatenate(y),
                                                                   sample_weight=np.concatenate(sw))
        return self

    def proba(self, idx):
        out = []
        for i in idx:
            s = self.clf.decision_function(self.feat[i])
            z = np.exp(s - s.max())
            out.append(z / z.sum())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bfcl_items.jsonl")
    ap.add_argument("--annot-dir", default="data/bfcl_annot")
    ap.add_argument("--emb", default="bge_small")
    ap.add_argument("--view", choices=["names", "full"], default="names")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-seed", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--out", default="results/cpu_bfcl_breakeven")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = [json.loads(line) for line in open(a.items)]
    ids = [it["id"] for it in items]
    preds = {}
    for s in SOURCES:
        d = {json.loads(line)["id"]: json.loads(line)["pred"] for line in open(f"{a.annot_dir}/annot_{s}.jsonl")}
        preds[s] = np.array([d[i] for i in ids])
    texts = {it["query"] for it in items} | {PairLearner.ctext(c, a.view) for it in items for c in it["candidates"]}
    E = embed_texts(sorted(texts), a.emb, a.out)
    L = PairLearner(items, E, a.view, a.C)
    gold_all = np.array([it["gold"] for it in items])
    rows = []
    t0 = time.time()
    for s in range(a.seeds):
        for rho in RHO_CURVE:
            te, sd, ad, adapt = make_split(items, preds, s, a.n_test, a.n_seed, rho)
            ver = adapt.verified.to_numpy()
            unv = ~ver
            obs = adapt.drop(columns=["y_true"]).copy()
            obs["y_ver"] = np.where(ver, adapt.y_true, -1)
            seed = PairLearner.fit(L, sd, gold_all[sd], np.ones(len(sd)))
            p_ad = seed.proba(ad)
            p_seed_fb = np.array([p[y] for p, y in zip(p_ad, adapt.y_fb)])

            def evaluate(m, lab, w):
                L.fit(np.r_[sd, ad], np.r_[gold_all[sd], lab], np.r_[np.ones(len(sd)), w])
                pred = np.array([int(np.argmax(p)) for p in L.proba(te)])
                wu, cu = w[unv], (lab == adapt.y_true.to_numpy())[unv]
                rows.append(dict(learner=a.emb, view=a.view, seed=s, rho=rho, method=m, n_ver=int(ver.sum()),
                                 acc=float((pred == gold_all[te]).mean()), w_unv=float(wu.sum()),
                                 prec_unv=float((wu * cu).sum() / max(wu.sum(), 1e-9)),
                                 realized_noise=float((adapt.y_fb != adapt.y_true).mean())))

            if rho == 0.0:
                pred = np.array([int(np.argmax(p)) for p in seed.proba(te)])
                rows.append(dict(learner=a.emb, view=a.view, seed=s, rho=0.0, method="seed_only", n_ver=0,
                                 acc=float((pred == gold_all[te]).mean())))
            meths = ["verified_only"] + (METHODS if rho in RHO_METHODS else [])
            for m in meths:
                if m == "self_train":  # model on seed + verified pseudo-labels the unverified requests
                    lab = np.where(ver, obs.y_ver, 0)
                    L.fit(np.r_[sd, ad[ver]], np.r_[gold_all[sd], lab[ver]], np.ones(len(sd) + ver.sum()))
                    pp = L.proba(ad)
                    q = np.array([p.max() for p in pp])
                    lab = np.where(ver, obs.y_ver, [int(np.argmax(p)) for p in pp])
                    w = np.where(ver, 1.0, np.where(q >= a.tau, q ** a.gamma, 0.0))
                else:
                    lab, w, _ = method_weights(m, obs, p_seed_fb, tau=a.tau, gamma=a.gamma,
                                               y_true_for_oracle=adapt.y_true.to_numpy() if m == "oracle_clean"
                                               else None)
                evaluate(m, np.asarray(lab), np.asarray(w, float))
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    json.dump(dict(args=vars(a), sources=SOURCES, rho_curve=RHO_CURVE, rho_methods=RHO_METHODS,
                   sec=time.time() - t0), open(f"{a.out}/run_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
