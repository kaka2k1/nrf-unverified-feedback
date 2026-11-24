"""PLAN_Q1 A1: which items should the verifier check? Ablation on the break-even testbed (same seeds as breakeven_cpu).

Budget B = round(rho * N_adapt) is fixed; only the CHOICE of verified items changes.
  random      uniform sample (default in every other experiment)
  suspicious  the B items whose feedback the seed model finds least plausible (lowest p_seed(y_fb | x)); active label
              cleaning style
  strat_ctx   stratified by context (API family / domain): proportional allocation per domain, random within domain
  strat_cell  stratified by (source, domain) cell: round-robin over cells, random within a cell

Reliability-estimate bias (uses hidden truth, REPORTING ONLY, after the estimate is formed):
  bias_global  mean_q(unverified) - true accuracy of the unverified feedback
  mae_cell     mean over (source, domain) cells of |mean q in cell - true feedback accuracy in cell| (unverified items)
  ver_acc_gap  feedback accuracy on the verified sample - on the whole pool (selection bias of the sample itself)
for q from source_prior and ctx_eb; plus downstream test accuracy of verified_only / source_prior / ctx_eb / ca_eb.

python q1_a1_cpu.py --learners tfidf bge_small --seeds 20 --out results/q1_a1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from breakeven_cpu import EMB, embed_texts, featurize, fit, proba4, stack
from nrf import make_benchmark, method_weights, observable_view

STRATS = ["random", "suspicious", "strat_ctx", "strat_cell"]
CONDS = [("context", 0.4), ("source", 0.4)]
RHOS = [0.05, 0.1, 0.25]
METHODS = ["verified_only", "source_prior", "ctx_eb", "ca_eb"]


def choose(strategy, adapt, p_seed_fb, budget, rng):
    n = len(adapt)
    ver = np.zeros(n, bool)
    if strategy == "random":
        ver[rng.choice(n, budget, replace=False)] = True
    elif strategy == "suspicious":
        ver[np.argsort(p_seed_fb + 1e-9 * rng.random(n), kind="stable")[:budget]] = True
    elif strategy == "strat_ctx":
        dom = adapt.domain.to_numpy()
        doms, cnt = np.unique(dom, return_counts=True)
        alloc = np.floor(budget * cnt / n).astype(int)
        rem = budget - alloc.sum()
        frac = budget * cnt / n - alloc
        alloc[np.argsort(-frac + 1e-9 * rng.random(len(doms)))[:rem]] += 1
        for d, k in zip(doms, alloc):
            idx = np.flatnonzero(dom == d)
            ver[rng.choice(idx, min(k, len(idx)), replace=False)] = True
    elif strategy == "strat_cell":
        cells = (adapt.source.astype(str) + "|" + adapt.domain.astype(str)).to_numpy()
        order = {c: list(rng.permutation(np.flatnonzero(cells == c))) for c in np.unique(cells)}
        keys = list(order)
        while ver.sum() < budget:
            for c in rng.permutation(keys):
                if order[c] and ver.sum() < budget:
                    ver[order[c].pop()] = True
    else:
        raise ValueError(strategy)
    return ver


def job(args):
    kind, nt, nz, s = args
    rows = []
    for rho in RHOS:
        b = make_benchmark(seed=1000 + s, noise=nz, noise_type=nt, verify_rate=rho)
        seed_df, adapt, test = b["seed"], b["adapt"], b["test"]
        Xs, Xa, Xt = featurize(kind, seed_df, adapt, test)
        ys, yt = seed_df.y_true.to_numpy(), test.y_true.to_numpy()
        p_seed = proba4(fit(kind, Xs, ys), Xa)
        yfb = adapt.y_fb.to_numpy()
        p_fb = p_seed[np.arange(len(adapt)), yfb]
        ok_fb = yfb == adapt.y_true.to_numpy()
        cells = (adapt.source.astype(str) + "|" + adapt.domain.astype(str)).to_numpy()
        budget = int(round(rho * len(adapt)))
        for st in STRATS:
            rng = np.random.default_rng(77_000 + 97 * s + int(rho * 1000))
            ver = choose(st, adapt, p_fb, budget, rng)
            a2 = adapt.copy()
            a2["verified"] = ver
            obs = observable_view(a2)
            unv = ~ver
            base = dict(learner=kind, noise_type=nt, noise=nz, seed=s, rho=rho, strategy=st, n_ver=int(ver.sum()),
                        ver_acc_gap=100 * (ok_fb[ver].mean() - ok_fb.mean()))
            for m in METHODS:
                lab, w, q = method_weights(m, obs, p_seed, tau=0.5, gamma=2.0)
                keep = w > 0
                clf = fit(kind, stack(Xs, Xa[keep]), np.r_[ys, lab[keep]], np.r_[np.ones(len(ys)), w[keep]])
                r = dict(base, method=m, acc=100 * float((clf.predict(Xt) == yt).mean()))
                if q is not None and m in ("source_prior", "ctx_eb"):
                    qu, ou, cu = q[unv], ok_fb[unv], cells[unv]
                    r["bias_global"] = 100 * (qu.mean() - ou.mean())
                    g = pd.DataFrame(dict(q=qu, o=ou, c=cu)).groupby("c").mean()
                    r["mae_cell"] = 100 * float((g.q - g.o).abs().mean())
                    r["bias_cell_mean"] = 100 * float((g.q - g.o).mean())
                rows.append(r)
    return rows


def _init(emb):
    EMB.update(emb)
    os.environ["OMP_NUM_THREADS"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="+", default=["tfidf", "bge_small"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache", default="results/q1_e4/cache")
    ap.add_argument("--out", default="results/q1_a1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    texts = set()
    for s in range(a.seeds):
        b = make_benchmark(seed=1000 + s, noise=0.4, noise_type="context", verify_rate=0.1)
        texts |= set(b["seed"].text) | set(b["adapt"].text) | set(b["test"].text)
    emb = {k: embed_texts(sorted(texts), k, a.cache) for k in a.learners if k != "tfidf"}
    jobs = [(k, nt, nz, s) for k in a.learners for nt, nz in CONDS for s in range(a.seeds)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init, initargs=(emb,)) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            rows += r
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/raw_results.csv", index=False)
    json.dump(dict(args=vars(a), strategies=STRATS, conds=CONDS, rhos=RHOS, methods=METHODS, sec=time.time() - t0),
              open(f"{a.out}/run_meta.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
