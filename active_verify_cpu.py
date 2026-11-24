"""Which interactions should the verifier check?  CPU study on the synthetic benchmark.

The verification budget B = round(rho * N_adapt) is held fixed; only the *choice* of verified items changes.
Strategies
  random      uniform sample (what every other experiment uses)
  balanced    round-robin over (source, domain) cells, random within a cell
  cell_active half the budget random; the rest allocated greedily to the (source, domain) cells whose reliability is
              most uncertain, weighted by how many unverified items the cell would re-weight (random within a cell,
              so per-cell accuracy estimates stay unbiased)
  disagree    verify the items whose feedback the seed model finds least plausible (active label cleaning style);
              this biases the verified sample towards wrong feedback, so reliability estimates are biased low

python active_verify_cpu.py --seeds 8 --out results/cpu_active
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from nrf import make_benchmark, observable_view
from nrf.reliability import eb_context_reliability
from pilot_cpu import featurize, fit_lr, run_one

STRATEGIES = ["random", "balanced", "cell_active", "disagree"]


def _cells(adapt):
    return (adapt.source.astype(str) + "|" + adapt.domain.astype(str)).to_numpy()


def choose_verified(bench, strategy, rho, rng):
    adapt = bench["adapt"]
    n = len(adapt)
    budget = int(round(rho * n))
    ver = np.zeros(n, bool)
    if strategy == "random":
        ver[rng.choice(n, budget, replace=False)] = True
        return ver
    cells = _cells(adapt)
    order = {c: list(rng.permutation(np.flatnonzero(cells == c))) for c in np.unique(cells)}
    if strategy == "balanced":
        keys = list(order)
        while ver.sum() < budget:
            for c in rng.permutation(keys):
                if order[c] and ver.sum() < budget:
                    ver[order[c].pop()] = True
        return ver
    if strategy == "disagree":
        seed_df = bench["seed"]
        vec = featurize(pd.concat([seed_df.text, adapt.text]))
        clf = fit_lr(vec.transform(seed_df.text), seed_df.y_true)
        proba = np.zeros((n, 4))
        proba[:, clf.classes_] = clf.predict_proba(vec.transform(adapt.text))
        p_fb = proba[np.arange(n), adapt.y_fb.to_numpy()]
        ver[np.argsort(p_fb + 1e-9 * rng.random(n))[:budget]] = True
        return ver
    if strategy == "cell_active":
        first = rng.choice(n, budget // 2, replace=False)
        ver[first] = True
        for c in order:
            order[c] = [i for i in order[c] if not ver[i]]
        tmp = adapt.copy()
        tmp["verified"] = ver
        q, info = eb_context_reliability(observable_view(tmp))
        obs_ok = (adapt.y_fb.to_numpy() == adapt.y_true.to_numpy()) & ver  # only verified rows are used
        stats = {}
        for c in order:
            idx = np.flatnonzero(cells == c)
            prior = float(np.mean(q[idx]))  # LOO-shrunk estimate ~ cell posterior mean
            m = float(min(info["m_ctx"], 64.0))  # cap so that a large m does not freeze all cells
            a = prior * m + obs_ok[idx].sum()
            b = (1 - prior) * m + (ver[idx].sum() - obs_ok[idx].sum())
            stats[c] = [a, b, len(order[c])]
        while ver.sum() < budget:
            best, val = None, -1.0
            for c, (a, b, u) in stats.items():
                if u == 0:
                    continue
                s = a + b
                var = a * b / (s * s * (s + 1))
                gain = u * var / (s + 1)  # expected variance reduction of one more label, times items affected
                if gain > val:
                    best, val = c, gain
            if best is None:
                break
            i = order[best].pop()
            ver[i] = True
            a, b, u = stats[best]
            p = a / (a + b)
            stats[best] = [a + p, b + (1 - p), u - 1]  # expected update (labels are not peeked)
        return ver
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--noise-type", nargs="+", default=["source", "context"])
    ap.add_argument("--verify-rate", type=float, nargs="+", default=[0.1, 0.25])
    ap.add_argument("--methods", nargs="+", default=["verified_only", "source_prior", "ctx_eb", "ca_eb", "oracle_clean"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--out", default="results/cpu_active")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for nt in a.noise_type:
        for vr in a.verify_rate:
            for s in range(a.seeds):
                for st in a.strategies:
                    b = make_benchmark(seed=1000 + s, noise=a.noise, noise_type=nt, verify_rate=vr)
                    rng = np.random.default_rng(77_000 + s)
                    ver = choose_verified(b, st, vr, rng)
                    b["adapt"]["verified"] = ver
                    b["meta"]["verified_frac"] = float(ver.mean())
                    ad = b["adapt"]
                    frac_wrong_ver = float((ad.y_fb != ad.y_true)[ver].mean())
                    for r in run_one(b, a.methods, 0.5, 2.0):
                        r.update(strategy=st, frac_wrong_verified=frac_wrong_ver)
                        rows.append(r)
                print(nt, vr, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/raw_results.csv", index=False)
    json.dump(vars(a), open(f"{a.out}/run_meta.json", "w"), indent=1)
    # paired difference of each strategy vs random, per method
    piv = df.pivot_table(index=["noise_type", "verify_rate", "method", "seed"], columns="strategy", values="acc")
    out = []
    for key, g in piv.groupby(level=[0, 1, 2]):
        for st in a.strategies:
            d = (g[st] - g["random"]) * 100
            out.append(dict(noise_type=key[0], verify_rate=key[1], method=key[2], strategy=st,
                            acc=100 * g[st].mean(), d_vs_random=d.mean(),
                            ci=1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    s = pd.DataFrame(out)
    s.to_csv(f"{a.out}/summary.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
