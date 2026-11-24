"""PLAN_Q1 E5: statistics for the CPU break-even testbeds (numbers come only from raw_results.csv files).

Uses the self-contained E4 runs (results/q1_e4/{synth,bfcl}: every 18/11 method + verified_only curve + new baselines,
one run, so all paired comparisons share the exact same learner fits); the 18/11 runs (results/cpu_breakeven,
results/cpu_bfcl_breakeven) are only used for a reproduction check of overlapping (method, seed, rho) rows.

For every cell (testbed, learner, noise condition, rho) and every pair (proposed P in {ctx_eb, ca_eb}, comparator B):
  d_s = acc_P(seed s) - acc_B(seed s)                    paired over the 20 seeds (same data, same verify mask)
  mean, 95% percentile bootstrap CI over seeds (10,000 resamples)
  p   = EXACT two-sided paired sign-flip permutation test (all 2^n sign patterns)
  d_z = mean(d) / sd(d)  (standardised paired effect size), win = fraction of seeds with d > 0
  Holm step-down correction (a) over ALL comparisons of the testbed ("p_holm_all", primary, conservative)
                            (b) within the cell ("p_holm_cell", across comparators and both proposed methods)

python scripts/q1_stats.py      ->  results/q1_e5/{synth,bfcl}_comparisons.csv, results/q1_e5/summary.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_breakeven import analyze  # noqa: E402

PROPOSED = ["ctx_eb", "ca_eb"]
COMPARATORS = ["verified_only", "source_prior", "naive", "verifier_corrected", "conf_filter", "self_train",
               "ds_source", "crowd_seed", "ctx_crowd", "mace", "rgpo_style", "small_loss", "coteach"]
OUT = "results/q1_e5"


def merge(old_path, new_path, keys):
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    ov = old.merge(new, on=keys + ["method", "seed", "rho"], suffixes=("_old", "_new"))
    diff = (ov.acc_old - ov.acc_new).abs()
    rep = dict(n_overlap=len(ov), n_mismatch=int((diff > 1e-9).sum()), max_abs_diff=float(diff.max()) if len(ov) else 0)
    rep["mean_diff_points"] = float(100 * (ov.acc_new - ov.acc_old).mean()) if len(ov) else 0
    return new, rep  # all analyses use the single self-contained E4 run; the 18/11 run is only a reproduction check


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def signs(n):
    """All 2^n sign patterns, as int8 [2^n, n] (n <= 22)."""
    idx = np.arange(2 ** n, dtype=np.int64)[:, None]
    return (1 - 2 * ((idx >> np.arange(n)) & 1)).astype(np.int8)


def exact_signflip_p(D):
    """D: [n_seeds, n_comp] paired differences -> two-sided exact sign-flip p-values."""
    n, k = D.shape
    S = signs(n)
    obs = np.abs(D.sum(0))
    cnt = np.zeros(k)
    Df = D.astype(np.float64)
    for i in range(0, len(S), 1 << 15):
        T = np.abs(S[i:i + (1 << 15)].astype(np.float64) @ Df)
        cnt += (T >= obs[None, :] - 1e-12).sum(0)
    return cnt / len(S)


def bootstrap_means(values, rng, replicates=10_000):
    """Resample the seed axis jointly for all columns."""
    n = len(values)
    indices = rng.integers(n, size=(replicates, n))
    return values[indices].mean(axis=1)


def compare(df, cell_cols, B=10_000):
    rng = np.random.default_rng(0)
    rows, Ds = [], []
    for key, g in df.groupby(cell_cols + ["rho"]):
        piv = g.pivot_table(index="seed", columns="method", values="acc")
        for P in PROPOSED:
            if P not in piv:
                continue
            for C in COMPARATORS:
                if C not in piv:
                    continue
                d = (piv[P] - piv[C]).dropna().to_numpy() * 100
                n = len(d)
                bs = bootstrap_means(d, rng, B)
                sd = d.std(ddof=1)
                r = dict(zip(cell_cols + ["rho"], key))
                r.update(proposed=P, comparator=C, n=n, acc_P=100 * piv[P].mean(), acc_C=100 * piv[C].mean(),
                         diff=d.mean(), ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975),
                         d_z=d.mean() / sd if sd > 0 else np.nan, win=float((d > 0).mean()), cell="|".join(map(str, key)))
                rows.append(r)
                Ds.append(d)
    out = pd.DataFrame(rows)
    ns = out.n.unique()
    assert len(ns) == 1, ns
    out["p"] = exact_signflip_p(np.stack(Ds, 1))
    out["p_holm_all"] = holm(out.p)
    out["p_holm_cell"] = 0.0
    for c, g in out.groupby("cell"):
        out.loc[g.index, "p_holm_cell"] = holm(g.p)
    return out.drop(columns="cell")


def star(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def fmt(r):
    return f"{r['diff']:+.1f} [{r.ci_lo:+.1f},{r.ci_hi:+.1f}]{star(r.p_holm_all)}"


def md_tables(res, cell_cols, title):
    lines = [f"## {title}", "",
             "Cell = mean paired difference (proposed − comparator, accuracy points) [95% bootstrap CI over seeds]; "
             "`*`/`**`: Holm-adjusted p < 0.05 / 0.01 over ALL comparisons of this testbed (exact sign-flip test).", ""]
    for P in PROPOSED:
        r = res[res.proposed == P]
        if r.empty:
            continue
        comps = [c for c in COMPARATORS if c in set(r.comparator)]
        lines += [f"### {P} vs ...", "", "| " + " | ".join(cell_cols + ["rho", f"acc {P}"] + comps) + " |",
                  "|" + "---|" * (len(cell_cols) + 2 + len(comps))]
        for key, g in r.groupby(cell_cols + ["rho"]):
            g = g.set_index("comparator")
            cells = [fmt(g.loc[c]) if c in g.index else "" for c in comps]
            lines.append("| " + " | ".join([str(k) for k in key] + [f"{g.acc_P.iloc[0]:.1f}"] + cells) + " |")
        lines.append("")
    return lines


KEY = [("ctx_eb", "verified_only"), ("ca_eb", "verified_only"), ("ctx_eb", "source_prior"), ("ca_eb", "source_prior"),
       ("ctx_eb", "ds_source"), ("ctx_eb", "rgpo_style"), ("ctx_eb", "small_loss"), ("ctx_eb", "coteach"),
       ("ca_eb", "coteach")]


def key_tables():
    """Compact table (written to results/q1_e5/key_comparisons.md) from the *_comparisons.csv files."""
    lines = ["# Key comparisons (E5)", "", "diff = proposed − comparator in accuracy points, mean over 20 paired seeds "
             "[95% bootstrap CI]; p_H = Holm-adjusted exact sign-flip p over ALL comparisons of the testbed "
             "(synthetic: 1248, BFCL: 80); d_z = paired standardised effect size. Generated by scripts/q1_stats.py.", ""]
    for name, cols in (("synth", ["learner", "noise_type", "noise"]), ("bfcl", ["learner", "view"])):
        path = f"{OUT}/{name}_comparisons.csv"
        if not os.path.exists(path):
            continue
        c = pd.read_csv(path)
        lines += [f"## {name}", ""]
        for P, C in KEY:
            g = c[(c.proposed == P) & (c.comparator == C)]
            if g.empty:
                continue
            lines += [f"### {P} vs {C}", "", "| " + " | ".join(cols) + " | rho | acc P | acc C | diff [95% CI] | d_z | p_H |",
                      "|" + "---|" * (len(cols) + 6)]
            for _, r in g.iterrows():
                lines.append("| " + " | ".join(str(r[k]) for k in cols) + f" | {r.rho:g} | {r.acc_P:.1f} | {r.acc_C:.1f} | "
                             f"{r['diff']:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] | {r.d_z:+.2f} | {r.p_holm_all:.3g} |")
            lines.append("")
    open(f"{OUT}/key_comparisons.md", "w").write("\n".join(lines))


def main():
    if "--key-only" in sys.argv:
        key_tables()
        return
    os.makedirs(OUT, exist_ok=True)
    md = ["# Q1 E4/E5 — baselines + statistics on the CPU break-even testbeds",
          "", "Generated by `scripts/q1_stats.py` from raw_results.csv files only. 20 seeds, paired by seed.", ""]
    setups = [("synth", "results/cpu_breakeven/raw_results.csv", "results/q1_e4/synth/raw_results.csv",
               ["learner", "noise_type", "noise"]),
              ("bfcl", "results/cpu_bfcl_breakeven/raw_results.csv", "results/q1_e4/bfcl/raw_results.csv",
               ["learner", "view"])]
    for name, old_p, new_p, cols in setups:
        if not os.path.exists(new_p):
            print("missing", new_p)
            continue
        df, rep = merge(old_p, new_p, cols)
        df.to_csv(f"{OUT}/{name}_merged_raw.csv", index=False)
        md += [f"Reproduction check `{name}` (rows present in both the 18/11 run and the E4 re-run): {rep}", ""]
        res = compare(df[df.rho.isin([0.02, 0.05, 0.1, 0.25])], cols)
        res.to_csv(f"{OUT}/{name}_comparisons.csv", index=False)
        md += md_tables(res, cols, f"{name}: paired comparisons")
        # mean accuracy table of every method (for the record)
        acc = df[df.rho.isin([0.02, 0.05, 0.1, 0.25])].groupby(cols + ["rho", "method"]).acc.mean().unstack() * 100
        md += [f"### {name}: mean accuracy (%) per method", "", acc.round(1).to_markdown(), ""]
        # exchange rates for all methods (verified_only curve from the 18/11 run)
        N = 640 if name == "synth" else int(df[df.method == "verified_only"].n_ver.max())
        ex = analyze(df, cols, N=N)
        ex.to_csv(f"{OUT}/{name}_exchange_rates.csv", index=False)
    open(f"{OUT}/summary.md", "w").write("\n".join(md))
    key_tables()
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
