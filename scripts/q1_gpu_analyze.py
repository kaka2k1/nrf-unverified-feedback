"""Q1 GPU exchange-rate analysis (Qwen3-0.6B QLoRA, synthetic benchmark, 40% noise, seeds 0-4).

Reads ONLY results/q1_gpu/{context,source}/{rho}/results.jsonl (the oom_partial folder is never read).
For each noise type and rho in {0.02, 0.05}:
  * accuracy mean with 95% paired-bootstrap CI over seeds (10,000 resamples), per method;
  * paired difference to verified_only (same seed), 95% bootstrap CI and an exact two-sided sign-flip p-value
    (with 5 seeds the smallest attainable two-sided p is 2/32 = 0.0625, so no cell can reach p < 0.05);
  * exchange rate rho_eq/rho, where rho_eq = a_V^{-1}(a_M) inverts the verified_only curve over the rho grid
    {0, .02, .05, .1, .25, .5, 1} (inversion, running max and bootstrap CI are imported from
    scripts/analyze_breakeven.py; the CI resamples seeds jointly for the curve and the method).
The verified_only curve is checked for monotonicity; where the raw seed mean decreases, the running max used by
the inversion is reported explicitly.
Embedding-learner exchange rates (E5, 20 seeds) are read from results/q1_e5/synth_exchange_rates.csv for the
comparison "stronger learner -> lower exchange rate".

python scripts/q1_gpu_analyze.py  ->  results/q1_gpu/summary.csv, results/q1_gpu/curves.csv, results/q1_gpu/summary.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.analyze_breakeven import inv_curve  # noqa: E402
from scripts.q1_stats import bootstrap_means, exact_signflip_p  # noqa: E402

OUT = os.path.join(ROOT, "results", "q1_gpu")
NOISE_TYPES = ["context", "source"]
RHOS_ALL = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
RHOS_LOW = [0.02, 0.05]
METHODS = ["source_prior", "ctx_eb", "ca_eb"]
FOLDER = {0.0: "0", 0.02: ".02", 0.05: ".05", 0.1: ".1", 0.25: ".25", 0.5: ".5", 1.0: "1"}
N_BOOT = 10_000
E5_PATH = os.path.join(ROOT, "results", "q1_e5", "synth_exchange_rates.csv")


def load_runs():
    rows = []
    for nt in NOISE_TYPES:
        for rho in RHOS_ALL:
            path = os.path.join(OUT, nt, FOLDER[rho], "results.jsonl")
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    rows.append(dict(noise_type=nt, rho=float(r["verify_rate"]), seed=int(r["seed"]),
                                     method=r["method"], acc=float(r["acc"]), path=path))
    df = pd.DataFrame(rows)
    # every file must carry the rho of its folder
    assert set(df.rho.unique()) == set(RHOS_ALL), sorted(df.rho.unique())
    return df


def curve_table(df):
    """verified_only curve per noise type: raw seed means, running max (used by the inversion), monotone flag."""
    out = []
    for nt in NOISE_TYPES:
        g = df[(df.noise_type == nt) & (df.method == "verified_only")].groupby("rho").acc
        mean = g.mean().reindex(RHOS_ALL)
        runmax = np.maximum.accumulate(mean.to_numpy())
        for rho, m, rm in zip(RHOS_ALL, mean.to_numpy(), runmax):
            out.append(dict(noise_type=nt, rho=rho, n=int(g.count().get(rho, 0)), acc_raw=m, acc_runmax=rm,
                            is_dip=bool(rm > m + 1e-12)))
    return pd.DataFrame(out)


def paired_table(df, B=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for nt in NOISE_TYPES:
        vo_all = df[(df.noise_type == nt) & (df.method == "verified_only")].pivot_table(
            index="seed", columns="rho", values="acc")
        rhos = vo_all.columns.to_numpy(float)
        seeds = vo_all.index.to_numpy()
        vo_mean = vo_all.mean().to_numpy()
        # seed-only reference (not part of the exchange-rate curve)
        for rho in RHOS_LOW:
            vo = vo_all[rho].to_numpy()
            so_r = df[(df.noise_type == nt) & (df.method == "seed_only") & (df.rho == rho)].set_index("seed").acc
            rows.append(_row(nt, rho, "seed_only", None, so_r.reindex(seeds).to_numpy(), vo, None, None, rng, B))
            for m in ["verified_only"] + METHODS:
                acc = df[(df.noise_type == nt) & (df.method == m) & (df.rho == rho)].set_index("seed").acc
                acc = acc.reindex(seeds).to_numpy()
                rows.append(_row(nt, rho, m, rhos, acc, vo, vo_all, vo_mean, rng, B))
    return pd.DataFrame(rows)


def _row(nt, rho, m, rhos, acc, vo, vo_all, vo_mean, rng, B):
    d = acc - vo
    r = dict(noise_type=nt, rho=rho, method=m, n=len(acc), acc=100 * acc.mean(),
             acc_lo=100 * np.quantile(bootstrap_means(acc, rng, B), .025),
             acc_hi=100 * np.quantile(bootstrap_means(acc, rng, B), .975))
    if m == "verified_only":
        r.update(diff=0.0, diff_lo=np.nan, diff_hi=np.nan, p_exact=np.nan, rate=np.nan, rate_lo=np.nan,
                 rate_hi=np.nan, rate_reliable=np.nan)
        return r
    bd = bootstrap_means(d, rng, B) * 100
    r.update(diff=100 * d.mean(), diff_lo=np.quantile(bd, .025), diff_hi=np.quantile(bd, .975),
             p_exact=float(exact_signflip_p(d.reshape(-1, 1))[0]))
    if m == "seed_only":
        r.update(rate=np.nan, rate_lo=np.nan, rate_hi=np.nan, rate_reliable=np.nan)
        return r
    if m == "verified_only":
        return r
    cur = vo_all.to_numpy()  # seeds x rhos
    if acc.mean() <= vo_mean[0]:
        # at or below the verified_only point at rho=0: the inversion would clip to 0, which is not an exchange rate
        r.update(rate=np.nan, rate_lo=np.nan, rate_hi=np.nan, rate_reliable=False, status="below_rho0")
        return r
    r_eq = inv_curve(rhos, vo_mean, acc.mean())
    boots = np.empty(B)
    n = len(acc)
    ix = rng.integers(n, size=(B, n))
    curves = cur[ix].mean(axis=1)  # B x rhos
    accs = acc[ix].mean(axis=1)
    for b in range(B):
        boots[b] = inv_curve(rhos, curves[b], accs[b]) / rho
    rate = r_eq / rho
    r.update(rate=rate, rate_lo=np.quantile(boots, .025, method="inverted_cdf"),
             rate_hi=np.quantile(boots, .975, method="inverted_cdf"),
             rate_reliable=bool(np.isfinite(rate)), status="ok" if np.isfinite(rate) else "above_max")
    return r


def e5_table():
    """Embedding learners (20 seeds), same noise levels, from the E5 exchange-rate file."""
    e = pd.read_csv(E5_PATH)
    e = e[(e.noise == 0.4) & e.rho.isin(RHOS_LOW) & e.method.isin(METHODS)]
    rows = []
    for (lrn, nt, rho, m), g in e.groupby(["learner", "noise_type", "rho", "method"]):
        rows.append(dict(learner=lrn, noise_type=nt, rho=rho, method=m, acc=100 * g.acc.iloc[0],
                         acc_vo=100 * g.acc_vo.iloc[0], rate=g.rate.iloc[0], rate_lo=g.rate_lo.iloc[0],
                         rate_hi=g.rate_hi.iloc[0], d_vo=100 * g.d_vo.iloc[0], d_ci=100 * g.d_ci.iloc[0]))
    return pd.DataFrame(rows)


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if np.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def rate_str(r):
    if r.method in ("verified_only", "seed_only"):
        return "--"
    if r.status == "below_rho0":
        return "not identified: acc <= verified_only at rho=0"
    if pd.isna(r.rate):
        return "--"
    if not np.isfinite(r.rate):
        return f">{1 / r.rho:.0f} (above curve max)"
    return f"{r.rate:.2f} [{r.rate_lo:.2f}, {r.rate_hi:.2f}]"


def main():
    df = load_runs()
    curves = curve_table(df)
    res = paired_table(df)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    curves.to_csv(os.path.join(OUT, "curves.csv"), index=False)

    L = ["# Q1 GPU exchange-rate analysis (Qwen3-0.6B QLoRA)", "",
         "Generated by `scripts/q1_gpu_analyze.py` from `results/q1_gpu/{context,source}/{rho}/results.jsonl` only "
         "(`source/0.oom_partial` is not read). Learner: Qwen3-0.6B, QLoRA 4-bit, fp16, RTX 3050 (local). "
         "Synthetic benchmark, feedback noise 40%, **5 seeds (0-4)**.", "",
         "**Read the intervals with care:** with 5 paired seeds the bootstrap CIs are wide and the exact sign-flip "
         "test cannot go below p = 0.0625, so no single cell is significant at 0.05. The results are a sparse-budget "
         "pilot, not a confirmatory test.", "",
         "Accuracy = mean over seeds (%), CI = seed bootstrap (10,000). Δ = paired difference to verified_only "
         "(points), CI = paired bootstrap; p = exact two-sided sign-flip (5 seeds). Exchange rate ρ_eq/ρ: the "
         "verified_only curve is inverted (running max), CI = joint seed bootstrap.", ""]
    for nt in NOISE_TYPES:
        L += [f"## {nt} noise", "", "verified_only curve (mean over seeds, ρ: acc):", ""]
        c = curves[curves.noise_type == nt]
        L.append(", ".join(f"ρ={r.rho:g}: {100 * r.acc_raw:.1f}" + (" (dip)" if r.is_dip else "")
                           for r in c.itertuples()))
        dips = c[c.is_dip]
        if len(dips):
            L += ["", "Non-monotone: " + "; ".join(
                f"ρ={r.rho:g} raw {100 * r.acc_raw:.1f} < running max {100 * r.acc_runmax:.1f}" for r in dips.itertuples())
                  + ". The inversion uses the running max, i.e. a dip is replaced by the best accuracy at a smaller ρ "
                    "(conservative for the exchange rate of methods that match that plateau)."]
        else:
            L += ["", "Monotone (no running-max adjustment needed on the mean curve)."]
        L += ["", "| ρ | method | n | acc % [95% CI] | Δ vs verified [95% CI] | p (exact) | exchange rate [95% CI] |",
              "|---|---|---|---|---|---|---|"]
        for r in res[res.noise_type == nt].itertuples():
            delta = "--" if r.method in ("verified_only", "seed_only") else \
                f"{r.diff:+.1f} [{r.diff_lo:+.1f}, {r.diff_hi:+.1f}]"
            p = "--" if r.method in ("verified_only", "seed_only") else f"{r.p_exact:.3f}"
            name = {"verified_only": "verified_only (ref.)", "seed_only": "seed_only (ref.)"}.get(r.method, r.method)
            L.append(f"| {r.rho:g} | {name} | {r.n} | {r.acc:.1f} [{r.acc_lo:.1f}, {r.acc_hi:.1f}] | {delta} | {p} | "
                     f"{rate_str(r)} |")
        L.append("")

    e5 = e5_table()
    mono = []
    for (nt, rho, m), g in e5.groupby(["noise_type", "rho", "method"]):
        g = g.sort_values("acc_vo")  # weakest -> strongest learner
        rates = g.rate.to_numpy()
        mono.append(dict(noise_type=nt, rho=rho, method=m, learners=" < ".join(g.learner),
                         rates=" , ".join(f"{x:.2f}" for x in rates),
                         monotone_decreasing=bool(np.all(np.diff(rates) < 0))))
    mono = pd.DataFrame(mono)
    mono.to_csv(os.path.join(OUT, "e5_monotonicity.csv"), index=False)
    e5.to_csv(os.path.join(OUT, "e5_embedding_compare.csv"), index=False)
    L += ["## Comparison with embedding learners (E5, 20 seeds, noise 40%)", "",
          "Learner strength is the verified_only accuracy at the same ρ (higher = stronger). E5 rates use the same "
          "inversion on a 20-seed curve (results/q1_e5/synth_exchange_rates.csv). Different learners and different "
          "seed counts; the comparison is directional.", "",
          "| noise | ρ | learner | verified acc % | method | acc % | rate [95% CI] | Δ vs verified |",
          "|---|---|---|---|---|---|---|---|"]
    for r in e5.sort_values(["noise_type", "rho", "acc_vo"]).itertuples():
        L.append(f"| {r.noise_type} | {r.rho:g} | {r.learner} | {r.acc_vo:.1f} | {r.method} | {r.acc:.1f} | "
                 f"{r.rate:.2f} [{r.rate_lo:.2f}, {r.rate_hi:.2f}] | {r.d_vo:+.1f} ±{r.d_ci:.1f} |")
    L += ["", "Monotonicity check (rate must fall as learner strength rises; learners ordered by verified acc):", "",
          "| noise | ρ | method | learners (weak < strong) | rates | strictly decreasing |", "|---|---|---|---|---|---|"]
    for r in mono.itertuples():
        L.append(f"| {r.noise_type} | {r.rho:g} | {r.method} | {r.learners} | {r.rates} | "
                 f"{'yes' if r.monotone_decreasing else 'no'} |")
    L.append("")
    # Qwen3-0.6B row for the same comparison
    vo_acc = {(r.noise_type, r.rho): r.acc for r in res.itertuples() if r.method == "verified_only"}
    L += ["Qwen3-0.6B verified-only acc % at ρ=0.02 / 0.05: " + "; ".join(
        f"{nt}: " + " / ".join(f"{vo_acc[(nt, rr)]:.1f}" for rr in RHOS_LOW) for nt in NOISE_TYPES) + ".", ""]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
