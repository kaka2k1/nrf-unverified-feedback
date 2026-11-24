"""Summarize completed E1 or local QLoRA runs using paired seed bootstrap."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from q1_e4_cpu import NEW, REPRO, RHO_CURVE, RHOS
from scripts.analyze_breakeven import analyze
from scripts.q1_stats import bootstrap_means, compare
from nrf.reliability import CORRECTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["metatool", "qlora"])
    args = parser.parse_args()
    gpu = args.experiment == "qlora"
    out = Path("results/q1_gpu" if gpu else "results/q1_e1")
    if gpu:
        paths = sorted(out.glob("*/*/results.jsonl"))
        raw = pd.concat([pd.read_json(path, lines=True) for path in paths], ignore_index=True)
        raw = raw.rename(columns={"verify_rate": "rho"})
        groups, rhos, nseed = ["noise_type"], [0.02, 0.05], 5
        grid = [0., .02, .05, .1, .25, .5, 1.]
        expected = ["source_prior", "ctx_eb", "ca_eb"]
        for noise in ("context", "source"):
            data = raw[raw.noise_type == noise]
            for method in ["verified_only", *expected]:
                rates = grid if method == "verified_only" else rhos
                cells = data[data.method == method].pivot(index="seed", columns="rho", values="acc")
                if list(cells.index) != list(range(nseed)) or list(cells.columns) != rates or cells.isna().any().any():
                    raise ValueError(f"Incomplete QLoRA grid for {noise}/{method}")
        raw.to_csv(out / "raw_results.csv", index=False)
    else:
        raw = pd.read_csv(out / "raw_results.csv")
        groups, rhos, nseed = ["learner", "view"], RHOS, 20
        if set(raw.learner) != {"bge_small"} or set(raw.view) != {"names"}:
            raise ValueError("Expected the bge-small MetaTool learner with tool names")
        for method in [*REPRO, *NEW]:
            if method in CORRECTION:
                continue
            rates = RHO_CURVE if method == "verified_only" else rhos
            cells = raw[raw.method == method].pivot(index="seed", columns="rho", values="acc")
            if list(cells.index) != list(range(nseed)) or list(cells.columns) != rates or cells.isna().any().any():
                raise ValueError(f"Incomplete MetaTool grid for {method}")

    rng = np.random.default_rng(0)
    accuracy = []
    for key, data in raw[raw.rho.isin(rhos)].groupby(groups + ["rho"]):
        pivot = data.pivot(index="seed", columns="method", values="acc")
        if list(pivot.index) != list(range(nseed)) or pivot.isna().any().any():
            raise ValueError(f"Incomplete seed grid for {key}")
        boot = bootstrap_means(pivot.to_numpy(), rng)
        lo, hi = np.quantile(boot, [.025, .975], axis=0)
        for method, mean, low, high in zip(pivot.columns, pivot.mean(), lo, hi):
            accuracy.append(dict(zip(groups + ["rho"], key), method=method,
                                 accuracy=f"{mean:.2%} [{low:.2%}, {high:.2%}]"))
    rates = analyze(raw, groups, N=640 if gpu else 631, nboot=2000)
    rates.to_csv(out / "exchange_rates.csv", index=False)
    rates["exchange rate"] = [
        f"{row.rate:.2f} [{row.rate_lo:.2f}, {row.rate_hi:.2f}]" for row in rates.itertuples()
    ]
    columns = groups + ["rho", "method"]
    results = pd.DataFrame(accuracy).merge(rates[columns + ["exchange rate"]], on=columns, how="left")
    compare(raw[raw.rho.isin(rhos)], groups).to_csv(out / "comparisons.csv", index=False)
    (out / "summary.md").write_text(
        f"# {'Qwen3-0.6B QLoRA' if gpu else 'E1 MetaTool'}\n\n"
        f"Seeds 0–{nseed - 1}. Accuracy and exchange rates have pointwise 95% seed-bootstrap intervals. "
        "Exchange rates invert the monotone verified-only curve. Infinite values exceed the reference curve. "
        "See comparisons.csv for paired differences, effect sizes, and Holm-adjusted p-values.\n\n"
        + results.fillna("reference").to_markdown(index=False) + "\n"
    )


if __name__ == "__main__":
    main()
