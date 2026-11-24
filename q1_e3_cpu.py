"""Synthetic online feedback with frozen bge-small and cumulative LR refits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from breakeven_cpu import RHO_CURVE, fit, proba4
from nrf import make_benchmark, method_weights, observable_view
from nrf.baselines import small_loss_select
from q1_e4_cpu import RHOS
from scripts.analyze_breakeven import inv_curve
from scripts.q1_stats import bootstrap_means

METHODS = ("verified_only", "source_prior", "ctx_eb", "ca_eb", "coteach", "small_loss")
CONDITIONS = ("context", "source")
ROUNDS = 4
KIND = "bge_small"


def trajectory(method, rho, seed, noise, adapt, Xs, ys, Xa, Xt, yt, seed_model):
    model = seed_model
    logs, confidences, rows = [], [], []
    rng = np.random.default_rng(8000 + seed)
    verify_draw, echo_draw = rng.random((2, len(adapt)))
    for rnd, batch in enumerate(np.array_split(np.arange(len(adapt)), ROUNDS), 1):
        current = adapt.iloc[batch].copy()
        p = proba4(model, Xa[batch])
        action = p.argmax(axis=1)
        # At source failures, an incorrect action can be echoed. Other failures
        # retain the benchmark's pre-drawn confusion. Gold stays in the simulator.
        gold = current.y_true.to_numpy()
        suggestion = current.y_fb.to_numpy()
        echo = (suggestion != gold) & (action != gold) & (echo_draw[batch] < 0.5)
        current["y_fb"] = np.where(echo, action, suggestion)
        current["verified"] = verify_draw[batch] < rho
        observed = observable_view(current)
        logs.append(observed)
        confidences.append(p[np.arange(len(batch)), observed.y_fb.to_numpy()])
        history = pd.concat(logs, ignore_index=True)

        def fit_score(tr, ytr, wtr, ev, yev):
            clf = fit(KIND, np.vstack([Xs, Xa[tr]]), np.r_[ys, ytr], np.r_[np.ones(len(ys)), wtr])
            return proba4(clf, Xa[ev])[np.arange(len(ev)), yev]

        if method in ("small_loss", "coteach"):
            labels, weights, _ = small_loss_select(
                history, fit_score, rounds=2 if method == "coteach" else 0, rng_seed=seed
            )
        else:
            labels, weights, _ = method_weights(method, history, np.concatenate(confidences))
        keep = np.flatnonzero(weights > 0)
        model = fit(KIND, np.vstack([Xs, Xa[keep]]), np.r_[ys, labels[keep]],
                    np.r_[np.ones(len(ys)), weights[keep]])
        rows.append(dict(seed=seed, noise_type=noise, rho=rho, method=method, round=rnd,
                         n_ver=int(history.verified.sum()), acc=float((model.predict(Xt) == yt).mean()),
                         action_acc=float((action == gold).mean()), echo_rate=float(echo.mean())))
    return rows


def summarize(raw, out):
    rng = np.random.default_rng(0)
    rows = []
    for (noise, rnd), cell in raw.groupby(["noise_type", "round"]):
        pivot = cell.pivot(index="seed", columns=["method", "rho"], values="acc")
        values = pivot.to_numpy()
        if np.isnan(values).any():
            raise ValueError("Incomplete paired seed grid")
        boot = bootstrap_means(values, rng)
        ref = pivot["verified_only"]
        grid = ref.columns.to_numpy()
        ref_boot = boot[:, [pivot.columns.get_loc(("verified_only", r)) for r in grid]]
        for j, (method, rho) in enumerate(pivot.columns):
            mean = values[:, j].mean()
            acc_lo, acc_hi = np.quantile(boot[:, j], [0.025, 0.975])
            delta = boot[:, j] - boot[:, pivot.columns.get_loc(("verified_only", rho))]
            d_lo, d_hi = np.quantile(delta, [0.025, 0.975])
            row = dict(noise_type=noise, round=rnd, method=method, rho=rho, n=len(pivot),
                       acc=mean, acc_lo=acc_lo, acc_hi=acc_hi, delta=mean - ref[rho].mean(),
                       delta_lo=d_lo, delta_hi=d_hi)
            if method != "verified_only":
                rates = np.array([inv_curve(grid, curve, a) / rho for curve, a in zip(ref_boot, boot[:, j])])
                row.update(rate=inv_curve(grid, ref.mean().to_numpy(), mean) / rho,
                           rate_lo=np.quantile(rates, 0.025, method="inverted_cdf"),
                           rate_hi=np.quantile(rates, 0.975, method="inverted_cdf"))
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    table = summary[summary.rho.isin(RHOS)].copy()
    table["Accuracy % [95% CI]"] = [
        f"{100*r.acc:.2f} [{100*r.acc_lo:.2f}, {100*r.acc_hi:.2f}]" for r in table.itertuples()
    ]
    table["Exchange rate [95% CI]"] = [
        f"{r.rate:.2f} [{r.rate_lo:.2f}, {r.rate_hi:.2f}]" if r.method != "verified_only" else "reference"
        for r in table.itertuples()
    ]
    cols = ["noise_type", "round", "rho", "method", "Accuracy % [95% CI]", "Exchange rate [95% CI]"]
    seeds = ", ".join(map(str, sorted(raw.seed.unique())))
    (out / "summary.md").write_text(
        "# E3 online synthetic experiment\n\n"
        f"Seeds: {seeds}. Benchmark seed = 1000 + seed. Context and source noise target 40%.\n\n"
        "Four batches of 160 items, cumulative refits including 80 trusted seed items. "
        "CA-EB uses probabilities logged before feedback, and evaluation uses the fixed 500-item test split. "
        "Exchange rates invert the monotone verified-only curve at the same round. "
        "An infinite estimate exceeds the highest reference accuracy. "
        "Intervals are pointwise paired seed-bootstrap percentiles (10,000 replicates).\n\n"
        + table[cols].to_markdown(index=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--cache", type=Path, default=Path("results/q1_e4/cache/emb_bge_small.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/q1_e3"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=True) as data:
        embeddings = dict(zip(data["texts"].tolist(), data["emb"]))
    rows = []
    for seed in range(args.seeds):
        for noise in CONDITIONS:
            b = make_benchmark(1000 + seed, noise=0.4, noise_type=noise)
            Xs, Xa, Xt = (np.stack([embeddings[text] for text in b[k].text]) for k in ("seed", "adapt", "test"))
            ys = b["seed"].y_true.to_numpy()
            yt = b["test"].y_true.to_numpy()
            model = fit(KIND, Xs, ys)
            for rho in RHO_CURVE:
                for method in (METHODS if rho in RHOS else ("verified_only",)):
                    rows.extend(trajectory(method, rho, seed, noise, b["adapt"], Xs, ys, Xa, Xt, yt, model))
        pd.DataFrame(rows).to_csv(args.out / "raw_results.csv", index=False)
        print(f"Seed {seed} complete", flush=True)
    summarize(pd.DataFrame(rows), args.out)
    (args.out / "run_meta.json").write_text(json.dumps({
        "seeds": list(range(args.seeds)), "rounds": ROUNDS, "cache": str(args.cache),
        "methods": METHODS, "reference_rhos": RHO_CURVE, "conditions": CONDITIONS
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
