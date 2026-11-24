"""Figures for the paper. python scripts/make_figures.py"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
SHORT = {"SmolLM2-360M-Instruct": "SmolLM2-0.36B", "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
         "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen3-1.7B": "Qwen3-1.7B",
         "granite-3.1-2b-instruct": "Granite-3.1-2B", "Phi-3.5-mini-instruct": "Phi-3.5-mini"}
CTX = {0: "restaurants", 1: "payments", 2: "music/media", 3: "services", 4: "travel", 5: "compute/weather/API",
       6: "events/tickets", 7: "home/alarms"}


def heatmap():
    items = [json.loads(line) for line in open(os.path.join(ROOT, "data", "bfcl_items.jsonl"))]
    ids = [i["id"] for i in items]
    gold = np.array([i["gold"] for i in items])
    ctx = np.array([i["context"] for i in items])
    acc = {}
    for f in glob.glob(os.path.join(ROOT, "data", "bfcl_annot", "annot_*.jsonl")):
        s = os.path.basename(f)[6:-6]
        if s not in SHORT:
            continue
        d = {json.loads(line)["id"]: json.loads(line)["pred"] for line in open(f)}
        p = np.array([d[i] for i in ids])
        acc[s] = [100 * (p == gold)[ctx == c].mean() for c in range(8)] + [100 * (p == gold).mean()]
    order = sorted(acc, key=lambda s: acc[s][-1])
    M = np.array([acc[s] for s in order])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    im = ax.imshow(M, cmap="RdYlGn", vmin=45, vmax=100, aspect="auto")
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{CTX[c]}\n(n={int((ctx == c).sum())})" for c in range(8)] + ["ALL"], fontsize=6.5,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[s] for s in order], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="black" if 60 < M[i, j] < 92 else "white" if M[i, j] <= 60 else "black")
    ax.axvline(7.5, color="black", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("accuracy of corrections (%)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bfcl_heatmap.pdf"))
    fig.savefig(os.path.join(OUT, "bfcl_heatmap.png"), dpi=160)
    print("wrote bfcl_heatmap")


def verify_curve():
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "results", "cpu_final", "raw_results.csv"))
    df = df[np.isclose(df.noise, 0.4)]
    ms = [("naive", "Naive", "#999999", ":"), ("verified_only", "Verified only", "#7f7f7f", "--"),
          ("conf_filter", "Confidence filter", "#d62728", "-."), ("source_prior", "Source prior", "#1f77b4", "-"),
          ("instance_rel", "Instance-level rel.", "#9467bd", "--"), ("ctx_eb", "Ctx-EB (ours)", "#2ca02c", "-"),
          ("oracle_clean", "Oracle", "black", ":")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, nt in zip(axes, ["uniform", "source", "context"]):
        sub = df[df.noise_type == nt]
        for m, lab, col, ls in ms:
            g = sub[sub.method == m].groupby("verify_rate").acc
            mu, se = 100 * g.mean(), 100 * g.std() / np.sqrt(g.count())
            ax.errorbar(mu.index, mu.values, yerr=1.96 * se.values, label=lab, color=col, ls=ls, marker="o", ms=3,
                        lw=1.2, capsize=2)
        ax.set_title(f"{nt} noise", fontsize=8)
        ax.set_xlabel("verified fraction $\\rho$", fontsize=7)
        ax.set_xticks([0.1, 0.25, 0.5])
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)", fontsize=7)
    axes[-1].legend(fontsize=5.8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cpu_verify.pdf"))
    fig.savefig(os.path.join(OUT, "cpu_verify.png"), dpi=160)
    print("wrote cpu_verify")


if __name__ == "__main__":
    heatmap()
    verify_curve()
