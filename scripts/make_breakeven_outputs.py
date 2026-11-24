"""Table + figure for the break-even (exchange-rate) analysis, generated ONLY from results/*/exchange_rates.csv.

python scripts/analyze_breakeven.py results/cpu_breakeven results/cpu_bfcl_breakeven
python scripts/make_breakeven_outputs.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"naive": "Naive", "source_prior": "Source prior", "conf_filter": "Conf.\\ filter", "ctx_eb": "\\textsc{Ctx-EB}",
         "ca_eb": "\\textsc{CA-EB}", "self_train": "Self-training", "oracle_clean": "\\textit{Oracle}"}
LRN = {"tfidf": "TF-IDF", "bge_small": "bge-small", "bge_base": "bge-base"}
RHOS = [0.02, 0.05, 0.1, 0.25]


def cell(r):
    rate = f"{r.rate:.1f}" if np.isfinite(r.rate) else f"$>${1 / r.rho:.0f}"
    sig = abs(r.d_vo) > r.d_ci
    d = f"{100 * r.d_vo:+.1f}"
    d = f"\\textbf{{{d}}}" if sig else d
    return f"{rate}{{\\scriptsize\\,({d})}}"


def table(s):
    conds = [("context", 0.4), ("source", 0.4)]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{[New]} Exchange rate of the unverified feedback pool: $\rho_{\text{eq}}/\rho$, where "
             r"verified-only training would need a verify rate $\rho_{\text{eq}}$ to match the method at verify rate "
             r"$\rho$ ($>1$: the unverified corrections are worth extra verifications; $<1$: using them costs "
             r"verifications). In brackets: accuracy difference to verified-only training at the same $\rho$ (points; "
             r"bold: 95\% CI excludes 0). Synthetic benchmark, 40\% noise, CPU learners of increasing strength "
             r"(seed-only accuracy: TF-IDF 59.8\%, bge-small 74.2\%, bge-base 77.8\%), 20 seeds.}\label{tab:breakeven}",
             r"\setlength{\tabcolsep}{2.5pt}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{ll" + "r" * 8 + "}", r"\toprule",
             r" & & \multicolumn{4}{c}{context noise} & \multicolumn{4}{c}{source noise}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             "Method & Learner & " + " & ".join([f"$\\rho$={r:g}" for r in RHOS] * 2) + r"\\", r"\midrule"]
    for m in ["source_prior", "conf_filter", "self_train", "ctx_eb", "ca_eb"]:
        for i, lr in enumerate(["tfidf", "bge_small", "bge_base"]):
            row = [NAMES[m] if i == 0 else "", LRN[lr]]
            for nt, nz in conds:
                for rho in RHOS:
                    r = s[(s.method == m) & (s.learner == lr) & (s.noise_type == nt) & (s.noise == nz)
                          & np.isclose(s.rho, rho)].iloc[0]
                    row.append(cell(r))
            lines.append(" & ".join(row) + r"\\")
        lines.append(r"\midrule" if m != "ca_eb" else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(ROOT, "paper", "tables", "breakeven.tex"), "w").write("\n".join(lines) + "\n")


def figure(s, sb):
    conds = [("context", 0.4), ("source", 0.4), ("uniform", 0.2)]
    meths = ["naive", "source_prior", "conf_filter", "self_train", "ctx_eb", "ca_eb"]
    col = dict(zip(meths, ["#999999", "#1f77b4", "#d62728", "#8c564b", "#2ca02c", "#9467bd"]))
    lab = {"naive": "Naive", "source_prior": "Source prior", "conf_filter": "Conf. filter", "self_train": "Self-training",
           "ctx_eb": "Ctx-EB", "ca_eb": "CA-EB"}
    npan = len(conds) + (1 if sb is not None else 0)
    fig, axes = plt.subplots(1, npan, figsize=(1.8 * npan, 2.4), sharey=True)
    for ax, (nt, nz) in zip(axes, conds):
        for m in meths:
            g = s[(s.method == m) & (s.learner == "bge_base") & (s.noise_type == nt) & (s.noise == nz)].sort_values("rho")
            y = np.clip(g.rate.replace(np.inf, 50).to_numpy(), 0.1, 50)
            ax.plot(g.rho, y, "o-", ms=3, lw=1.2, color=col[m], label=lab[m])
        ax.set_title(f"{nt}, {int(100 * nz)}% noise", fontsize=8)
    if sb is not None:
        ax = axes[-1]
        for m in meths:
            g = sb[sb.method == m].sort_values("rho")
            y = np.clip(g.rate.replace(np.inf, 50).to_numpy(), 0.1, 50)
            ax.plot(g.rho, y, "o-", ms=3, lw=1.2, color=col[m])
        ax.set_title("BFCL real feedback", fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(RHOS)
        ax.set_xticklabels([f"{r:g}" for r in RHOS], fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(r"verify rate $\rho$", fontsize=7)
        ax.set_ylim(0.1, 50)
    axes[0].set_ylabel(r"exchange rate $\rho_{eq}/\rho$", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(ROOT, "paper", "figures", "breakeven.pdf"))


def main():
    s = pd.read_csv(os.path.join(ROOT, "results", "cpu_breakeven", "exchange_rates.csv"))
    f = os.path.join(ROOT, "results", "cpu_bfcl_breakeven", "exchange_rates.csv")
    sb = pd.read_csv(f) if os.path.exists(f) else None
    table(s)
    figure(s, sb)
    print("wrote paper/tables/breakeven.tex, paper/figures/breakeven.pdf")


if __name__ == "__main__":
    main()
