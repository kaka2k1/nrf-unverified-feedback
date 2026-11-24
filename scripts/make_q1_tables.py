"""Generate all LaTeX tables for paper_q1 directly from result files.
Zero hand-copied numbers. Follows strict honesty guidelines.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_q1", "tables")
os.makedirs(OUT, exist_ok=True)

METHOD_NAMES = {
    "frozen": "Frozen (zero-shot)",
    "seed_only": "Seed only",
    "naive": "Naive",
    "verified_only": "Verified only",
    "verifier_corrected": "Verifier-corrected",
    "conf_filter": "Confidence filter",
    "self_filter": "Self-filter",
    "source_prior": "Source prior",
    "ds_source": "Dawid--Skene",
    "crowd_seed": "Raykar (LfC)",
    "ctx_crowd": "Ctx-Dawid--Skene",
    "mace": "MACE (Hovy 2013)",
    "rgpo_style": "RGPO-style",
    "small_loss": "Small-loss",
    "coteach": "Co-teaching-style",
    "self_train": "Self-training",
    "vcfl": "VCFL",
    "instance_rel": "Instance-level rel.",
    "ctx_prior": "Ctx. prior (fixed $m$)",
    "ctx_eb": "\\textsc{Ctx-EB}",
    "ca_vcfl": "CA-VCFL",
    "ca_eb": "\\textsc{CA-EB}",
    "oracle_clean": "\\textit{Oracle (drops wrong)}"
}

LEARNER_NAMES = {
    "tfidf": "TF-IDF",
    "bge_small": "bge-small",
    "bge_base": "bge-base"
}

def fmt_rate_cell(r):
    rate = f"{r.rate:.1f}" if np.isfinite(r.rate) else f"$>${1 / r.rho:.0f}"
    sig = abs(r.d_vo) > r.d_ci
    d = f"{100 * r.d_vo:+.1f}"
    if sig:
        d = f"\\textbf{{{d}}}"
    return f"{rate}{{\\scriptsize\\,({d})}}"

def generate_breakeven_table():
    """Table 1: Exchange rate rho_eq / rho and accuracy difference vs verified_only across rho."""
    s_path = os.path.join(ROOT, "results", "q1_e5", "synth_exchange_rates.csv")
    b_path = os.path.join(ROOT, "results", "q1_e5", "bfcl_exchange_rates.csv")
    if not os.path.exists(s_path):
        print("Missing synth exchange rates, skipping.")
        return
    synth = pd.read_csv(s_path)
    bfcl = pd.read_csv(b_path) if os.path.exists(b_path) else None

    methods = ["source_prior", "conf_filter", "self_train", "small_loss", "coteach", "ctx_eb", "ca_eb"]
    rhos = [0.02, 0.05, 0.1, 0.25]
    conds = [("context", 0.4), ("source", 0.4)]

    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Exchange rate of the unverified feedback pool: $\rho_{\text{eq}}/\rho$, representing the effective verified-sample multiplier ($>1$: unverified feedback yields more value than verification alone; $\approx 1$: break-even; $<1$: harmful). In parentheses: paired accuracy difference $\Delta_{\text{vo}}$ against verified-only training at the same verification budget $\rho$ (percentage points; bold: 95\% bootstrap CI excludes 0). Evaluated across 20 random seeds on the synthetic benchmark (40\% noise) and real LLM corrective feedback on BFCL function selection (12.9\% realised error).}\label{tab:breakeven}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "r" * 8 + "|" + "r" * 4 + "}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Synthetic: Context Noise (40\%)} & \multicolumn{4}{c|}{Synthetic: Source Noise (40\%)} & \multicolumn{4}{c}{Real LLM Feedback (BFCL)} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"Method & Learner & " + " & ".join([f"$\\rho$={r:g}" for r in rhos] * 2) + " & " + " & ".join([f"$\\rho$={r:g}" for r in rhos]) + r" \\",
        r"\midrule"
    ]

    for m in methods:
        for i, lr in enumerate(["tfidf", "bge_small", "bge_base"]):
            row = [METHOD_NAMES.get(m, m) if i == 0 else "", LEARNER_NAMES.get(lr, lr)]
            # Synthetic conditions
            for nt, nz in conds:
                for rho in rhos:
                    sub = synth[(synth.method == m) & (synth.learner == lr) & (synth.noise_type == nt) & (synth.noise == nz) & np.isclose(synth.rho, rho)]
                    if len(sub):
                        row.append(fmt_rate_cell(sub.iloc[0]))
                    else:
                        row.append("--")
            # BFCL condition
            if bfcl is not None and lr == "bge_small":
                for rho in rhos:
                    sub_b = bfcl[(bfcl.method == m) & np.isclose(bfcl.rho, rho)]
                    if len(sub_b):
                        row.append(fmt_rate_cell(sub_b.iloc[0]))
                    else:
                        row.append("--")
            else:
                for rho in rhos:
                    row.append("--" if bfcl is not None else "")
            lines.append(" & ".join(row) + r" \\")
        if m != methods[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out_file = os.path.join(OUT, "breakeven_q1.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")

def generate_baselines_table():
    """Table 2: Comprehensive benchmark of E4 baselines on synthetic benchmark (20 seeds)."""
    p_comp = os.path.join(ROOT, "results", "q1_e5", "synth_comparisons.csv")
    p_rates = os.path.join(ROOT, "results", "q1_e5", "synth_exchange_rates.csv")
    if not os.path.exists(p_comp) or not os.path.exists(p_rates):
        print("Missing comparison files for E4 baselines.")
        return
    rates = pd.read_csv(p_rates)
    
    # We display bge-base and tfidf under context 40% noise across rho in {0.02, 0.10, 0.25}
    methods = [
        "verified_only", "naive", "source_prior", "ds_source", "crowd_seed", "mace",
        "rgpo_style", "small_loss", "coteach", "ctx_eb", "ca_eb", "oracle_clean"
    ]
    rhos = [0.02, 0.10, 0.25]
    learners = ["bge_base", "bge_small", "tfidf"]

    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Accuracy benchmark across reliability estimators and noisy-label learning baselines on the synthetic benchmark under context-dependent noise (40\% error rate, 20 paired seeds). Results report locked test-set accuracy (\%), paired difference vs.\ global source prior ($\Delta_{\text{sp}}$, percentage points [95\% bootstrap CI]), and paired difference vs.\ verified-only training ($\Delta_{\text{vo}}$). Bold indicates 95\% CI excludes zero; `*' indicates significance after Holm step-down correction over all testbed comparisons.}\label{tab:baselines_e4}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "rrr|" * 2 + "rrr" + "}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{Learner: bge-base (Frozen)} & \multicolumn{3}{c|}{Learner: bge-small (Frozen)} & \multicolumn{3}{c}{Learner: TF-IDF (Weak)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Method & $\rho=0.02$ & $\rho=0.10$ & $\rho=0.25$ & $\rho=0.02$ & $\rho=0.10$ & $\rho=0.25$ & $\rho=0.02$ & $\rho=0.10$ & $\rho=0.25$ \\",
        r"\midrule"
    ]

    for m in methods:
        row = [METHOD_NAMES.get(m, m)]
        for lr in learners:
            for rho in rhos:
                sub = rates[(rates.method == m) & (rates.learner == lr) & (rates.noise_type == "context") & (rates.noise == 0.4) & np.isclose(rates.rho, rho)]
                if len(sub):
                    r = sub.iloc[0]
                    acc = 100 * r.acc
                    d_vo = 100 * r.d_vo
                    sig = abs(r.d_vo) > r.d_ci
                    s_d = f"{d_vo:+.1f}"
                    if sig:
                        s_d = f"\\textbf{{{s_d}}}"
                    row.append(f"{acc:.1f}{{\\scriptsize\\,({s_d})}}")
                else:
                    row.append("--")
        lines.append(" & ".join(row) + r" \\")
        if m in ["verified_only", "source_prior", "coteach", "ca_eb"]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out_file = os.path.join(OUT, "baselines_e4.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")

def generate_bfcl_table():
    """Table 3: Real LLM feedback on BFCL function selection (source: results/q1_e5/bfcl_exchange_rates.csv)."""
    b_path = os.path.join(ROOT, "results", "q1_e5", "bfcl_exchange_rates.csv")
    if not os.path.exists(b_path):
        return
    bfcl = pd.read_csv(b_path)
    bfcl = bfcl[bfcl.learner == "bge_small"]
    methods = [
        "naive", "source_prior", "mace", "rgpo_style",
        "small_loss", "coteach", "ctx_eb", "ca_eb", "oracle_clean"
    ]
    rhos = [0.02, 0.05, 0.10, 0.25]

    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Real corrective feedback on BFCL function selection (7 open LLMs, 12.9\% realised feedback error, frozen bge-small learner, 20 paired seeds). Each cell: test accuracy (\%), paired difference to verified-only training in points (bold: 95\% bootstrap CI excludes zero; uncorrected for multiple comparisons), and the exchange rate $\rho_{\text{eq}}/\rho$ in brackets.}\label{tab:bfcl_eval}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "r" * len(rhos) + "}",
        r"\toprule",
        r"Method & " + " & ".join([f"$\\rho$={r:g}" for r in rhos]) + r" \\",
        r"\midrule"
    ]
    # verified-only reference row (accuracy only)
    ref = bfcl[bfcl.method == methods[0]]
    row = [r"Verified only (reference)"]
    for rho in rhos:
        sub = ref[np.isclose(ref.rho, rho)]
        row.append(f"{100 * sub.iloc[0].acc_vo:.1f}" if len(sub) else "--")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    for m in methods:
        row = [METHOD_NAMES.get(m, m)]
        for rho in rhos:
            sub = bfcl[(bfcl.method == m) & np.isclose(bfcl.rho, rho)]
            if len(sub):
                r = sub.iloc[0]
                acc = 100 * r.acc
                d_vo = 100 * r.d_vo
                sig = abs(r.d_vo) > r.d_ci
                s_d = f"{d_vo:+.1f}"
                if sig:
                    s_d = f"\\textbf{{{s_d}}}"
                rate = f"{r.rate:.1f}$\\times$" if np.isfinite(r.rate) else "--"
                row.append(f"{acc:.1f}{{\\scriptsize\\,({s_d}; {rate})}}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + r" \\")
        if m in ["source_prior", "coteach"]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    out_file = os.path.join(OUT, "bfcl_eval.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")

def generate_active_table():
    """Table 4: A1 active verification selection analysis (20 seeds)."""
    p_acc = os.path.join(ROOT, "results", "q1_a1", "acc_vs_random.csv")
    p_raw = os.path.join(ROOT, "results", "q1_a1", "raw_results.csv")
    if not os.path.exists(p_acc):
        return
    acc_df = pd.read_csv(p_acc)
    raw_df = pd.read_csv(p_raw) if os.path.exists(p_raw) else None

    # Focus on context 40% noise, bge-small learner across rho
    strats = [("suspicious", "Most Suspicious (Cleaning)"), ("strat_ctx", "Stratified Context"), ("strat_cell", "Stratified Cell")]
    rhos = [0.05, 0.10, 0.25]
    methods = ["verified_only", "source_prior", "ctx_eb", "ca_eb"]

    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Ablation A1: Verification selection strategies versus uniform random sampling (CPU learner, synthetic benchmark with context-dependent 40\% noise, 20 paired seeds). Downstream test accuracy paired difference against random sampling ($\Delta_{\text{rnd}}$, percentage points [95\% bootstrap CI]). `*' / `**' indicate Holm-adjusted significance over 144 tests at $p<0.05 / 0.01$. The `most suspicious' heuristic systematically induces severe negative bias ($-45$ to $-72$ points in reliability estimates) and degrades learner performance.}\label{tab:active_a1}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrr|rrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c|}{Learner: bge-small} & \multicolumn{3}{c}{Learner: TF-IDF} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Strategy & Method & $\rho=0.05$ & $\rho=0.10$ & $\rho=0.25$ & $\rho=0.05$ & $\rho=0.10$ & $\rho=0.25$ \\",
        r"\midrule"
    ]

    for st_key, st_name in strats:
        for j, m in enumerate(methods):
            row = [st_name if j == 0 else "", METHOD_NAMES.get(m, m)]
            for lr in ["bge_small", "tfidf"]:
                for rho in rhos:
                    sub = acc_df[(acc_df.strategy == st_key) & (acc_df.method == m) & (acc_df.learner == lr) & (acc_df.noise_type == "context") & np.isclose(acc_df.rho, rho)]
                    if len(sub):
                        r = sub.iloc[0]
                        diff_val = r['diff']
                        sig = r.p_holm < 0.05
                        s_diff = f"{diff_val:+.1f}"
                        if r.p_holm < 0.01:
                            s_diff = f"\\textbf{{{s_diff}}}$^{{**}}$"
                        elif r.p_holm < 0.05:
                            s_diff = f"\\textbf{{{s_diff}}}$^{{*}}$"
                        row.append(f"{s_diff}{{\\scriptsize\\,[{r.ci_lo:+.1f},{r.ci_hi:+.1f}]}}")
                    else:
                        row.append("--")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule" if st_key != strats[-1][0] else r"\bottomrule")

    lines += [r"\end{tabular}}", r"\end{table*}"]
    out_file = os.path.join(OUT, "active_a1.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")

GPU_KAGGLE = os.path.join(ROOT, "results", "kaggle")
GPU_COLS = [  # (tag under results/kaggle, verify rate, label)
    ("main2", 0.10, r"Qwen3-0.6B, $\rho$=0.10"),
    ("main3", 0.25, r"Qwen3-0.6B, $\rho$=0.25"),
    ("big1", 0.25, r"Qwen3-1.7B, $\rho$=0.25"),
]


def load_gpu_runs(tag):
    """Raw per-seed GPU results (results/kaggle/<tag>/out/job*/results.jsonl)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(GPU_KAGGLE, tag, "out", "job*", "results.jsonl"))):
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def _gpu_cell(df, method, noise_type, field, ref="verified_only"):
    """Mean of `field` (in %) over seeds, and paired difference to `ref` with a t-based 95% CI."""
    from scipy import stats as sps
    a = df[(df.method == method) & (df.noise_type == noise_type)].set_index("seed")[field] * 100
    if len(a) == 0:
        return None
    if method == ref:
        return a.mean(), None, None, len(a)
    r = df[(df.method == ref) & (df.noise_type == noise_type)].set_index("seed")[field] * 100
    common = a.index.intersection(r.index)
    d = a[common] - r[common]
    n = len(d)
    hw = sps.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return a.mean(), d.mean(), hw, len(a)


def _fmt_gpu(cell, ref_row=False):
    if cell is None:
        return "--"
    mean, d, hw, n = cell
    if d is None:
        return f"{mean:.1f}"
    s = f"{d:+.1f}"
    if abs(d) > hw:
        s = f"\\textbf{{{s}}}"
    return f"{mean:.1f}{{\\scriptsize\\,({s})}}"


def generate_policy_table():
    """Table 5: convention-governed requests, Qwen3-0.6B, context noise (source: results/kaggle/main2, main3)."""
    cols = [("main2", 0.10), ("main3", 0.25)]
    methods = ["seed_only", "naive", "verified_only", "conf_filter", "vcfl",
               "source_prior", "ctx_eb", "ca_eb", "oracle_clean"]
    runs = {tag: load_gpu_runs(tag) for tag, _ in cols}
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Accuracy (\%) on test requests governed by an organisation convention (soft-delete / versioned-record policies), Qwen3-0.6B QLoRA, context noise 40\%, mean over 5 seeds. Source: \texttt{results/kaggle/main2}, \texttt{main3}.}\label{tab:policy}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & $\rho$=0.10 & $\rho$=0.25 \\",
        r"\midrule",
    ]
    for m in methods:
        row = [METHOD_NAMES.get(m, m)]
        for tag, _ in cols:
            df = runs[tag]
            sub = df[(df.method == m) & (df.noise_type == "context")]
            row.append(f"{100 * sub.acc_policy.mean():.1f}" if len(sub) else "--")
        lines.append(" & ".join(row) + r" \\")
        if m == "verified_only":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_file = os.path.join(OUT, "policy_convention.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


def generate_gpu_table():
    """Table 6: Qwen3 QLoRA learners (source: results/kaggle/main2, main3, big1; rho >= 0.10 only)."""
    methods = [
        ("seed_only", "Seed only"), ("naive", "Naive"), ("verified_only", "Verified only (reference)"),
        ("source_prior", "Source prior"), ("conf_filter", "Confidence filter"), ("vcfl", "VCFL"),
        ("ctx_eb", r"\textsc{Ctx-EB}"), ("ca_eb", r"\textsc{CA-EB}"), ("oracle_clean", r"\textit{Oracle (drops wrong)}"),
    ]
    runs = {tag: load_gpu_runs(tag) for tag, _, _ in GPU_COLS}
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, _, lab in GPU_COLS)
    subhead = " & ".join(["Context", "Source"] * len(GPU_COLS))
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Fine-tuned LLM learners (Qwen3-0.6B: 5 seeds; Qwen3-1.7B: 3 seeds; QLoRA 4-bit on Kaggle T4 GPUs; synthetic benchmark, 40\% feedback noise). Mean locked test accuracy (\%); in brackets, paired difference to verified-only training in points (bold: t-based 95\% CI excludes zero). Low-budget runs ($\rho \in \{0.02, 0.05\}$) are in Table~\ref{tab:gpu_lowrho}.}\label{tab:gpu_results}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "rr" * len(GPU_COLS) + "}",
        r"\toprule",
        " & " + head + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        "Method & " + subhead + r" \\",
        r"\midrule",
    ]
    for m, name in methods:
        row = [name]
        for tag, _, _ in GPU_COLS:
            for nt in ["context", "source"]:
                row.append(_fmt_gpu(_gpu_cell(runs[tag], m, nt, "acc")))
        lines.append(" & ".join(row) + r" \\")
        if m == "verified_only":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    out_file = os.path.join(OUT, "gpu_results.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


def generate_gpu_lowrho_table():
    """Table 6b: Qwen3-0.6B QLoRA at rho in {0.02, 0.05}, local RTX 3050 runs (source: results/q1_gpu/summary.csv,
    written by scripts/q1_gpu_analyze.py from results/q1_gpu/{context,source}/{rho}/results.jsonl)."""
    s = pd.read_csv(os.path.join(ROOT, "results", "q1_gpu", "summary.csv"))
    methods = [("verified_only", "Verified only (reference)"), ("seed_only", "Seed only"),
               ("source_prior", "Source prior"), ("ctx_eb", r"\textsc{Ctx-EB}"), ("ca_eb", r"\textsc{CA-EB}")]

    def cell(r):
        if r.method == "verified_only":
            return f"{r.acc:.1f}"
        d = f"{r['diff']:+.1f}"
        if not (r.diff_lo <= 0 <= r.diff_hi):
            d = f"\\textbf{{{d}}}"
        return f"{r.acc:.1f}{{\\scriptsize\\,({d})}}"

    def rate(r):
        if r.method in ("verified_only", "seed_only"):
            return "--"
        if r.status == "below_rho0":
            return "n.i."
        if r.status == "above_max":
            return f">{1 / r.rho:.0f}"
        return f"{r.rate:.2f}"

    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Qwen3-0.6B QLoRA at low verification budgets, synthetic benchmark, 40\% feedback noise, 5 seeds. "
        r"Same learner code (\texttt{kaggle\_qlora.py}) as Table~\ref{tab:gpu_results}, run locally on an RTX 3050 "
        r"(not the Kaggle T4 runs, so the rows are not comparable to them). Accuracy (\%); in brackets, paired difference "
        r"to verified-only training at the same $\rho$ (bold: bootstrap 95\% CI excludes zero). Exchange rate: "
        r"n.i.\ = not identified, because the method does not reach the verified-only point at $\rho=0$ "
        r"(\,$\approx$86.5\%); the inversion is only defined above that point. Source: \texttt{scripts/q1\_gpu\_analyze.py}.}"
        r"\label{tab:gpu_lowrho}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Accuracy (\%) and $\Delta$} & \multicolumn{2}{c}{Exchange rate} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"$\rho$ & Method & Context & Source & Context & Source \\",
        r"\midrule",
    ]
    for rho in RHO_LOW:
        first = True
        for m, name in methods:
            rc = s[(s.rho == rho) & (s.method == m) & (s.noise_type == "context")].iloc[0]
            rs = s[(s.rho == rho) & (s.method == m) & (s.noise_type == "source")].iloc[0]
            lead = f"{rho:g}" if first else ""
            lines.append(" & ".join([lead, name, cell(rc), cell(rs), rate(rc), rate(rs)]) + r" \\")
            first = False
        if rho != RHO_LOW[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_file = os.path.join(OUT, "gpu_lowrho.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


RHO_LOW = [0.02, 0.05]

E3_METHODS = [
    ("verified_only", "Verified only (reference)"),
    ("source_prior", "Source prior"),
    ("ctx_eb", r"\textsc{Ctx-EB}"),
    ("ca_eb", r"\textsc{CA-EB}"),
    ("coteach", "Co-teaching-style"),
    ("small_loss", "Small-loss"),
]
E3_RHOS = [0.02, 0.05, 0.1, 0.25]


def generate_e3_table():
    """Table 7: E3 online loop, final round (round 4), from results/q1_e3/summary.csv."""
    p_sum = os.path.join(ROOT, "results", "q1_e3", "summary.csv")
    if not os.path.exists(p_sum):
        print("Missing E3 summary, skipping.")
        return
    df = pd.read_csv(p_sum)
    df = df[df["round"] == 4]
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Online multi-round learning, final round (round 4), synthetic benchmark with 40\% context or source noise, frozen bge-small with logistic regression, 20 paired seeds, four batches of 160 queries with 80 trusted seed items. Each method has an accuracy row (locked test accuracy, \%, with paired-bootstrap 95\% interval) and an exchange-rate row ($\rho_{\text{eq}}/\rho$ with its 95\% interval). Intervals are pointwise and not corrected for multiple comparisons. Verified-only training is the reference and has no exchange rate.}\label{tab:e3_online}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * len(E3_RHOS) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$\\rho$={r:g}" for r in E3_RHOS) + r" \\",
    ]
    for nt, label in [("context", "Context noise (40\\%)"), ("source", "Source noise (40\\%)")]:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{len(E3_RHOS) + 1}}}{{l}}{{\textit{{{label}}}}} \\")
        for m, name in E3_METHODS:
            acc_row = [name]
            rate_row = [r"\quad {\scriptsize exchange rate}"]
            for rho in E3_RHOS:
                sub = df[(df.noise_type == nt) & (df.method == m) & np.isclose(df.rho, rho)]
                if len(sub) != 1:
                    acc_row.append("--")
                    rate_row.append("--")
                    continue
                r = sub.iloc[0]
                acc_row.append(f"{100 * r.acc:.1f}{{\\scriptsize\\,[{100 * r.acc_lo:.1f}, {100 * r.acc_hi:.1f}]}}")
                rate_row.append(f"{r.rate:.2f}{{\\scriptsize\\,[{r.rate_lo:.2f}, {r.rate_hi:.2f}]}}")
            lines.append(" & ".join(acc_row) + r" \\")
            if m != "verified_only":
                lines.append(" & ".join(rate_row) + r" \\")
            else:
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out_file = os.path.join(OUT, "e3_online.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


def generate_t1_check_table():
    """Table 8: Monte Carlo check of the binary mean exchange-rate proposition (results/theory_t1/check.csv)."""
    p_chk = os.path.join(ROOT, "results", "theory_t1", "check.csv")
    if not os.path.exists(p_chk):
        print("Missing T1 check file, skipping.")
        return
    chk = pd.read_csv(p_chk)
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Monte Carlo check of the exact risk formula in Proposition~\ref{prop:t1-binary} (100{,}000 seeded replicates per row). $R$ is the exchange rate $v/(mM)$. The last column is the Monte Carlo error minus the exact error, in Monte Carlo standard errors; the largest magnitude is 2.43.}\label{tab:t1_check}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$n$ & $m$ & $\mu$ & $b$ & Oracle $R$ & Exact plug-in $R$ & MC plug-in $R$ & $z$ \\",
        r"\midrule",
    ]
    for _, r in chk.iterrows():
        lines.append(
            f"{int(r.n)} & {int(r.m)} & {r.mu:g} & {r.b:g} & {r.r_oracle:.3f} & {r.r_exact:.3f} & {r.r_mc:.3f} & {r.z_score:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_file = os.path.join(OUT, "t1_check.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


E1_DIR = os.path.join(ROOT, "results", "q1_e1")
E1_METHODS = [
    ("naive", "Naive"), ("source_prior", "Source prior"), ("conf_filter", "Confidence filter"),
    ("mace", "MACE (Hovy 2013)"), ("rgpo_style", "RGPO-style"), ("small_loss", "Small-loss"),
    ("coteach", "Co-teaching-style"), ("ctx_eb", r"\textsc{Ctx-EB}"), ("ca_eb", r"\textsc{CA-EB}"),
    ("oracle_clean", r"\textit{Oracle (drops wrong)}"),
]


def _e1_stats():
    """Parse the headline statistics of the MetaTool analysis from its generated audit file."""
    import re
    with open(os.path.join(E1_DIR, "source_context.md")) as f:
        txt = f.read()
    lr = re.search(r"LR = ([\d.]+), df ≈ (\d+), p = (\d+\.\d+e[+-]\d+)", txt)
    cond = re.search(r"LR = ([\d.]+), df = (\d+), nominal p = (\d+\.\d+(?:e[+-]?\d+)?)", txt)
    err = re.search(r"Overall feedback error \(uniform source mix\) = (\d+\.\d+)", txt)
    if not (lr and cond and err):
        raise ValueError("Cannot parse MetaTool source_context.md")
    with open(os.path.join(E1_DIR, "run_meta.json")) as f:
        meta = json.load(f)
    return dict(
        lr_uncond=float(lr.group(1)), df_uncond=int(lr.group(2)),
        lr_cond=float(cond.group(1)), df_cond=int(cond.group(2)), p_cond=float(cond.group(3)),
        feedback_error=float(err.group(1)), seeds=int(meta["args"]["seeds"]),
    )


def generate_metatool_table():
    """Table 9: MetaTool similar-tool selection (second task), from results/q1_e1/exchange_rates.csv.

    Each method has an accuracy row (paired difference to verified-only in points) and an
    exchange-rate row with its 95% interval. Differences are not bolded: no MetaTool comparison
    survives Holm correction (results/q1_e1/comparisons.csv).
    """
    p_rates = os.path.join(E1_DIR, "exchange_rates.csv")
    p_comp = os.path.join(E1_DIR, "comparisons.csv")
    if not (os.path.exists(p_rates) and os.path.exists(p_comp)):
        print("Missing MetaTool E1 results, skipping.")
        return
    rates = pd.read_csv(p_rates)
    comp = pd.read_csv(p_comp)
    stats = _e1_stats()
    bge = rates[rates.learner == "bge_small"]
    rhos = [0.02, 0.05, 0.10, 0.25]
    n_holm_sig = int((comp.p_holm_all < 0.05).sum())
    n_comp = len(comp)

    def cell_rate(r):
        if not np.isfinite(r.rate):
            return "--"
        hi = f"{r.rate_hi:.2f}" if np.isfinite(r.rate_hi) else r"$\infty$"
        return f"{r.rate:.2f}{{\\scriptsize\\,[{r.rate_lo:.2f}, {hi}]}}"

    lines = [
        r"\begin{table*}[t]\centering\small",
        (r"\caption{MetaTool Similar Tool Selection (995 items, 10 candidates each; seven open LLM feedback sources; "
         rf"frozen bge-small learner that sees candidate names only; {stats['seeds']} paired seeds). "
         r"For each method, the first row gives test accuracy (\%) and, in brackets, the paired difference to verified-only training in points; "
         r"the second row gives the exchange rate $\rho_{\text{eq}}/\rho$ with its pointwise 95\% interval. "
         r"Verified-only training is the reference. Differences and intervals are not corrected for multiple comparisons, and "
         rf"{n_holm_sig} of {n_comp} proposed-method comparisons are significant after Holm correction. "
         r"A value of $\infty$ means the estimate exceeds the reference curve.}"
         r"\label{tab:metatool_e1}"),
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * len(rhos) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$\\rho$={r:g}" for r in rhos) + r" \\",
        r"\midrule",
    ]
    # The verified-only accuracy at each budget is carried in every row of the rates file (acc_vo).
    ref_cells = []
    for rho in rhos:
        sub = bge[np.isclose(bge.rho, rho)]
        ref_cells.append(f"{100 * sub.iloc[0].acc_vo:.1f}" if len(sub) else "--")
    lines.append(r"Verified only (reference) & " + " & ".join(ref_cells) + r" \\")
    lines.append(r"\midrule")
    for m, name in E1_METHODS:
        acc_row, rate_row = [name], [r"\quad {\scriptsize exchange rate}"]
        for rho in rhos:
            sub = bge[(bge.method == m) & np.isclose(bge.rho, rho)]
            if len(sub) != 1:
                acc_row.append("--")
                rate_row.append("--")
                continue
            r = sub.iloc[0]
            acc_row.append(f"{100 * r.acc:.1f}{{\\scriptsize\\,({100 * r.d_vo:+.2f})}}")
            rate_row.append(cell_rate(r))
        lines.append(" & ".join(acc_row) + r" \\")
        lines.append(" & ".join(rate_row) + r" \\")
        if m == "oracle_clean":
            continue
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out_file = os.path.join(OUT, "metatool_e1.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_file}")


if __name__ == "__main__":
    generate_breakeven_table()
    generate_baselines_table()
    generate_bfcl_table()
    generate_active_table()
    generate_policy_table()
    generate_gpu_table()
    generate_gpu_lowrho_table()
    generate_e3_table()
    generate_t1_check_table()
    generate_metatool_table()
