"""Generate LaTeX tables for the paper directly from result files (no hand-copied numbers).

python scripts/make_tables.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
NAMES = {"frozen": "Frozen (zero-shot)", "seed_only": "Seed only", "naive": "Naive", "verified_only": "Verified only",
         "verifier_corrected": "Verifier-corrected", "conf_filter": "Confidence filter", "self_filter": "Self-filter",
         "source_prior": "Source prior", "ds_source": "Dawid--Skene", "crowd_seed": "Learning from crowds",
         "ctx_crowd": "Ctx. confusion (LfC)", "vcfl": "VCFL", "instance_rel": "Instance-level rel.",
         "ctx_prior": "Ctx. prior (fixed $m$)", "ctx_eb": "\\textsc{Ctx-EB} (ours)", "ca_vcfl": "CA-VCFL",
         "ca_eb": "\\textsc{CA-EB} (ours)", "agree_eb": "Agree-EB", "ca_agree_eb": "CA-Agree-EB",
         "oracle_clean": "\\textit{Oracle (drops wrong)}"}
REF = "source_prior"
# main1 (rho=0.25, 3 seeds, before Ctx-EB existed) is superseded by main3 (rho=0.25, 5 seeds, all methods)
GPU_SYNTH = ["results/kaggle/main2", "results/kaggle/main3"] if os.path.isdir(os.path.join(ROOT, "results/kaggle/main3/out")) \
    else ["results/kaggle/main1", "results/kaggle/main2"]


def fmt_diff(d):
    if d is None or np.isnan(d[0]):
        return "--"
    m, ci = d
    s = f"{m:+.1f}"
    if abs(m) > ci and ci > 0:
        s = f"\\textbf{{{s}}}"
    return f"{s}{{\\scriptsize$\\pm${ci:.1f}}}"


def paired(df, m, keycols):
    piv = df.pivot_table(index=keycols, columns="method", values="acc")
    if m not in piv or REF not in piv:
        return None
    d = (piv[m] - piv[REF]).dropna() * 100
    if len(d) < 2:
        return (float(d.mean()), float("nan"))
    return (float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))))


def table(df, cols, methods, keycol, caption, label, colnames, path, extra_note=""):
    """cols: list of (filter-dict) per column group; each group shows acc and diff."""
    lines = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3.5pt}",
             ("\\resizebox{\\linewidth}{!}{" if len(cols) > 3 else "") + "\\begin{tabular}{l" + "rr" * len(cols) + "}",
             "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in colnames) + "\\\\",
             " " + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             "Method & " + " & ".join(["Acc & $\\Delta$ vs SP"] * len(cols)) + "\\\\", "\\midrule"]
    for m in methods:
        row = [NAMES.get(m, m)]
        for flt in cols:
            sub = df
            for k, v in flt.items():
                sub = sub[np.isclose(sub[k], v)] if isinstance(v, float) else sub[sub[k] == v]
            a = sub[sub.method == m].acc
            row.append(f"{100 * a.mean():.1f}" if len(a) else "--")
            row.append("0" if m == REF else fmt_diff(paired(sub, m, keycol)) if len(a) else "--")
        lines.append(" & ".join(row) + "\\\\")
        if m in ("seed_only", "verifier_corrected"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if len(cols) > 3 else "")]
    if extra_note:
        lines.append(f"\\par\\vspace{{2pt}}\\footnotesize {extra_note}")
    lines.append("\\end{table}")
    open(os.path.join(OUT, path), "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def load_gpu(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(ROOT, d, "**", "results.jsonl"), recursive=True):
            for line in open(f):
                r = json.loads(line)
                if "acc" in r:
                    r.pop("losses", None)
                    r.pop("acc_by_ctx", None)
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    note = ("$\\Delta$: paired difference to the global source prior (SP) over seeds, $\\pm$95\\% CI; bold = CI excludes 0.")
    # ---- CPU synthetic
    cpu = pd.read_csv(os.path.join(ROOT, "results", "cpu_final", "raw_results.csv"))
    cpu = cpu[np.isclose(cpu.noise, 0.4)]
    mc = ["seed_only", "naive", "verified_only", "verifier_corrected", "conf_filter", "self_filter", "ds_source",
          "vcfl", "instance_rel", "ctx_prior", "ca_vcfl", "source_prior", "ctx_eb", "ca_eb", "oracle_clean"]
    cols = [dict(noise_type=nt, verify_rate=vr) for nt in ["uniform", "source", "context"] for vr in [0.1]]
    table(cpu, cols, mc, ["seed"], "CPU learner (TF--IDF + logistic regression), synthetic benchmark, 40\\% feedback noise, "
          "10\\% verified, 8 seeds. Accuracy (\\%) on the locked test set.", "tab:cpu10",
          ["uniform noise", "source noise", "context noise"], "cpu_v10.tex", note)
    cols = [dict(noise_type=nt, verify_rate=vr) for nt in ["uniform", "source", "context"] for vr in [0.25]]
    table(cpu, cols, mc, ["seed"], "As Table~\\ref{tab:cpu10} with 25\\% verified.", "tab:cpu25",
          ["uniform noise", "source noise", "context noise"], "cpu_v25.tex", note)
    # ---- GPU synthetic
    g = load_gpu(GPU_SYNTH)
    if len(g):
        mg = ["frozen", "seed_only", "naive", "verified_only", "verifier_corrected", "conf_filter", "vcfl",
              "instance_rel", "ctx_prior", "ca_vcfl", "source_prior", "ctx_eb", "ca_eb", "oracle_clean"]
        cols, names = [], []
        for vr in [0.1, 0.25]:
            for nt in ["source", "context"]:
                if len(g[(g.noise_type == nt) & np.isclose(g.verify_rate, vr)]):
                    cols.append(dict(noise_type=nt, verify_rate=vr))
                    ns = g[(g.noise_type == nt) & np.isclose(g.verify_rate, vr)].seed.nunique()
                    names.append(f"{nt}, $\\rho$={vr:g} ({ns} seeds)")
        table(g, cols, mg, ["seed"], "Qwen3-0.6B QLoRA learner (Kaggle T4), synthetic benchmark, 40\\% feedback noise. "
              "Accuracy (\\%) by constrained scoring on the locked test set.", "tab:gpu", names, "gpu_synth.tex", note)
    # ---- GPU real feedback
    r = load_gpu([d for d in ["results/kaggle/real1", "results/kaggle/real2"] if os.path.isdir(os.path.join(ROOT, d))])
    if len(r):
        mr = ["frozen", "seed_only", "naive", "verified_only", "verifier_corrected", "conf_filter", "vcfl",
              "instance_rel", "ctx_prior", "source_prior", "ctx_eb", "ca_eb", "oracle_clean"]
        cols, names = [], []
        for vr in sorted(r.verify_rate.unique()):
            cols.append(dict(verify_rate=float(vr)))
            names.append(f"$\\rho$={vr:g} ({r[np.isclose(r.verify_rate, vr)].seed.nunique()} seeds)")
        table(r, cols, mr, ["seed"], "Real LLM feedback on BFCL function selection (7 open LLM sources, mean correction "
              f"error {100 * r.realized_noise.mean():.1f}\\%). Qwen3-0.6B QLoRA learner that sees function names only. "
              "Accuracy (\\%) on 300 locked test requests.", "tab:real", names, "gpu_real.tex", note)
    # ---- GPU synthetic, larger learner (Qwen3-1.7B)
    b = load_gpu(["results/kaggle/big1"]) if os.path.isdir(os.path.join(ROOT, "results/kaggle/big1")) else pd.DataFrame()
    if len(b):
        mb = ["frozen", "seed_only", "naive", "verified_only", "conf_filter", "ca_vcfl", "source_prior", "ctx_eb",
              "ca_eb", "oracle_clean"]
        cols, names = [], []
        for nt in ["source", "context"]:
            sub = b[b.noise_type == nt]
            if len(sub):
                cols.append(dict(noise_type=nt))
                names.append(f"{nt}, $\\rho$=0.25 ({sub.seed.nunique()} seeds)")
        table(b, cols, mb, ["seed"], "Larger learner: Qwen3-1.7B QLoRA, synthetic benchmark, 40\\% feedback noise, 25\\% "
              "verified. Accuracy (\\%) on the locked test set.", "tab:big", names, "gpu_big.tex", note)


def policy_table():
    """Accuracy on convention (policy) items: does filtering by model confidence discard novel-but-correct corrections?"""
    cpu = pd.read_csv(os.path.join(ROOT, "results", "cpu_final", "raw_results.csv"))
    cpu = cpu[np.isclose(cpu.noise, 0.4) & (cpu.noise_type == "context")]
    g = load_gpu(GPU_SYNTH)
    g = g[(g.noise_type == "context")] if len(g) else g
    ms = ["seed_only", "naive", "verified_only", "conf_filter", "vcfl", "source_prior", "ctx_eb", "ca_eb", "oracle_clean"]
    cols = []
    for vr in [0.1, 0.25]:
        c = cpu[np.isclose(cpu.verify_rate, vr)]
        cols.append((f"CPU, $\\rho$={vr:g}", c))
    if len(g):
        for vr in sorted(g.verify_rate.unique()):
            cols.append((f"Qwen3, $\\rho$={vr:g}", g[np.isclose(g.verify_rate, vr)]))
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Accuracy (\\%) on test requests governed by an organisation convention (soft-delete / versioned "
             "records), context noise 40\\%. These are exactly the corrections a seed model finds implausible.}",
             "\\label{tab:policy}", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(c[0] for c in cols) + "\\\\", "\\midrule"]
    for m in ms:
        row = [NAMES[m]]
        for _, c in cols:
            v = c[c.method == m].acc_policy
            row.append(f"{100 * v.mean():.1f}" if len(v) and v.notna().any() else "--")
        lines.append(" & ".join(row) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "policy.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote policy.tex")


def gpu_detail_table():
    g = load_gpu(["results/kaggle/main2"])
    if not len(g):
        return
    ms = ["frozen", "seed_only", "naive", "verified_only", "conf_filter", "vcfl", "instance_rel", "source_prior",
          "ctx_eb", "ca_eb", "oracle_clean"]
    cols = [("acc_seen", "seen"), ("acc_held", "held-out"), ("acc_policy", "convention"), ("gen_acc", "gen.")]
    lines = ["\\begin{table}[h]\\centering\\small",
             "\\caption{Qwen3-0.6B, synthetic benchmark, 40\\% noise, $\\rho=0.1$, 5 seeds: accuracy (\\%) on test "
             "requests from training families (seen), held-out families, convention requests, and greedy-generation exact "
             "match on the first 64 test requests (gen.).}\\label{tab:gpu-detail}",
             "\\begin{tabular}{l" + "rrrr" * 2 + "}", "\\toprule",
             " & \\multicolumn{4}{c}{source noise} & \\multicolumn{4}{c}{context noise}\\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Method & " + " & ".join([c[1] for c in cols] * 2) + "\\\\", "\\midrule"]
    for m in ms:
        row = [NAMES[m]]
        for nt in ["source", "context"]:
            sub = g[(g.noise_type == nt) & (g.method == m)]
            for c, _ in cols:
                row.append(f"{100 * sub[c].mean():.1f}" if len(sub) else "--")
        lines.append(" & ".join(row) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "gpu_detail.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote gpu_detail.tex")


def correction_table():
    s = pd.read_csv(os.path.join(ROOT, "results", "cpu_v2", "raw_results.csv"))
    s = s[np.isclose(s.noise, 0.4)]
    ms = ["source_prior", "ds_source", "crowd_seed", "ctx_crowd", "self_filter"]
    cols = [dict(noise_type=nt, verify_rate=vr) for nt in ["uniform", "source", "context"] for vr in [0.1, 0.25]]
    table(s, cols, ms, ["seed"], "Label-correction and self-filtering baselines, CPU learner, 40\\% noise, 8 seeds.",
          "tab:correction", [f"{c['noise_type']}, $\\rho$={c['verify_rate']:g}" for c in cols], "correction.tex",
          "$\\Delta$: paired difference to the source prior, $\\pm$95\\% CI.")


def active_table():
    """Which items to verify: paired difference of each selection strategy to random selection (same budget)."""
    f = os.path.join(ROOT, "results", "cpu_active", "raw_results.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    piv = d.pivot_table(index=["noise_type", "verify_rate", "method", "seed"], columns="strategy", values="acc")
    strats = [("balanced", "Cell-balanced"), ("cell_active", "Cell-uncertainty"), ("disagree", "Most implausible")]
    conds = [("source", 0.1), ("source", 0.25), ("context", 0.1), ("context", 0.25)]
    ms = ["verified_only", "source_prior", "ctx_eb", "ca_eb"]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Which corrections to verify (CPU learner, 40\\% noise, 8 seeds). Paired difference in accuracy "
             "(points, $\\pm$95\\% CI) of each selection strategy against random selection with the same budget, per "
             "condition and pooled over the four conditions (32 pairs). Bold: CI excludes 0. Last row: share of wrong "
             "feedback among the verified items (random: about 40\\%).}\\label{tab:active}",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\linewidth}{!}{\\begin{tabular}{ll" + "r" * (len(conds) + 1) + "}", "\\toprule",
             "Strategy & Method & " + " & ".join(f"{nt}, $\\rho$={vr:g}" for nt, vr in conds) + " & pooled\\\\",
             "\\midrule"]
    for st, sn in strats:
        for j, m in enumerate(ms):
            row = [sn if j == 0 else "", NAMES[m]]
            alld = []
            for nt, vr in conds:
                g = piv.xs((nt, vr, m), level=["noise_type", "verify_rate", "method"])
                x = (g[st] - g["random"]).to_numpy() * 100
                alld.append(x)
                row.append(fmt_diff((x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)))))
            x = np.concatenate(alld)
            row.append(fmt_diff((x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)))))
            lines.append(" & ".join(row) + "\\\\")
        fw = [100 * d[(d.strategy == st) & (d.noise_type == nt) & np.isclose(d.verify_rate, vr)].frac_wrong_verified.mean()
              for nt, vr in conds]
        lines.append(" & \\textit{wrong among verified} & " + " & ".join(f"{v:.0f}\\%" for v in fw) + " & \\\\")
        lines.append("\\midrule" if st != strats[-1][0] else "\\bottomrule")
    lines += ["\\end{tabular}}", "\\end{table}"]
    open(os.path.join(OUT, "active.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote active.tex")


if __name__ == "__main__":
    active_table()
    gpu_detail_table()
    correction_table()
    policy_table()
    main()
