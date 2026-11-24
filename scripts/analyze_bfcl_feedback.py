"""Go/no-go test on REAL LLM feedback: is feedback reliability context-dependent, and do reliability estimators
pick better training labels? (CPU only, no learner training.)

python scripts/analyze_bfcl_feedback.py --annot-dir results/kaggle/annot1 --out results/bfcl_feedback_analysis.md
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrf.reliability import eb_context_reliability, method_weights, weight_diagnostics  # noqa: E402


def loglik(X, y):
    clf = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/bfcl_items.jsonl")
    ap.add_argument("--annot-dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=["Qwen3-0.6B"], help="learner model, not a feedback source")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.items)]
    ids = [it["id"] for it in items]
    gold = np.array([it["gold"] for it in items])
    ctx = np.array([it["context"] for it in items])
    files = sorted(glob.glob(os.path.join(a.annot_dir, "**", "annot_*.jsonl"), recursive=True))
    preds = {}
    for f in files:
        s = os.path.basename(f)[6:-6]
        d = {json.loads(line)["id"]: json.loads(line)["pred"] for line in open(f)}
        if all(i in d for i in ids):
            preds[s] = np.array([d[i] for i in ids])
    out = []
    terms = {it["context"]: ", ".join(it["context_terms"][:3]) for it in items}
    tab = pd.DataFrame({s: [(p == gold)[ctx == c].mean() for c in sorted(terms)] for s, p in preds.items()},
                       index=[f"{c}: {terms[c]} (n={int((ctx == c).sum())})" for c in sorted(terms)])
    tab.loc["ALL"] = [(p == gold).mean() for p in preds.values()]
    out.append("## Accuracy of each feedback source by context (%)\n\n" + (100 * tab).round(1).to_markdown())

    sources = [s for s in preds if s not in a.exclude]
    # likelihood-ratio test: correct ~ source  vs  correct ~ source + source x context (all items, gold known)
    rows = [(s, c, int(pc == g)) for s in sources for pc, g, c in zip(preds[s], gold, ctx)]
    df = pd.DataFrame(rows, columns=["s", "c", "y"])
    df["sc"] = df.s + "|" + df.c.astype(str)
    oh_s = OneHotEncoder().fit_transform(df[["s"]])
    oh_sc = OneHotEncoder().fit_transform(df[["s", "c", "sc"]])
    l0, k0 = loglik(oh_s, df.y.to_numpy())
    l1, k1 = loglik(oh_sc, df.y.to_numpy())
    lr = 2 * (l1 - l0)
    out.append(f"\n## Is reliability context-dependent? (sources = {sources})\n\nLikelihood-ratio test, "
               f"correct ~ source vs ~ source + context + source×context: LR = {lr:.1f}, df ≈ {k1 - k0}, "
               f"p = {chi2.sf(lr, k1 - k0):.2e}. Overall feedback error (uniform source mix) = "
               f"{1 - np.mean([(preds[s] == gold).mean() for s in sources]):.3f}.")
    spread = (tab[sources].iloc[:-1].max() - tab[sources].iloc[:-1].min())
    out.append("Per-source accuracy range across contexts (max−min, pts): "
               + ", ".join(f"{s}: {100 * v:.1f}" for s, v in spread.items()))

    # Compare the saturated cell model against additive source and context effects.
    additive = OneHotEncoder().fit_transform(df[["s", "c"]])
    l_add, _ = loglik(additive, df.y.to_numpy())
    lr_int = max(0.0, 2 * (l1 - l_add))
    df_int = (len(sources) - 1) * (len(terms) - 1)
    out.append(f"Interaction conditional on both main effects: LR = {lr_int:.1f}, "
               f"df = {df_int}, nominal p = {chi2.sf(lr_int, df_int):.2e}. "
               "Chi-square p-values are approximate and treat repeated annotations of an item as independent.")

    # label-quality simulation at verify 10/25%
    res = []
    for vr in [0.1, 0.25]:
        for r in range(a.reps):
            rng = np.random.default_rng(r)
            src = rng.integers(len(sources), size=len(items))
            y_fb = np.array([preds[sources[s]][i] for i, s in enumerate(src)])
            ver = rng.random(len(items)) < vr
            obs = pd.DataFrame(dict(source=src, domain=ctx, y_fb=y_fb, verified=ver,
                                    y_ver=np.where(ver, gold, -1), text=[it["query"] for it in items]))
            for m in ["naive", "verifier_corrected", "source_prior", "instance_rel", "ctx_prior", "ctx_eb"]:
                lab, w, q = method_weights(m, obs, np.full(len(items), 0.5))
                dg = weight_diagnostics(obs, gold, lab, w, q)
                res.append(dict(vr=vr, rep=r, method=m, **dg))
            res[-1]["m_ctx"] = eb_context_reliability(obs)[1]["m_ctx"]
    R = pd.DataFrame(res)
    g = R.groupby(["vr", "method"]).agg(label_precision=("weighted_label_precision", "mean"),
                                         lp_sd=("weighted_label_precision", "std"),
                                         accepted_noisy=("accepted_noisy_rate", "mean"),
                                         accepted_clean=("accepted_clean_rate", "mean"),
                                         n_eff=("n_train_eff", "mean"), brier=("brier_unverified", "mean"))
    out.append(f"\n## Weighted label precision of the training set ({a.reps} random source assignments)\n\n"
               + g.round(4).to_markdown())
    piv = R.pivot_table(index=["vr", "rep"], columns="method", values="weighted_label_precision")
    for vr in [0.1, 0.25]:
        d = (piv.loc[vr, "ctx_eb"] - piv.loc[vr, "source_prior"]) * 100
        out.append(f"verify {vr}: ctx_eb − source_prior label precision = {d.mean():+.2f} ± {1.96 * d.std() / np.sqrt(len(d)):.2f} pts; "
                   f"median EB m_ctx = {R[(R.vr == vr) & R.m_ctx.notna()].m_ctx.median()}")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        open(a.out, "w").write("# Real LLM feedback on BFCL — go/no-go analysis (auto-generated)\n\n" + txt + "\n")


if __name__ == "__main__":
    main()
