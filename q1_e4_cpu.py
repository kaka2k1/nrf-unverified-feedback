"""PLAN_Q1 E4: noisy-label / crowdsourcing baselines on the SAME CPU break-even testbeds (same seeds, same rho grid).

Synthetic: identical to breakeven_cpu.py (make_benchmark(seed=1000+s), 3 learners, 4 noise conditions, 20 seeds,
rho in {0.02, 0.05, 0.1, 0.25}). BFCL real feedback: identical to bfcl_breakeven_cpu.py (bge_small, view names, C=1000).
Besides the new baselines we RE-RUN verified_only / source_prior / ctx_eb / ca_eb, so that (i) the paired comparison
is computed inside one run and (ii) reproduction of the 18/11 numbers can be checked.

New baselines (see nrf/baselines.py; already-existing crowd baselines are run here for the first time on this testbed):
  ds_source   semi-supervised Dawid-Skene (class prior), label correction          (nrf/reliability.py)
  crowd_seed  Raykar et al. 2010 learning-from-crowds (seed-model prior)            (nrf/reliability.py)
  ctx_crowd   Dawid-Skene with (source x context) confusion, shrunk to source       (nrf/reliability.py)
  mace        MACE, semi-supervised                                                 (nrf/baselines.py)
  rgpo_style  RGPO reliability component: per-annotator scalar from DS-EM           (nrf/baselines.py)
  small_loss  cross-fitted small-loss selection, forget rate estimated on verified  (nrf/baselines.py)
  coteach     co-teaching-style exchange between learners on disjoint halves, R=2   (nrf/baselines.py)

python q1_e4_cpu.py --setup synth --learners tfidf bge_small bge_base --seeds 20 --out results/q1_e4/synth
python q1_e4_cpu.py --setup bfcl --seeds 20 --out results/q1_e4/bfcl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from breakeven_cpu import CONDS, EMB, RHO_CURVE, embed_texts, featurize, fit, proba4, self_train_weights, stack
from nrf import make_benchmark, method_weights, observable_view
from nrf.baselines import small_loss_select

RHOS = [0.02, 0.05, 0.1, 0.25]
REPRO = ["verified_only", "naive", "verifier_corrected", "conf_filter", "source_prior", "ctx_eb", "ca_eb", "self_train",
         "oracle_clean"]  # everything of the 18/11 run, re-run so that all paired comparisons come from ONE run
NEW = ["ds_source", "crowd_seed", "ctx_crowd", "mace", "rgpo_style", "small_loss", "coteach"]
TAU, GAMMA = 0.5, 2.0


def synth_job(job):
    kind, nt, nz, s, methods = job
    rows = []
    for rho in RHO_CURVE:
        b = make_benchmark(seed=1000 + s, noise=nz, noise_type=nt, verify_rate=rho)
        seed_df, adapt, test = b["seed"], b["adapt"], b["test"]
        obs = observable_view(adapt)
        Xs, Xa, Xt = featurize(kind, seed_df, adapt, test)
        ys, yt = seed_df.y_true.to_numpy(), test.y_true.to_numpy()
        y_true_a = adapt.y_true.to_numpy()
        unv = ~obs.verified.to_numpy().astype(bool)
        p_seed = proba4(fit(kind, Xs, ys), Xa)

        def fit_score(tr, ytr, wtr, ev, yev):
            clf = fit(kind, stack(Xs, Xa[tr]), np.r_[ys, ytr], np.r_[np.ones(len(ys)), wtr])
            return proba4(clf, Xa[ev])[np.arange(len(ev)), yev]

        if rho == 0.0:
            ok = fit(kind, Xs, ys).predict(Xt) == yt
            rows.append(dict(learner=kind, noise_type=nt, noise=nz, seed=s, rho=0.0, method="seed_only", n_ver=0,
                             acc=float(ok.mean())))
        for m in (methods if rho in RHOS else ["verified_only"]):
            t0 = time.time()
            if m in ("small_loss", "coteach"):
                lab, w, q = small_loss_select(obs, fit_score, rounds=0 if m == "small_loss" else 2, rng_seed=s)
            elif m == "self_train":
                lab, w, q = self_train_weights(kind, Xs, ys, Xa, obs, TAU, GAMMA)
            else:
                lab, w, q = method_weights(m, obs, p_seed, tau=TAU, gamma=GAMMA,
                                           y_true_for_oracle=y_true_a if m == "oracle_clean" else None)
            keep = w > 0
            clf = fit(kind, stack(Xs, Xa[keep]), np.r_[ys, lab[keep]], np.r_[np.ones(len(ys)), w[keep]])
            ok = clf.predict(Xt) == yt
            wu, cu = w[unv], (lab == y_true_a)[unv]
            rows.append(dict(learner=kind, noise_type=nt, noise=nz, seed=s, rho=rho, method=m,
                             n_ver=int((~unv).sum()), acc=float(ok.mean()),
                             acc_seen=float(ok[test.split.to_numpy() == "seen"].mean()),
                             acc_held=float(ok[test.split.to_numpy() == "held"].mean()),
                             w_unv=float(wu.sum()), prec_unv=float((wu * cu).sum() / max(wu.sum(), 1e-9)),
                             realized_noise=b["meta"]["realized_noise"], sec=time.time() - t0))
    return rows


def _init(emb):
    EMB.update(emb)
    os.environ["OMP_NUM_THREADS"] = "1"


def run_synth(a, methods):
    texts = set()
    for s in range(a.seeds):
        b = make_benchmark(seed=1000 + s, noise=0.4, noise_type="context", verify_rate=0.1)
        texts |= set(b["seed"].text) | set(b["adapt"].text) | set(b["test"].text)
    emb = {k: embed_texts(sorted(texts), k, a.cache) for k in a.learners if k != "tfidf"}
    if a.embed_only:
        return []
    jobs = [(k, nt, nz, s, methods) for k in a.learners for nt, nz in CONDS for s in range(a.seeds)]
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init, initargs=(emb,)) as pool:
        for i, r in enumerate(pool.imap_unordered(synth_job, jobs)):
            rows += r
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    return rows


def run_bfcl(a, methods):
    from bfcl_breakeven_cpu import SOURCES, PairLearner, make_split
    items = [json.loads(line) for line in open(a.items)]
    ids = [it["id"] for it in items]
    preds = {}
    for s in SOURCES:
        d = {json.loads(line)["id"]: json.loads(line)["pred"] for line in open(os.path.join(a.annot_dir, f"annot_{s}.jsonl"))}
        preds[s] = np.array([d[i] for i in ids])
    texts = {it["query"] for it in items} | {PairLearner.ctext(c, "names") for it in items for c in it["candidates"]}
    E = embed_texts(sorted(texts), "bge_small", a.cache)
    L = PairLearner(items, E, "names", 1000.0)
    gold = np.array([it["gold"] for it in items])
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        for rho in RHO_CURVE:
            te, sd, ad, adapt = make_split(items, preds, s, 300, 64, rho)
            ver = adapt.verified.to_numpy()
            unv = ~ver
            obs = observable_view(adapt)
            obs["n_cand"] = [len(items[i]["candidates"]) for i in ad]
            seed = PairLearner.fit(L, sd, gold[sd], np.ones(len(sd)))
            p_seed_fb = np.array([p[y] for p, y in zip(seed.proba(ad), adapt.y_fb)])

            def fit_score(tr, ytr, wtr, ev, yev):
                L.fit(np.r_[sd, ad[tr]], np.r_[gold[sd], ytr], np.r_[np.ones(len(sd)), wtr])
                return np.array([p[y] for p, y in zip(L.proba(ad[ev]), yev)])

            if rho == 0.0:
                pred = np.array([int(np.argmax(p)) for p in seed.proba(te)])
                rows.append(dict(learner="bge_small", view="names", seed=s, rho=0.0, method="seed_only", n_ver=0,
                                 acc=float((pred == gold[te]).mean())))
            for m in (methods if rho in RHOS else ["verified_only"]):
                if m in ("small_loss", "coteach"):
                    lab, w, _ = small_loss_select(obs, fit_score, rounds=0 if m == "small_loss" else 2, rng_seed=s)
                elif m == "self_train":  # identical to bfcl_breakeven_cpu.py
                    lab0 = np.where(ver, obs.y_ver, 0)
                    L.fit(np.r_[sd, ad[ver]], np.r_[gold[sd], lab0[ver]], np.ones(len(sd) + ver.sum()))
                    pp = L.proba(ad)
                    q = np.array([p.max() for p in pp])
                    lab = np.where(ver, obs.y_ver, [int(np.argmax(p)) for p in pp])
                    w = np.where(ver, 1.0, np.where(q >= TAU, q ** GAMMA, 0.0))
                elif m in ("ds_source", "crowd_seed", "ctx_crowd"):
                    continue  # full confusion matrices need a fixed label space; BFCL candidate sets vary per item
                else:
                    lab, w, _ = method_weights(m, obs, p_seed_fb, tau=TAU, gamma=GAMMA,
                                               y_true_for_oracle=adapt.y_true.to_numpy() if m == "oracle_clean"
                                               else None)
                lab, w = np.asarray(lab), np.asarray(w, float)
                L.fit(np.r_[sd, ad], np.r_[gold[sd], lab], np.r_[np.ones(len(sd)), w])
                pred = np.array([int(np.argmax(p)) for p in L.proba(te)])
                wu, cu = w[unv], (lab == adapt.y_true.to_numpy())[unv]
                rows.append(dict(learner="bge_small", view="names", seed=s, rho=rho, method=m, n_ver=int(ver.sum()),
                                 acc=float((pred == gold[te]).mean()), w_unv=float(wu.sum()),
                                 prec_unv=float((wu * cu).sum() / max(wu.sum(), 1e-9)),
                                 realized_noise=float((adapt.y_fb != adapt.y_true).mean())))
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["synth", "bfcl"], required=True)
    ap.add_argument("--items", default="data/bfcl_items.jsonl", help="BFCL-format selection items")
    ap.add_argument("--annot-dir", default="data/bfcl_annot")
    ap.add_argument("--learners", nargs="+", default=["tfidf", "bge_small", "bge_base"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=REPRO + NEW)
    ap.add_argument("--cache", default="results/q1_e4/cache")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    rows = (run_synth if a.setup == "synth" else run_bfcl)(a, a.methods)
    if a.embed_only:
        return
    pd.DataFrame(rows).to_csv(f"{a.out}/raw_results.csv", index=False)
    json.dump(dict(args=vars(a), rhos=RHOS, rho_curve=RHO_CURVE, tau=TAU, gamma=GAMMA, sec=time.time() - t0),
              open(f"{a.out}/run_meta.json", "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
