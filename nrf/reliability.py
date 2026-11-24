"""Feedback-weighting methods. All take the *observable* adapt view plus seed-model probabilities.

Every method returns (labels, weights) for the adapt pool. Verified items get the verifier label with
weight 1 in every method except `naive` (which learns all feedback as given). Unverified items keep the
feedback label and get w = f(q) = q**gamma * 1[q >= tau], with q the method's reliability estimate, so the
reliability methods differ ONLY in how q is estimated (fair comparison).

  naive               learn all feedback as given
  verified_only       only verified items
  verifier_corrected  verified -> verifier label, unverified kept with weight 1
  conf_filter         q = p_seed(y_fb | x)
  source_prior        q = Beta-smoothed accuracy of the source on verified items           (strong baseline)
  vcfl                q = logistic calibration of [p_seed, source_prior] fit on verified items
  ctx_prior           q = hierarchical (source x domain) reliability, shrunk to an additive
                          logit(source) + logit(domain) prior (no seed model)                 (ablation)
  ca_vcfl             q = logistic calibration of [p_seed, source_prior, ctx_prior]          (proposed)
  oracle_clean        uses hidden truth: drop wrong feedback. REFERENCE UPPER BOUND ONLY.

Features of verified items used to fit the calibrator are leave-one-out (an item's own verifier result never
enters its own feature), so the calibrator is fit on honest out-of-sample features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

METHODS = ["naive", "verified_only", "verifier_corrected", "conf_filter", "source_prior", "vcfl",
           "ctx_prior", "ca_vcfl", "instance_rel", "ds_source", "crowd_seed", "ctx_crowd", "ctx_eb", "ca_eb", "agree_eb", "ca_agree_eb",
           "mace", "rgpo_style"]  # mace/rgpo_style: nrf/baselines.py (Q1 E4)
REFERENCE = ["oracle_clean"]
EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _group_loo_rate(keys, correct, verified, prior, m):
    """Beta-smoothed accuracy per group with leave-one-out for verified rows.
    prior: per-row prior mean (array)."""
    keys = pd.Series(keys).astype(str).to_numpy()
    c = pd.Series(np.where(verified, correct, 0)).groupby(keys).transform("sum").to_numpy().astype(float)
    n = pd.Series(verified.astype(float)).groupby(keys).transform("sum").to_numpy()
    c = c - np.where(verified, correct, 0)
    n = n - verified.astype(float)
    return (c + m * prior) / (n + m)


def reliability_features(obs: pd.DataFrame, m_src: float = 5.0, m_dom: float = 5.0, m_ctx: float = 5.0):
    ver = obs.verified.to_numpy().astype(bool)
    corr = (obs.y_fb.to_numpy() == obs.y_ver.to_numpy()) & ver
    n_ver = ver.sum()
    mu = (corr.sum() + 1.0) / (n_ver + 2.0)  # global verified accuracy, Laplace
    mu_row = np.full(len(obs), mu)
    q_src = _group_loo_rate(obs.source.to_numpy(), corr, ver, mu_row, m_src)
    q_dom = _group_loo_rate(obs.domain.to_numpy(), corr, ver, mu_row, m_dom)
    # additive-logit prior for the (source, domain) cell, then shrink the cell's own evidence toward it
    prior_ctx = _sigmoid(_logit(q_src) + _logit(q_dom) - _logit(mu_row))
    cell = obs.source.astype(str) + "|" + obs.domain.astype(str)
    q_ctx = _group_loo_rate(cell.to_numpy(), corr, ver, prior_ctx, m_ctx)
    return pd.DataFrame(dict(q_src=q_src, q_dom=q_dom, q_ctx=q_ctx), index=obs.index), corr


M_GRID = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 512, 4096], float)


def _betabinom_ll(c, n, mu, m):
    from scipy.special import betaln
    a, b = m * mu, m * (1 - mu)
    return float(np.sum(betaln(c + a, n - c + b) - betaln(a, b)))


def _eb_strength(keys, corr, ver, prior_mean):
    """Empirical-Bayes shrinkage strength: argmax_m of the beta-binomial marginal likelihood of the verified
    correct/incorrect counts per group, given each group's prior mean. Large m <=> no group-level variation."""
    df = pd.DataFrame(dict(k=keys, c=np.where(ver, corr, 0).astype(float), n=ver.astype(float), mu=prior_mean))
    g = df.groupby("k").agg(c=("c", "sum"), n=("n", "sum"), mu=("mu", "mean"))
    g = g[g.n > 0]
    if len(g) == 0:
        return float(M_GRID[-1])
    mu = np.clip(g.mu.to_numpy(), 1e-3, 1 - 1e-3)
    ll = [_betabinom_ll(g.c.to_numpy(), g.n.to_numpy(), mu, m) for m in M_GRID]
    return float(M_GRID[int(np.argmax(ll))])


def eb_context_reliability(obs):
    """Proposed: hierarchical source -> (source, domain) reliability whose shrinkage strengths are chosen by
    empirical Bayes on the verified items, so it reduces to the global source prior when the verifier evidence
    shows no context dependence. Returns (q, info)."""
    ver = obs.verified.to_numpy().astype(bool)
    corr = (obs.y_fb.to_numpy() == obs.y_ver.to_numpy()) & ver
    mu = (corr.sum() + 1.0) / (ver.sum() + 2.0)
    mu_row = np.full(len(obs), mu)
    src = obs.source.astype(str).to_numpy()
    dom = obs.domain.astype(str).to_numpy()
    cell = np.char.add(np.char.add(src, "|"), dom)
    m_src = _eb_strength(src, corr, ver, mu_row)
    m_dom = _eb_strength(dom, corr, ver, mu_row)
    q_src = _group_loo_rate(src, corr, ver, mu_row, m_src)
    q_dom = _group_loo_rate(dom, corr, ver, mu_row, m_dom)
    prior_ctx = _sigmoid(_logit(q_src) + _logit(q_dom) - _logit(mu_row))
    m_ctx = _eb_strength(cell, corr, ver, prior_ctx)
    q = _group_loo_rate(cell, corr, ver, prior_ctx, m_ctx)
    return q, dict(m_src=m_src, m_dom=m_dom, m_ctx=m_ctx)


def _agreement_feature(obs, p_seed_fb, m=5.0):
    """LOO mean of p_seed(y_fb|x) within the (source, context) cell, shrunk toward the source mean (unsupervised)."""
    p = np.asarray(p_seed_fb, float)
    src = obs.source.astype(str).to_numpy()
    cell = (obs.source.astype(str) + "|" + obs.domain.astype(str)).to_numpy()
    ser = pd.Series(p)
    s_sum = ser.groupby(src).transform("sum").to_numpy() - p
    s_n = ser.groupby(src).transform("size").to_numpy() - 1
    s_mean = s_sum / np.maximum(s_n, 1)
    c_sum = ser.groupby(cell).transform("sum").to_numpy() - p
    c_n = ser.groupby(cell).transform("size").to_numpy() - 1
    return (c_sum + m * s_mean) / (c_n + m)


def _calibrate(X, ver, corr):
    """Fit P(feedback correct | X) on verified rows; predict for all rows."""
    Xl = _logit(X)
    if ver.sum() < 12 or len(np.unique(corr[ver])) < 2:
        return _sigmoid(Xl.mean(1))
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(Xl[ver], corr[ver].astype(int))
    return clf.predict_proba(Xl)[:, 1]


def estimate_q(method: str, obs: pd.DataFrame, p_seed_fb: np.ndarray, **kw) -> np.ndarray | None:
    feats, corr = reliability_features(obs, **{k: v for k, v in kw.items() if k.startswith("m_")})
    ver = obs.verified.to_numpy().astype(bool)
    if method == "conf_filter":
        return np.asarray(p_seed_fb)
    if method == "source_prior":
        return feats.q_src.to_numpy()
    if method == "ctx_prior":
        return feats.q_ctx.to_numpy()
    if method == "vcfl":
        return _calibrate(np.c_[p_seed_fb, feats.q_src], ver, corr)
    if method == "ca_vcfl":
        return _calibrate(np.c_[p_seed_fb, feats.q_src, feats.q_ctx], ver, corr)
    if method == "ca_eb":  # proposed full: calibrated [seed confidence, EB context reliability]
        q_eb, _ = eb_context_reliability(obs)
        return _calibrate(np.c_[p_seed_fb, q_eb], ver, corr)
    if method in ("agree_eb", "ca_agree_eb"):
        # cell-level agreement between the source's corrections and the agent's own model, computed on ALL items of
        # the (source, context) cell (no labels needed, leave-one-out), shrunk to the source-level agreement;
        # a calibrator fit on verified items maps it (with the EB reliability) to P(correct).
        q_eb, _ = eb_context_reliability(obs)
        a_cell = _agreement_feature(obs, p_seed_fb)
        X = np.c_[q_eb, a_cell] if method == "agree_eb" else np.c_[p_seed_fb, q_eb, a_cell]
        return _calibrate(X, ver, corr)
    return None


def _instance_rel(obs, C=1.0):
    """Li et al. (2019)-style instance-level reliability, supervised by verified items:
    P(feedback correct | source, domain, source x domain, source x text n-grams), L2-regularised logistic."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder

    ver = obs.verified.to_numpy().astype(bool)
    corr = (obs.y_fb.to_numpy() == obs.y_ver.to_numpy()) & ver
    if ver.sum() < 12 or len(np.unique(corr[ver])) < 2:
        return np.full(len(obs), 0.5)
    cat = pd.DataFrame(dict(s=obs.source.astype(str), d=obs.domain.astype(str),
                            sd=obs.source.astype(str) + "|" + obs.domain.astype(str)))
    oh = OneHotEncoder(handle_unknown="ignore").fit_transform(cat)
    tf = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True).fit_transform(obs.text)
    # source-specific text weights: one block per source
    src = obs.source.to_numpy()
    blocks = [tf.multiply(csr_matrix((src == s).astype(float)).T) for s in np.unique(src)]
    X = hstack([oh, tf] + blocks).tocsr()
    # cross-fitted predictions for verified items (honest), full fit for unverified
    q = np.zeros(len(obs))
    vi = np.where(ver)[0]
    rng = np.random.default_rng(0)
    folds = rng.integers(5, size=len(vi))
    for k in range(5):
        tr, te = vi[folds != k], vi[folds == k]
        if len(np.unique(corr[tr])) < 2:
            q[te] = corr[tr].mean()
            continue
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], corr[tr])
        q[te] = clf.predict_proba(X[te])[:, 1]
    clf = LogisticRegression(C=C, max_iter=2000).fit(X[vi], corr[vi])
    q[~ver] = clf.predict_proba(X[~ver])[:, 1]
    return q


def crowd_posterior(obs, prior, context=False, m_ctx=5.0, alpha_diag=4.0, alpha_off=0.5, iters=30, K=4):
    """Semi-supervised Dawid-Skene / learning-from-crowds (Raykar et al. 2010) with one feedback label per item.
    prior: [N,K] prior over the true label (class prior -> Dawid-Skene; seed model -> Raykar-style).
    Confusion pi[s][t, j] = P(feedback j | true t, source s); verified items are clamped to their verifier label.
    context=True: per (source, domain) confusion, shrunk toward the source confusion with strength m_ctx
    (the proposed context-conditioned label correction). Returns posterior [N,K]."""
    ver = obs.verified.to_numpy().astype(bool)
    yv = obs.y_ver.to_numpy()
    yf = obs.y_fb.to_numpy()
    src = obs.source.astype(str).to_numpy()
    cell = (obs.source.astype(str) + "|" + obs.domain.astype(str)).to_numpy()
    n = len(obs)
    post = np.asarray(prior, float).copy()
    post[ver] = np.eye(K)[yv[ver]]
    A = np.full((K, K), alpha_off) + np.eye(K) * (alpha_diag - alpha_off)
    for _ in range(iters):
        # M-step: source confusion
        pis = {}
        for s in np.unique(src):
            m = src == s
            cnt = A.copy()
            np.add.at(cnt.T, yf[m], post[m])  # cnt[t, j] += post_i(t) for items with feedback j
            pis[s] = cnt / cnt.sum(1, keepdims=True)
        if context:
            pic = {}
            for c in np.unique(cell):
                m = cell == c
                s = src[m][0]
                cnt = m_ctx * pis[s]
                cnt = cnt.copy()
                np.add.at(cnt.T, yf[m], post[m])
                pic[c] = cnt / cnt.sum(1, keepdims=True)
            lik = np.stack([pic[c][:, j] for c, j in zip(cell, yf)])
        else:
            lik = np.stack([pis[s][:, j] for s, j in zip(src, yf)])
        new = prior * lik
        new /= new.sum(1, keepdims=True)
        new[ver] = np.eye(K)[yv[ver]]
        if np.abs(new - post).max() < 1e-5:
            post = new
            break
        post = new
    return post


CORRECTION = ["ds_source", "crowd_seed", "ctx_crowd"]


def method_weights(method: str, obs: pd.DataFrame, p_seed: np.ndarray, tau: float = 0.5,
                   gamma: float = 2.0, y_true_for_oracle: np.ndarray | None = None, **kw):
    """p_seed: [N,4] seed-model probabilities over the 4 tools (or [N] = p_seed(y_fb|x), then correction
    methods are unavailable). Returns labels, weights, q (q may be None)."""
    ver = obs.verified.to_numpy().astype(bool)
    y_fb = obs.y_fb.to_numpy()
    y_cor = np.where(ver, obs.y_ver.to_numpy(), y_fb)
    n = len(obs)
    p_seed = np.asarray(p_seed, float)
    p_seed_fb = p_seed[np.arange(n), y_fb] if p_seed.ndim == 2 else p_seed
    if method in ("mace", "rgpo_style"):
        from .baselines import mace_weights, rgpo_style_weights
        return (mace_weights if method == "mace" else rgpo_style_weights)(obs, tau, gamma)
    if method in CORRECTION:
        assert p_seed.ndim == 2, "correction methods need the full seed distribution"
        if method == "ds_source":
            cp = np.bincount(y_cor, minlength=4) + 1.0
            prior = np.tile(cp / cp.sum(), (n, 1))
        else:
            prior = np.clip(p_seed, 1e-3, 1)
            prior /= prior.sum(1, keepdims=True)
        post = crowd_posterior(obs, prior, context=(method == "ctx_crowd"))
        lab = post.argmax(1)
        q = post.max(1)
        w = np.where(q >= tau, q ** gamma, 0.0)
        w = np.where(ver, 1.0, w)
        return np.where(ver, y_cor, lab), w, q
    if method in ("instance_rel", "ctx_eb"):
        q = _instance_rel(obs) if method == "instance_rel" else eb_context_reliability(obs)[0]
        w = np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0))
        return y_cor, w, q
    if method == "naive":
        return y_fb, np.ones(n), None
    if method == "verified_only":
        return y_cor, ver.astype(float), None
    if method == "verifier_corrected":
        return y_cor, np.ones(n), None
    if method == "oracle_clean":
        assert y_true_for_oracle is not None
        return y_fb, (y_fb == y_true_for_oracle).astype(float), None
    q = estimate_q(method, obs, p_seed_fb, **kw)
    if q is None:
        raise ValueError(method)
    w = np.where(q >= tau, q ** gamma, 0.0)
    w = np.where(ver, 1.0, w)
    return y_cor, w, q


def weight_diagnostics(obs, y_true, labels, w, q):
    """Diagnostics computed AFTER weighting, for reporting only (uses hidden truth)."""
    unv = ~obs.verified.to_numpy().astype(bool)
    wrong = (labels != y_true)
    out = dict(
        n_train_eff=float(w.sum()),
        accepted_noisy_rate=float(((w > 0) & wrong & unv).sum() / max(1, (wrong & unv).sum())),
        accepted_clean_rate=float(((w > 0) & ~wrong & unv).sum() / max(1, (~wrong & unv).sum())),
        weighted_label_precision=float((w * ~wrong).sum() / max(EPS, w.sum())),
    )
    if q is not None:
        tgt = (~wrong[unv]).astype(float)
        qq = np.clip(q[unv], 0, 1)
        out["brier_unverified"] = float(((qq - tgt) ** 2).mean())
        out["ece_unverified"] = float(_ece(qq, tgt))
    return out


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
