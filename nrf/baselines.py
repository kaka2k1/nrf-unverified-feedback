"""Noisy-label / crowdsourcing baselines added for the Q1 revision (PLAN_Q1 E4).

All methods use only the OBSERVABLE view (feedback label, source, context, verifier label where verified) and are
covered by tests/test_nrf.py::test_no_leakage_from_unverified_truth.

  mace          MACE (Hovy et al., 2013) adapted to one feedback label per item and semi-supervised with the sparse
                verifier labels: each source s is "competent" with prob theta_s (then it reports the true label) and
                otherwise "spams" from a source-level distribution xi_s. MAP-EM with Beta/Dirichlet pseudo-counts;
                verified items are clamped to their verifier label. Output: posterior label correction (like ds_source).
  rgpo_style    the reliability component of RGPO (Huang et al., ICML 2026): one scalar per annotator = mean diagonal
                of its Dawid-Skene confusion matrix (EM, here semi-supervised by the verifier), used to weight the
                (uncorrected) feedback. The GRPO policy-optimisation part is NOT reproduced (our learners are
                classifiers); this isolates "global per-annotator reliability estimated by EM".
  small_loss    cross-fitted small-loss selection (Han et al. 2018 / confident-learning style) for convex CPU learners:
                the unverified pool is split in two halves; a learner trained on seed + verified + half A scores the
                feedback labels of half B (and vice versa); the (1 - eps_hat) fraction with the smallest loss is kept,
                eps_hat = feedback error rate estimated on the verified items (Laplace).
  coteach       co-teaching-style exchange (Han et al. 2018) adapted to convex learners: as small_loss, then for R
                rounds the learner trained on the samples SELECTED in half A picks the small-loss samples of half B and
                vice versa (each model selects data for its peer). With logistic regression two networks with
                different initialisations would coincide, so the two "peers" are learners on disjoint halves.

small_loss / coteach need the learner, so they take a callback rather than seed-model probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reliability import crowd_posterior


def _class_prior(obs, K):
    """Class prior from the corrected labels; if obs has `n_cand` (variable candidate sets, BFCL) the prior is uniform
    over each item's own candidates instead (label indices are not comparable across items)."""
    if "n_cand" in obs:
        nc = obs.n_cand.to_numpy()
        P = (np.arange(K)[None, :] < nc[:, None]).astype(float)
        return P / P.sum(1, keepdims=True)
    ver = obs.verified.to_numpy().astype(bool)
    y_cor = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
    cp = np.bincount(y_cor, minlength=K)[:K] + 1.0
    return np.tile(cp / cp.sum(), (len(obs), 1))


def _K(obs, K):
    return int(obs.n_cand.max()) if "n_cand" in obs else K


def mace_posterior(obs, prior, a_comp=(2.0, 1.0), b_spam=1.0, iters=50, K=4):
    """Semi-supervised MACE with one label per item. Returns (posterior [N,K], theta per source)."""
    ver = obs.verified.to_numpy().astype(bool)
    yv = obs.y_ver.to_numpy()
    yf = obs.y_fb.to_numpy()
    src = obs.source.astype(str).to_numpy()
    srcs = np.unique(src)
    sidx = np.searchsorted(srcs, src)
    S = len(srcs)
    theta = np.full(S, 0.7)
    xi = np.full((S, K), 1.0 / K)
    eyeK = np.eye(K)
    post = np.asarray(prior, float).copy()
    for _ in range(iters):
        # E-step: joint over (T=t, spam) given the observed label j
        th = theta[sidx][:, None]
        lik_comp = th * eyeK[yf]                       # [N,K]: theta * 1[t == j]
        lik_spam = (1 - th) * xi[sidx, yf][:, None]    # [N,1] broadcast over t
        joint_c = prior * lik_comp
        joint_s = prior * lik_spam
        z = joint_c.sum(1) + joint_s.sum(1)
        pc = joint_c / z[:, None]
        ps = joint_s / z[:, None]
        if ver.any():  # clamp verified: T known
            t = yv[ver]
            jc = theta[sidx[ver]] * (t == yf[ver])
            js = (1 - theta[sidx[ver]]) * xi[sidx[ver], yf[ver]]
            pr_c = jc / (jc + js)
            pc[ver] = eyeK[t] * pr_c[:, None]
            ps[ver] = eyeK[t] * (1 - pr_c)[:, None]
        new = pc + ps
        e_comp = pc.sum(1)
        e_spam = ps.sum(1)
        # M-step (MAP with Beta(a_comp) on theta, symmetric Dirichlet(b_spam) on xi)
        num = np.bincount(sidx, weights=e_comp, minlength=S) + a_comp[0] - 1
        den = np.bincount(sidx, minlength=S) + a_comp[0] + a_comp[1] - 2
        theta = np.clip(num / den, 1e-3, 1 - 1e-3)
        xi = np.full((S, K), b_spam)
        np.add.at(xi, (sidx, yf), e_spam)
        xi /= xi.sum(1, keepdims=True)
        if np.abs(new - post).max() < 1e-6:
            post = new
            break
        post = new
    return post, dict(zip(srcs, theta))


def mace_weights(obs, tau, gamma, K=4):
    K = _K(obs, K)
    ver = obs.verified.to_numpy().astype(bool)
    y_cor = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
    post, _ = mace_posterior(obs, _class_prior(obs, K), K=K)
    lab = post.argmax(1)
    q = post.max(1)
    w = np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0))
    return np.where(ver, y_cor, lab), w, q


def rgpo_style_weights(obs, tau, gamma, K=4):
    """Per-source scalar reliability = mean diagonal of the (semi-supervised) Dawid-Skene confusion; feedback label
    kept (no correction), weight f(r_s) as for every reliability method.
    With variable candidate sets (obs.n_cand, BFCL) a per-position confusion matrix is meaningless, so the one-coin
    version is used: r_s = MACE competence theta_s (semi-supervised EM)."""
    if "n_cand" in obs:
        K = _K(obs, K)
        ver = obs.verified.to_numpy().astype(bool)
        _, theta = mace_posterior(obs, _class_prior(obs, K), K=K)
        q = np.array([theta[s] for s in obs.source.astype(str)])
        y_cor = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
        return y_cor, np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0)), q
    ver = obs.verified.to_numpy().astype(bool)
    y_cor = np.where(ver, obs.y_ver.to_numpy(), obs.y_fb.to_numpy())
    yf = obs.y_fb.to_numpy()
    src = obs.source.astype(str).to_numpy()
    post = crowd_posterior(obs, _class_prior(obs, K), context=False, K=K)
    A = np.full((K, K), 0.5) + np.eye(K) * 3.5  # same Dirichlet pseudo-counts as crowd_posterior
    r = {}
    for s in np.unique(src):
        m = src == s
        cnt = A.copy()
        np.add.at(cnt.T, yf[m], post[m])
        pi = cnt / cnt.sum(1, keepdims=True)
        r[s] = float(np.mean(np.diag(pi)))
    q = np.array([r[s] for s in src])
    w = np.where(ver, 1.0, np.where(q >= tau, q ** gamma, 0.0))
    return y_cor, w, q


def small_loss_select(obs, fit_score, rounds=0, rng_seed=0):
    """Cross-fitted small-loss selection (rounds=0) / co-teaching-style exchange (rounds>0).

    fit_score(train_idx, train_labels, train_w, eval_idx, eval_labels) -> p(eval_label | x) for eval items; indices
    refer to rows of `obs` (the adapt pool); the callback adds the trusted seed set itself.
    Returns labels, weights (verified 1; selected unverified 1; rest 0), q = cross-fitted p(y_fb|x) (nan if verified)."""
    ver = obs.verified.to_numpy().astype(bool)
    yf = obs.y_fb.to_numpy()
    y_cor = np.where(ver, obs.y_ver.to_numpy(), yf)
    corr = (yf == obs.y_ver.to_numpy()) & ver
    eps = 1.0 - (corr.sum() + 1.0) / (ver.sum() + 2.0)
    vi = np.flatnonzero(ver)
    ui = np.flatnonzero(~ver)
    rng = np.random.default_rng(rng_seed)
    half = rng.permutation(len(ui)) % 2
    H = [ui[half == 0], ui[half == 1]]
    sel = [H[0].copy(), H[1].copy()]  # round 0: every feedback item of the half is used for training
    q = np.full(len(obs), np.nan)
    for _ in range(rounds + 1):
        new_sel = []
        for a, b in ((0, 1), (1, 0)):
            tr = np.r_[vi, sel[a]]
            p = np.asarray(fit_score(tr, y_cor[tr], np.ones(len(tr)), H[b], yf[H[b]]), float)
            q[H[b]] = p
            k = int(round((1 - eps) * len(H[b])))
            order = np.argsort(-p, kind="stable")  # small loss = large p(y_fb|x)
            new_sel.append((b, H[b][order[:k]]))
        for b, s in new_sel:
            sel[b] = s
    w = ver.astype(float)
    w[np.r_[sel[0], sel[1]].astype(int)] = 1.0
    return y_cor, w, q
