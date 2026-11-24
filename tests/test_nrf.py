import numpy as np
import pytest

from nrf import METHODS, make_benchmark, method_weights, observable_view


def test_benchmark_shapes_and_group_split():
    b = make_benchmark(seed=3, noise=0.4, noise_type="context")
    held = set(b["meta"]["heldout_domains"])
    assert not held & set(b["seed"].domain) and not held & set(b["adapt"].domain)
    assert set(b["test"][b["test"].split == "held"].domain) <= held
    assert len(b["adapt"]) == 640 and len(b["seed"]) == 80


@pytest.mark.parametrize("nt", ["uniform", "source", "context"])
def test_realized_noise_close_to_target(nt):
    r = np.mean([make_benchmark(seed=s, noise=0.4, noise_type=nt)["meta"]["realized_noise"] for s in range(6)])
    assert abs(r - 0.4) < 0.05


def test_zero_noise_is_clean():
    b = make_benchmark(seed=1, noise=0.0)
    assert (b["adapt"].y_fb == b["adapt"].y_true).all()


def test_observable_view_hides_unverified_truth():
    b = make_benchmark(seed=2)
    obs = observable_view(b["adapt"])
    assert "y_true" not in obs.columns
    assert (obs.y_ver[~obs.verified] == -1).all()


@pytest.mark.parametrize("m", METHODS)
def test_no_leakage_from_unverified_truth(m):
    """Weights must not change when the hidden labels of unverified items are permuted."""
    b = make_benchmark(seed=5)
    a = b["adapt"]
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), len(a))
    a2 = a.copy()
    unv = ~a2.verified.to_numpy()
    a2.loc[unv, "y_true"] = rng.permutation(a2.loc[unv, "y_true"].to_numpy())
    y1, w1, _ = method_weights(m, observable_view(a), p)
    y2, w2, _ = method_weights(m, observable_view(a2), p)
    assert np.array_equal(y1, y2) and np.allclose(w1, w2)


def test_verified_items_get_verifier_label_and_weight_one():
    b = make_benchmark(seed=4)
    obs = observable_view(b["adapt"])
    p = np.full((len(obs), 4), 0.25)
    for m in ["source_prior", "vcfl", "ctx_prior", "ca_vcfl", "conf_filter", "instance_rel", "ds_source",
              "crowd_seed", "ctx_crowd", "mace", "rgpo_style"]:
        y, w, _ = method_weights(m, obs, p)
        v = obs.verified.to_numpy()
        assert (y[v] == b["adapt"].y_true.to_numpy()[v]).all() and (w[v] == 1).all()


def test_self_train_no_leakage():
    """breakeven_cpu.self_train_weights only sees seed + verified labels (feedback and hidden truth ignored)."""
    from breakeven_cpu import featurize, self_train_weights
    b = make_benchmark(seed=6, verify_rate=0.25)
    a = b["adapt"]
    a2 = a.copy()
    unv = ~a2.verified.to_numpy()
    rng = np.random.default_rng(1)
    a2.loc[unv, "y_true"] = rng.permutation(a2.loc[unv, "y_true"].to_numpy())
    Xs, Xa, _ = featurize("tfidf", b["seed"], a, b["test"])
    ys = b["seed"].y_true.to_numpy()
    y1, w1, _ = self_train_weights("tfidf", Xs, ys, Xa, observable_view(a), 0.5, 2.0)
    y2, w2, _ = self_train_weights("tfidf", Xs, ys, Xa, observable_view(a2), 0.5, 2.0)
    assert np.array_equal(y1, y2) and np.allclose(w1, w2)


def _tfidf_fit_score(b):
    """Learner callback for small_loss/coteach: TF-IDF+LR on seed + given adapt rows (only observable inputs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    seed, adapt = b["seed"], b["adapt"]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(list(seed.text) + list(adapt.text))
    Xs, Xa = vec.transform(seed.text), vec.transform(adapt.text)
    ys = seed.y_true.to_numpy()

    def fit_score(tr, ytr, wtr, ev, yev):
        from scipy.sparse import vstack
        clf = LogisticRegression(C=5.0, max_iter=2000).fit(vstack([Xs, Xa[tr]]), np.r_[ys, ytr],
                                                            sample_weight=np.r_[np.ones(len(ys)), wtr])
        P = np.zeros((len(ev), 4))
        P[:, clf.classes_] = clf.predict_proba(Xa[ev])
        return P[np.arange(len(ev)), yev]
    return fit_score


@pytest.mark.parametrize("rounds", [0, 2])
def test_small_loss_coteach_no_leakage(rounds):
    """Q1 E4: small_loss (rounds=0) / coteach (rounds>0) must not depend on hidden truth of unverified items."""
    from nrf.baselines import small_loss_select
    b = make_benchmark(seed=7, verify_rate=0.2)
    a = b["adapt"]
    a2 = a.copy()
    unv = ~a2.verified.to_numpy()
    rng = np.random.default_rng(2)
    a2.loc[unv, "y_true"] = rng.permutation(a2.loc[unv, "y_true"].to_numpy())
    fs = _tfidf_fit_score(b)
    y1, w1, _ = small_loss_select(observable_view(a), fs, rounds=rounds)
    y2, w2, _ = small_loss_select(observable_view(a2), fs, rounds=rounds)
    assert np.array_equal(y1, y2) and np.allclose(w1, w2)
    v = a.verified.to_numpy()
    assert (w1[v] == 1).all() and (y1[v] == a.y_true.to_numpy()[v]).all()
    # keeps roughly (1 - estimated noise) of the unverified pool, and the kept feedback is cleaner than the pool
    keep = (w1 > 0) & ~v
    wrong = (a.y_fb != a.y_true).to_numpy()
    assert 0.3 < keep[~v].mean() < 0.9 and wrong[keep].mean() < wrong[~v].mean()


def test_mace_recovers_source_competence():
    """MACE competence should rank sources like their true feedback accuracy under source noise."""
    from nrf.baselines import mace_posterior, _class_prior
    b = make_benchmark(seed=3, noise=0.4, noise_type="source", verify_rate=0.25)
    a = b["adapt"]
    obs = observable_view(a)
    _, theta = mace_posterior(obs, _class_prior(obs, 4))
    acc = (a.y_fb == a.y_true).groupby(a.source.astype(str)).mean()
    th = np.array([theta[s] for s in acc.index])
    assert np.corrcoef(th, acc.to_numpy())[0, 1] > 0.7
