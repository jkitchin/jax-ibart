"""UQ tests: OLS intervals, conformal coverage, stability, CV RMSE, BART PPD.

Fast methods use the diff backend; the BART posterior test needs bartz.
"""

import numpy as np
import pytest

pytest.importorskip("optax")

from jax_ibart import IBart, uncertainty


def _data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
    X = np.column_stack([a, b, c, d])
    y = 5.0 * (a * b) + rng.normal(0, 0.2, n)
    return X, y, ["a", "b", "c", "d"]


def _factory():
    return IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=800)


def test_ols_intervals_structure():
    X, y, names = _data()
    model = _factory().fit(X, y, names)
    out = uncertainty.ols_intervals(model, X, y, X_new=X[:5])
    terms = {c["term"] for c in out["coefficients"]}
    assert "intercept" in terms
    for c in out["coefficients"]:
        assert c["ci_low"] <= c["coef"] <= c["ci_high"]
    pi = out["prediction"]
    assert np.all(pi["lower"] <= pi["point"]) and np.all(pi["point"] <= pi["upper"])


def test_conformal_coverage_is_honest():
    X, y, names = _data(n=240)
    # hold out a test set to measure realized coverage
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(y))
    te, tr = idx[:80], idx[80:]
    conf = uncertainty.conformal_intervals(
        _factory, X[tr], y[tr], names, alpha=0.10, cal_frac=0.4, seed=0)
    cov = uncertainty.empirical_coverage(conf, X[te], y[te])
    # marginal guarantee is ~0.90; allow finite-sample slack
    assert cov >= 0.80, f"coverage {cov:.2f} far below nominal 0.90"
    assert conf.half_width > 0


def test_selection_stability_finds_true_features():
    X, y, names = _data()
    out = uncertainty.selection_stability(_factory, X, y, names, n_boot=8, seed=0)
    # a and b drive y; they should be the most frequently selected features
    ff = out["feature_freq"]
    assert ff.get("a", 0) >= 0.5 and ff.get("b", 0) >= 0.5


def test_repeated_split_rmse_summary():
    X, y, names = _data()
    out = uncertainty.repeated_split_rmse(_factory, X, y, names, n_splits=6, seed=0)
    assert out["rmse"].shape == (6,)
    assert out["q05"] <= out["mean"] <= out["q95"]


def test_bart_posterior_intervals():
    pytest.importorskip("bartz")
    X, y, names = _data()
    model = _factory().fit(X, y, names)
    out = uncertainty.bart_posterior_intervals(
        model, X, y, X[:10], alpha=0.10, n_tree=30, n_draw=200, n_burn=200)
    assert np.all(out["lower"] <= out["median"]) and np.all(out["median"] <= out["upper"])
    assert out["lower"].shape == (10,)
