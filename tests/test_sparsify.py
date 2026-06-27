"""M3 tests: LASSO screening and l0 best-subset (host-side f64, no bartz)."""

import numpy as np

from jax_ibart import sparsify


def _linear(n=80, d=8, seed=0):
    # y = 3*x0 - 2*x1 + noise; x2..x7 irrelevant.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


def test_lasso_screen_surfaces_true_features():
    X, y = _linear()
    idx = sparsify.lasso_screen(X, y, n_keep=4)
    assert 0 in idx and 1 in idx


def test_l0_recovers_subset_and_coefficients():
    X, y = _linear()
    idx, coef = sparsify.l0_best_subset(X, y, max_dim=3, score="bic")
    assert set(idx.tolist()) == {0, 1}
    # coef = [intercept, slope_x0, slope_x1]
    np.testing.assert_allclose(coef, [0.0, 3.0, -2.0], atol=0.1)


def test_l0_runs_in_f64():
    X, y = _linear()
    _, coef = sparsify.l0_best_subset(X.astype(np.float32), y.astype(np.float32))
    assert coef.dtype == np.float64


def test_l0_cv_scoring_also_works():
    X, y = _linear()
    idx, _ = sparsify.l0_best_subset(X, y, max_dim=2, score="cv")
    assert set(idx.tolist()) == {0, 1}
