"""Final parametric stage: LASSO screening then l0 best-subset.

Once iterative nonparametric screening has shrunk the pool to O(10) candidates,
finish with a linear model: LASSO to a sparse candidate set, then an exhaustive
l0 best-subset OLS over small combinations, scored by BIC (or CV). Mirrors the
``LASSO -> l0`` tail of the iBART schedule.

This is the precision-load-bearing stage: small n, collinear descriptors. It
runs entirely host-side in numpy **float64**, independent of the JAX x64 flag.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.linear_model import LassoLarsCV


def _as_f64(X, y):
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def lasso_screen(X, y, n_keep: int = 10) -> np.ndarray:
    """Return indices of up to ``n_keep`` descriptors ranked by LASSO entry.

    Features are standardized for ranking only. Variables are ordered by
    absolute standardized coefficient at the cross-validated LASSO solution;
    any shortfall is filled by univariate correlation so exactly
    ``min(n_keep, d)`` indices come back.
    """
    X, y = _as_f64(X, y)
    n, d = X.shape
    if d <= n_keep:
        return np.arange(d)

    mu, sd = X.mean(0), X.std(0) + 1e-300
    Xs = (X - mu) / sd
    yc = y - y.mean()

    coef = LassoLarsCV(cv=min(5, n), max_iter=2000).fit(Xs, yc).coef_
    ranked = list(np.argsort(np.abs(coef))[::-1])
    chosen = [j for j in ranked if coef[j] != 0.0][:n_keep]

    if len(chosen) < n_keep:  # fill from univariate correlation
        corr = np.abs(Xs.T @ yc) / n
        for j in np.argsort(corr)[::-1]:
            if j not in chosen:
                chosen.append(int(j))
            if len(chosen) == n_keep:
                break
    return np.asarray(chosen[:n_keep], dtype=int)


def _ols(X, y):
    """OLS with intercept (f64). Returns (coef_with_intercept, rss)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta, float(resid @ resid)


def _bic(rss, n, k):
    # k = number of fitted parameters (including intercept)
    rss = max(rss, 1e-300)
    return n * np.log(rss / n) + k * np.log(n)


def _cv_mse(X, y, n_splits=5):
    n = len(y)
    folds = np.array_split(np.arange(n), min(n_splits, n))
    se = 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        beta, _ = _ols(X[tr], y[tr])
        pred = np.column_stack([np.ones(len(f)), X[f]]) @ beta
        se += float(((y[f] - pred) ** 2).sum())
    return se / n


def l0_best_subset(X, y, max_dim: int = 3, score: str = "bic"):
    """Exhaustive l0: best OLS subset up to ``max_dim`` columns.

    Returns ``(indices, coef)`` where ``indices`` are column positions into
    ``X`` and ``coef`` is the OLS coefficient vector ``[intercept, *slopes]``
    for that subset (f64).
    """
    if score not in ("bic", "cv"):
        raise ValueError(f"score must be 'bic' or 'cv', got {score!r}")
    X, y = _as_f64(X, y)
    n, d = X.shape
    if d == 0:
        raise ValueError("no candidate descriptors to choose from")
    max_dim = min(max_dim, d)

    def score_subset(cols):
        Xs = X[:, list(cols)]
        if score == "cv":
            return _cv_mse(Xs, y)
        _, rss = _ols(Xs, y)
        return _bic(rss, n, k=len(cols) + 1)

    best_cols, best_score = None, np.inf
    for k in range(1, max_dim + 1):
        for cols in combinations(range(d), k):
            s = score_subset(cols)
            if s < best_score:
                best_score, best_cols = s, cols

    coef, _ = _ols(X[:, list(best_cols)], y)
    return np.asarray(best_cols, dtype=int), coef
