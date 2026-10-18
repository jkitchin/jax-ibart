"""Uncertainty quantification for iBART models.

iBART is a *selection* method, so its dominant uncertainty is which descriptors
get chosen, not just the spread around a fixed model. This module provides, from
cheap to rigorous:

- :func:`ols_intervals` -- classical OLS coefficient SEs / CIs and Gaussian
  prediction intervals for the final model. Fast, but **optimistic**: it
  conditions on the selected descriptors and ignores selection, so its intervals
  under-cover. Use it for a quick read on coefficient magnitudes, not for honest
  coverage.
- :func:`conformal_intervals` -- **split conformal** prediction intervals. Wraps
  the *entire* fit (selection included) as a black box and calibrates on
  held-out data, giving a finite-sample, distribution-free marginal coverage
  guarantee regardless of post-selection effects. This is the honest interval.
- :func:`selection_stability` -- bootstrap inclusion frequencies: how often each
  descriptor / primary feature is selected across resamples. Captures the
  selection uncertainty point intervals miss.
- :func:`repeated_split_rmse` -- the papers' protocol: out-of-sample RMSE over
  many random train/test splits, as a distribution.
- :func:`bart_posterior_intervals` -- Bayesian posterior predictive intervals
  from a BART fit on the selected descriptors (propagates function + noise
  uncertainty, conditional on the selected descriptors).

The factory-based functions take ``factory``, a zero-arg callable returning a
fresh unfitted estimator, e.g. ``factory=model.clone`` or
``lambda: IBart(backend="diff", ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# 1. Classical OLS intervals (optimistic; conditional on selected descriptors)
# --------------------------------------------------------------------------- #


def ols_intervals(model, X_train, y_train, X_new=None, alpha: float = 0.10):
    """OLS coefficient SEs/CIs and Gaussian prediction intervals.

    OPTIMISTIC: conditions on ``model.descriptors_`` and ignores that they were
    selected from data, so coverage is below nominal. Use
    :func:`conformal_intervals` for honest intervals.
    """
    y = np.asarray(y_train, dtype=np.float64)
    D = model._design(X_train)  # (n, k) selected descriptors
    A = np.column_stack([np.ones(len(y)), D])  # design with intercept
    n, p = A.shape
    beta = model.coef_
    resid = y - A @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)

    names = ["intercept"] + [str(e) for e in model.descriptors_]
    coefs = [
        {
            "term": nm,
            "coef": float(b),
            "se": float(s),
            "ci_low": float(b - tcrit * s),
            "ci_high": float(b + tcrit * s),
        }
        for nm, b, s in zip(names, beta, se)
    ]
    out = {"coefficients": coefs, "sigma": float(np.sqrt(sigma2)), "dof": dof}

    if X_new is not None:
        Dn = model._design(X_new)
        An = np.column_stack([np.ones(len(Dn)), Dn])
        pred = An @ beta
        # prediction interval: includes both estimation and noise variance
        lev = np.einsum("ij,jk,ik->i", An, XtX_inv, An)
        half = tcrit * np.sqrt(sigma2 * (1.0 + lev))
        out["prediction"] = {"point": pred, "lower": pred - half, "upper": pred + half}
    return out


# --------------------------------------------------------------------------- #
# 2. Split conformal intervals (honest, calibrated coverage)
# --------------------------------------------------------------------------- #


@dataclass
class ConformalModel:
    """A fitted model plus a calibrated conformal half-width.

    ``interval(X)`` returns ``(lower, point, upper)`` with marginal coverage
    >= ``1 - alpha`` (finite-sample, distribution-free).
    """

    model: object
    half_width: float
    alpha: float

    def interval(self, X):
        point = np.asarray(self.model.predict(X), dtype=np.float64)
        return point - self.half_width, point, point + self.half_width


def conformal_intervals(
    factory: Callable,
    X,
    y,
    feature_names,
    units=None,
    alpha: float = 0.10,
    cal_frac: float = 0.3,
    seed: int = 0,
):
    """Split-conformal prediction intervals around an iBART fit.

    The estimator (selection + final model) is fit on a proper-training split
    and scored on a disjoint calibration split, so the residual quantile is an
    honest, post-selection-valid calibration. Returns a :class:`ConformalModel`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(int(round(cal_frac * n)), 1)
    cal, tr = idx[:n_cal], idx[n_cal:]

    u_tr = None if units is None else units
    model = factory().fit(X[tr], y[tr], feature_names, units=u_tr)
    resid = np.abs(y[cal] - np.asarray(model.predict(X[cal])))

    # finite-sample conformal quantile level
    level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    qhat = float(np.quantile(resid, level, method="higher"))
    return ConformalModel(model=model, half_width=qhat, alpha=alpha)


def empirical_coverage(conformal: ConformalModel, X_test, y_test):
    """Fraction of test points whose true y falls inside the conformal interval."""
    lo, _, hi = conformal.interval(X_test)
    y_test = np.asarray(y_test, dtype=np.float64)
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


# --------------------------------------------------------------------------- #
# 3. Selection stability (bootstrap inclusion frequencies)
# --------------------------------------------------------------------------- #


def selection_stability(
    factory: Callable, X, y, feature_names, units=None, n_boot: int = 25, seed: int = 0
):
    """Bootstrap inclusion frequencies for descriptors and primary features.

    Refits the whole pipeline on ``n_boot`` bootstrap resamples and records how
    often each descriptor (by canonical form) and each primary feature is
    selected. Frequencies near 1 are stable signals; low frequencies are
    selection noise.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)

    desc_counts: dict[str, int] = {}
    feat_counts: dict[str, int] = {}
    for _ in range(n_boot):
        bs = rng.integers(0, n, n)
        m = factory().fit(X[bs], y[bs], feature_names, units=units)
        seen_feats = set()
        for e in m.descriptors_:
            key = str(e)
            desc_counts[key] = desc_counts.get(key, 0) + 1
            seen_feats |= {str(s) for s in e.free_symbols}
        for f in seen_feats:
            feat_counts[f] = feat_counts.get(f, 0) + 1

    def freq(d):
        return {k: v / n_boot for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    return {
        "n_boot": n_boot,
        "descriptor_freq": freq(desc_counts),
        "feature_freq": freq(feat_counts),
    }


# --------------------------------------------------------------------------- #
# 4. Repeated-split out-of-sample RMSE (the papers' protocol)
# --------------------------------------------------------------------------- #


def repeated_split_rmse(
    factory: Callable,
    X,
    y,
    feature_names,
    units=None,
    n_splits: int = 20,
    test_frac: float = 0.25,
    seed: int = 0,
):
    """Out-of-sample RMSE over ``n_splits`` random train/test partitions."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_test = max(int(round(test_frac * n)), 1)

    rmses = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        te, tr = idx[:n_test], idx[n_test:]
        m = factory().fit(X[tr], y[tr], feature_names, units=units)
        err = y[te] - np.asarray(m.predict(X[te]))
        rmses.append(float(np.sqrt(np.mean(err**2))))
    rmses = np.array(rmses)
    return {
        "rmse": rmses,
        "mean": float(rmses.mean()),
        "std": float(rmses.std()),
        "q05": float(np.quantile(rmses, 0.05)),
        "q95": float(np.quantile(rmses, 0.95)),
    }


# --------------------------------------------------------------------------- #
# 5. Bayesian posterior predictive intervals (from BART on selected descriptors)
# --------------------------------------------------------------------------- #


def bart_posterior_intervals(
    model,
    X_train,
    y_train,
    X_new,
    alpha: float = 0.10,
    n_tree: int = 50,
    n_draw: int = 500,
    n_burn: int = 500,
    seed: int = 0,
):
    """Posterior predictive intervals from a BART fit on the selected descriptors.

    Propagates BART's function-value posterior and the error-scale posterior
    (sigma) to a predictive interval at ``X_new``. Conditional on the selected
    descriptors (it does not propagate selection uncertainty -- use
    :func:`selection_stability` for that). Requires ``bartz``.
    """
    from .selection import bartz_layout, fit_gbart

    D_tr = np.asarray(model._design(X_train), dtype=np.float32)
    D_new = np.asarray(model._design(X_new), dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    m = fit_gbart(
        D_tr,
        y,
        ntree=n_tree,
        ndpost=n_draw,
        nskip=n_burn,
        seed=int(seed),
        printevery=None,
    )
    fhat = np.asarray(
        m.predict(bartz_layout(D_new)), dtype=np.float64
    )  # (ndpost, n_new)
    sigma = np.asarray(m.sigma, dtype=np.float64)[-n_draw:]  # post-burn draws

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(fhat.shape) * sigma[:, None]
    ppd = fhat + noise  # posterior predictive
    lo = np.quantile(ppd, alpha / 2, axis=0)
    med = np.quantile(ppd, 0.5, axis=0)
    hi = np.quantile(ppd, 1 - alpha / 2, axis=0)
    return {"lower": lo, "median": med, "upper": hi}
