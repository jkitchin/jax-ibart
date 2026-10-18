"""Tests for the faithful BART-G.SE selection backend (requires bartz).

PAN acceptance: on a problem where only x0, x1 drive y, the selector must keep
both (no false negatives) and reject pure-noise features.
"""

import jax
import numpy as np
import pytest

pytest.importorskip("bartz")

from jax_ibart.selection import BartGSESelector, make_selector


def _friedman(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + rng.normal(0, 0.5, n)
    return X, y


def test_gse_keeps_signal_rejects_noise():
    X, y = _friedman()
    # Small but adequate MCMC / permutation budget for a fast test.
    sel = BartGSESelector(n_perm=15, alpha=0.05, n_tree=40, n_draw=300, n_burn=300)
    keep = np.asarray(sel.select(X, y, key=jax.random.PRNGKey(0)))

    assert keep.shape == (X.shape[1],)
    # PAN: both true drivers retained (no false negatives).
    assert keep[0] and keep[1]
    # Most of the 6 noise features rejected (allow at most one false positive).
    assert keep[2:].sum() <= 1

    # Diagnostics expose the evidence behind the decision.
    dg = sel.diagnostics_
    assert dg["null"].shape == (15, X.shape[1])
    assert np.array_equal(keep, dg["q_obs"] > dg["threshold"])


def test_factory_returns_bart_backend():
    assert isinstance(make_selector("bart"), BartGSESelector)
