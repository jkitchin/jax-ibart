"""M2b tests: the differentiable (diff) selection backend.

Fast (no bartz): the gated MLP trains in ~1 s. Acceptance:
1. keeps the true drivers and rejects noise (PAN: no false negatives);
2. agrees with the bart oracle on which true features matter;
3. is differentiable end to end (gradients flow through the gates);
4. drives IBart end-to-end to recover a product descriptor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

pytest.importorskip("optax")

from jax_ibart import IBart
from jax_ibart._diffscreen import _open_prob, _sample_gate, screen_importance
from jax_ibart.selection import DiffSelector, make_selector


def _friedman(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, p))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + rng.normal(0, 0.5, n)
    return X, y


def test_diff_keeps_signal_rejects_noise():
    X, y = _friedman()
    keep = np.asarray(DiffSelector(steps=1500).select(X, y, key=jax.random.PRNGKey(0)))
    assert keep.shape == (X.shape[1],)
    assert keep[0] and keep[1]          # PAN: both true drivers kept
    assert keep[2:].sum() <= 1          # noise rejected


def test_factory_returns_diff_backend():
    assert isinstance(make_selector("diff"), DiffSelector)


def test_diff_agrees_with_bart_oracle():
    # The differentiable backend must select the same true drivers the faithful
    # BART-G.SE backend does (validated against the oracle on the fixture).
    bartz = pytest.importorskip("bartz")  # noqa: F841
    from jax_ibart.selection import BartGSESelector

    X, y = _friedman()
    diff = np.asarray(DiffSelector(steps=1500).select(X, y, key=jax.random.PRNGKey(0)))
    bart = np.asarray(BartGSESelector(n_perm=15, n_tree=40, n_draw=300, n_burn=300)
                      .select(X, y, key=jax.random.PRNGKey(0)))
    # Both keep the true drivers; neither commits a false negative there.
    assert diff[0] and diff[1] and bart[0] and bart[1]


def test_screening_is_differentiable():
    # The defining property: gradient of the gated-MLP loss w.r.t. log_alpha
    # exists and is finite (gradients flow through the reparameterized gates).
    X, y = _friedman(n=40, p=5)
    Xs = jnp.asarray((X - X.mean(0)) / (X.std(0) + 1e-6), jnp.float32)
    yc = jnp.asarray((y - y.mean()) / (y.std() + 1e-6), jnp.float32)
    log_alpha = jnp.zeros(5)

    def loss(la, key):
        z = _sample_gate(la, key)
        pred = (Xs * z).sum(axis=1)  # trivial differentiable predictor
        return jnp.mean((yc - pred) ** 2) + 0.03 * jnp.sum(_open_prob(la))

    g = jax.grad(loss)(log_alpha, jax.random.PRNGKey(0))
    assert g.shape == (5,)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.0)


def test_ibart_diff_backend_end_to_end():
    rng = np.random.default_rng(0)
    n = 90
    a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
    X = np.column_stack([a, b, c, d])
    y = 5.0 * (a * b) + rng.normal(0, 0.05, n)

    model = IBart(backend="diff", schedule=("b",), max_dim=1, seed=0,
                  steps=1200).fit(X, y, feature_names=["a", "b", "c", "d"])

    target = sp.Symbol("a") * sp.Symbol("b")
    assert any(sp.simplify(e - target) == 0 for e in model.descriptors_)
    pred = model.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.95
