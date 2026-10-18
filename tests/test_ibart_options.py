"""IBart options and fitted attributes: history_, pool_, custom operator sets,
carry_selected, and clone round-tripping. Uses the diff backend (no bartz)."""

import jax.numpy as jnp
import numpy as np
import sympy as sp

from jax_ibart import IBart
from jax_ibart.operators import Operator, default_unary


def _product_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 4))
    y = 5.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y, ["a", "b", "c", "d"]


def test_history_and_pool():
    X, y, names = _product_data()
    m = IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=800)
    m.fit(X, y, names)
    stages = [h["stage"] for h in m.history_]
    assert stages == ["primary", "b", "l0"]
    assert m.history_[0] == {"stage": "primary", "generated": 4, "kept": 4}
    b = m.history_[1]
    assert b["generated"] > b["kept"] >= 1
    assert m.history_[-1]["kept"] == len(m.descriptors_)
    # pool_ is the screened pool the l0 stage chose from
    assert m.pool_.n_descriptors == b["kept"]
    assert all(
        any(sp.simplify(d - e) == 0 for e in m.pool_.exprs) for d in m.descriptors_
    )


def test_custom_unary_ops():
    rng = np.random.default_rng(0)
    T = rng.uniform(1.0, 3.0, 80)
    X = np.column_stack([T, rng.uniform(0, 1, 80)])
    y = 2.0 * T**4
    pow4 = Operator(
        name="pow4",
        arity=1,
        fn=lambda x: x**4,
        sym=lambda e: e**4,
        valid=lambda x: jnp.ones(x.shape[-1], bool),
        unit=lambda u: 4 * u,
    )
    m = IBart(
        backend="diff",
        schedule=("u",),
        max_dim=1,
        seed=0,
        steps=800,
        unary_ops=default_unary() + [pow4],
    ).fit(X, y, ["T", "d"])
    assert m.descriptors_ == [sp.Symbol("T") ** 4]
    assert np.isclose(m.coef_[1], 2.0, rtol=1e-3)


def test_carry_selected_keeps_earlier_survivors():
    X, y, names = _product_data()
    m = IBart(
        backend="diff",
        schedule=("b", "u"),
        max_dim=1,
        seed=0,
        steps=800,
        carry_selected=True,
    ).fit(X, y, names)
    kept = [h["kept"] for h in m.history_[1:-1]]
    assert kept == sorted(kept)  # the pool only accumulates
    target = sp.Symbol("a") * sp.Symbol("b")
    assert any(sp.simplify(e - target) == 0 for e in m.pool_.exprs)


def test_clone_preserves_options():
    m = IBart(backend="diff", unary_ops=default_unary(), carry_selected=True, steps=10)
    c = m.clone()
    assert c.carry_selected is True
    assert c.unary_ops == m.unary_ops
    assert c.get_params() == m.get_params()
