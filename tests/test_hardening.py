"""Edge cases: operator domains, dynamics arguments, defaults and warnings."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jax_ibart import IBart, dynamics, engineer, sparsify
from jax_ibart.operators import BINARY, Operator, u_same


def _pool(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 2.0, size=(30, 3))
    X[:, 2] = np.abs(X[:, 2]) + 0.5  # only c is strictly positive
    return engineer.from_primary(X, ["a", "b", "c"])


def test_binary_valid_mask_filters_pairs():
    # Only keep pairs whose right operand is strictly positive.
    pos_div = dataclasses.replace(
        BINARY["div"], name="pdiv", valid=lambda a, b: jnp.all(b > 0, axis=0)
    )
    pool = _pool()
    X = np.asarray(pool.X)
    out = engineer.apply_binary(pool, [pos_div])
    assert out.n_descriptors > 0
    for e in out.exprs:
        _, den = sp.fraction(e)
        assert str(den) == "c", e
    assert (X[:, 2] > 0).all()


def test_binary_valid_scalar_false_skips_op():
    never = Operator(
        "never", 2, lambda a, b: a + b, lambda e, f: e + f, lambda a, b: False, u_same
    )
    assert engineer.apply_binary(_pool(), [never]).n_descriptors == 0


def test_finite_difference_guards():
    with pytest.raises(ValueError, match="at least 2"):
        dynamics.finite_difference(np.ones((1, 2)), np.zeros(1))
    with pytest.raises(ValueError, match="t must have shape"):
        dynamics.finite_difference(np.ones((5, 2)), np.arange(4.0))


def test_dynamics_argument_validation():
    X = np.random.default_rng(0).uniform(1, 2, size=(20, 2))
    with pytest.raises(ValueError, match="degree"):
        dynamics.polynomial_library(X, ["x", "y"], degree=3)
    with pytest.raises(ValueError, match="screen"):
        dynamics.discover_ode(X, np.arange(20.0), ["x", "y"], screen="lasso")


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lasso_screen_tiny_n(n):
    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, 8))
    idx = sparsify.lasso_screen(X, X[:, 0] + 0.1 * rng.normal(size=n), n_keep=3)
    assert len(idx) == 3 and len(set(idx)) == 3


def test_ols_is_public_and_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    beta, rss = sparsify.ols(X, 1.5 + X @ np.array([2.0, -3.0]))
    np.testing.assert_allclose(beta, [1.5, 2.0, -3.0], atol=1e-10)
    assert rss < 1e-20


def _xy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    return X, 3.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)


def _fast(**kw):
    pytest.importorskip("optax")
    return IBart(backend="diff", steps=200, max_dim=1, schedule=("b",), seed=0, **kw)


def test_feature_names_default():
    X, y = _xy()
    m = _fast().fit(X, y)
    assert {str(s) for s in m.expression().free_symbols} <= {"x0", "x1"}


def test_feature_names_from_dataframe():
    pd = pytest.importorskip("pandas")
    X, y = _xy()
    m = _fast().fit(pd.DataFrame(X, columns=["T", "P"]), y)
    assert {str(s) for s in m.expression().free_symbols} <= {"T", "P"}


def test_predict_warns_on_nonfinite():
    X, y = _xy()
    m = _fast(binary_ops=[BINARY["div"]]).fit(X, y, ["a", "b"])
    Xbad = X.copy()
    Xbad[:, :] = 0.0
    with pytest.warns(UserWarning, match="NaN or inf"):
        m.predict(Xbad)


def test_fn_sym_mismatch_warns():
    lying_mul = dataclasses.replace(BINARY["mul"], sym=lambda e, f: e + f)
    X, y = _xy()
    with pytest.warns(UserWarning, match="symbolic and numeric forms disagree"):
        _fast(binary_ops=[lying_mul]).fit(X, y, ["a", "b"])
