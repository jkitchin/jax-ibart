"""Precision-policy guards: x64 stays off; host-side decisions use f64."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_ibart import engineer
from jax_ibart.operators import UNARY


def test_jax_x64_is_off():
    # The package must never enable global x64 (it breaks bartz suite-wide).
    assert jax.config.read("jax_enable_x64") is False


def test_pool_columns_are_f32():
    X = np.random.default_rng(0).uniform(1, 2, size=(8, 3))
    pool = engineer.from_primary(X, ["a", "b", "c"])
    assert pool.X.dtype == np.float32


def test_collinearity_dedup_drops_numeric_duplicate():
    # sqrt(x**2) == abs(x): symbolically distinct, numerically identical.
    # The f64 collinearity pass must drop one of the pair.
    X = np.random.default_rng(1).uniform(-2, 2, size=(40, 1))
    base = engineer.from_primary(X, ["x"])

    abs_pool = engineer.apply_unary(base, [UNARY["abs"]])  # abs(x)
    sq = engineer.apply_unary(base, [UNARY["sq"]])  # x**2
    sqrt_sq = engineer.apply_unary(sq, [UNARY["sqrt"]])  # sqrt(x**2)

    merged = engineer.Pool(
        X=jnp.concatenate([abs_pool.X, sqrt_sq.X], axis=1),
        exprs=abs_pool.exprs + sqrt_sq.exprs,
    )
    assert merged.n_descriptors == 2
    cleaned = engineer.clean(merged)
    assert cleaned.n_descriptors == 1
