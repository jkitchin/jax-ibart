"""M1 tests: operator algebra and the feature-engineering engine.

These run without bartz (selection backends are M2). They check numeric
correctness, domain protection, symbolic provenance, and that the engineered
pool can express the target catalysis descriptor dH_ox,fm * O_vac.
"""

import numpy as np
import sympy as sp

# NOTE: do not enable jax_enable_x64 here. The package runs in float32 because
# bartz (the selection backend) breaks under global x64; enabling it in any
# imported test module pollutes global JAX state for the whole suite.

from jax_ibart import engineer
from jax_ibart.operators import BINARY, UNARY, default_binary, default_unary


def _toy_pool(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, 3))  # positive -> log/sqrt valid
    return engineer.from_primary(X, ["a", "b", "c"]), X


def test_unary_numeric_and_symbolic():
    pool, X = _toy_pool()
    out = engineer.apply_unary(pool, [UNARY["sq"]])
    # square of every primary feature, provenance preserved
    assert out.n_descriptors == 3
    assert {sp.srepr(e) for e in out.exprs} == {sp.srepr(sp.Symbol(s) ** 2) for s in "abc"}
    np.testing.assert_allclose(np.asarray(out.X), X**2, rtol=1e-5)  # float32


def test_domain_protection_drops_invalid_columns():
    # negative column -> log/sqrt domain-invalid, must be dropped, not NaN
    X = np.array([[-1.0, 2.0], [-3.0, 4.0], [-2.0, 1.0]])
    pool = engineer.from_primary(X, ["x", "y"])
    out = engineer.apply_unary(pool, [UNARY["log"]])
    assert out.n_descriptors == 1  # only 'y' survives
    assert np.all(np.isfinite(np.asarray(out.X)))


def test_binary_absdiff_commutative_dedup():
    pool, _ = _toy_pool()
    out = engineer.apply_binary(pool, [BINARY["absdiff"]])
    # commutative: C(3,2) = 3 unique pairs
    assert out.n_descriptors == 3


def test_can_express_target_descriptor():
    # dH_ox,fm * O_vac : a multiplicative binary descriptor of two primaries
    X = np.random.default_rng(1).uniform(1, 2, size=(10, 2))
    pool = engineer.from_primary(X, ["dH_ox_fm", "O_vac"])
    out = engineer.apply_binary(pool, [BINARY["mul"]])
    target = sp.Symbol("dH_ox_fm") * sp.Symbol("O_vac")
    assert any(sp.simplify(e - target) == 0 for e in out.exprs)


def test_default_sets_sizes():
    assert len(default_unary()) == 9
    assert len(default_binary()) == 5


def test_unit_filter_drops_inconsistent_sums():
    # x: unit [1,0] (e.g. length), z: unit [0,1] (e.g. speed). x*y... here:
    # x+z is dimensionally invalid; x*z and x/z are valid.
    X = np.random.default_rng(0).uniform(1, 2, size=(12, 2))
    units = np.array([[1, 0], [0, 1]], dtype=np.int64)  # (n_features, n_dims)
    pool = engineer.from_primary(X, ["x", "z"], units=units)

    # add: matching-unit only -> x+z rejected, so add yields nothing here.
    out_add = engineer.apply_binary(pool, [BINARY["add"]])
    assert out_add.n_descriptors == 0

    # mul: always valid -> x*z, with unit [1,1].
    out_mul = engineer.apply_binary(pool, [BINARY["mul"]])
    assert out_mul.n_descriptors == 1
    assert np.array_equal(out_mul.units[0], np.array([1, 1]))

    # div: x/z valid both directions, units [1,-1] and [-1,1].
    out_div = engineer.apply_binary(pool, [BINARY["div"]])
    assert out_div.n_descriptors == 2


def test_unit_filter_allows_matching_units():
    # two eV-valued features can be added; result keeps the eV unit.
    X = np.random.default_rng(1).uniform(1, 2, size=(12, 2))
    units = np.array([[1, 0], [1, 0]], dtype=np.int64)  # both same unit
    pool = engineer.from_primary(X, ["a", "b"], units=units)
    out = engineer.apply_binary(pool, [BINARY["add"]])
    assert out.n_descriptors == 1
    assert np.array_equal(out.units[0], np.array([1, 0]))


def test_square_doubles_unit_sqrt_zeros_it():
    X = np.random.default_rng(2).uniform(1, 2, size=(12, 1))
    units = np.array([[1, 2]], dtype=np.int64)
    pool = engineer.from_primary(X, ["x"], units=units)
    sq = engineer.apply_unary(pool, [UNARY["sq"]])
    assert np.array_equal(sq.units[0], np.array([2, 4]))
    rt = engineer.apply_unary(pool, [UNARY["sqrt"]])
    assert np.array_equal(rt.units[0], np.array([0, 0]))  # sqrt -> unitless
