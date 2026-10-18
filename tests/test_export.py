"""Embedding fitted models in discopt (jax_ibart.export)."""

import warnings

import numpy as np
import pytest
import sympy as sp

import jax

# Importing discopt turns on jax_enable_x64 for the whole process. Test modules
# are all imported at collection, so restore the setting at once (the rest of
# the suite must run in float32) and turn x64 on only while these tests run.
_x64 = jax.config.jax_enable_x64
dm = pytest.importorskip("discopt.modeling")
jax.config.update("jax_enable_x64", _x64)


@pytest.fixture(autouse=True, scope="module")
def _discopt_x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _x64)


from jax_ibart import IBart
from jax_ibart.export import to_discopt

a, b, c = sp.symbols("a b c")


class _Fitted(IBart):
    """An IBart with hand-written descriptors (no fit), for exact tests."""

    def __init__(self, descriptors, coef, X):
        super().__init__()
        self._feature_names = ["a", "b", "c"]
        self.descriptors_ = list(descriptors)
        self.coef_ = np.asarray(coef, dtype=float)
        self._X_train = np.asarray(X, dtype=float)


def _X(lo=(0.5, 1.5, -1.0), hi=(2.0, 3.0, 1.0), n=50, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))


def _solve(model, sense="max", **kw):
    m = dm.Model("t")
    x, y = to_discopt(model, m, **kw)
    (m.maximize if sense == "max" else m.minimize)(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60)
    xv = {k: float(r.value(v)) for k, v in x.items()}
    return r, xv


def _predict_at(model, xv):
    row = np.array([[xv.get(n, 0.0) for n in model._feature_names]])
    return float(model.predict(row)[0])


@pytest.mark.parametrize("sense", ["max", "min"])
def test_every_operator_matches_predict(sense):
    d = [
        sp.Abs(a - c) * sp.exp(c / b),
        sp.sin(sp.pi * a * c) + sp.cos(sp.pi * b),
        sp.sqrt(a * b) / sp.log(b + 1),
        a**2 / b,
    ]
    model = _Fitted(d, [0.3, 1.0, -0.5, 2.0, 0.7], _X())
    r, xv = _solve(model, sense)
    # The solver's point is a real prediction of the model.
    assert r.status in ("optimal", "feasible")
    assert r.objective == pytest.approx(_predict_at(model, xv), abs=1e-5)
    # Its proven bound is valid: no sampled point in the box gets past it. The
    # box is the training range, so drop samples that fall outside it.
    X = _X(n=20000, seed=1)
    lo, hi = model._X_train.min(axis=0), model._X_train.max(axis=0)
    p = model.predict(X[((X >= lo) & (X <= hi)).all(axis=1)])
    if sense == "max":
        assert r.bound >= p.max() - 1e-6
        # Max certifies quickly; min needs ~5 minutes, too slow for a test.
        assert r.status == "optimal" and r.gap_certified
        assert r.objective >= p.max() - 1e-6
    else:
        assert r.bound <= p.min() + 1e-6


def test_inputs_bounded_to_training_range():
    X = _X()
    model = _Fitted([a * b], [0.0, 1.0], X)
    r, xv = _solve(model, "max")
    assert xv["a"] == pytest.approx(X[:, 0].max(), abs=1e-6)
    assert xv["b"] == pytest.approx(X[:, 1].max(), abs=1e-6)
    assert set(xv) == {"a", "b"}  # only the inputs the model uses


def test_denominator_keeps_training_sign():
    # Training b - 1 is in [0.5, 2]; without the guard, maximizing 1/(b - 1)
    # over a box that reaches b = 1 would run into the pole.
    model = _Fitted([1 / (b - 1)], [0.0, 1.0], _X())
    r, xv = _solve(model, "max", bounds=([0, 0, -1], [3, 3, 1]))
    assert xv["b"] >= _X()[:, 1].min() - 1e-6
    assert np.isfinite(r.objective) and r.objective <= 2.0 + 1e-6


def test_mixed_sign_denominator_excludes_pole():
    model = _Fitted([1 / c], [0.0, 1.0], _X())  # training c spans [-1, 1]
    delta = np.abs(_X()[:, 2]).min()
    r, xv = _solve(model, "max")
    assert abs(xv["c"]) >= delta - 1e-6
    assert r.objective == pytest.approx(1 / delta, rel=1e-4)


def test_log_argument_stays_positive():
    model = _Fitted([sp.log(c + 1.5)], [0.0, 1.0], _X())
    r, xv = _solve(model, "min", bounds=([0, 0, -2], [3, 3, 1]))
    assert xv["c"] + 1.5 >= (_X()[:, 2] + 1.5).min() - 1e-6


def test_domain_false_skips_guards():
    model = _Fitted([1 / (b - 1) + sp.log(a)], [0.0, 1.0], _X())
    counts = {}
    for domain in (True, False):
        m = dm.Model("t")
        to_discopt(model, m, domain=domain)
        counts[domain] = len(m._constraints)  # no public count in discopt
    assert counts[True] >= counts[False] + 2


def test_user_supplied_inputs():
    model = _Fitted([a * b], [1.0, 2.0], _X())
    m = dm.Model("t")
    v = m.continuous("v", shape=(3,), lb=0, ub=1)
    x, y = to_discopt(model, m, x=v, y_bounds=(0, 10))
    m.maximize(y)
    r = m.solve(time_limit=30)
    assert r.objective == pytest.approx(3.0, abs=1e-6)

    m = dm.Model("t")
    p = m.continuous("p", lb=0, ub=2)
    with pytest.raises(ValueError, match="missing inputs"):
        to_discopt(model, m, x={"a": p})


def test_unsupported_function_raises():
    model = _Fitted([sp.gamma(a)], [0.0, 1.0], _X())
    with pytest.raises(NotImplementedError, match="gamma"):
        to_discopt(model, dm.Model("t"))


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        to_discopt(IBart(), dm.Model("t"))


def test_fit_is_independent_of_global_x64():
    # Importing discopt turns on jax_enable_x64; fit must not change because
    # of that, and must leave the global setting as it found it.
    import jax

    pytest.importorskip("optax")
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.0, size=(40, 3))
    y = 3.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 40)
    kw = dict(backend="diff", steps=200, max_dim=1, seed=0)
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        e32 = IBart(**kw).fit(X, y).expression()
        jax.config.update("jax_enable_x64", True)
        e64 = IBart(**kw).fit(X, y).expression()
        assert jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", prev)
    assert str(e32) == str(e64)


def test_prediction_bounds_contain_every_prediction():
    # y gets finite bounds from interval arithmetic over the input box; they
    # must be conservative or they would cut off real optima.
    from jax_ibart.export import _interval

    d = [
        sp.Abs(a - c) * sp.exp(c / b),
        sp.sin(sp.pi * a * c) + sp.cos(sp.pi * b),
        sp.sqrt(a * b) / sp.log(b + 1),
        a ** sp.Rational(3, 2) / sp.sqrt(b),
        (a - 1.2) ** 2 - sp.tanh(c) / a**2,
    ]
    X = _X()
    model = _Fitted(d, [0.3, 1.0, -0.5, 2.0, 0.7, -1.3], X)
    lo, hi = X.min(axis=0), X.max(axis=0)
    box = {n: (float(l), float(h)) for n, l, h in zip("abc", lo, hi)}
    y_lb, y_ub = _interval(model, box)
    p = model.predict(np.random.default_rng(2).uniform(lo, hi, size=(50000, 3)))
    assert y_lb <= p.min() and p.max() <= y_ub

    # A pole inside the box leaves y unbounded rather than wrongly bounded.
    pole = _Fitted([1 / c], [0.0, 1.0], X)
    assert _interval(pole, box) == (-np.inf, np.inf)


def test_bad_bounds_arguments_raise():
    model = _Fitted([a * b], [0.0, 1.0], _X())
    with pytest.raises(ValueError, match="bounds must be"):
        to_discopt(model, dm.Model("t"), bounds="train")
    with pytest.raises(ValueError, match="y_bounds must be"):
        to_discopt(model, dm.Model("t"), y_bounds="data")
