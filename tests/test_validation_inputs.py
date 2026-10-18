"""Input validation and edge cases for IBart.fit and the helpers it uses."""

import numpy as np
import pytest

pytest.importorskip("optax")

from jax_ibart import IBart, sparsify
from jax_ibart.uncertainty import conformal_intervals


def _data(n=40, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 2.0, size=(n, p))
    y = 3.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y, [f"x{i}" for i in range(p)]


def _fast(**kw):
    return IBart(backend="diff", steps=200, max_dim=1, seed=0, **kw)


def test_name_count_mismatch_raises():
    X, y, names = _data()
    with pytest.raises(ValueError, match="feature names"):
        _fast().fit(X, y, names[:2])


def test_y_length_mismatch_raises():
    X, y, names = _data()
    with pytest.raises(ValueError, match="y must have shape"):
        _fast().fit(X, y[:-1], names)


@pytest.mark.parametrize("where", ["X", "y"])
def test_nonfinite_raises(where):
    X, y, names = _data()
    if where == "X":
        X[3, 0] = np.nan
    else:
        y[3] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        _fast().fit(X, y, names)


def test_constant_y_raises():
    X, _, names = _data()
    with pytest.raises(ValueError, match="constant"):
        _fast().fit(X, np.ones(len(X)), names)


def test_bad_units_shape_raises():
    X, y, names = _data()
    with pytest.raises(ValueError, match="units"):
        _fast().fit(X, y, names, units=np.zeros((2, 4)))


def test_bad_score_raises():
    X, y, names = _data()
    with pytest.raises(ValueError, match="score"):
        _fast(score="aic").fit(X, y, names)
    with pytest.raises(ValueError, match="score"):
        sparsify.l0_best_subset(X, y, score="aic")


def test_single_feature_does_not_crash():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.0, size=(40, 1))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.01, 40)
    m = _fast(schedule=("b",)).fit(X, y, ["a"])
    assert m.history_[1] == {"stage": "b", "generated": 0, "kept": 1}
    assert np.isfinite(m.predict(X)).all()


def test_unfitted_methods_raise():
    m = _fast()
    for call in (lambda: m.predict(np.ones((2, 3))), m.expression, m.latex):
        with pytest.raises(RuntimeError, match="not fitted"):
            call()


def test_backend_change_after_init_is_used():
    X, y, names = _data()
    m = IBart(backend="bart")
    m.set_params(backend="diff", steps=200, max_dim=1)
    m.fit(X, y, names)
    assert type(m.selector).__name__ == "DiffSelector"


def test_failed_refit_keeps_previous_fit():
    X, y, names = _data()
    m = _fast().fit(X, y, names)
    before = list(m.descriptors_)
    with pytest.raises(ValueError):
        m.fit(X, y, ["a", "b"])
    assert m.descriptors_ == before
    assert m._feature_names == names


def test_transform_is_public_design_matrix():
    X, y, names = _data()
    m = _fast().fit(X, y, names)
    D = m.transform(X)
    assert D.shape == (len(X), len(m.descriptors_))
    np.testing.assert_allclose(m.predict(X), m.coef_[0] + D @ m.coef_[1:])


def test_conformal_too_few_calibration_points_is_infinite():
    X, y, names = _data(n=20)
    with pytest.warns(UserWarning, match="too few"):
        cm = conformal_intervals(
            lambda: _fast(), X, y, names, alpha=0.1, cal_frac=0.3, seed=0
        )
    assert cm.half_width == float("inf")
