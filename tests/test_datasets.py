"""Dataset loader tests (fast, no bartz)."""

import numpy as np

from jax_ibart.datasets import load_metal_support


def test_metal_support_shape_and_names():
    X, y, names, units = load_metal_support()
    assert X.shape == (91, 59)
    assert y.shape == (91,)
    assert len(names) == 59
    assert X.dtype == np.float64 and y.dtype == np.float64
    # The two features behind the published descriptor must be present.
    assert "Hfo" in names  # oxide formation enthalpy (dH_ox,fm)
    assert "Oxv" in names  # oxygen vacancy energy (O_vac)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))


def test_metal_support_units():
    X, y, names, units = load_metal_support()
    assert units.shape == (59, 12)  # (n_features, n_base_dims)
    # Hfo and Oxv are both pure-eV (identical unit vectors).
    hfo, oxv = names.index("Hfo"), names.index("Oxv")
    assert np.array_equal(units[hfo], units[oxv])
    assert units[hfo].sum() == 1  # single base dimension (eV)
