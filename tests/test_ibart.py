"""End-to-end M3 integration: IBart.fit recovers a product descriptor.

Uses the bart backend (requires bartz); kept small for speed.
"""

import numpy as np
import pytest
import sympy as sp

pytest.importorskip("bartz")

from jax_ibart import IBart


def test_ibart_recovers_product_descriptor():
    # y = 5 * (a * b) + noise; primaries a, b, c, d.
    rng = np.random.default_rng(0)
    n = 90
    a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
    X = np.column_stack([a, b, c, d])
    y = 5.0 * (a * b) + rng.normal(0, 0.05, n)
    names = ["a", "b", "c", "d"]

    model = IBart(
        backend="bart",
        schedule=("b",),
        max_dim=1,
        n_keep=8,
        seed=0,
        n_perm=12,
        n_tree=40,
        n_draw=300,
        n_burn=300,
    ).fit(X, y, feature_names=names)

    # The single best descriptor should be the product a*b.
    target = sp.Symbol("a") * sp.Symbol("b")
    assert any(sp.simplify(e - target) == 0 for e in model.descriptors_)

    # And it should predict well.
    pred = model.predict(X)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.95
