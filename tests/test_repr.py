"""Tests for richer equation representations (LaTeX / HTML)."""

import numpy as np
import pytest

pytest.importorskip("optax")

from jax_ibart import IBart


def _fit():
    rng = np.random.default_rng(0)
    n = 90
    a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
    X = np.column_stack([a, b, c, d])
    y = 5.0 * (a * b) + rng.normal(0, 0.05, n)
    return IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=800).fit(
        X, y, ["a", "b", "c", "d"]
    )


def test_ibart_latex_and_html():
    m = _fit()
    tex = m.latex()
    assert tex.startswith("y = ")
    assert "a b" in tex or "a*b" in tex.replace(" ", "")  # product present
    # rich repr for Jupyter
    assert m._repr_latex_().startswith("$") and m._repr_latex_().endswith("$")
    # html snippet wraps a MathJax span
    assert "\\(" in m.to_html() and "\\)" in m.to_html()


def test_expression_rounds_coefficients():
    m = _fit()
    # default display rounds to 4 sig figs -> short string
    assert "0.0000000" not in m.latex()


def test_discovered_system_latex():
    from scipy.integrate import solve_ivp

    from jax_ibart import dynamics

    def f(t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

    t = np.arange(0, 10, 0.01)
    sol = solve_ivp(f, [0, t[-1]], [-8, 8, 27], t_eval=t, rtol=1e-9, atol=1e-9)
    sysd = dynamics.discover_ode(sol.y.T, t, ["x", "y", "z"], max_terms=3)
    tex = sysd.latex()
    assert r"\frac{dx}{dt}" in tex and r"\begin{aligned}" in tex
    assert sysd._repr_latex_().startswith("$$")
