"""Tests for SINDy-style ODE discovery and the implicit-constraint foundation."""

import numpy as np
import pytest
import sympy as sp

from jax_ibart import dynamics


def _lorenz_traj(dt=0.002, T=20.0):
    from scipy.integrate import solve_ivp

    def f(t, s):
        x, y, z = s
        return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]

    t = np.arange(0, T, dt)
    sol = solve_ivp(f, [t[0], t[-1]], [-8, 8, 27], t_eval=t,
                    rtol=1e-9, atol=1e-9)
    return sol.y.T, t


def _support(expr):
    return {str(s) for s in sp.sympify(expr).free_symbols}


def test_finite_difference_linear():
    t = np.linspace(0, 1, 50)
    X = np.column_stack([3 * t, t ** 2])
    dX = dynamics.finite_difference(X, t)
    np.testing.assert_allclose(dX[:, 0], 3.0, atol=1e-6)
    np.testing.assert_allclose(dX[1:-1, 1], 2 * t[1:-1], atol=1e-2)


def test_discover_lorenz():
    X, t = _lorenz_traj()
    sys = dynamics.discover_ode(X, t, ["x", "y", "z"], max_terms=3)

    assert all(r > 0.999 for r in sys.r2)
    # correct sparse support per equation (the Lorenz terms)
    sx, sy, sz = (_support(e) for e in sys.rhs)
    assert sx == {"x", "y"}              # 10*(y - x)
    assert sy == {"x", "y", "z"}         # 28x - y - x*z  (x*z -> {x,z}, plus y)
    assert sz == {"x", "y", "z"}         # x*y - (8/3) z

    # coefficient sanity on the linear dx/dt = 10y - 10x
    poly = sp.Poly(sys.rhs[0], *sp.symbols("x y z"))
    coeffs = {m: float(c) for m, c in poly.terms()}
    assert abs(coeffs.get((1, 0, 0), 0) + 10) < 0.1   # x coefficient ~ -10
    assert abs(coeffs.get((0, 1, 0), 0) - 10) < 0.1   # y coefficient ~ +10


def test_discover_reaction_rhs():
    # Rediscover the mole-balance RHS from the example reaction dataset.
    from jax_ibart.datasets import load_reaction

    C, dCdt, species, _ = load_reaction()
    sys = dynamics.discover_ode(C, t=None, feature_names=species, dXdt=dCdt,
                                max_terms=2, screen=None)

    def dominant_support(expr):
        # support of the largest-magnitude term in the polynomial
        poly = sp.Poly(sp.expand(expr), *sp.symbols("CA CB CC CD"))
        mono, coef = max(poly.terms(), key=lambda mc: abs(mc[1]))
        names = ["CA", "CB", "CC", "CD"]
        return {names[i] for i, p in enumerate(mono) if p > 0}

    # bimolecular rate law: dCA/dt dominated by C_A * C_B
    assert dominant_support(sys.rhs[0]) == {"CA", "CB"}
    # first-order C -> D: dCD/dt dominated by C_C
    assert dominant_support(sys.rhs[3]) == {"CC"}
    # the strong equations fit well even with measurement noise; the weak,
    # slow dCD/dt (small intermediate) is genuinely hard from noisy data, so we
    # assert its *structure* (above) rather than a high R^2.
    assert all(r > 0.9 for r in sys.r2[:3])


def test_implicit_constraint_circle():
    th = np.linspace(0, 2 * np.pi, 200)
    C = np.column_stack([np.cos(th), np.sin(th)])
    r = dynamics.implicit_constraints(C, ["x", "y"], degree=2)
    # x^2 + y^2 - 1 = 0 : an exact constraint, residual ~ 0
    assert r["residual"] < 1e-6
    assert _support(r["constraint"]) == {"x", "y"}
    # the constraint should contain x**2 and y**2 with equal sign, minus a const
    expr = sp.expand(r["constraint"])
    x, y = sp.symbols("x y")
    assert abs(float(expr.coeff(x, 2)) - float(expr.coeff(y, 2))) < 1e-3
