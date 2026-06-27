"""Data-driven discovery of dynamical systems (SINDy-style), built on iBART.

This reuses iBART's descriptor machinery for sparse identification of governing
equations:

- :func:`discover_ode` -- explicit ODE discovery (Brunton, Proctor & Kutz 2016).
  For each state x_i it sparse-regresses the time derivative dx_i/dt on a
  library of candidate terms (linear, squares, pairwise products by default).
  iBART's operator engine builds a richer, dimensionally-filtered library than a
  fixed polynomial basis, and the LASSO -> l0 stage does the sparse selection.

- :func:`implicit_constraints` -- the *foundation* for differential-algebraic
  (DAE) discovery: finding sparse combinations of library terms that are ~0
  (algebraic constraints / conserved quantities). This is the implicit / null
  space problem (Implicit-SINDy, Mangan et al. 2016; SINDy-PI, Kaheman et al.
  2020), which is fundamentally different from regressing on a target. The
  function here is a documented prototype (SVD null space); a production version
  needs sparse-null-space selection and a weak formulation (see the docstring
  and docs/references.md).

References: see docs/references.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import sympy as sp

from . import engineer, sparsify
from .operators import BINARY, UNARY


# --------------------------------------------------------------------------- #
# Derivative estimation
# --------------------------------------------------------------------------- #

def finite_difference(X_t, t):
    """Central finite-difference time derivatives of a trajectory.

    ``X_t`` is (n_time, n_state); ``t`` is (n_time,). Returns dX/dt with the same
    shape (forward/backward differences at the endpoints). For noisy data prefer
    a smoothing or weak/integral formulation (see docs/references.md); this is
    the clean-data baseline.
    """
    X_t = np.asarray(X_t, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dX = np.empty_like(X_t)
    dX[1:-1] = (X_t[2:] - X_t[:-2]) / (t[2:] - t[:-2])[:, None]
    dX[0] = (X_t[1] - X_t[0]) / (t[1] - t[0])
    dX[-1] = (X_t[-1] - X_t[-2]) / (t[-1] - t[-2])
    return dX


# --------------------------------------------------------------------------- #
# Library construction (reuses the iBART engineer + unit filter)
# --------------------------------------------------------------------------- #

def polynomial_library(X, feature_names, degree: int = 2, units=None,
                       dedup_collinear: bool = True):
    """A degree-<=2 polynomial library: linear, squares, pairwise products.

    Built with the iBART feature engine, so the dimensional-analysis filter
    applies when ``units`` is given (a real asset for physical systems).

    ``dedup_collinear`` controls the numeric collinearity pruning. Leave it on
    for explicit regression (ODEs). Turn it OFF for implicit constraint
    discovery: there, collinearity among library columns *is* the constraint we
    want to find, so dropping it would discard the answer.
    """
    cmd = 10000 if dedup_collinear else 0
    lin = engineer.from_primary(X, feature_names, units=units)
    blocks = [lin]
    if degree >= 2:
        blocks.append(engineer.apply_unary(lin, [UNARY["sq"]], corr_max_d=cmd))   # x^2
        blocks.append(engineer.apply_binary(lin, [BINARY["mul"]], corr_max_d=cmd))  # x*y
    Xcat = jnp.concatenate([b.X for b in blocks if b.n_descriptors], axis=1)
    exprs = [e for b in blocks for e in b.exprs]
    u = None
    if units is not None:
        u = np.concatenate([b.units for b in blocks if b.n_descriptors], axis=0)
    # corr_max_d=0 disables the O(d^2) collinearity pass (keeps collinear cols)
    return engineer.clean(engineer.Pool(X=Xcat, exprs=exprs, units=u), corr_max_d=cmd)


@dataclass
class DiscoveredSystem:
    """A discovered ODE system: one symbolic RHS per state variable."""

    state_names: list
    rhs: list            # list[sympy.Expr], dx_i/dt = rhs[i]
    r2: list             # per-equation in-sample R^2

    def __str__(self):
        def fmt(e):
            return sp.nsimplify(e, rational=False).evalf(4)
        return "\n".join(f"d{n}/dt = {fmt(e)}  (R^2={r:.3f})"
                         for n, e, r in zip(self.state_names, self.rhs, self.r2))


# --------------------------------------------------------------------------- #
# Explicit ODE discovery (SINDy)
# --------------------------------------------------------------------------- #

def discover_ode(X_t, t, feature_names, degree: int = 2, max_terms: int = 3,
                 units=None, dXdt=None, coef_threshold: float = 1e-3,
                 screen: str | None = None, seed: int = 0):
    """Discover dx_i/dt = f_i(x) by sparse regression on a polynomial library.

    Parameters
    ----------
    X_t : (n_time, n_state) trajectory samples.
    t : (n_time,) time stamps.
    feature_names : names of the state variables.
    degree : polynomial library degree (<=2 here).
    max_terms : max non-zero terms per equation (l0 best subset).
    units : optional (n_state, n_dims) unit-exponent matrix.
    dXdt : optional precomputed derivatives; if None, central finite differences.
        (For multi-trajectory data, compute derivatives per trajectory and pass
        the stacked array here.)
    screen : None or "bart". With "bart", a BART-G.SE nonparametric screen prunes
        the library against each derivative before the l0 search, i.e. BART finds
        which terms enter each mole-balance RHS. Needs ``bartz``.

    Returns a :class:`DiscoveredSystem`.
    """
    import jax

    X_t = np.asarray(X_t, dtype=np.float64)
    dX = finite_difference(X_t, t) if dXdt is None else np.asarray(dXdt, np.float64)

    lib = polynomial_library(X_t, feature_names, degree=degree, units=units)
    L = np.asarray(lib.X, dtype=np.float64)

    bart = None
    if screen == "bart":
        from .selection import BartGSESelector
        bart = BartGSESelector(n_perm=30, alpha=0.20, n_tree=40,
                               n_draw=400, n_burn=400)

    rhs, r2 = [], []
    for i in range(len(feature_names)):
        yi = dX[:, i]
        if bart is not None:
            key = jax.random.PRNGKey(seed + i)
            keep = np.asarray(bart.select(L, yi, key=key))
            cand = np.where(keep)[0] if keep.any() else np.arange(L.shape[1])
        else:
            cand = sparsify.lasso_screen(
                L, yi, n_keep=min(2 * max_terms + 2, L.shape[1]))
        idx, coef = sparsify.l0_best_subset(L[:, cand], yi, max_dim=max_terms,
                                            score="bic")
        # STLSQ-style cleanup: drop terms whose coefficient is negligible
        # relative to the largest, then refit OLS on the survivors.
        slopes = coef[1:]
        big = np.abs(slopes) > coef_threshold * (np.abs(slopes).max() + 1e-30)
        idx = [j for j, keep in zip(idx, big) if keep]
        cols = [L[:, cand[j]] for j in idx]
        A = np.column_stack([np.ones(len(yi))] + cols) if cols else np.ones((len(yi), 1))
        coef, _ = sparsify._ols(np.column_stack(cols), yi) if cols else (
            np.array([yi.mean()]), 0.0)

        expr = sp.Float(coef[0])
        for c, j in zip(coef[1:], idx):
            expr = expr + sp.Float(c) * lib.exprs[cand[j]]
        pred = A @ coef
        ss = 1 - ((yi - pred) ** 2).sum() / (((yi - yi.mean()) ** 2).sum() + 1e-30)
        rhs.append(expr)
        r2.append(float(ss))
    return DiscoveredSystem(list(feature_names), rhs, r2)


# --------------------------------------------------------------------------- #
# DAE foundation: implicit / null-space constraint discovery (PROTOTYPE)
# --------------------------------------------------------------------------- #

def implicit_constraints(X, feature_names, degree: int = 2, units=None,
                         tol: float = 1e-2):
    """PROTOTYPE: find algebraic constraints g(x) ~ 0 (the DAE algebraic part).

    A differential-algebraic system couples dynamics dx/dt = f with algebraic
    constraints 0 = g(x, z). Discovering g is an *implicit* problem: find a
    sparse coefficient vector xi != 0 with Theta(x) @ xi ~ 0, i.e. a vector in
    (or near) the null space of the library matrix. This is NOT the regress-on-a-
    target problem iBART/SINDy solve, which is why it needs separate machinery.

    This prototype:
      1. builds the iBART library Theta (with the unit filter),
      2. standardizes columns and takes the SVD,
      3. returns the smallest right-singular vector as the leading candidate
         constraint, with its normalized residual ||Theta xi|| / ||Theta||.

    A production DAE module (the "something special to develop") would add:
      - *sparse* null-space selection (only a few terms active), e.g. SINDy-PI's
        per-term implicit regressions or sparse-null-space algorithms, instead of
        the dense smallest singular vector;
      - a *weak / integral* formulation to handle noise and avoid derivatives
        (Messenger & Bortz 2021);
      - separation of differential vs algebraic relations and DAE index/
        consistency handling.

    See docs/references.md (Mangan et al. 2016; Kaheman et al. 2020; Messenger &
    Bortz 2021).

    Returns a dict with the candidate constraint expression, its coefficient
    vector over the library terms, and the residual (smaller = closer to an
    exact constraint).
    """
    lib = polynomial_library(X, feature_names, degree=degree, units=units,
                             dedup_collinear=False)
    Theta = np.asarray(lib.X, dtype=np.float64)
    exprs = list(lib.exprs)
    # augment with a constant column so constraints like x^2 + y^2 - 1 = 0 are
    # representable (the engineer drops constant columns, so add it here).
    Theta = np.column_stack([Theta, np.ones(len(Theta))])
    exprs = exprs + [sp.Integer(1)]

    # scale columns to unit norm (not standardize: that breaks the constant
    # column), then the smallest right-singular vector is the best near-null
    # combination.
    norms = np.linalg.norm(Theta, axis=0) + 1e-30
    _, S, Vt = np.linalg.svd(Theta / norms, full_matrices=False)
    xi = Vt[-1] / norms
    residual = float(S[-1] / (S[0] + 1e-30))

    # normalize so the largest coefficient is 1, then threshold small terms
    xi = xi / (np.abs(xi).max() + 1e-30)
    expr = sp.Float(0.0)
    for c, e in zip(xi, exprs):
        if abs(c) > tol:
            expr = expr + sp.Float(round(c, 4)) * e
    return {"constraint": expr, "coef": xi, "terms": [str(e) for e in exprs],
            "residual": residual,
            "note": "PROTOTYPE: dense SVD null space; production needs sparse "
                    "null-space selection + weak form (see docstring)."}
