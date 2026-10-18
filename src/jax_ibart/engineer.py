"""Feature-engineering engine: apply operators to a descriptor pool.

Produces an expanded numeric matrix plus matching SymPy provenance, then filters
invalid / constant / duplicate columns. Binary operators are the O(d^2) blowup
that makes this the prime JAX/GPU target, so each operator is evaluated over all
columns or column pairs in one batched JAX call.

Optional dimensional analysis: when a pool carries unit-exponent vectors,
operators propagate units and additive operators drop unit-mismatched pairs
(e.g. "volume + speed"), mirroring the R iBART unit filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import sympy as sp

from .operators import Operator, u_add, u_same, u_sub


@dataclass
class Pool:
    """A descriptor pool: numeric columns paired with symbolic provenance.

    ``units`` is optional. When present it has shape (d, n_dims): row j is the
    integer unit-exponent vector of column j. When None, dimensional analysis is
    disabled and no unit filtering is done.
    """

    X: jnp.ndarray  # (n, d)
    exprs: list[sp.Expr]  # length d
    units: np.ndarray | None = None  # (d, n_dims) or None

    @property
    def n_descriptors(self) -> int:
        return len(self.exprs)


def from_primary(X, names: list[str], units=None) -> Pool:
    """Build the initial pool from primary features and their symbols.

    Numeric columns are float32: this is the dtype bartz consumes, and global
    ``jax_enable_x64`` must stay off (it breaks bartz's MCMC). Higher-precision
    work (final LASSO/OLS) happens host-side in numpy, not in JAX traces.

    ``units`` (optional) is (d, n_dims): the unit-exponent vector per feature.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    u = None if units is None else np.asarray(units, dtype=np.int64)
    return Pool(X=X, exprs=[sp.Symbol(n) for n in names], units=u)


def apply_unary(pool: Pool, ops: list[Operator], corr_max_d: int = 10000) -> Pool:
    """Apply unary operators to every column.

    Vectorized: each operator is evaluated on all valid columns in one batched
    JAX call (``op.fn`` broadcasts over the (n, k) slice), and unit vectors are
    transformed with a single array op. The symbolic provenance is built in a
    host-side loop (SymPy is not vectorizable). ``corr_max_d`` is forwarded to
    :func:`clean` (set 0 to keep collinear columns, e.g. for implicit discovery).
    """
    has_units = pool.units is not None
    blocks, exprs, units = [], [], ([] if has_units else None)
    for op in ops:
        idx = np.where(np.asarray(op.valid(pool.X)))[0]
        if idx.size == 0:
            continue
        blocks.append(op.fn(pool.X[:, idx]))  # (n, k) batched
        exprs.extend(op.sym(pool.exprs[j]) for j in idx)
        if has_units:
            units.append(op.unit(pool.units[idx]))  # vectorized unit rule
    return _concat(blocks, exprs, units, corr_max_d)


def apply_binary(pool: Pool, ops: list[Operator], corr_max_d: int = 10000) -> Pool:
    """Apply binary operators to all column pairs.

    Vectorized over pairs: index arrays (I, J) select the operands, ``op.fn``
    evaluates all pairs at once via broadcasting (the O(d^2) blowup that is the
    prime JAX/GPU target), and the unit filter is a vectorized numpy mask.
    """
    d = pool.n_descriptors
    has_units = pool.units is not None
    U = pool.units
    blocks, exprs, units = [], [], ([] if has_units else None)

    for op in ops:
        if op.commutative:
            I, J = np.triu_indices(d, k=1)
        else:
            ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
            off = ii != jj
            I, J = ii[off], jj[off]
        if I.size == 0:
            continue

        new_u = None
        if has_units:
            if op.unit is u_same:  # additive: keep if equal
                ok = np.all(U[I] == U[J], axis=1)
                I, J = I[ok], J[ok]
                new_u = U[I]
            elif op.unit is u_add:  # multiply: add exponents
                new_u = U[I] + U[J]
            elif op.unit is u_sub:  # divide: subtract
                new_u = U[I] - U[J]
            else:  # generic fallback
                pairs = [(op.unit(U[a], U[b]), a, b) for a, b in zip(I, J)]
                keep = [(u, a, b) for (u, a, b) in pairs if u is not None]
                if not keep:
                    continue
                new_u = np.array([u for u, _, _ in keep])
                I = np.array([a for _, a, _ in keep])
                J = np.array([b for _, _, b in keep])
            if I.size == 0:
                continue

        ok = np.asarray(op.valid(pool.X[:, I], pool.X[:, J]), dtype=bool)
        if ok.ndim == 0:
            if not ok:
                continue
        else:
            I, J = I[ok], J[ok]
            if has_units:
                new_u = new_u[ok]
            if I.size == 0:
                continue

        blocks.append(op.fn(pool.X[:, I], pool.X[:, J]))  # (n, npairs) batched
        exprs.extend(op.sym(pool.exprs[a], pool.exprs[b]) for a, b in zip(I, J))
        if has_units:
            units.append(new_u)
    return _concat(blocks, exprs, units, corr_max_d)


def _concat(blocks, exprs, units, corr_max_d: int = 10000) -> Pool:
    if not blocks:
        return Pool(
            X=jnp.zeros((0, 0)),
            exprs=[],
            units=None if units is None else np.zeros((0, 0), dtype=np.int64),
        )
    X = jnp.concatenate(blocks, axis=1)
    u = None if units is None else np.concatenate(units, axis=0).astype(np.int64)
    return clean(Pool(X=X, exprs=exprs, units=u), corr_max_d=corr_max_d)


def clean(pool: Pool, corr_thresh: float = 1 - 1e-8, corr_max_d: int = 10000) -> Pool:
    """Drop invalid (NaN/inf), near-constant, and duplicate/collinear columns.

    All numeric *decisions* are made in float64 (host-side numpy), even though
    the columns are stored f32 for the bartz pipeline: collinearity/rank
    judgements on small-n, highly correlated engineered descriptors are exactly
    where f32's ~7 digits are not enough. The returned pool keeps the original f32 columns; only the
    keep/drop choices use f64.
    """
    X64 = np.asarray(pool.X, dtype=np.float64)
    n, d = X64.shape
    keep = np.ones(d, dtype=bool)

    # 1. Finite and non-constant (f64).
    finite = np.all(np.isfinite(X64), axis=0)
    std = np.where(finite, X64.std(axis=0), 0.0)
    keep &= finite & (std > 1e-12)

    # 2. Symbolic-duplicate pruning (structural, keep first occurrence).
    #    SymPy auto-canonicalizes commutative args on construction, so srepr
    #    catches reorderings (a+b == b+a) directly. Deeper algebraic identities
    #    (e.g. sqrt(x**2) == abs(x)) are left to the numeric collinearity pass
    #    below -- using sp.simplify here instead costs ~100x more (it dominated
    #    feature-engineering time) for no gain over the numeric dedup.
    seen: set = set()
    for j, e in enumerate(pool.exprs):
        if not keep[j]:
            continue
        key = sp.srepr(e)
        if key in seen:
            keep[j] = False
        else:
            seen.add(key)

    # 3. Numeric collinearity dedup (f64): greedily drop columns that are
    #    near-perfectly correlated with an already-kept column. Catches
    #    numeric duplicates symbolic canonicalization misses (e.g. sqrt(x**2)
    #    vs abs(x)) and ill-conditioning before it reaches the linear solve.
    #    This greedy pass is O(d^2 n); skip it above corr_max_d (pre-selection,
    #    exact duplicates there are harmless and BART screening follows anyway).
    surv = np.where(keep)[0]
    if 1 < surv.size <= corr_max_d:
        Z = X64[:, surv]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-300)
        kept_cols: list[int] = []
        for col, j in enumerate(surv):
            if kept_cols:
                corr = np.abs(Z[:, kept_cols].T @ Z[:, col] / n)
                if corr.max() > corr_thresh:
                    keep[j] = False
                    continue
            kept_cols.append(col)

    idx = np.where(keep)[0]
    u = None if pool.units is None else pool.units[idx]
    return Pool(X=pool.X[:, idx], exprs=[pool.exprs[j] for j in idx], units=u)
