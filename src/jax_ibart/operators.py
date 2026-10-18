"""Operator algebra for iBART feature engineering.

Each operator bundles three views that must stay consistent:

* ``fn``      : a JAX-jittable numeric function over feature columns.
* ``sym``     : a SymPy-level symbolic function (provenance / interpretability).
* ``valid``   : a domain-validity predicate returning a boolean mask of columns
                (or column-pairs) for which the operator is defined on the data.

The numeric side lives inside JAX traces; the symbolic side lives on the host
and is indexed in lockstep with the numeric columns. The two are never mixed
inside a trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np
import sympy as sp

# Small constant guarding protected division / log domains.
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Unit-propagation rules (dimensional analysis). Units are integer-exponent
# vectors over a fixed set of base dimensions. These mirror the R iBART package
# (R/operations.R): additive operators only combine matching units (returning
# None rejects the pair, which is how nonsensical descriptors like
# "volume + speed" are filtered); multiply/divide add/subtract exponents;
# transcendental and root unary operators yield unitless results.
# --------------------------------------------------------------------------- #


def u_keep(u):  # abs, neg, id
    return u


def u_zero(u):  # sqrt, log, exp, sin, cos -> unitless
    return np.zeros_like(u)


def u_double(u):  # square
    return 2 * u


def u_neg(u):  # inverse
    return -u


def u_same(a, b):  # add, sub, absdiff: require matching units
    return a if np.array_equal(a, b) else None


def u_add(a, b):  # multiply
    return a + b


def u_sub(a, b):  # divide
    return a - b


@dataclass(frozen=True)
class Operator:
    name: str
    arity: int  # 1 (unary) or 2 (binary)
    fn: Callable  # numeric: (col,) or (a, b) -> col   [jnp arrays]
    sym: Callable  # symbolic: (expr,) or (e1, e2) -> sympy expr
    valid: Callable  # domain predicate -> bool mask (jnp)
    unit: Callable  # unit rule: (u,) or (u_a, u_b) -> vector | None (None=reject)
    commutative: bool = False  # binary ops: skip duplicate (j, i) pairs


# --------------------------------------------------------------------------- #
# Unary operators (O1)
# --------------------------------------------------------------------------- #

UNARY: dict[str, Operator] = {
    "id": Operator(
        "id",
        1,
        lambda x: x,
        lambda e: e,
        lambda x: jnp.ones(x.shape[-1], bool),
        unit=u_keep,
    ),
    "neg": Operator(
        "neg",
        1,
        lambda x: -x,
        lambda e: -e,
        lambda x: jnp.ones(x.shape[-1], bool),
        unit=u_keep,
    ),
    "sq": Operator(
        "sq",
        1,
        lambda x: x * x,
        lambda e: e**2,
        lambda x: jnp.ones(x.shape[-1], bool),
        unit=u_double,
    ),
    "abs": Operator(
        "abs", 1, jnp.abs, sp.Abs, lambda x: jnp.ones(x.shape[-1], bool), unit=u_keep
    ),
    # Domain-restricted: only columns satisfying the predicate are emitted.
    "sqrt": Operator(
        "sqrt",
        1,
        lambda x: jnp.sqrt(jnp.abs(x)),
        lambda e: sp.sqrt(e),
        lambda x: jnp.all(x >= 0, axis=0),
        unit=u_zero,
    ),
    "log": Operator(
        "log",
        1,
        lambda x: jnp.log(jnp.abs(x) + EPS),
        lambda e: sp.log(e),
        lambda x: jnp.all(x > 0, axis=0),
        unit=u_zero,
    ),
    "exp": Operator(
        "exp",
        1,
        jnp.exp,
        sp.exp,
        # guard against overflow: only well-scaled columns
        lambda x: jnp.all(jnp.abs(x) < 50, axis=0),
        unit=u_zero,
    ),
    "inv": Operator(
        "inv",
        1,
        lambda x: 1.0 / jnp.where(jnp.abs(x) < EPS, jnp.nan, x),
        lambda e: 1 / e,
        lambda x: jnp.all(jnp.abs(x) > EPS, axis=0),
        unit=u_neg,
    ),
    "sin": Operator(
        "sin",
        1,
        lambda x: jnp.sin(jnp.pi * x),
        lambda e: sp.sin(sp.pi * e),
        lambda x: jnp.ones(x.shape[-1], bool),
        unit=u_zero,
    ),
    "cos": Operator(
        "cos",
        1,
        lambda x: jnp.cos(jnp.pi * x),
        lambda e: sp.cos(sp.pi * e),
        lambda x: jnp.ones(x.shape[-1], bool),
        unit=u_zero,
    ),
}


# --------------------------------------------------------------------------- #
# Binary operators (O2)
# --------------------------------------------------------------------------- #


def _safe_div(a, b):
    return a / jnp.where(jnp.abs(b) < EPS, jnp.nan, b)


BINARY: dict[str, Operator] = {
    "add": Operator(
        "add",
        2,
        lambda a, b: a + b,
        lambda e, f: e + f,
        lambda a, b: True,
        unit=u_same,
        commutative=True,
    ),
    "sub": Operator(
        "sub", 2, lambda a, b: a - b, lambda e, f: e - f, lambda a, b: True, unit=u_same
    ),
    "mul": Operator(
        "mul",
        2,
        lambda a, b: a * b,
        lambda e, f: e * f,
        lambda a, b: True,
        unit=u_add,
        commutative=True,
    ),
    "div": Operator(
        "div", 2, _safe_div, lambda e, f: e / f, lambda a, b: True, unit=u_sub
    ),
    # |a - b|: a single, directly interpretable operator (Ye et al. 2024).
    "absdiff": Operator(
        "absdiff",
        2,
        lambda a, b: jnp.abs(a - b),
        lambda e, f: sp.Abs(e - f),
        lambda a, b: True,
        unit=u_same,
        commutative=True,
    ),
}


def default_unary() -> list[Operator]:
    """The O1 set used in the catalysis study (Liu et al. 2022)."""
    return [
        UNARY[k] for k in ("id", "neg", "sq", "sqrt", "log", "exp", "abs", "sin", "cos")
    ]


def default_binary() -> list[Operator]:
    """The O2 set used in the catalysis study (Liu et al. 2022)."""
    return [BINARY[k] for k in ("add", "sub", "mul", "div", "absdiff")]
