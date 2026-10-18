"""Embed a fitted iBART model in an optimization model.

A fitted :class:`~jax_ibart.IBart` is a linear combination of a few closed-form
descriptors, so it can be written exactly as algebra in an equation-oriented
modeling language and optimized over globally: find the inputs that maximize or
minimize the prediction, hit a target value, or use the model as a constraint
inside a larger design problem.

The translation walks each descriptor's sympy tree and emits native nodes
(no opaque callbacks), so a global solver can relax it and certify the result.
Shared subexpressions (e.g. ``Abs(Hfo/Oxv)`` appearing in two descriptors) map
to a single node. The prediction is returned as a *variable* ``y`` defined by
the equality ``y == expression`` (an epigraph form): global solvers bound a
plain variable objective much better than a composite nonlinear one, and in
testing this both found better optima and certified ones the direct form could
not.

The descriptors are only meaningful where the training pipeline could evaluate
them, so by default the export also adds *domain constraints* derived from the
reference data (the training ``X`` unless ``X_ref`` is given):

* ``log(a)``: ``a >= min(a on X_ref)``  (strictly positive)
* ``sqrt(a)``: ``a >= 0``
* ``a / b`` or ``b**-k``: ``b`` keeps its sign on ``X_ref`` with
  ``|b| >= min(|b| on X_ref)``, which excludes poles. If ``b`` changes sign on
  ``X_ref`` the constraint is ``abs(b) >= min(|b|)`` instead.

and bounds each input variable to its range on ``X_ref``. Extrapolating outside
the data a model was fitted on is rarely what you want; widen ``bounds``
deliberately if you do.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

# sympy function -> discopt.modeling function name
_DISCOPT_FUNCS = {
    sp.exp: "exp",
    sp.log: "log",
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.Abs: "abs",
    sp.tanh: "tanh",
}


def to_discopt(
    model,
    m,
    x=None,
    *,
    bounds="data",
    domain=True,
    X_ref=None,
    name="y",
    y_bounds="auto",
):
    """Add a fitted iBART model to a discopt model.

    Parameters
    ----------
    model : IBart
        A fitted model.
    m : discopt.modeling.Model
        The optimization model to add variables and constraints to.
    x : dict or sequence, optional
        The inputs. A dict maps feature names to discopt expressions; a
        sequence (list, array, or a vector variable) is indexed in
        ``feature_names`` order. If None, one continuous variable per feature
        the model uses is created, named after the feature.
    bounds : "data", tuple of arrays, or None
        Bounds for created variables: ``"data"`` uses the range of ``X_ref``,
        ``(lb, ub)`` gives arrays over all features. Ignored when ``x`` is
        given.
    domain : bool
        Add the domain constraints described in the module docstring.
    X_ref : array, optional
        Reference data for bounds and domain constraints; defaults to the
        training ``X``.
    name : str
        Name of the prediction variable (must be unique in ``m``).
    y_bounds : "auto", (lb, ub), or None
        Bounds for the prediction variable. ``"auto"`` derives them by interval
        arithmetic over the input box when the inputs are created here (they
        are conservative, so they never exclude a reachable prediction, and
        they help the solver); with user-supplied ``x`` the box is not known
        and ``y`` is left unbounded, so pass ``(lb, ub)`` to bound it yourself.
        ``None`` leaves ``y`` unbounded.

    Returns
    -------
    x : dict[str, Expression]
        The inputs the model uses, by feature name.
    y : Variable
        The model prediction, constrained to equal
        ``coef_[0] + sum(coef_[k] * descriptor_k)``.

    Examples
    --------
    >>> import discopt.modeling as dm
    >>> m = dm.Model("design")
    >>> x, y = to_discopt(fitted, m)
    >>> m.maximize(y)
    >>> res = m.solve()
    >>> {k: float(res.value(v)) for k, v in x.items()}
    """
    try:
        import discopt.modeling as dm
    except ImportError as e:  # pragma: no cover - exercised without discopt
        raise ImportError(
            "to_discopt needs discopt (Python >= 3.12): "
            "pip install 'jax-ibart[discopt]'"
        ) from e

    model._check_fitted()
    names = list(model._feature_names)
    X_ref = model._X_train if X_ref is None else np.asarray(X_ref, dtype=np.float64)
    if X_ref.ndim != 2 or X_ref.shape[1] != len(names):
        raise ValueError(f"X_ref must have shape (n, {len(names)})")

    used = sorted(
        {str(s) for d in model.descriptors_ for s in d.free_symbols},
        key=names.index,
    )
    if x is None:
        lb, ub = _bounds(bounds, X_ref, len(names))
        inputs = {
            n: m.continuous(n, lb=lb[names.index(n)], ub=ub[names.index(n)])
            for n in used
        }
    elif isinstance(x, dict):
        missing = [n for n in used if n not in x]
        if missing:
            raise ValueError(f"x is missing inputs the model uses: {missing}")
        inputs = {n: x[n] for n in used}
    else:
        inputs = {n: x[names.index(n)] for n in used}

    walker = _Walker(dm, m, inputs, names, X_ref, domain)
    expr = float(model.coef_[0])
    for c, d in zip(model.coef_[1:], model.descriptors_):
        expr = expr + float(c) * walker(d)
    if isinstance(y_bounds, str) and y_bounds != "auto":
        raise ValueError(f"y_bounds must be 'auto', (lb, ub) or None, not {y_bounds!r}")
    y_lb, y_ub = -np.inf, np.inf
    if y_bounds is not None and not isinstance(y_bounds, str):
        y_lb, y_ub = (float(v) for v in y_bounds)
    elif y_bounds == "auto" and x is None:
        box = {n: (float(lb[names.index(n)]), float(ub[names.index(n)])) for n in used}
        y_lb, y_ub = _interval(model, box)
    kw = {}
    if np.isfinite(y_lb):
        kw["lb"] = y_lb
    if np.isfinite(y_ub):
        kw["ub"] = y_ub
    y = m.continuous(name, **kw)
    m.subject_to(y == expr, name=f"{name}_ibart")
    return inputs, y


def _interval(model, box):
    """Bounds on the prediction over the input box, by interval arithmetic.

    The bounds are conservative (dependency between repeated inputs only makes
    them wider), so they never cut off a feasible prediction; they just give
    the solver a finite range for ``y``. Infinite when any descriptor is
    unbounded on the box (e.g. a pole inside it).
    """
    lo = hi = float(model.coef_[0])
    for c, d in zip(model.coef_[1:], model.descriptors_):
        a, b = _imul((float(c), float(c)), _ieval(d, box))
        lo, hi = lo + a, hi + b
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return -np.inf, np.inf
    pad = 1e-6 * (1.0 + max(abs(lo), abs(hi)))  # float rounding
    return lo - pad, hi + pad


_INF = (-np.inf, np.inf)


def _imul(x, y):
    with np.errstate(invalid="ignore"):
        p = [x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]]
    if any(np.isnan(p)):  # 0 * inf
        return _INF
    return min(p), max(p)


def _ipow(x, n):
    """Integer power n >= 0."""
    if n % 2 == 1 or x[0] >= 0:
        return min(x[0] ** n, x[1] ** n), max(x[0] ** n, x[1] ** n)
    if x[1] <= 0:
        return x[1] ** n, x[0] ** n
    return 0.0, max(x[0] ** n, x[1] ** n)


def _iinv(x):
    if x[0] > 0 or x[1] < 0:
        return 1.0 / x[1], 1.0 / x[0]
    return _INF


def _iperiodic(f, x, peak, trough):
    """sin/cos over x: endpoints plus any interior extremum."""
    if not (np.isfinite(x[0]) and np.isfinite(x[1])) or x[1] - x[0] >= 2 * np.pi:
        return -1.0, 1.0
    lo, hi = sorted((f(x[0]), f(x[1])))
    two_pi = 2 * np.pi
    if np.ceil((x[0] - peak) / two_pi) <= np.floor((x[1] - peak) / two_pi):
        hi = 1.0
    if np.ceil((x[0] - trough) / two_pi) <= np.floor((x[1] - trough) / two_pi):
        lo = -1.0
    return lo, hi


def _ieval(e, box):
    """Interval of sympy expression ``e`` with inputs in ``box``."""
    if e.is_Symbol:
        return box[str(e)]
    if e.is_Number or e.is_NumberSymbol:
        return float(e), float(e)
    if e.is_Add:
        lo = hi = 0.0
        for a in e.args:
            a, b = _ieval(a, box)
            lo, hi = lo + a, hi + b
        return lo, hi
    if e.is_Mul:
        out = (1.0, 1.0)
        for a in e.args:
            out = _imul(out, _ieval(a, box))
        return out
    if e.is_Pow:
        base, p = e.args
        if not p.is_number:
            return _INF
        x = _ieval(base, box)
        if p.is_integer:
            k = int(p)
            v = _ipow(x, abs(k))
            return _iinv(v) if k < 0 else v
        # fractional power: real only on the non-negative part of the base
        lo, hi, q = max(x[0], 0.0), max(x[1], 0.0), float(p)
        if q > 0:
            return lo**q, hi**q
        if hi == 0:
            return _INF
        return hi**q, (lo**q if lo > 0 else np.inf)
    if e.func in _DISCOPT_FUNCS:
        x = _ieval(e.args[0], box)
        if e.func is sp.exp:
            return float(np.exp(x[0])), float(np.exp(x[1]))
        if e.func is sp.tanh:
            return float(np.tanh(x[0])), float(np.tanh(x[1]))
        if e.func is sp.log:
            if x[1] <= 0:
                return _INF
            return (float(np.log(x[0])) if x[0] > 0 else -np.inf), float(np.log(x[1]))
        if e.func is sp.Abs:
            if x[0] >= 0:
                return x
            if x[1] <= 0:
                return -x[1], -x[0]
            return 0.0, max(-x[0], x[1])
        if e.func is sp.sin:
            return _iperiodic(np.sin, x, np.pi / 2, -np.pi / 2)
        if e.func is sp.cos:
            return _iperiodic(np.cos, x, 0.0, np.pi)
    return _INF


def _bounds(bounds, X_ref, p):
    if isinstance(bounds, str):
        if bounds != "data":
            raise ValueError(f"bounds must be 'data', (lb, ub) or None, not {bounds!r}")
        return X_ref.min(axis=0), X_ref.max(axis=0)
    if bounds is None:
        return np.full(p, -np.inf), np.full(p, np.inf)
    lb, ub = (np.broadcast_to(np.asarray(b, dtype=np.float64), (p,)) for b in bounds)
    return lb, ub


class _Walker:
    """sympy expression -> discopt expression, with shared nodes and domains."""

    def __init__(self, dm, m, inputs, names, X_ref, domain):
        self.dm, self.m, self.inputs, self.domain = dm, m, inputs, domain
        self.syms = [sp.Symbol(n) for n in names]
        self.cols = [X_ref[:, i] for i in range(X_ref.shape[1])]
        self.cache: dict = {}
        self.guarded: set = set()

    def __call__(self, e):
        if e not in self.cache:
            self.cache[e] = self._build(e)
        return self.cache[e]

    def _ref(self, e):
        """Numeric values of subexpression ``e`` on the reference data."""
        f = sp.lambdify(self.syms, e, modules="numpy")
        with np.errstate(all="ignore"):
            v = np.asarray(f(*self.cols), dtype=np.float64)
        return np.broadcast_to(v, self.cols[0].shape)

    def _guard(self, kind, e, node):
        if not self.domain or (kind, e) in self.guarded:
            return
        self.guarded.add((kind, e))
        v = self._ref(e)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        if kind == "sqrt":
            self.m.subject_to(node >= 0)
        elif kind == "log":
            self.m.subject_to(node >= float(v.min()) if v.min() > 0 else node >= 0)
        else:  # denominator
            delta = float(np.abs(v).min())
            if delta == 0.0:
                return
            if (v > 0).all():
                self.m.subject_to(node >= delta)
            elif (v < 0).all():
                self.m.subject_to(node <= -delta)
            else:
                self.m.subject_to(self.dm.abs(node) >= delta)

    def _build(self, e):
        dm = self.dm
        if e.is_Symbol:
            return self.inputs[str(e)]
        if e.is_Number or e.is_NumberSymbol:
            return float(e)
        if e.is_Add:
            out = self(e.args[0])
            for a in e.args[1:]:
                out = out + self(a)
            return out
        if e.is_Mul:
            num, den = [], []
            for a in e.args:
                if a.is_Pow and a.exp.is_number and a.exp < 0:
                    den.append(a.base**-a.exp)
                else:
                    num.append(a)
            out = self(num[0]) if num else 1.0
            for a in num[1:]:
                out = out * self(a)
            for d in den:
                node = self(d)
                self._guard("den", d, node)
                out = out / node
            return out
        if e.is_Pow:
            base, p = e.args
            if not p.is_number:
                raise NotImplementedError(f"non-constant exponent in {e}")
            if p < 0:
                d = base**-p
                node = self(d)
                self._guard("den", d, node)
                return 1.0 / node
            b = self(base)
            if p == sp.Rational(1, 2):
                self._guard("sqrt", base, b)
                return dm.sqrt(b)
            if p.is_integer:
                return b ** int(p)
            # fractional power: real only on a non-negative base
            self._guard("sqrt", base, b)
            return b ** float(p)
        if e.func in _DISCOPT_FUNCS:
            (arg,) = e.args
            a = self(arg)
            if e.func is sp.log:
                self._guard("log", arg, a)
            return getattr(dm, _DISCOPT_FUNCS[e.func])(a)
        raise NotImplementedError(
            f"no discopt translation for {e.func.__name__} in {e}"
        )
