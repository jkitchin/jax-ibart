"""The iBART orchestrator: a scikit-learn-style estimator.

Drives the interleaved feature-engineering / feature-selection schedule, e.g.

    BART -> O2 -> BART -> O1 -> BART -> O2 -> LASSO -> l0

keeping the active descriptor pool small at every step.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import sympy as sp

from . import engineer, sparsify
from .operators import default_binary, default_unary
from .selection import _x64_off, make_selector

# Default schedule: tokens "u" (apply unary) / "b" (apply binary); a selection
# pass runs after each engineering step.
DEFAULT_SCHEDULE = ("b", "u", "b")


class IBart:
    """Iterative BART descriptor discovery.

    Parameters
    ----------
    backend : {"bart", "diff"}
        Selection backend; see :mod:`jax_ibart.selection`.
    schedule : tuple of {"u", "b"}
        Feature-engineering steps to interleave with selection.
    max_dim : int
        Maximum size of the final descriptor (l0 best-subset).
    n_keep : int
        Candidates passed from LASSO screening into the l0 search.
    score : {"bic", "cv"}
        Selection score for the l0 best-subset.
    seed : int
        PRNG seed for the (functional) selection randomness.
    unary_ops, binary_ops : list of :class:`~jax_ibart.operators.Operator`, optional
        Operator sets for the ``"u"`` and ``"b"`` steps. Default to
        :func:`~jax_ibart.operators.default_unary` and
        :func:`~jax_ibart.operators.default_binary` (the catalysis-study sets).
    carry_selected : bool
        If True, each screen's survivors are unioned with everything kept by
        earlier screens, and that union feeds the next engineering step and the
        final LASSO/l0 stage. This is what the R iBART package does
        (``X_selected`` accumulates across iterations). If False (default), a
        binary step replaces the pool, so a descriptor kept in one round can
        drop out of the final pool.

    Attributes (after ``fit``)
    --------------------------
    descriptors_ : list[sympy.Expr]
        The selected descriptors (symbolic, interpretable).
    coef_ : np.ndarray
        OLS coefficients ``[intercept, *slopes]`` aligned with ``descriptors_``.
    history_ : list[dict]
        One record per pipeline stage, ``{"stage", "generated", "kept"}``: the
        pool size after feature engineering and after screening. Shows the
        grow-then-prune rhythm that keeps iBART tractable.
    pool_ : engineer.Pool
        The final screened descriptor pool that the LASSO/l0 stage chose from.
    """

    def __init__(
        self,
        backend: str = "bart",
        schedule=DEFAULT_SCHEDULE,
        max_dim: int = 3,
        n_keep: int = 10,
        score: str = "bic",
        screen_primaries: bool = False,
        seed: int = 0,
        unary_ops=None,
        binary_ops=None,
        carry_selected: bool = False,
        **selector_kwargs,
    ):
        self.backend = backend
        self.schedule = schedule
        self.max_dim = max_dim
        self.n_keep = n_keep
        self.score = score
        # Leading BART screen on the primary features before the first
        # engineering step (the "BART ->" prefix of the paper's schedule). The R
        # iBART package selects primaries first (e.g. 59 -> 12) and only then
        # expands. This is more faithful and much faster, but at small MCMC
        # budgets the leading screen can commit a PAN false negative and drop an
        # important primary before it is ever combined (observed: it dropped Oxv
        # on catalysis). Default off; recommended with a larger MCMC budget.
        # (Found via the R-iBART oracle cross-check.)
        self.screen_primaries = screen_primaries
        self.seed = seed
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.carry_selected = carry_selected
        self._selector_kwargs = dict(selector_kwargs)
        self.selector = make_selector(backend, **selector_kwargs)
        self.descriptors_ = None
        self.coef_ = None
        self.history_ = None
        self.pool_ = None
        self._feature_names = None

    def get_params(self, deep: bool = True) -> dict:
        """Constructor kwargs, for cloning a fresh unfitted estimator."""
        return dict(
            backend=self.backend,
            schedule=self.schedule,
            max_dim=self.max_dim,
            n_keep=self.n_keep,
            score=self.score,
            screen_primaries=self.screen_primaries,
            seed=self.seed,
            unary_ops=self.unary_ops,
            binary_ops=self.binary_ops,
            carry_selected=self.carry_selected,
            **self._selector_kwargs,
        )

    def set_params(self, **params) -> "IBart":
        """Update constructor kwargs (scikit-learn style); returns ``self``."""
        own = set(self.get_params()) - set(self._selector_kwargs)
        for k, v in params.items():
            if k in own:
                setattr(self, k, v)
            else:
                self._selector_kwargs[k] = v
        self.selector = make_selector(self.backend, **self._selector_kwargs)
        return self

    def clone(self) -> "IBart":
        """A fresh, unfitted copy with the same configuration."""
        return IBart(**self.get_params())

    def fit(self, X, y, feature_names: list[str] | None = None, units=None):
        """Fit iBART.

        ``feature_names`` defaults to the columns of a pandas DataFrame ``X``,
        or ``x0, x1, ...``. Names become the symbols in the discovered
        expressions.

        ``units`` (optional): an (n_features, n_dims) integer unit-exponent
        matrix. When given, descriptors that are dimensionally inconsistent
        (e.g. summing two different units) are never generated.

        The fit always runs JAX in float32, even if 64-bit mode is enabled
        globally (some libraries, e.g. discopt, enable it on import), so results
        do not depend on what else is imported. The final regression is float64.
        """
        if feature_names is None:
            cols = getattr(X, "columns", None)
            p = np.shape(X)[1] if np.ndim(X) == 2 else 0
            feature_names = (
                [str(c) for c in cols]
                if cols is not None
                else [f"x{i}" for i in range(p)]
            )
        X, y, feature_names, units = _check_inputs(X, y, feature_names, units)
        with _x64_off():
            return self._fit(X, y, feature_names, units)

    def _fit(self, X, y, feature_names, units):
        if self.score not in ("bic", "cv"):
            raise ValueError(f"score must be 'bic' or 'cv', got {self.score!r}")
        # Built here so changes to ``backend`` after construction take effect.
        self.selector = make_selector(self.backend, **self._selector_kwargs)
        key = jax.random.PRNGKey(self.seed)
        unary = default_unary() if self.unary_ops is None else list(self.unary_ops)
        binary = default_binary() if self.binary_ops is None else list(self.binary_ops)
        pool = engineer.from_primary(X, feature_names, units=units)
        n = pool.n_descriptors
        history = [{"stage": "primary", "generated": n, "kept": n}]

        # Union of everything kept so far (used when carry_selected).
        selected = None

        # Leading screen on the primary features (matches R iBART).
        if self.screen_primaries and pool.n_descriptors > 1:
            key, sk = jax.random.split(key)
            pool = self._screen(pool, y, sk)
            history[0]["kept"] = pool.n_descriptors
            selected = pool

        for step in self.schedule:
            if step not in ("u", "b"):
                raise ValueError(f"schedule steps must be 'u' or 'b', got {step!r}")
            grown = (
                engineer.apply_unary(pool, unary)
                if step == "u"
                else engineer.apply_binary(pool, binary)
            )
            generated = grown.n_descriptors
            if generated == 0:
                # Nothing valid to engineer (e.g. one feature, or every value
                # outside the operators' domains): stop growing, keep the pool.
                history.append(
                    {"stage": step, "generated": 0, "kept": pool.n_descriptors}
                )
                break
            key, sk = jax.random.split(key)
            pool = self._screen(grown, y, sk)
            if self.carry_selected:
                selected = pool if selected is None else _union(selected, pool)
                pool = selected
            history.append(
                {"stage": step, "generated": generated, "kept": pool.n_descriptors}
            )

        # Final parametric stage: LASSO screen -> l0 best subset (f64 host-side).
        Xpool = np.asarray(pool.X, dtype=np.float64)
        cand = sparsify.lasso_screen(Xpool, y, n_keep=self.n_keep)
        local, coef = sparsify.l0_best_subset(
            Xpool[:, cand], y, max_dim=self.max_dim, score=self.score
        )
        history.append({"stage": "l0", "generated": len(cand), "kept": len(local)})

        # Set fitted state only once the fit has succeeded.
        self._feature_names = feature_names
        self._X_train = X
        self.pool_ = pool
        self.descriptors_ = [pool.exprs[cand[i]] for i in local]
        self.coef_ = coef
        self.history_ = history
        self._check_consistency(X, [cand[i] for i in local])
        return self

    def _check_consistency(self, X, cols):
        """Warn if the symbolic descriptors don't reproduce their numeric columns.

        That happens when a custom operator's ``fn`` and ``sym`` disagree, and
        would make ``predict`` silently differ from the fitted model.
        """
        num = np.asarray(self.pool_.X, dtype=np.float64)[:, cols]
        with np.errstate(all="ignore"):
            sym = self.transform(X)
        scale = np.abs(num).max(axis=0) + 1e-12
        bad = [
            str(self.descriptors_[k])
            for k in range(len(cols))
            if not np.allclose(sym[:, k] / scale[k], num[:, k] / scale[k], atol=1e-3)
        ]
        if bad:
            warnings.warn(
                "symbolic and numeric forms disagree on the training data for "
                f"{bad}; check custom operators' fn and sym",
                stacklevel=3,
            )

    def _screen(self, pool, y, key):
        """Run the selection backend and reduce the pool, preserving units.

        Safeguard: if screening would empty the pool, keep it unchanged.
        """
        keep = np.asarray(self.selector.select(pool.X, y, key=key))
        if not keep.any():
            return pool
        return engineer.Pool(
            X=pool.X[:, keep],
            exprs=[e for e, k in zip(pool.exprs, keep) if bool(k)],
            units=None if pool.units is None else pool.units[keep],
        )

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("this IBart is not fitted yet; call fit first")

    def transform(self, X):
        """Evaluate the selected descriptors on raw primary features ``X``.

        Returns the ``(n, k)`` design matrix the final linear model uses.
        """
        self._check_fitted()
        syms = [sp.Symbol(n) for n in self._feature_names]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(syms):
            raise ValueError(
                f"X must have shape (n, {len(syms)}), got {tuple(X.shape)}"
            )
        cols = []
        for expr in self.descriptors_:
            f = sp.lambdify(syms, expr, modules="numpy")
            cols.append(
                np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64)
            )
        return np.column_stack(cols)

    _design = transform  # backward-compatible alias

    def predict(self, X):
        """Predict ``y`` for raw primary features ``X``.

        The symbolic descriptors are evaluated as written, so operators such as
        ``log`` and ``sqrt`` give NaN outside their real domain (the training
        pipeline evaluates them on ``|x|``).
        """
        with np.errstate(all="ignore"):
            D = self.transform(X)
        pred = self.coef_[0] + D @ self.coef_[1:]
        if not np.isfinite(pred).all():
            warnings.warn(
                f"{int((~np.isfinite(pred)).sum())} predictions are NaN or inf; "
                "inputs are probably outside a descriptor's domain (e.g. log of a "
                "non-positive value)",
                stacklevel=2,
            )
        return pred

    def expression(self, digits: int = 4) -> sp.Expr:
        """The fitted model as a single symbolic expression.

        Coefficients are rounded to ``digits`` significant figures for display
        (pass ``digits=None`` for full precision).
        """
        self._check_fitted()
        expr = sp.Float(self.coef_[0])
        for c, d in zip(self.coef_[1:], self.descriptors_):
            expr = expr + sp.Float(c) * d
        return sp.N(expr, digits) if digits else expr

    def to_discopt(self, m, x=None, **kwargs):
        """Add this model to a discopt optimization model.

        Returns ``(x, y)``: the input variables by feature name and a variable
        constrained to equal the prediction. Keyword arguments are passed on;
        see :func:`jax_ibart.export.to_discopt`.
        """
        from .export import to_discopt

        return to_discopt(self, m, x, **kwargs)

    def latex(self, lhs: str = "y", digits: int = 4) -> str:
        """The fitted model as a LaTeX string, e.g. ``y = -0.099 BVS Hfo + ...``."""
        return f"{lhs} = {sp.latex(self.expression(digits))}"

    def to_html(self, lhs: str = "y", digits: int = 4) -> str:
        """An HTML snippet rendering the model (MathJax/KaTeX delimiters)."""
        return f"<p>\\({self.latex(lhs, digits)}\\)</p>"

    def _repr_latex_(self) -> str:
        """Rich LaTeX display in Jupyter/notebooks."""
        if self.coef_ is None:
            return "IBart(unfitted)"
        return f"${self.latex()}$"


def _check_inputs(X, y, feature_names, units):
    """Validate shapes and values before fitting; returns normalized inputs."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got {X.ndim}-D")
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    feature_names = list(feature_names)
    if len(feature_names) != p:
        raise ValueError(f"got {len(feature_names)} feature names for {p} columns")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or inf")
    if np.ptp(y) == 0:
        raise ValueError("y is constant")
    if units is not None:
        units = np.asarray(units)
        if units.ndim != 2 or units.shape[0] != p:
            raise ValueError(
                f"units must have shape ({p}, n_dims), got {tuple(units.shape)}"
            )
    return X, y, feature_names, units


def _union(a, b):
    """Columns of ``a`` followed by those of ``b``, duplicates removed."""
    units = None if a.units is None else np.concatenate([a.units, b.units], axis=0)
    return engineer.clean(
        engineer.Pool(
            X=jax.numpy.concatenate([a.X, b.X], axis=1),
            exprs=list(a.exprs) + list(b.exprs),
            units=units,
        )
    )
