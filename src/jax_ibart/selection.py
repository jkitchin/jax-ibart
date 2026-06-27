"""Nonparametric descriptor screening (the FS half of iBART).

Two interchangeable backends sit behind one :class:`Selector` interface:

* :class:`BartGSESelector` (``backend="bart"``) -- the *faithful* path. BART-G.SE
  via ``bartz`` with a permutation null (Bleich et al. 2014 global-SE threshold).
  This is the correctness oracle, validated against the R ``iBART`` package.
* :class:`DiffSelector` (``backend="diff"``) -- the *differentiable* path. A
  sparse-gated neural importance screener (hard-concrete L0 gates, Louizos et
  al. 2018) whose selection is differentiable end to end. Validated against
  ``BartGSESelector`` on the fixtures before being trusted alone.

Both must satisfy the PAN criterion: over-selection (false positives) is fine,
but a needed intermediate descriptor must not be dropped (no false negatives).
See PLAN-jax-ibart.md, decision (2).
"""

from __future__ import annotations

from typing import Protocol

import jax
import numpy as np

_EPS = 1e-9


def _bart_varprop(X32, y32, seed, *, n_tree, n_draw, n_burn):
    """Fit bartz and return per-variable inclusion *proportions* (sums to 1).

    ``varcount_mean`` is the histogram of predictor usage in decision rules;
    normalizing to a proportion makes fits on real and permuted data
    comparable. bartz expects ``(p, n)`` layout and an int seed, and must run
    in float32 (x64 breaks its MCMC scan carry).
    """
    from bartz.BART import gbart

    m = gbart(X32.T, y32, ntree=n_tree, ndpost=n_draw, nskip=n_burn,
              seed=int(seed), printevery=None)
    vc = np.asarray(m.varcount_mean, dtype=np.float64)
    total = vc.sum()
    return vc / total if total > 0 else vc


class Selector(Protocol):
    """Screen descriptor columns; return a boolean keep-mask over columns.

    Implementations take a numeric design matrix ``X`` of shape (n, d) and a
    response ``y`` of shape (n,), and return a boolean array of shape (d,)
    marking which columns survive screening for the next iteration.
    """

    def select(self, X, y, *, key: jax.Array) -> jax.Array:  # -> bool[d]
        ...


class BartGSESelector:
    """Faithful BART-G.SE screening backend (default).

    Parameters
    ----------
    n_perm : int
        Number of label permutations for the null distribution of inclusion
        proportions.
    alpha : float
        Simultaneous coverage level for the global standard-error (G.SE)
        threshold. Larger alpha lowers the threshold and keeps *more*
        descriptors. This is the key knob: the PAN criterion wants intermediate
        screening to tolerate false positives but never drop a true building
        block (no false negatives), so a permissive (larger) alpha is correct
        for iBART's interleaved screening. An over-conservative alpha
        over-prunes and was found (via the R-iBART oracle cross-check) to be the
        cause of the exact-descriptor gap on catalysis: at alpha=0.25 the
        pipeline recovers the published canonical descriptor abs(Hfo/Oxv), which
        alpha=0.05 screened out. Default 0.10; use ~0.2-0.25 for the most
        faithful reproduction. (Clean, well-separated problems are insensitive
        to alpha; it matters on collinear, high-dimensional screens.)
    n_tree, n_draw, n_burn : int
        BART ensemble size and MCMC schedule (passed through to ``bartz``).
        Note: a *larger* MCMC budget does not close the catalysis gap; the
        screening threshold (alpha) does. See docs/oracle-crosscheck.md.
    """

    def __init__(self, n_perm: int = 50, alpha: float = 0.10,
                 n_tree: int = 50, n_draw: int = 1000, n_burn: int = 1000):
        self.n_perm = n_perm
        self.alpha = alpha
        self.n_tree = n_tree
        self.n_draw = n_draw
        self.n_burn = n_burn

    def select(self, X, y, *, key):
        """BART-G.SE screening (Bleich et al. 2014, Global SE rule).

        Returns a boolean keep-mask over the columns of ``X`` (n, d). A column
        is kept if its observed inclusion proportion exceeds a permutation-null
        threshold ``mu_j + g * sd_j`` with a single global multiplier ``g``
        calibrated for simultaneous (1-alpha) coverage.
        """
        X32 = np.asarray(X, dtype=np.float32)
        y32 = np.asarray(y, dtype=np.float32)
        n, d = X32.shape
        kw = dict(n_tree=self.n_tree, n_draw=self.n_draw, n_burn=self.n_burn)

        # Derive reproducible int seeds from the functional key.
        base = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(base)

        # 1. Observed inclusion proportions.
        q_obs = _bart_varprop(X32, y32, base, **kw)
        assert q_obs.shape == (d,), (q_obs.shape, d)

        # 2. Permutation null: refit on permuted y.
        null = np.empty((self.n_perm, d), dtype=np.float64)
        for b in range(self.n_perm):
            yb = y32[rng.permutation(n)]
            null[b] = _bart_varprop(X32, yb, base + 1 + b, **kw)

        # 3. Per-variable null mean/sd, then the global SE multiplier: the
        #    (1-alpha) quantile of the per-permutation max standardized score.
        mu = null.mean(axis=0)
        sd = null.std(axis=0) + _EPS
        z_null = (null - mu) / sd            # (n_perm, d)
        g = float(np.quantile(z_null.max(axis=1), 1.0 - self.alpha))

        # 4. Select where the observed score clears the global threshold.
        keep = (q_obs - mu) / sd > g
        return jax.numpy.asarray(keep)


class DiffSelector:
    """Differentiable end-to-end screening backend.

    Replaces the discrete BART importance with a sparse-gated neural importance
    model (hard-concrete L0 gates, Louizos et al. 2018; see
    :mod:`jax_ibart._diffscreen`) so the whole selection step admits gradients.
    Acceptance gate: keep the true drivers on the synthetic fixtures and agree
    with :class:`BartGSESelector` (PAN: no false negatives).

    Parameters
    ----------
    steps, lr, l0, hidden : training hyperparameters for the gated MLP.
    threshold : float
        Keep features whose learned gate-open probability exceeds this. Kept low
        to over-select (PAN-safe: tolerate false positives, avoid false
        negatives).
    min_keep : int
        Always keep at least this many top-importance features (never empties
        the pool).
    """

    def __init__(self, steps: int = 1500, lr: float = 1e-2, l0: float = 0.03,
                 hidden: int = 16, threshold: float = 0.5, min_keep: int = 2):
        self.steps = steps
        self.lr = lr
        self.l0 = l0
        self.hidden = hidden
        self.threshold = threshold
        self.min_keep = min_keep

    def select(self, X, y, *, key):
        import numpy as np

        from ._diffscreen import screen_importance

        imp = np.asarray(screen_importance(
            X, y, key, steps=self.steps, lr=self.lr, l0=self.l0,
            hidden=self.hidden))
        keep = imp > self.threshold
        # PAN safeguard: never drop below min_keep (keep top-importance features).
        if keep.sum() < min(self.min_keep, imp.size):
            top = np.argsort(imp)[::-1][:min(self.min_keep, imp.size)]
            keep = np.zeros_like(keep)
            keep[top] = True
        return jax.numpy.asarray(keep)


def make_selector(backend: str = "bart", **kwargs) -> Selector:
    """Factory: ``"bart"`` -> BART-G.SE (faithful), ``"diff"`` -> differentiable."""
    if backend == "bart":
        return BartGSESelector(**kwargs)
    if backend == "diff":
        return DiffSelector(**kwargs)
    raise ValueError(f"unknown selection backend: {backend!r}")
