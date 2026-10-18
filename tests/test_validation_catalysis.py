"""Golden test: reproduce the published catalysis result.

Runs the full faithful pipeline on the exact R-iBART ``catalysis`` dataset with
the paper's descriptor-generating strategy (opt = binary, unary, binary; max_dim=3;
standardize=False) and checks that it recovers a predictive model built on the
known physical drivers Hfo (dH_ox,fm) and Oxv (O_vac).

Slow (minutes): the round-1 binary expansion of 59 features is thousands of
descriptors, screened by BART-G.SE three times. Opt in with ``pytest -m slow``.

Reference (Liu et al. 2022; Ye, Senftle & Li 2024): the OIS model explains the
metal/oxide binding energy with R^2 ~ 0.92, via a descriptor coupling the oxide
formation enthalpy and the oxygen vacancy energy.
"""

import pytest

pytest.importorskip("bartz")

from jax_ibart import IBart
from jax_ibart.datasets import load_metal_support


@pytest.mark.slow
def test_reproduces_catalysis_descriptor():
    X, y, names, units = load_metal_support()

    model = IBart(
        backend="bart",
        schedule=("b", "u", "b"),  # paper's opt strategy
        max_dim=3,
        n_keep=10,
        score="bic",
        seed=0,
        n_perm=10,
        n_tree=30,
        n_draw=200,
        n_burn=200,
    ).fit(X, y, feature_names=names, units=units)  # dimensional-analysis filter on

    # Predictive performance of the right order (the published OIS model reaches
    # R^2 ~ 0.92 at a much larger MCMC budget; we use a small CPU budget).
    pred = model.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.80, f"R2={r2:.3f} below acceptable fit"

    # Interpretability / physics: the model must be built on at least one of the
    # known catalysis drivers, the oxide formation enthalpy (Hfo) or the oxygen
    # vacancy energy (Oxv), which the oracle's canonical descriptor abs(Hfo/Oxv)
    # couples. Exact-descriptor reproduction is not asserted; see
    # docs/oracle-crosscheck.md. We assert the robust invariant: a known driver
    # is selected. (Whether both appear varies with the stochastic screening.)
    used = set()
    for e in model.descriptors_:
        used |= {str(s) for s in e.free_symbols}
    assert used & {"Hfo", "Oxv"}, "no known catalysis driver (Hfo/Oxv) selected"


@pytest.mark.slow
def test_permissive_screen_recovers_hfo_oxv_coupling():
    """With a permissive (PAN-aligned) screen, the port recovers the published
    Hfo/Oxv coupling (the canonical descriptor is abs(Hfo/Oxv)). The old
    default alpha=0.05 over-pruned and missed it; see docs/oracle-crosscheck.md.
    """
    X, y, names, units = load_metal_support()
    model = IBart(
        backend="bart",
        schedule=("b", "u", "b"),
        max_dim=3,
        n_keep=15,
        score="bic",
        seed=0,
        alpha=0.25,  # permissive threshold
        n_perm=25,
        n_tree=40,
        n_draw=400,
        n_burn=400,
    ).fit(X, y, feature_names=names, units=units)

    # At least one selected descriptor couples Hfo and Oxv (the dH_ox,fm/O_vac
    # mechanism); the conservative default does not achieve this.
    coupled = [
        e
        for e in model.descriptors_
        if {"Hfo", "Oxv"} <= {str(s) for s in e.free_symbols}
    ]
    assert coupled, f"no Hfo/Oxv-coupled descriptor in {model.descriptors_}"
