"""Validation datasets: the public catalysis benchmarks used by iBART.

These are the fixtures for the M4 validation gate (reproduce the published
descriptor and CV error, cross-check against the R ``iBART`` package).

Sources (Senftle group):
* M-S: 91 binding energies, 13 metals x 7 oxide supports.
       https://github.com/tsenftle/Metal-Oxide-LASSO-lo
* M-D: 19 metals on doped MgO(100).
       https://github.com/tsenftle/MgO_SL

Target descriptor to recover on M-S (Liu et al. 2022): dH_ox,fm * O_vac.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MS_URL = "https://github.com/tsenftle/Metal-Oxide-LASSO-lo"
MD_URL = "https://github.com/tsenftle/MgO_SL"


def load_metal_support():
    """Load the M-S single-atom catalysis dataset.

    Returns ``(X, y, feature_names, units)``:

    * ``X``     -- 91 x 59 primary-feature matrix.
    * ``y``     -- 91 metal/oxide binding energies (eV).
    * ``feature_names`` -- the 59 feature names.
    * ``units`` -- 59 x 12 integer unit-exponent matrix (row per feature, column
      per base dimension), for the dimensional-analysis filter. Pass it to
      ``IBart.fit(..., units=units)`` to drop unit-inconsistent descriptors.

    This is the exact ``catalysis`` dataset shipped with the R ``iBART`` package
    (converted from ``data/catalysis.rda``), so results are directly comparable.
    Notable features for the published descriptor (dH_ox,fm * O_vac):
    ``Hfo`` (oxide formation enthalpy) and ``Oxv`` (oxygen vacancy energy); both
    carry unit ``eV``.
    """
    Xdf = pd.read_csv(DATA_DIR / "catalysis_X.csv")
    y = pd.read_csv(DATA_DIR / "catalysis_y.csv")["y"].to_numpy(dtype=np.float64)
    X = Xdf.to_numpy(dtype=np.float64)
    names = list(Xdf.columns)

    # unit CSV: rows = base dimensions, cols = features (+ a row-name column).
    udf = pd.read_csv(DATA_DIR / "catalysis_unit.csv", index_col=0)
    units = udf[names].to_numpy(dtype=np.int64).T  # -> (n_features, n_dims)
    return X, y, names, units


def load_metal_dopant():
    """Load the M-D (doped MgO) dataset -> (X, y, feature_names)."""
    raise NotImplementedError("M4+: wire up the MgO_SL loader")


def load_reaction():
    """Load the example batch-reactor kinetics dataset.

    Returns ``(C, dCdt, species, batch)``:

    * ``C``       -- (n, 4) measured concentrations [CA, CB, CC, CD].
    * ``dCdt``    -- (n, 4) time derivatives, estimated by central finite
      differences *within each batch* (the mole-balance RHS targets).
    * ``species`` -- ["CA", "CB", "CC", "CD"].
    * ``batch``   -- (n,) batch index of each row.

    Synthetic data from a liquid-phase batch reactor with A+B->C (bimolecular)
    and C->D (first order); see ``scripts/make_reaction_data.py``. Use it to
    rediscover the mole-balance RHS (e.g. with
    ``jax_ibart.dynamics.discover_ode(C, t=None, species, dXdt=dCdt,
    screen="bart")``).
    """
    from .dynamics import finite_difference

    df = pd.read_csv(DATA_DIR / "reaction_kinetics.csv")
    species = ["CA", "CB", "CC", "CD"]
    C, dCdt, batch = [], [], []
    for b, g in df.groupby("batch"):
        g = g.sort_values("t")
        Cb = g[species].to_numpy(dtype=np.float64)
        C.append(Cb)
        dCdt.append(finite_difference(Cb, g["t"].to_numpy(dtype=np.float64)))
        batch.append(np.full(len(g), b))
    return (np.vstack(C), np.vstack(dCdt), species, np.concatenate(batch))
