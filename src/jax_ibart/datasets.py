"""Validation datasets: the public catalysis benchmarks used by iBART.

The CSV files ship inside the package (``jax_ibart/data``), so the loaders work
from a plain ``pip install``.

Sources (Senftle group):
* M-S: 91 binding energies, 13 metals x 7 oxide supports.
       https://github.com/tsenftle/Metal-Oxide-LASSO-lo

Published descriptor on M-S (Liu et al. 2022): abs(Hfo/Oxv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

MS_URL = "https://github.com/tsenftle/Metal-Oxide-LASSO-lo"


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
    (converted from the R package's ``data/catalysis.rda``), so results are directly comparable.
    Notable features for the published descriptor abs(Hfo/Oxv):
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
