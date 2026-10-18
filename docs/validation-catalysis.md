# Validation: single-atom catalysis (metal-support binding energy)

Result of running `jax-ibart` on the exact `catalysis` dataset shipped with the
R `iBART` package (`data/catalysis.rda`, converted to
`src/jax_ibart/data/catalysis_X.csv` / `catalysis_y.csv`). This is the dataset
and task from Liu et al. 2022 (*J. Chem. Phys.* 156:164105) and Ye, Senftle & Li
2024 (*JASA* 119:81-94). The full, executed analysis is the
[catalysis notebook](examples/02_catalysis.ipynb).

## Setup

- Data: n = 91 metal/oxide binding energies, p = 59 primary features.
- Strategy: `schedule = ("b", "u", "b")` (the paper's eq. 8 schedule),
  `max_dim = 3` (l0 best subset, BIC), unit filter on.
- Backend: BART-G.SE via `bartz`, with the permissive screen recommended in the
  [oracle cross-check](oracle-crosscheck.md):

```python
from jax_ibart import IBart
from jax_ibart.datasets import load_metal_support

X, y, names, units = load_metal_support()
model = IBart(backend="bart", alpha=0.25, n_perm=25, n_tree=40, n_draw=400,
              n_burn=400, n_keep=15, max_dim=3, seed=0)
model.fit(X, y, names, units=units)
```

The fit takes about 20-30 s on a laptop CPU.

## The dimensional-analysis (unit) filter

The R iBART package eliminates dimensionally invalid descriptors (e.g.
`volume + speed`) by propagating physical units. We reproduce this: each feature
carries an integer unit-exponent vector (12 base dimensions for this dataset);
additive operators only combine matching units, multiply/divide add/subtract
exponents, and root/transcendental unary operators return unitless results
(matching `R/operations.R`). Enable it with `IBart.fit(..., units=units)`.

On this dataset the filter cuts the round-1 binary space from 11,370 to 3,570
descriptors (about 70%), and it is applied in every engineering round. See the
[custom operators and units notebook](examples/04_custom_operators_units.ipynb).

## Result

In-sample **RMSE = 0.822 eV, R^2 = 0.919**. Selected 3-descriptor model:

```
y_hat = -0.3507
        - 0.002754 * BVB**2*Hfo**2/log(Hs*m_Chi_MB)
        - 0.4844   * exp(BVS/Hfo)*Abs(Hfo/Oxv)
        - 0.3449   * sin(pi*Oxv*s_phi)*Abs(Hfo/Oxv)
```

Two of the three descriptors contain the published `|Hfo/Oxv|` as a factor.
Compared with the published l0 models of the same size:

| k | jax-ibart RMSE (eV) | published RMSE (eV) |
|---|---------------------|---------------------|
| 1 | 1.012 | 0.929 |
| 2 | 0.930 | 0.664 |
| 3 | 0.822 | 0.624 |

The port finds the same physical drivers (`Hfo`, oxide formation enthalpy, and
`Oxv`, oxygen vacancy energy, coupled through their ratio) but is less accurate
than the published models at every size. The bare ratio `abs(Hfo/Oxv)` is
generated during the run but is not in the final pool by default; set
`carry_selected=True` to keep every round's survivors, as R iBART does. Which
descriptors appear varies with the stochastic BART screen and the seed.

## Differentiable backend (`backend="diff"`)

The same schedule with the hard-concrete-L0 screener instead of BART-G.SE
(`steps=1000, l0=0.03`, `n_keep=10`) runs in a few seconds, several times faster than
the bart run above, with in-sample RMSE 1.090 eV. It also finds
`Oxv`-normalized descriptors (`Abs(Hf/Oxv)`), but its selection is noisier
than BART's permutation-null screen on this collinear problem (n = 91,
thousands of candidates). It is validated against the bart backend on synthetic
fixtures (`tests/test_diffselect.py`); use bart when fidelity matters. See the
[backends notebook](examples/03_backends.ipynb).

## Reproduce

Run the [catalysis notebook](examples/02_catalysis.ipynb), or the slow tests
(`pytest -m slow`): one checks that a small-budget fit reaches R^2 > 0.80 on
the known drivers, the other that the permissive screen above selects a
descriptor coupling `Hfo` and `Oxv`.
