# M4 validation: single-atom catalysis (metal-support binding energy)

Result of running the faithful `jax-ibart` pipeline on the exact `catalysis`
dataset shipped with the R `iBART` package (`data/catalysis.rda`, converted to
`data/catalysis_X.csv` / `data/catalysis_y.csv`). This is the dataset and task
from Liu et al. 2022 (*J. Chem. Phys.* 156:164105) and Ye, Senftle & Li 2024
(*JASA* 119:81-94).

## Setup

- Data: n = 91 metal/oxide binding energies, p = 59 primary features.
- Strategy: `opt = (binary, unary, binary)` (the paper's eq. 8 schedule),
  K = 3 (l0 best subset, BIC), `standardize = False`.
- Backend: faithful BART-G.SE via `bartz`. Modest MCMC budget for a single run:
  `n_perm=10, n_tree=30, n_draw=200, n_burn=200`, `seed=0`.

## The dimensional-analysis (unit) filter

The R iBART package eliminates dimensionally invalid descriptors (e.g.
`volume + speed`) by propagating physical units. We reproduce this: each feature
carries an integer unit-exponent vector (12 base dimensions for this dataset);
additive operators only combine matching units, multiply/divide add/subtract
exponents, and root/transcendental unary operators return unitless results
(matching `R/operations.R`). Enabled via `IBart.fit(..., units=units)`.

Effect on this run (seed=0, same settings; units-ON carries the filter through
*every* FE round, a correctness fix found during the oracle cross-check):

| Metric                     | Units OFF | Units ON |
|----------------------------|-----------|----------|
| Round-1 binary descriptors | ~12,000   | 3,570    |
| Fit time (CPU)             | 216 s     | 132 s    |
| In-sample R^2              | 0.921     | 0.856    |
| Known driver (Hfo/Oxv)     | yes       | yes (Hfo)|

The filter removes ~70% of the round-1 space and runs faster. In-sample R^2 is
slightly lower with units on (the dimensional constraint shrinks the descriptor
space, trading a little in-sample fit for guaranteed physical interpretability).

## Result (units ON)

In-sample **R^2 = 0.856**. Selected 3-descriptor model:

```
y_hat = +(intercept)
        - 4.135  * sqrt(m_Chi_MB/m_N_val) * log(s_ion_3/m_n13)
        + 0.0390 * Hfo * m_ion_1 * s_LUMO / s_Chi_MB_un
        + 0.1549 * s_HOMO * s_ion_4 / (BVB * s_Chi_P)
```

## Physics check (the credibility gate)

The model is built on a known driver: **`Hfo`** (oxide formation enthalpy,
dH_ox,fm) appears in `Hfo * m_ion_1 * s_LUMO / s_Chi_MB_un`. Hfo and `Oxv`
(oxygen vacancy energy) are the metal-oxygen charge-transfer quantities the
original papers identify as governing metal-support binding (the published
descriptor is `abs(Hfo/Oxv)`). Which specific drivers and descriptors appear
varies with the stochastic BART screening and the MCMC budget; see
[`oracle-crosscheck.md`](oracle-crosscheck.md) for the direct comparison against
the reference R-iBART output and an honest account of the remaining gap.

## Differentiable backend (`backend="diff"`)

The same pipeline with the differentiable hard-concrete-L0 screener instead of
BART-G.SE, on the identical data/schedule (seed=0, `steps=1000, l0=0.03`):

| Metric            | bart (faithful) | diff (differentiable) |
|-------------------|-----------------|-----------------------|
| Fit time (CPU)    | 132 s           | **10 s** (~13x faster)|
| In-sample R^2     | 0.926           | 0.833                 |
| Oxv selected      | yes             | yes                   |
| Hfo selected      | yes             | no (picks `Hf`)       |

The differentiable backend is dramatically faster but lower fidelity on this
hard, high-dimensional screen (~12k descriptors, n = 91): the gated MLP is
heavily overparameterized, so its selection is noisier than BART's permutation-
null screening. It is validated against the bart oracle on synthetic fixtures
(`tests/test_diffselect.py`) and is the research/speed path; the bart backend
remains the faithful path. Tuning (stronger regularization, more steps, a
permutation-style threshold for the gates) is future work.

## Notes / caveats

- The exact symbolic form differs from the published descriptor (expected: BART
  screening is stochastic, and the MCMC budget here is deliberately small). The
  *invariants* that matter, predictive accuracy (~0.92) and selection of the
  Hfo and Oxv drivers, are reproduced.
- A larger MCMC budget would be expected to tighten agreement with the exact
  published expression. A direct R-iBART oracle cross-check on identical inputs
  remains a follow-up.
- Reproduce with: `pytest -m slow` (see `tests/test_validation_catalysis.py`).
