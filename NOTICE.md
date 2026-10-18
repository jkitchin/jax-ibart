# Notices and attributions

`jax-ibart` is licensed under the MIT License (see `LICENSE`). It is an
independent reimplementation in Python/JAX of the iBART method; it does not
copy, link, or distribute any code from the GPL-licensed R `iBART` package.

## Method and prior work

The iBART algorithm and the operator-induced-structure framework are due to:

- Chun-Yen Liu, Shengbin Ye, Meng Li, Thomas P. Senftle.
  "A rapid feature selection method for catalyst design: Iterative Bayesian
  additive regression trees (iBART)." J. Chem. Phys. 156, 164105 (2022).
- Shengbin Ye, Thomas P. Senftle, Meng Li.
  "Operator-induced structural variable selection for identifying materials
  genes." J. Am. Stat. Assoc. 119(545):81-94 (2024); arXiv:2110.10195.

The reference implementation is the R package `mattsheng/iBART` (GPL >= 3). The
unit-propagation rules in `jax_ibart.operators` were reimplemented from the
*description* of that package's behaviour (its `R/operations.R`); no source code
was copied. Please cite the papers above when using this software.

## Runtime dependencies

Runtime dependencies are all permissively licensed:

- `bartz` (MIT) -- the BART engine used by the faithful selection backend.
  Petrillo, "Very fast Bayesian Additive Regression Trees on GPU,"
  arXiv:2410.23244 (2024).
- `jax`, `optax` (Apache-2.0); `sympy`, `scikit-learn`, `numpy`, `pandas` (BSD).

This project does **not** depend on the GPL `bartMachine` package.

## Bundled data

- `src/jax_ibart/data/catalysis_X.csv`, `catalysis_y.csv`, `catalysis_unit.csv`: the
  single-atom catalysis primary features, metal/oxide binding energies, and the
  unit exponents of each feature. They were converted from `data/catalysis.rda` in the R package
  in the R `iBART` package (GPL >= 3) so results are directly comparable. The
  underlying data originate from the Senftle group's
  `tsenftle/Metal-Oxide-LASSO-lo` repository, which is **MIT licensed**
  (Copyright (c) 2018 tsenftle). Numeric DFT results
  are facts and are not subject to copyright.
- `src/jax_ibart/data/oracle_ibart.json`: a small summary (selected descriptor names, RMSEs,
  and per-iteration sizes) of the reference R-iBART output, extracted from the
  iBART package's `iBART_real_data.rda` for validation/comparison only. These
  are the published results reported in the papers above. Used here under fair
  use for interoperability testing, with attribution.
