<p align="center"><img src="docs/assets/logo.svg" alt="jax-ibart" width="200"></p>

# jax-ibart

[![CI](https://github.com/jkitchin/jax-ibart/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/jax-ibart/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-jkitchin.github.io%2Fjax--ibart-indigo)](https://jkitchin.github.io/jax-ibart/)

A JAX-native reimplementation of **iBART** (iterative Bayesian Additive
Regression Trees) for symbolic descriptor discovery in materials informatics.
Pure Python, GPU-capable, `pip`-installable (no JVM).

iBART finds compact, interpretable nonlinear descriptors by **interleaving
feature engineering** (applying algebraic operators) with **nonparametric
feature selection**, keeping the active descriptor pool small at every step
instead of enumerating the full O(10^17) candidate space.

References: Liu et al. 2022 (*J. Chem. Phys.* 156:164105); Ye, Senftle & Li 2024
(*JASA* 119:81-94). Reference implementation: the R package
[`mattsheng/iBART`](https://github.com/mattsheng/iBART).

## Features

- Operator algebra and a vectorized feature-engineering engine with symbolic
  (SymPy) provenance for every descriptor.
- Two selection backends behind one interface: faithful BART-G.SE screening via
  [`bartz`](https://github.com/Gattocrucco/bartz), the GPU JAX BART, and a
  differentiable hard-concrete-L0 gated screener (Louizos et al. 2018).
- LASSO screening plus l0 best-subset final models; `IBart.fit` / `.predict`.
- A dimensional-analysis (unit) filter that drops descriptors like
  `volume + speed` (about 70% of the round-1 space on the catalysis data).
- Validated on the single-atom catalysis dataset against the R iBART authors'
  own output: the selected models are built on the published `|Hfo/Oxv|`
  descriptor ([validation](docs/validation-catalysis.md),
  [oracle cross-check](docs/oracle-crosscheck.md)).
- Uncertainty quantification: split-conformal intervals, selection stability,
  repeated-split RMSE, OLS intervals, and the BART posterior.
- Sparse identification of dynamics (SINDy) and implicit algebraic constraints.

## Install

Not on PyPI yet; install from GitHub:

```bash
pip install "jax-ibart[bart] @ git+https://github.com/jkitchin/jax-ibart"
```

The default `backend="bart"` needs the `bart` extra (`bartz<0.11`; see
[installation](docs/installation.md)); `backend="diff"` needs the `diff` extra.
For development:

```bash
git clone https://github.com/jkitchin/jax-ibart
cd jax-ibart
uv sync --extra all
uv run pytest            # core tests run without bartz
```

## Quick taste

Find the hidden product `a*b` among six features (the
[quickstart notebook](docs/examples/01_quickstart.ipynb) walks through it):

```python
import numpy as np
from jax_ibart import IBart

rng = np.random.default_rng(0)
names = ["a", "b", "c", "d", "e", "f"]
X = rng.uniform(0.5, 2.0, size=(90, 6))
y = 5.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 90)  # only a*b matters

model = IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=1200)
model.fit(X, y, feature_names=names)
print(model.expression())   # 4.99*a*b + 0.01599
```

## Documentation

Full documentation, with the API reference and executed example notebooks, is
at <https://jkitchin.github.io/jax-ibart/>. To build it locally:
`uv sync --extra docs && uv run mkdocs serve`.

- Worked notebooks under [`docs/examples/`](docs/examples/), with rendered
  outputs:
  1. [Quickstart](docs/examples/01_quickstart.ipynb): the grow-and-prune loop
     on a hidden `a*b`.
  2. [Catalysis case study](docs/examples/02_catalysis.ipynb): 91 points, 59
     features, compared with the published `|Hfo/Oxv|` result.
  3. [`bart` vs `diff` backends](docs/examples/03_backends.ipynb): what each
     screen computes, and when to use which.
  4. [Custom operators and units](docs/examples/04_custom_operators_units.ipynb):
     the unit filter, and adding a `T^4` operator for Stefan-Boltzmann.
  5. [Dynamical systems](docs/examples/05_dynamics.ipynb): SINDy on Lorenz and
     implicit constraints.
  6. [Reaction kinetics](docs/examples/06_reaction_kinetics.ipynb): rate laws
     from noisy batch data, and why derivative smoothing matters.
  7. [What iBART can and can't find](docs/examples/07_nonlinear.ipynb).
  8. [Uncertainty quantification](docs/examples/08_uncertainty.ipynb).
- Validation: [`docs/validation-catalysis.md`](docs/validation-catalysis.md),
  the [R-iBART oracle cross-check](docs/oracle-crosscheck.md), and
  [benchmarks](docs/benchmark.md).
- Rebuild the example notebooks: `uv run --extra all --with nbformat python scripts/build_notebooks.py` then
  `uv run --extra all --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace docs/examples/*.ipynb`.
  The docs show the committed outputs (notebooks are not executed at build time).

## Comparison to [jaxsr](https://github.com/jkitchin/jaxsr)

`jaxsr` is a broad, general-purpose symbolic regression toolkit with a fixed,
user-specified basis; `jax-ibart` is a specialized implementation of iBART that
*grows* its descriptor library by composing operators and pruning between
rounds. See [the comparison](docs/comparison.md) for when to use which.

## Dynamical systems (SINDy / DAEs)

`jax_ibart.dynamics` reuses the descriptor engine for sparse identification of
governing equations:

- `discover_ode` -- explicit ODE discovery (SINDy, Brunton et al. 2016); recovers
  the Lorenz system to R^2=1.0. iBART's adaptive, unit-filtered library is a
  richer alternative to a fixed polynomial basis.
- `implicit_constraints` -- the **foundation** for differential-algebraic (DAE)
  discovery: finds algebraic constraints g(x)~0 via the library null space
  (recovers `x^2 + y^2 - 1 = 0` exactly). A production DAE module needs sparse
  null-space selection and a weak formulation (see [References](docs/references.md),
  Mangan 2016 / Kaheman 2020 / Messenger & Bortz 2021).

## References

This work builds on the iBART papers, BART, SISSO, conformal prediction, and
SINDy literature. See [`docs/references.md`](docs/references.md) for the full,
cited bibliography.

## License and attribution

MIT (see [`LICENSE`](LICENSE)). This is an independent reimplementation of the
iBART method; it copies no code from the GPL R `iBART` package and depends only
on permissively licensed libraries (MIT `bartz`, Apache `jax`/`optax`, BSD
`sympy`/`scikit-learn`). Please cite the original iBART papers and `bartz`; see
[`NOTICE.md`](NOTICE.md). Bundled data provenance is in
[`src/jax_ibart/data/README.md`](src/jax_ibart/data/README.md).
