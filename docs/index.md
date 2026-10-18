# jax-ibart

A **JAX-native reimplementation of iBART** (iterative Bayesian Additive
Regression Trees) for symbolic descriptor discovery in materials informatics.
Pure Python, GPU-capable, `pip`-installable (no JVM).

iBART finds compact, interpretable nonlinear descriptors by **interleaving
feature engineering** (applying algebraic operators) with **nonparametric
feature selection**, keeping the active descriptor pool small at every step
instead of enumerating the full O(10^17) candidate space.

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

## Why it exists

The reference implementation is the R package
[`mattsheng/iBART`](https://github.com/mattsheng/iBART) (R + Java via
`bartMachine`). `jax-ibart` re-expresses the method in JAX so it runs on GPU,
installs without a JVM, and exposes a scikit-learn-style API, with a
differentiable selection backend as a research extension.

## Design in one paragraph

JAX traces carry the **numeric** descriptor columns; plain Python + SymPy carry
each descriptor's **symbolic** expression, kept in lockstep but never mixed
inside a trace. Selection is pluggable behind one interface with two backends:

- **`bart`** — faithful BART-G.SE (Bleich 2014 permutation test) via
  [`bartz`](https://github.com/Gattocrucco/bartz), the GPU JAX BART. The
  correctness oracle.
- **`diff`** — a differentiable hard-concrete-L0 gate screener (Louizos et al.
  2018); gradients flow end to end. Faster, lower fidelity on hard problems.

An optional **dimensional-analysis filter** drops physically meaningless
descriptors (e.g. `volume + speed`).

## Validation

On the single-atom catalysis dataset the port selects models built on the
published descriptor `|Hfo/Oxv|` (oxide formation enthalpy over oxygen vacancy
energy), with in-sample RMSE 0.82 eV against the published 0.62 eV for three
descriptors. See [the catalysis writeup](validation-catalysis.md) and the
[R-iBART oracle cross-check](oracle-crosscheck.md).

## Start here

- [Installation](installation.md)
- [Quickstart notebook](examples/01_quickstart.ipynb)
- [API reference](api.md)
- [How it compares to jaxsr](comparison.md)

## Citing

This is an independent reimplementation. Please cite the original iBART papers
(Liu et al. 2022, *J. Chem. Phys.* 156:164105; Ye, Senftle & Li 2024, *JASA*
119:81-94) and `bartz` (Petrillo 2024). See
[`NOTICE.md`](https://github.com/jkitchin/jax-ibart/blob/main/NOTICE.md).
