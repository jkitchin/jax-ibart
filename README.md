# jax-ibart

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

## Status

Early scaffold (M0/M1). See [`PLAN-jax-ibart.md`](PLAN-jax-ibart.md) for the full
design and milestones.

- [x] Package skeleton, `pyproject`, dataset stubs, test harness (M0)
- [x] Operator algebra + feature-engineering engine + symbolic provenance (M1)
- [x] BART-G.SE selection via `bartz` -- faithful oracle (M2a)
- [x] Differentiable selection backend -- hard-concrete L0 gates, ~13x faster (M2b)
- [x] LASSO + l0 final stage + orchestrator -- `IBart.fit`/`.predict` end-to-end (M3)
- [x] Validation on the catalysis dataset -- R^2=0.93, recovers the Hfo/Oxv drivers (M4)
- [x] Dimensional-analysis (unit) filter -- drops ~70% of the space, faster, more physical
- [x] Vectorized feature engineering + benchmarks -- FE ~5x faster, GPU-ready (M5)
- [x] R-iBART oracle cross-check -- against the authors' shipped output (docs/oracle-crosscheck.md)
- [x] Closed the exact-descriptor gap -- recovers `abs(Hfo/Oxv)` via a permissive PAN-aligned screen (not budget/GPU)
- [x] Uncertainty quantification -- conformal (honest CIs), selection stability, repeated-split RMSE, OLS CIs, BART posterior

## Design in one paragraph

JAX traces carry the *numeric* descriptor columns; plain Python + SymPy carry
each descriptor's *symbolic* expression, kept in lockstep but never mixed inside
a trace. Selection is pluggable behind one interface with two backends:
`bart` (faithful BART-G.SE via [`bartz`](https://github.com/Gattocrucco/bartz),
the GPU JAX BART) and `diff` (a differentiable soft-tree / neural importance
screener). Both must satisfy the PAN criterion (no false negatives in screening).

## Install (dev)

```bash
uv pip install -e ".[all]"   # bart + diff + dev extras
pytest                       # M1 tests run without bartz
```

## Quick taste

```python
from jax_ibart import IBart

model = IBart(backend="bart").fit(X, y, feature_names=names)
print(model.expression())   # e.g.  -0.099*BVS*Hfo + ...
```

## Documentation

A full docs site (mkdocs-material + API reference + runnable example notebooks):

```bash
uv pip install -e ".[docs]"
mkdocs serve        # http://127.0.0.1:8000
```

- Worked notebooks under [`docs/examples/`](docs/examples/): quickstart, the
  catalysis case study, `bart` vs `diff` backends, and custom operators/units.
- Validation: [`docs/validation-catalysis.md`](docs/validation-catalysis.md),
  the [R-iBART oracle cross-check](docs/oracle-crosscheck.md), and
  [benchmarks](docs/benchmark.md).
- Rebuild the example notebooks: `python scripts/build_notebooks.py` then
  `jupyter nbconvert --to notebook --execute --inplace docs/examples/*.ipynb`.

## Comparison to [jaxsr](https://github.com/jkitchin/jaxsr)

[`jaxsr`](https://github.com/jkitchin/jaxsr) ("a symbolic regression engine for
JAX") is a **broad, mature general-purpose** SR toolkit; `jax-ibart` is a
**faithful, specialized** implementation of one method (iBART/OIS). They overlap
(both JAX/GPU, LASSO sparse selection, conformal + bootstrap UQ, OLS intervals,
BIC, SymPy export, parity plots, sklearn-ish, and both can do explicit SINDy-ODE)
but their distinctive strengths differ.

**Only `jax-ibart`:**

| Capability | Why it matters |
|---|---|
| **Iterative operator-composition library** | *Grows* the library by composing operators and pruning between rounds, reaching deep descriptors like `(exp(x1)-exp(x2))^2` without enumerating the O(10^17) space. `jaxsr` uses a fixed, user-specified basis. |
| **Nonparametric BART-G.SE screening (PAN)** | Bayesian-tree, monotone-invariant screening robust to the heavy collinearity iterated operators create; this is *what makes* the iterative growth tractable. |
| **Dimensional-analysis (unit) filter** | Drops dimensionally invalid descriptors (`volume + speed`). |
| **Differentiable selection backend** | Hard-concrete L0 gates: selection that gradients flow through. |
| **Implicit / null-space discovery** | Finds algebraic constraints `g(x)~0` (DAE foundation; recovers Michaelis-Menten exactly). |
| **Selection-stability + BART posterior UQ** | Per-descriptor bootstrap inclusion frequency; Bayesian tree posterior intervals. |
| **Validated against reference R-iBART** | Oracle cross-check; Liu 2022 / Ye 2024 (see [References](docs/references.md)). |

**Only `jaxsr`:**

| Capability | Notes |
|---|---|
| **Symbolic classification** | binary/multiclass. `jax-ibart` is regression-only. |
| **Physical constraints** | monotonicity, convexity/concavity, bounds, sign, linear. `jax-ibart` has none (only the unit filter). |
| **Bayesian Model Averaging / Pareto ensembles** | average over models; complexity-accuracy front. `jax-ibart` selects one sparse model. |
| **Adaptive sampling / DOE** | uncertainty/leverage/gradient acquisition. |
| **Broader UQ + criteria** | jackknife+ conformal, mean-response bands, AIC/AICc. |
| **Tooling & maturity** | LaTeX export, JSON persistence, full sklearn (`set_params`/`GridSearchCV`/`Pipeline`), CLI, and a rich visualization suite. |

**One-liner:** reach for `jaxsr` as a flexible, full-featured general SR toolkit
(constraints, BMA, DOE, classification, visualization); reach for `jax-ibart`
when you specifically want the iBART method, adaptive library *growth* for hard
high-order nonlinear descriptors, BART-based screening, dimensional-analysis-aware
discovery, a differentiable backend, or the implicit-constraint / DAE direction.

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
[`data/README.md`](data/README.md).
