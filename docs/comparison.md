# Comparison to jaxsr

[`jaxsr`](https://github.com/jkitchin/jaxsr) ("a symbolic regression engine for
JAX") is a **broad, mature general-purpose** symbolic-regression toolkit;
`jax-ibart` is a **faithful, specialized** implementation of one method
(iBART/OIS). They overlap substantially but their distinctive strengths differ.

## Shared (real overlap)

Both are JAX (JIT + GPU), use LASSO sparse selection, provide conformal and
bootstrap uncertainty quantification and OLS coefficient intervals, score with
BIC, export to SymPy, draw parity plots, expose a scikit-learn-style API, and can
do explicit SINDy-style ODE discovery.

## Only `jax-ibart`

| Capability | Why it matters |
|---|---|
| **Iterative operator-composition library** | *Grows* the library by composing operators and pruning between rounds, reaching deep descriptors like `(exp(x1)-exp(x2))^2` without enumerating the O(10^17) candidate space. `jaxsr` uses a fixed, user-specified basis. |
| **Nonparametric BART-G.SE screening (PAN)** | Bayesian-tree, monotone-invariant screening robust to the heavy collinearity that iterated operators create; this is *what makes* the iterative growth tractable. |
| **Dimensional-analysis (unit) filter** | Drops dimensionally invalid descriptors (`volume + speed`). |
| **Differentiable selection backend** | Hard-concrete L0 gates: selection that gradients flow through. |
| **Implicit / null-space discovery** | Finds algebraic constraints `g(x)~0` (the DAE foundation; recovers Michaelis-Menten exactly). |
| **Selection-stability + BART posterior UQ** | Per-descriptor bootstrap inclusion frequency; Bayesian-tree posterior predictive intervals. |
| **Validated against reference R-iBART** | Oracle cross-check; Liu 2022 / Ye 2024 (see [References](references.md)). |

## Only `jaxsr`

| Capability | Notes |
|---|---|
| **Symbolic classification** | binary/multiclass. `jax-ibart` is regression-only. |
| **Physical constraints** | monotonicity, convexity/concavity, bounds, sign, linear. `jax-ibart` has none (only the unit filter). |
| **Bayesian Model Averaging / Pareto ensembles** | average over models; complexity-accuracy front. `jax-ibart` selects one sparse model. |
| **Adaptive sampling / DOE** | uncertainty / leverage / gradient acquisition functions. |
| **Broader UQ + criteria** | jackknife+ conformal, mean-response confidence bands, AIC/AICc. |
| **Tooling & maturity** | LaTeX export, JSON persistence, full sklearn (`set_params` / `GridSearchCV` / `Pipeline`), CLI tools, and a rich visualization suite (Pareto, residual, coefficient paths, forest plots, BMA weights). |

## Which to use

- **`jaxsr`** -- a flexible, full-featured general SR toolkit: when you can
  specify the basis and want constraints, Bayesian model averaging, design of
  experiments / active learning, classification, rich UQ, or visualization.
- **`jax-ibart`** -- when you specifically want the iBART method: adaptive
  library *growth* for hard high-order nonlinear descriptors, BART-based
  screening, dimensional-analysis-aware discovery, a differentiable selection
  backend, fidelity to the published catalysis results, or the
  implicit-constraint / DAE direction.

> Capabilities for `jaxsr` are summarized from its README; see the
> [jaxsr repository](https://github.com/jkitchin/jaxsr) for the authoritative,
> current feature set.
