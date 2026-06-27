"""Generate the worked-example notebooks under docs/examples/.

Run:  PYTHONPATH=src python scripts/build_notebooks.py
Then execute them with:  jupyter nbconvert --to notebook --execute --inplace docs/examples/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "docs" / "examples"
OUT.mkdir(parents=True, exist_ok=True)


def nb(*cells):
    n = nbf.v4.new_notebook()
    n.cells = list(cells)
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}}
    return n


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# --------------------------------------------------------------------------- #
# 01 - Quickstart
# --------------------------------------------------------------------------- #
quickstart = nb(
    md("""
# Quickstart: recover a hidden descriptor

`jax-ibart` discovers compact, interpretable nonlinear descriptors. Here we hide
one inside synthetic data and recover it from scratch.
"""),
    code("""
import numpy as np
from jax_ibart import IBart

rng = np.random.default_rng(0)
n = 90
a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
X = np.column_stack([a, b, c, d])
# the response depends only on the product a*b
y = 5.0 * (a * b) + rng.normal(0, 0.05, n)
names = ["a", "b", "c", "d"]
X.shape, y.shape
"""),
    md("Fit with the differentiable backend (fast, no Java needed):"),
    code("""
model = IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=1200)
model.fit(X, y, feature_names=names)
model.descriptors_
"""),
    md("The fitted symbolic model and its fit metrics:"),
    code("""
import numpy as np

pred = model.predict(X)
resid = y - pred
ss_res = float((resid ** 2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
metrics = {
    "expression": str(model.expression()),
    "R2": 1 - ss_res / ss_tot,
    "RMSE": float(np.sqrt((resid ** 2).mean())),
    "MAE": float(np.abs(resid).mean()),
    "max_abs_error": float(np.abs(resid).max()),
}
for k, v in metrics.items():
    print(f"{k:>14}: {v}")
"""),
    md("A parity plot (predicted vs. true) is the standard visual fit check:"),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.5, 4.5))
lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="parity")
ax.scatter(y, pred, s=18, alpha=0.7)
ax.set_xlabel("true y")
ax.set_ylabel("predicted y")
ax.set_title(f"Parity (R$^2$={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.3f})")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
plt.show()
"""),
    md("""
It recovers `a*b` and fits `~5*a*b`, the hidden relationship. The descriptor is
*symbolic*, so you can read the physics straight off the model, and the parity
plot lands on the diagonal.
"""),
)

# --------------------------------------------------------------------------- #
# 02 - Catalysis case study
# --------------------------------------------------------------------------- #
catalysis = nb(
    md("""
# Catalysis case study

We run the full pipeline on the single-atom catalysis dataset (91 metal/oxide
binding energies, 59 primary features) and compare to the published iBART result.
"""),
    code("""
import numpy as np
from jax_ibart import IBart
from jax_ibart.datasets import load_metal_support

X, y, names, units = load_metal_support()
print("X:", X.shape, " y:", y.shape, " features:", len(names))
print("two known drivers present:", "Hfo" in names, "Oxv" in names)
"""),
    md("""
We use the differentiable backend here so the notebook runs in seconds. The
faithful `bart` backend (BART-G.SE) is more accurate but takes minutes; its
results and the published comparison are in
[the validation writeup](../validation-catalysis.md) and
[the oracle cross-check](../oracle-crosscheck.md).

We pass `units=units` to enable the dimensional-analysis filter (it drops
physically meaningless descriptors like `volume + speed`).
"""),
    code("""
model = IBart(backend="diff", schedule=("b", "u", "b"), max_dim=3,
              n_keep=10, seed=0, steps=1000, l0=0.03).fit(X, y, names, units=units)

pred = model.predict(X)
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"in-sample R^2 = {r2:.3f}\\n")
for desc, coef in zip(model.descriptors_, model.coef_[1:]):
    print(f"  {coef:+.4g} * {desc}")
"""),
    md("""
### Comparison to the published (reference R-iBART) result
"""),
    code("""
import json
from pathlib import Path

oracle = json.loads(Path("../../data/oracle_ibart.json").read_text())
print("published canonical descriptor :", oracle["canonical_descriptor"])
print("published l0 1-descriptor model:", oracle["lzero_1"])
print("published l0 3-descriptor model:", oracle["lzero_3"])
print("published l0 RMSEs (k=1..5)    :", [round(v, 3) for v in oracle["lzero_in_sample_rmse"]])
"""),
    md("""
The reference model's canonical descriptor is `abs(Hfo/Oxv)`, coupling the oxide
formation enthalpy and the oxygen vacancy energy, the metal-oxygen
charge-transfer drivers of metal-support binding. Our port recovers models built
on these same physical drivers; exact-descriptor reproduction needs a permissive
PAN-aligned screen (see the oracle cross-check for the honest details).
"""),
    md("""
## References and data provenance

- **Dataset** (`load_metal_support`): the single-atom catalysis `catalysis` data
  (91 metal/oxide binding energies, 59 features) shipped with the R `iBART`
  package; the underlying DFT data are from the Senftle group's MIT-licensed
  [`Metal-Oxide-LASSO-lo`](https://github.com/tsenftle/Metal-Oxide-LASSO-lo)
  repository. See [`data/README.md`](https://github.com/jkitchin/jax-ibart/blob/main/data/README.md).
- **Reference result** (`oracle_ibart.json`): the published iBART l0 descriptors
  and RMSEs, from the papers below (citable facts; see `NOTICE.md`).
- **Method**: C.-Y. Liu, S. Ye, M. Li, T. P. Senftle, *J. Chem. Phys.* **156**,
  164105 (2022); S. Ye, T. P. Senftle, M. Li, *JASA* **119**:81-94 (2024),
  arXiv:2110.10195.
- Full bibliography: [docs/references.md](../references.md). Honest comparison to
  the reference implementation: [docs/oracle-crosscheck.md](../oracle-crosscheck.md).
"""),
)

# --------------------------------------------------------------------------- #
# 03 - bart vs diff backends
# --------------------------------------------------------------------------- #
backends = nb(
    md("""
# Faithful vs differentiable selection

`jax-ibart` ships two interchangeable selection backends behind one interface:

- **`bart`** -- faithful BART-G.SE (Bleich 2014 global-SE permutation test) via
  `bartz`. The correctness oracle; matches the reference R iBART.
- **`diff`** -- a differentiable hard-concrete-L0 gate screener (Louizos et al.
  2018). Gradients flow end to end; much faster, lower fidelity on hard problems.

Both must satisfy the PAN criterion (no false negatives in screening).
"""),
    code("""
import numpy as np
from jax_ibart.selection import BartGSESelector, DiffSelector
import jax

# y depends only on x0, x1 (Friedman); x2..x7 are noise.
rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(120, 8))
y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + rng.normal(0, 0.5, 120)
"""),
    md("Both backends keep the true drivers and reject noise:"),
    code("""
diff_keep = np.asarray(DiffSelector(steps=1500).select(X, y, key=jax.random.PRNGKey(0)))
bart_keep = np.asarray(BartGSESelector(n_perm=15, n_tree=40, n_draw=300, n_burn=300)
                       .select(X, y, key=jax.random.PRNGKey(0)))
print("diff keeps:", np.where(diff_keep)[0].tolist())
print("bart keeps:", np.where(bart_keep)[0].tolist())
print("truth      : [0, 1]")
"""),
    md("""
### When to use which

| | `bart` (faithful) | `diff` (differentiable) |
|---|---|---|
| Fidelity | high (matches R iBART) | lower on high-dim screens |
| Speed | minutes (CPU) | seconds |
| Differentiable | no (discrete MCMC) | yes, end to end |
| Use for | final/validated results | fast exploration, gradient-based work |

On the real catalysis screen, `diff` is ~13x faster but recovers fewer of the
exact drivers; see [the benchmark](../benchmark.md).
"""),
)

# --------------------------------------------------------------------------- #
# 04 - Custom operators and units
# --------------------------------------------------------------------------- #
custom = nb(
    md("""
# Custom operators and units

You can define your own operator set and a unit system for a new domain. An
operator bundles a numeric function, a SymPy function, a domain-validity
predicate, and a unit rule.
"""),
    code("""
import numpy as np
import jax.numpy as jnp
import sympy as sp
from jax_ibart import engineer
from jax_ibart.operators import Operator, u_keep, u_add, BINARY, UNARY

# Inspect a built-in operator
op = BINARY["mul"]
op.name, op.arity, op.commutative
"""),
    md("""
### Units: a worked dimensional example

Units are integer-exponent vectors over base dimensions. Additive operators only
combine *matching* units; multiply/divide add/subtract exponents. Here feature
`x` is a length and `z` is a time, so `x + z` is rejected but `x*z` and `x/z` are
allowed.
"""),
    code("""
X = np.random.default_rng(0).uniform(1, 2, size=(12, 2))
units = np.array([[1, 0],   # x: [length, time] exponents
                  [0, 1]])  # z
pool = engineer.from_primary(X, ["x", "z"], units=units)

add = engineer.apply_binary(pool, [BINARY["add"]])
mul = engineer.apply_binary(pool, [BINARY["mul"]])
print("x + z generated:", add.n_descriptors, "(rejected: mismatched units)")
print("x * z generated:", mul.n_descriptors, "-> unit", mul.units[0].tolist())
"""),
    md("""
### Defining your own operator

A cube operator: numeric `x**3`, symbolic `e**3`, valid everywhere, unit `3*u`.
"""),
    code("""
cube = Operator(
    name="cube", arity=1,
    fn=lambda x: x ** 3,
    sym=lambda e: e ** 3,
    valid=lambda x: jnp.ones(x.shape[-1], bool),
    unit=lambda u: 3 * u,
)
out = engineer.apply_unary(pool, [cube])
print([str(e) for e in out.exprs], "-> units", out.units.tolist())
"""),
    md("""
Pass custom operators by editing the operator sets used in `IBart.fit`, or use
them directly with the `engineer` module as shown. The same lockstep design
holds: numbers live in JAX arrays, symbols in SymPy, units in integer vectors.
"""),
)

# --------------------------------------------------------------------------- #
# 05 - Dynamical systems (SINDy / DAE foundation)
# --------------------------------------------------------------------------- #
dynamics_nb = nb(
    md("""
# Discovering dynamical systems (SINDy / DAEs)

`jax_ibart.dynamics` reuses the descriptor engine for sparse identification of
governing equations (Brunton, Proctor & Kutz 2016). For ODEs it sparse-regresses
each state's time derivative on a library of candidate terms; iBART's adaptive,
unit-filtered library is a richer alternative to a fixed polynomial basis.

References: see [docs/references.md](../references.md).
"""),
    md("## Explicit ODE discovery: the Lorenz system"),
    code("""
import numpy as np
from scipy.integrate import solve_ivp
from jax_ibart import dynamics

def lorenz(t, s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

t = np.arange(0.0, 20.0, 0.002)
sol = solve_ivp(lorenz, [t[0], t[-1]], [-8, 8, 27], t_eval=t, rtol=1e-9, atol=1e-9)
X = sol.y.T   # (n_time, 3) trajectory
X.shape
"""),
    md("Derivatives are estimated from the trajectory (finite differences), then each equation is recovered by sparse regression:"),
    code("""
system = dynamics.discover_ode(X, t, ["x", "y", "z"], degree=2, max_terms=3)
print(system)
print("\\ntrue: dx/dt = 10(y - x);  dy/dt = 28x - y - xz;  dz/dt = xy - 2.667 z")
"""),
    md("""
The Lorenz equations are recovered with R^2 = 1.0 and the correct sparse terms
and coefficients, directly from the trajectory.
"""),
    md("### The attractor we learned from"),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(X[:, 0], X[:, 2], lw=0.4)
ax.set_xlabel("x"); ax.set_ylabel("z")
ax.set_title("Lorenz trajectory (x-z projection)")
fig.tight_layout(); plt.show()
"""),
    md("""
### How the fit works: derivative parity

SINDy fits each `dx_i/dt` as a sparse linear combination of library terms. The
parity plots below show the finite-difference derivative (the target) vs. the
fitted RHS, per equation. Points on the diagonal mean the recovered equation
reproduces the dynamics.
"""),
    code("""
from jax_ibart.dynamics import finite_difference
import sympy as sp

dX = finite_difference(X, t)
syms = sp.symbols("x y z")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
    f = sp.lambdify(syms, system.rhs[i], "numpy")
    pred = f(X[:, 0], X[:, 1], X[:, 2])
    ax.scatter(dX[:, i], pred, s=4, alpha=0.3)
    lo, hi = dX[:, i].min(), dX[:, i].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"d{name}/dt  (R$^2$={system.r2[i]:.3f})")
    ax.set_xlabel("true (finite diff)"); ax.set_ylabel("fitted")
fig.tight_layout(); plt.show()
"""),
    md("""
### Does the discovered model reproduce the dynamics?

The strongest test: integrate the *discovered* equations from the same initial
condition and compare to the true trajectory.
"""),
    code("""
funcs = [sp.lambdify(syms, e, "numpy") for e in system.rhs]
def learned(t, s):
    return [float(f(*s)) for f in funcs]

sol2 = solve_ivp(learned, [t[0], t[-1]], [-8, 8, 27], t_eval=t, rtol=1e-9, atol=1e-9)
Xh = sol2.y.T

fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
    ax.plot(t, X[:, i], lw=1.0, label="true")
    ax.plot(t, Xh[:, i], "--", lw=1.0, label="discovered")
    ax.set_ylabel(name)
axes[0].legend(loc="upper right"); axes[-1].set_xlabel("t")
fig.suptitle("True vs. discovered Lorenz trajectory")
fig.tight_layout(); plt.show()
"""),
    md("""
The discovered system tracks the true trajectory (they eventually diverge only
because Lorenz is chaotic -- tiny numerical differences amplify exponentially,
which is a property of the system, not a flaw in the recovered equations).
"""),
    md("""
## Toward DAEs: implicit algebraic constraints

A differential-algebraic system also has algebraic constraints `0 = g(x, z)`.
Finding those is an *implicit* (null-space) problem, not a regression on a
target, so it needs different machinery. `implicit_constraints` is the
foundation: it finds combinations of library terms that are ~0. Here it recovers
the constraint of points sampled on a unit circle.
"""),
    code("""
theta = np.linspace(0, 2 * np.pi, 200)
C = np.column_stack([np.cos(theta), np.sin(theta)])   # x^2 + y^2 = 1
res = dynamics.implicit_constraints(C, ["x", "y"], degree=2)
print("constraint ~ 0:", res["constraint"])
print("residual (0 = exact):", round(res["residual"], 6))
print("note:", res["note"])
"""),
    md("""
It recovers `x^2 + y^2 - 1 = 0` exactly. A production DAE module would add sparse
null-space selection and a weak/integral formulation (Mangan et al. 2016;
Kaheman et al. 2020; Messenger & Bortz 2021) and separate differential from
algebraic relations. The descriptor generation, unit filter, screening, and
uncertainty tools all carry over.
"""),
)

# --------------------------------------------------------------------------- #
# 06 - Reaction kinetics: discover the mole-balance RHS with BART
# --------------------------------------------------------------------------- #
reaction = nb(
    md("""
# Reaction kinetics: discovering rate laws with BART

Given concentration-vs-time data from a batch reactor, can we recover the
mole-balance right-hand sides (the rate laws)? This is SINDy applied to chemical
kinetics, with **BART** doing the term selection.

The example dataset (`scripts/make_reaction_data.py`) is a liquid-phase batch
reactor with two reactions:

- `A + B -> C`,  rate `r1 = k1 * C_A * C_B`  (bimolecular, k1 = 0.5)
- `C -> D`,      rate `r2 = k2 * C_C`        (first order,  k2 = 0.2)

so the true mole balances are
`dCA/dt = -k1 CA CB`, `dCC/dt = k1 CA CB - k2 CC`, `dCD/dt = k2 CC`.
"""),
    md("""
The measured concentrations carry realistic Gaussian **measurement noise**
(~1%), so the derivatives estimated from them are noisy too -- exactly the
challenge of real kinetic data.
"""),
    code("""
import numpy as np
from jax_ibart.datasets import load_reaction

# C: concentrations, dCdt: per-batch finite-difference derivatives (the targets)
C, dCdt, species, batch = load_reaction()
print("samples:", C.shape, "| species:", species, "| batches:", len(set(batch)))
"""),
    md("The concentration trajectories for one batch (note the measurement scatter):"),
    code("""
import matplotlib.pyplot as plt

m = batch == 0
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for j, s in enumerate(species):
    ax.plot(np.arange(m.sum()), C[m, j], "o", ms=3, alpha=0.7, label=s)
ax.set_xlabel("time index (batch 0)")
ax.set_ylabel("concentration")
ax.set_title("Noisy batch-reactor data")
ax.legend()
fig.tight_layout()
plt.show()
"""),
    md("""
Now let BART find which library terms enter each mole-balance RHS
(`screen="bart"` runs the BART-G.SE nonparametric screen before the sparse l0
fit). Derivatives were estimated per batch, so we pass them via `dXdt`.
"""),
    code("""
from jax_ibart.dynamics import discover_ode

system = discover_ode(C, t=None, feature_names=species, dXdt=dCdt,
                      degree=2, max_terms=2, screen="bart", seed=0)
print(system)
print("\\ntrue: dCA/dt=-0.5 CA CB ; dCB/dt=-0.5 CA CB ; "
      "dCC/dt=0.5 CA CB - 0.2 CC ; dCD/dt=0.2 CC")
"""),
    md("""
BART recovers the **bimolecular rate law** `dCA/dt ~ -0.5 C_A C_B` and the
**first-order** `dCD/dt ~ 0.2 C_C` directly from noisy data. The intermediate
`C_C` is the hardest (its derivative is small and noisy, and two terms compete) —
a realistic reminder that intermediates and noisy derivatives are where kinetic
discovery is most delicate; a weak/integral formulation (Messenger & Bortz 2021,
see [references](../references.md)) is the robust next step.

The dimensional-analysis filter (pass `units=...`) would further constrain the
library to rate terms with consistent units, a natural fit for engineering
kinetics.
"""),
    md("Rate-fit parity per species (estimated dC/dt vs. the discovered RHS):"),
    code("""
import sympy as sp

syms = sp.symbols("CA CB CC CD")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for i, (ax, s) in enumerate(zip(axes, species)):
    f = sp.lambdify(syms, system.rhs[i], "numpy")
    pred = f(C[:, 0], C[:, 1], C[:, 2], C[:, 3])
    ax.scatter(dCdt[:, i], np.broadcast_to(pred, dCdt[:, i].shape), s=5, alpha=0.3)
    lo, hi = dCdt[:, i].min(), dCdt[:, i].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"d{s}/dt (R$^2$={system.r2[i]:.2f})")
    ax.set_xlabel("measured"); ax.set_ylabel("fitted")
fig.tight_layout(); plt.show()
"""),
    md("""
The bimolecular species (`A`, `B`) and the product `C` fit tightly; the weak,
slow `dCD/dt` is noise-dominated (small derivative), so its parity scatters --
the honest signature of a hard-to-identify slow step.
"""),
)

# --------------------------------------------------------------------------- #
# 07 - What iBART can and cannot find (nonlinearity guide)
# --------------------------------------------------------------------------- #
nonlinear = nb(
    md("""
# What iBART can (and can't) find

iBART discovers **nonlinear equations** -- that is its whole purpose. It builds
nonlinear descriptors by composing operators (`square, sqrt, log, exp, sin, cos,
abs, *, /, |a-b|`) on the inputs. The model is *linear in the coefficients on a
nonlinear basis* (like SINDy or polynomial regression): the discovered equation
is nonlinear, the fit is convex.

This notebook shows the wins, the one real limitation, and two workarounds.
"""),
    md("## 1. Nonlinear expressions it recovers exactly"),
    code("""
import numpy as np
from jax_ibart import IBart

rng = np.random.default_rng(0)
n = 200
x1, x2, x3 = (rng.uniform(0.1, 1, n) for _ in range(3))
X = np.column_stack([x1, x2, x3]); names = ["x1", "x2", "x3"]

def r2(model, y):
    p = model.predict(X)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# transcendental: y = sin(pi * x1 * x2)
y = np.sin(np.pi * x1 * x2)
m = IBart(backend="diff", schedule=("b", "u"), max_dim=1, seed=0, steps=1500).fit(X, y, names)
print(f"sin(pi*x1*x2)         -> {m.descriptors_}  R2={r2(m, y):.4f}")

# the OIS paper's example: y = (exp(x1) - exp(x2))^2
y = (np.exp(x1) - np.exp(x2)) ** 2
m = IBart(backend="diff", schedule=("u", "b", "u"), max_dim=1, seed=0, steps=1500).fit(X, y, names)
print(f"(exp(x1)-exp(x2))^2   -> {m.descriptors_}  R2={r2(m, y):.4f}")
"""),
    md("""
Both are recovered with R^2 = 1.0. Transcendental, rational, product, and
composed nonlinearities are all in scope as long as the operators can build them.
"""),
    md("""
## 2. The one real limitation: free parameters *inside* a nonlinearity

iBART fits **linear coefficients** on a fixed, operator-generated library. It
cannot fit a *continuous parameter buried inside* a term unless that exact term
is generated. A power law with a non-integer exponent is the classic case: there
is no arbitrary-power operator, so `x^2.3` is only *approximated* (here by `x^2`).
"""),
    code("""
xp = rng.uniform(0.5, 3.0, n)
Xp = np.column_stack([xp, rng.uniform(0, 1, n)])
y = xp ** 2.3                      # exponent 2.3 is not constructible
m = IBart(backend="diff", schedule=("u",), max_dim=1, seed=0, steps=1200).fit(Xp, y, ["x", "d"])
pr = m.predict(Xp)
print("x^2.3  -> found", m.descriptors_,
      f" R2={1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum():.4f}  (approximate, not exact)")
"""),
    md("""
Same story for **Arrhenius** `k = A exp(-E/RT)` with a fitted activation energy
`E` *inside* the exponential, or **Michaelis-Menten** `v = Vmax S/(Km+S)` with a
fitted `Km` *in the denominator*. These are nonlinear *in the parameters*. Two
standard fixes:
"""),
    md("### Workaround A -- linearize (Arrhenius)"),
    code("""
R = 8.314; A_true = 1e3; E_true = 50000.0
T = rng.uniform(300, 600, n)
k = A_true * np.exp(-E_true / (R * T))

# ln(k) = ln(A) - (E/R) * (1/T)  is LINEAR in 1/T
invT = 1.0 / T
slope, intercept = np.polyfit(invT, np.log(k), 1)
print(f"recovered E = {-slope * R:,.0f} J/mol   (true 50000)")
print(f"recovered A = {np.exp(intercept):.1f}      (true 1000)")
"""),
    md("""
Transforming to `ln k` vs `1/T` turns the nonlinear-in-parameter problem into a
linear one. iBART recovers the same structure when given `1/T` (its `inv`
operator) and the log-transformed target.
"""),
    md("### Workaround B -- implicit / null-space form (Michaelis-Menten)"),
    md("""
**What "null space" means, basically.** Usual regression fits `y = f(x)` -- one
column is the target, and we predict it from the others. But a constraint like
`x^2 + y^2 = 1` has *no* target: no single variable is "the output." Instead, a
particular weighted sum of the candidate terms is **always zero**.

So build a table whose columns are candidate terms (here `S, v, S^2, v^2, S*v`,
and a column of `1`s). We want weights `w` such that

    w1*S + w2*v + ... + w6*1  =  0   for every row.

That weight vector is called a **null-space vector** of the table: multiplying
the table by it gives all zeros. Standard regression can't find it (there's no
`y`); a singular value decomposition (SVD) can -- the direction the data *never*
varies along is exactly the constraint. The smallest singular value measures how
close to "exactly zero" it is (0 = an exact law).

For Michaelis-Menten, rearranging `v = Vmax*S/(Km+S)` gives
`Vmax*S - Km*v - S*v = 0` -- a sum of terms that is always zero, i.e. a
null-space vector of the `[S, v, S*v, ...]` table. That is what we recover:
"""),
    code("""
from jax_ibart.dynamics import implicit_constraints
import sympy as sp

Vmax, Km = 5.0, 2.0
S = rng.uniform(0.1, 10.0, n)
v = Vmax * S / (Km + S)

# v(Km + S) = Vmax S  ->  Vmax*S - Km*v - S*v = 0  : an algebraic constraint
res = implicit_constraints(np.column_stack([S, v]), ["S", "v"], degree=2, tol=1e-2)
print("recovered constraint:", sp.nsimplify(res["constraint"], rational=False).evalf(3))
print("residual (0 = exact):", round(res["residual"], 6))
print("\\ntrue: 5*S - 2*v - S*v = 0  ->  divide by 5:  S - 0.4*v - 0.2*S*v = 0")
"""),
    md("""
The implicit (null-space) formulation recovers the rational Michaelis-Menten
kinetics **exactly** as an algebraic constraint, from which `Vmax` and `Km` read
straight off the coefficients. This is the same machinery that underlies the
DAE foundation (`implicit_constraints`) and SINDy-PI (see
[references](../references.md)).

**Summary:** nonlinear *expressions* -- yes, that is the point. Free parameters
*inside* a nonlinearity -- only via the operator library, a linearizing
transform, or the implicit null-space form.
"""),
)

# --------------------------------------------------------------------------- #
# 08 - Uncertainty quantification
# --------------------------------------------------------------------------- #
uq = nb(
    md("""
# Uncertainty quantification

iBART is a *selection* method, so its dominant uncertainty is **which descriptors
get chosen**, not just the spread around a fixed model. `jax_ibart.uncertainty`
provides honest, calibrated tools for this. We use a noisy synthetic problem
where only `a*b` drives the response (`c`, `d` are irrelevant).
"""),
    code("""
import numpy as np
from jax_ibart import IBart, uncertainty

rng = np.random.default_rng(0)
n = 240
a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
X = np.column_stack([a, b, c, d]); names = ["a", "b", "c", "d"]
y = 5.0 * (a * b) + rng.normal(0, 0.4, n)        # noisy

# a factory makes a fresh, identically-configured estimator on demand
factory = lambda: IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=900)
"""),
    md("""
## 1. Conformal prediction intervals (honest, calibrated)

Split conformal wraps the **entire fit** (selection included) and calibrates on
held-out data, giving a finite-sample coverage guarantee even after selection.
We check the *realized* coverage on a fresh test set.
"""),
    code("""
idx = rng.permutation(n); te, tr = idx[:80], idx[80:]
conf = uncertainty.conformal_intervals(factory, X[tr], y[tr], names,
                                       alpha=0.10, cal_frac=0.4, seed=0)
cov = uncertainty.empirical_coverage(conf, X[te], y[te])
print(f"target coverage: 0.90   realized: {cov:.2f}   half-width: {conf.half_width:.3f}")
"""),
    code("""
import matplotlib.pyplot as plt

lo, point, hi = conf.interval(X[te])
order = np.argsort(point)
fig, ax = plt.subplots(figsize=(7, 3.6))
xs = np.arange(len(order))
ax.fill_between(xs, lo[order], hi[order], alpha=0.25, label="90% conformal band")
ax.plot(xs, point[order], lw=1, label="prediction")
ax.scatter(xs, y[te][order], s=12, color="k", label="true", zorder=3)
ax.set_xlabel("test point (sorted by prediction)"); ax.set_ylabel("y")
ax.legend(); fig.tight_layout(); plt.show()
"""),
    md("""
## 2. Selection stability

The most informative UQ for a selection method: how often is each feature chosen
across bootstrap resamples? The true drivers `a`, `b` should dominate.
"""),
    code("""
stab = uncertainty.selection_stability(factory, X, y, names, n_boot=20, seed=0)
ff = stab["feature_freq"]
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(list(ff.keys()), list(ff.values()))
ax.set_ylabel("inclusion frequency"); ax.set_ylim(0, 1)
ax.set_title("Feature selection stability (bootstrap)")
fig.tight_layout(); plt.show()
print(ff)
"""),
    md("""
## 3. Out-of-sample RMSE distribution

The papers' protocol: RMSE over many random train/test splits, as a distribution
rather than a single number.
"""),
    code("""
cv = uncertainty.repeated_split_rmse(factory, X, y, names, n_splits=20, seed=0)
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(cv["rmse"], bins=12)
ax.axvline(cv["mean"], color="k", ls="--", label=f"mean {cv['mean']:.2f}")
ax.set_xlabel("out-of-sample RMSE"); ax.set_ylabel("count"); ax.legend()
fig.tight_layout(); plt.show()
print(f"RMSE = {cv['mean']:.3f} +/- {cv['std']:.3f}  (90% CI [{cv['q05']:.2f}, {cv['q95']:.2f}])")
"""),
    md("""
## 4. Coefficient intervals (classical, optimistic)

Classical OLS confidence intervals on the final model -- fast, but *optimistic*
because they condition on the selected descriptors and ignore selection. Use the
conformal intervals (section 1) for honest predictive coverage.
"""),
    code("""
model = factory().fit(X, y, names)
ols = uncertainty.ols_intervals(model, X, y)
for term in ols["coefficients"]:
    print(f"{term['term']:>10}: {term['coef']:+.3f}  "
          f"[{term['ci_low']:+.3f}, {term['ci_high']:+.3f}]")
"""),
    md("""
**Takeaway:** report the conformal band for honest prediction intervals and the
selection-stability frequencies to show which descriptors are trustworthy --
together they capture the uncertainty a single point estimate hides. See also
`uncertainty.bart_posterior_intervals` for Bayesian posterior predictive bands.
"""),
)

for name, notebook in [
    ("01_quickstart", quickstart),
    ("02_catalysis", catalysis),
    ("03_backends", backends),
    ("04_custom_operators_units", custom),
    ("05_dynamics", dynamics_nb),
    ("06_reaction_kinetics", reaction),
    ("07_nonlinear", nonlinear),
    ("08_uncertainty", uq),
]:
    path = OUT / f"{name}.ipynb"
    nbf.write(notebook, path)
    print("wrote", path)
