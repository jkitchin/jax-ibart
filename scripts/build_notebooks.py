"""Generate the worked-example notebooks under docs/examples/.

Run:  PYTHONPATH=src python scripts/build_notebooks.py
Then execute them (the docs render the committed outputs; mkdocs does not
execute notebooks):

    uv run --extra all --with nbconvert --with ipykernel jupyter nbconvert \\
        --to notebook --execute --inplace docs/examples/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "docs" / "examples"
OUT.mkdir(parents=True, exist_ok=True)


def nb(*cells):
    n = nbf.v4.new_notebook()
    n.cells = list(cells)
    n.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
    }
    return n


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# Shared helper for the "grow then prune" figure (01, 02).
HISTORY_PLOT = '''
def plot_history(history, ax=None, title="Descriptor pool at each stage"):
    """Bar chart of pool size generated vs kept at each pipeline stage."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    labels = {"primary": "primary\\nfeatures", "u": "unary\\nops", "b": "binary\\nops",
              "l0": "LASSO + l0"}
    xs = np.arange(len(history))
    gen = [h["generated"] for h in history]
    kept = [h["kept"] for h in history]
    ax.bar(xs - 0.2, gen, 0.4, label="generated", color="C0")
    ax.bar(xs + 0.2, kept, 0.4, label="kept after screening", color="C1")
    for x, g, k in zip(xs, gen, kept):
        ax.text(x - 0.2, g, f"{g:,}", ha="center", va="bottom", fontsize=8)
        ax.text(x + 0.2, k, f"{k:,}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs, [labels.get(h["stage"], h["stage"]) for h in history])
    ax.set_ylabel("number of descriptors")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax
'''

# --------------------------------------------------------------------------- #
# 01 - Quickstart
# --------------------------------------------------------------------------- #
quickstart = nb(
    md("""
# Quickstart: recover a hidden descriptor

iBART looks for a *descriptor*: a short formula built from your input features
that explains a response. It does that by alternating two steps:

1. **Grow**: apply operators (`x^2`, `sqrt`, `log`, `a*b`, `a/b`, `|a-b|`, ...) to
   every descriptor in the pool, so the pool explodes combinatorially.
2. **Prune**: screen the pool with a variable-selection method (BART, or a
   differentiable gate model) and keep only the descriptors that matter.

After the last round, LASSO and an exhaustive l0 best-subset search pick the
final few descriptors, and an ordinary least-squares fit gives their
coefficients. The result is a small, symbolic, linear-in-the-coefficients model
that you can read.

In this notebook we hide a product `a*b` among six features and let iBART find it.
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from jax_ibart import IBart

rng = np.random.default_rng(0)
n = 90
names = ["a", "b", "c", "d", "e", "f"]
X = rng.uniform(0.5, 2.0, size=(n, len(names)))
a, b = X[:, 0], X[:, 1]
# the response depends only on the product a*b; c..f are distractors
y = 5.0 * a * b + rng.normal(0, 0.05, n)
X.shape
"""),
    md("""
## Look at the data first

Plotted against any single feature, `y` is a cloud. `a` and `b` each show a
trend, but neither explains `y` on its own, and `c`..`f` show nothing.
"""),
    code("""
fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey=True)
for j, ax in enumerate(axes.flat):
    r = np.corrcoef(X[:, j], y)[0, 1]
    ax.scatter(X[:, j], y, s=10, alpha=0.6)
    ax.set_xlabel(names[j])
    ax.set_title(f"r = {r:+.2f}", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("y")
fig.suptitle("y against each raw feature")
fig.tight_layout()
plt.show()
"""),
    md("""
## Fit

One binary round (`schedule=("b",)`) is enough to build `a*b`. We ask for a
single descriptor (`max_dim=1`) and use the differentiable `diff` backend, which
runs in seconds and needs nothing beyond JAX. (`backend="bart"` is the faithful
BART-G.SE screen; see the [backends notebook](03_backends.ipynb).)
"""),
    code("""
model = IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=1200)
model.fit(X, y, feature_names=names)
model.descriptors_
"""),
    md("""
## What happened inside: grow, then prune

`model.history_` records the pool size at each stage. The binary round turned 6
features into 90 candidate descriptors (pairwise sums, differences, products,
quotients and absolute differences, after removing duplicates). The screen kept
a handful, and the l0 search chose one.
"""),
    code(HISTORY_PLOT),
    code("""
for h in model.history_:
    print(h)
plot_history(model.history_)
plt.tight_layout()
plt.show()
"""),
    md("""
## The descriptor collapses the data

Plotted against the discovered descriptor, the cloud collapses onto a line: that
is what it means for `a*b` to be the right variable. The parity plot shows the
fitted model against the data.
"""),
    code("""
pred = model.predict(X)
resid = y - pred
r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
rmse = np.sqrt((resid ** 2).mean())
D = model.transform(X)[:, 0]   # the descriptor evaluated on the data

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(D, y, s=12, alpha=0.7)
axes[0].set_xlabel(f"descriptor: {model.descriptors_[0]}")
axes[0].set_ylabel("y")
axes[0].set_title("y against the discovered descriptor")

lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
axes[1].scatter(y, pred, s=12, alpha=0.7)
axes[1].set_xlabel("true y")
axes[1].set_ylabel("predicted y")
axes[1].set_title(f"Parity: R$^2$ = {r2:.4f}, RMSE = {rmse:.3f}")
axes[1].set_aspect("equal")
fig.tight_layout()
plt.show()
"""),
    md("""
## Read the model

The fitted model renders as LaTeX in a notebook, and can be exported as LaTeX or
HTML strings for papers and web pages. The coefficient comes out close to the
hidden 5.
"""),
    code("""
model   # rich LaTeX rendering (via _repr_latex_)
"""),
    code("""
print("expression:", model.expression())
print("latex     :", model.latex())
"""),
    md("""
## Where next

- [Catalysis case study](02_catalysis.ipynb): the real problem iBART was built
  for: 91 data points, 59 features, and a published answer to compare against.
- [Backends](03_backends.ipynb): what the `bart` and `diff` screens actually
  compute, and when to use which.
- [Custom operators and units](04_custom_operators_units.ipynb): add your own
  operators and use dimensional analysis to prune the search.
- [What iBART can and cannot find](07_nonlinear.ipynb) and
  [Uncertainty](08_uncertainty.ipynb).
"""),
)

# --------------------------------------------------------------------------- #
# 02 - Catalysis case study
# --------------------------------------------------------------------------- #
catalysis = nb(
    md("""
# Catalysis case study: small data, huge feature space

This is the problem iBART was designed for. We have the binding energies of 91
single metal atoms on oxide supports, and 59 primary features describing the
metal and the oxide (formation enthalpies, vacancy energies, electronegativities,
ionization energies, ...). The goal is a **compact, physically interpretable
formula** for the binding energy, not a black box.

That is hard for two reasons:

- the descriptor space is enormous: one round of binary operators on 59 features
  gives thousands of candidates, and a second round gives millions;
- there are only 91 data points, so most of those candidates will fit the noise.

iBART handles this by never building the full space: it screens after every
round, so each round of operators only acts on descriptors that survived.

The published result (Liu et al. 2022; Ye et al. 2024) is that the single best
descriptor is `|Hfo/Oxv|`, the ratio of the oxide formation enthalpy to the
oxygen vacancy energy.
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from jax_ibart import IBart, sparsify
from jax_ibart.datasets import load_metal_support

X, y, names, units = load_metal_support()
print("X:", X.shape, " y:", y.shape)
print("first features:", names[:8], "...")
print("units matrix:", units.shape, "(one row of unit exponents per feature)")
"""),
    md("""
## No single primary feature is enough

The best single primary feature has |r| of about 0.74 with the binding energy.
The answer has to be a combination.
"""),
    code("""
with np.errstate(invalid="ignore"):  # a few primaries are constant -> r undefined
    r = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
r = np.nan_to_num(r)
order = np.argsort(-np.abs(r))
print("top primaries by |r|:", [(names[j], round(abs(r[j]), 3)) for j in order[:5]])

fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(range(len(r)), np.abs(r[order]), color="C0")
ax.set_xticks(range(len(r)), [names[j] for j in order], rotation=90, fontsize=6)
ax.set_ylabel("|r| with binding energy")
ax.set_ylim(0, 1)
ax.set_title("Correlation of each primary feature with the binding energy")
fig.tight_layout()
plt.show()
"""),
    md("""
## Fit with the faithful BART backend

We use the `bart` backend (the BART-G.SE permutation screen of Bleich et al.
2014, run with [`bartz`](https://github.com/Gattocrucco/bartz)) with the default
schedule `("b", "u", "b")`: binary, then unary, then binary operators, with a
screen after each.

Two settings matter:

- `units=units` turns on the dimensional-analysis filter, so physically
  meaningless descriptors (adding an energy to a length) are never built.
- `alpha=0.25` makes the screen permissive. A screen in iBART must above all
  avoid **false negatives**: a descriptor dropped in round 1 can never be
  combined in round 2. Keeping a few extra descriptors is cheap; losing a true one
  is not.

This takes about half a minute on a laptop CPU.
"""),
    code("""
import time

t0 = time.perf_counter()
model = IBart(backend="bart", alpha=0.25, n_perm=25, n_tree=40, n_draw=400,
              n_burn=400, n_keep=15, max_dim=3, seed=0)
model.fit(X, y, names, units=units)
print(f"fit time: {time.perf_counter() - t0:.0f} s")
"""),
    md("""
## Grow, prune, grow, prune

The history shows why this is tractable. The first binary round builds a few
thousand candidates (the unit filter has already removed most of the
dimensionally inconsistent ones), and the screen keeps under a hundred. Each
later round starts from that small pool instead of the full space.
"""),
    code(HISTORY_PLOT),
    code("""
plot_history(model.history_, title="Catalysis: descriptor pool at each stage")
plt.tight_layout()
plt.show()
"""),
    md("""
## The discovered model

BIC chose three descriptors. Two of them are built on `|Hfo/Oxv|`, the published
canonical descriptor.
"""),
    code("""
pred = model.predict(X)
rmse = np.sqrt(((y - pred) ** 2).mean())
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"in-sample RMSE = {rmse:.3f} eV,  R^2 = {r2:.3f}\\n")
print(f"  {model.coef_[0]:+.4g}")
for desc, coef in zip(model.descriptors_, model.coef_[1:]):
    print(f"  {coef:+.4g} * {desc}")
"""),
    md("""
## Comparison with the published result

`model.pool_` is the screened pool the final stage chose from. Re-running the l0
search on it with 1, 2 and 3 descriptors gives the accuracy-versus-complexity
trade-off, which we compare with the published l0 models.
"""),
    code("""
import json

from jax_ibart.datasets import DATA_DIR

oracle = json.loads((DATA_DIR / "oracle_ibart.json").read_text())
published = oracle["lzero_in_sample_rmse"][:3]

P = np.asarray(model.pool_.X, dtype=np.float64)
cand = sparsify.lasso_screen(P, y, n_keep=15)
ours, best = [], {}
for k in (1, 2, 3):
    loc, coef = sparsify.l0_best_subset(P[:, cand], y, max_dim=k, score="bic")
    A = np.column_stack([np.ones(len(y))] + [P[:, cand[i]] for i in loc])
    ours.append(np.sqrt(((y - A @ coef) ** 2).mean()))
    best[k] = A[:, 1:]
    print(f"k={k}: RMSE {ours[-1]:.3f} (published {published[k-1]:.3f})  "
          f"{[str(model.pool_.exprs[cand[i]]) for i in loc]}")
print()
print("published k=1:", oracle["lzero_1"])
print("published k=3:", oracle["lzero_3"])
"""),
    code("""
H, O = X[:, names.index("Hfo")], X[:, names.index("Oxv")]
ratio = np.abs(H / O)
best1 = best[1][:, 0]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ks = [1, 2, 3]
axes[0].plot(ks, published, "o-", label="published (R iBART)")
axes[0].plot(ks, ours, "s-", label="this run")
axes[0].set_xticks(ks)
axes[0].set_xlabel("number of descriptors")
axes[0].set_ylabel("in-sample RMSE (eV)")
axes[0].set_title("Accuracy vs. complexity")
axes[0].legend()

for ax, v, lab in [(axes[1], ratio, "|Hfo/Oxv| (published)"),
                   (axes[2], best1, "best single descriptor (this run)")]:
    rr = np.corrcoef(v, y)[0, 1]
    ax.scatter(v, y, s=12, alpha=0.7)
    ax.set_xlabel(lab)
    ax.set_ylabel("binding energy (eV)")
    ax.set_title(f"r = {rr:+.3f}")
fig.tight_layout()
plt.show()
"""),
    md("""
Compare the middle panel with the bar chart above: no primary feature had |r|
above 0.75, and this one ratio of two of them has |r| of about 0.95. That
collapse is what iBART is for.

Every model this run finds is built on the same `Hfo`/`Oxv` coupling as the
published one, but the published models are more accurate at each size: our
three-descriptor model sits between the published one- and two-descriptor
models. Even our best *single* descriptor (`exp(BVS/Hfo)*|Hfo/Oxv|`) is a little
worse than the published `|Hfo/Oxv|`. The main reason: `|Hfo/Oxv|` survives the first
screen, but the last binary round replaces the pool with products of its
members, so the bare ratio is not in the final pool. The R package keeps a
running union of everything every screen has selected. `IBart(...,
carry_selected=True)` does the same; it keeps `|Hfo/Oxv|` in the pool and
changes the final descriptors. The default stays off because the validated
results in [the oracle cross-check](../oracle-crosscheck.md) were made without it.

All RMSEs here are in-sample. With 91 points and thousands of candidates,
check models out of sample before trusting the third descriptor (see the
[uncertainty notebook](08_uncertainty.ipynb)).
"""),
    md("""
## The fast differentiable backend

The `diff` backend replaces each BART screen with a gated neural network trained
by gradient descent. It is useful for exploring quickly, but on this
small, high-dimensional problem it finds a less accurate model than `bart`.
"""),
    code("""
t0 = time.perf_counter()
fast = IBart(backend="diff", max_dim=3, n_keep=10, seed=0, steps=1000,
             l0=0.03).fit(X, y, names, units=units)
fp = fast.predict(X)
print(f"diff: {time.perf_counter() - t0:.0f} s, "
      f"RMSE {np.sqrt(((y - fp) ** 2).mean()):.3f} eV  (bart above: {rmse:.3f})")
for d_ in fast.descriptors_:
    print("  ", d_)
"""),
    md("""
## References and data provenance

- **Dataset** (`load_metal_support`): the single-atom catalysis data (91
  metal/oxide binding energies, 59 features) shipped with the R `iBART`
  package; the underlying DFT data are from the Senftle group's MIT-licensed
  [`Metal-Oxide-LASSO-lo`](https://github.com/tsenftle/Metal-Oxide-LASSO-lo)
  repository. See [`data/README.md`](https://github.com/jkitchin/jax-ibart/blob/main/src/jax_ibart/data/README.md).
- **Reference result** (`oracle_ibart.json`): the published iBART l0 descriptors
  and RMSEs, from the papers below.
- **Method**: C.-Y. Liu, S. Ye, M. Li, T. P. Senftle, *J. Chem. Phys.* **156**,
  164105 (2022); S. Ye, T. P. Senftle, M. Li, *JASA* **119**:81-94 (2024),
  arXiv:2110.10195.
- Full bibliography: [docs/references.md](../references.md). Detailed comparison
  with the reference implementation: [docs/oracle-crosscheck.md](../oracle-crosscheck.md).
"""),
)

# --------------------------------------------------------------------------- #
# 03 - bart vs diff backends
# --------------------------------------------------------------------------- #
backends = nb(
    md("""
# Faithful vs. differentiable screening

The screen is the heart of iBART: after every round of operators it decides
which descriptors survive. `jax-ibart` has two interchangeable screens:

- **`bart`**: the BART-G.SE permutation test (Bleich et al. 2014), run with
  `bartz`. Fit a Bayesian additive regression tree model, record how often each
  feature is used in a split, then refit many times with `y` permuted to learn
  how often each feature is used *by chance*. Keep features used clearly more
  often than chance. This is what the R iBART package does.
- **`diff`**: a gated neural network. Each feature passes through a stochastic
  hard-concrete gate (Louizos et al. 2018) with an L0 penalty, trained by
  gradient descent. Keep features whose gate ends up open.

We test both on the Friedman function, where `y` depends on `x0` and `x1` only
and six other features are noise.
"""),
    code("""
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax_ibart.selection import BartGSESelector, DiffSelector

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(120, 8))
y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + rng.normal(0, 0.5, 120)
"""),
    code("""
diff = DiffSelector(steps=1500)
t0 = time.perf_counter()
diff_keep = np.asarray(diff.select(X, y, key=jax.random.PRNGKey(0)))
t_diff = time.perf_counter() - t0

bart = BartGSESelector(n_perm=15, n_tree=40, n_draw=300, n_burn=300)
t0 = time.perf_counter()
bart_keep = np.asarray(bart.select(X, y, key=jax.random.PRNGKey(0)))
t_bart = time.perf_counter() - t0

print(f"diff keeps {np.where(diff_keep)[0].tolist()}  ({t_diff:.1f} s)")
print(f"bart keeps {np.where(bart_keep)[0].tolist()}  ({t_bart:.1f} s)")
print("truth      [0, 1]")
"""),
    md("""
## What each screen sees

Both selectors expose their evidence after `select`: `DiffSelector.importance_`
holds the gate-open probabilities, and `BartGSESelector.diagnostics_` holds the
observed split proportions (`q_obs`), the permutation null draws and the
per-feature threshold.
"""),
    code("""
feat = [f"x{j}" for j in range(X.shape[1])]
xs = np.arange(len(feat))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

imp = np.asarray(diff.importance_)
axes[0].bar(xs, imp, color=["C1" if k else "C0" for k in diff_keep])
axes[0].axhline(diff.threshold, color="k", ls="--", lw=1, label="threshold")
axes[0].set_xticks(xs, feat)
axes[0].set_ylabel("gate-open probability")
axes[0].set_title("diff: learned gates")
axes[0].legend()

dg = bart.diagnostics_
null = np.asarray(dg["null"])
axes[1].boxplot([null[:, j] for j in xs], positions=xs, widths=0.5,
                showfliers=False)
axes[1].scatter(xs, dg["q_obs"], color=["C1" if k else "C0" for k in bart_keep],
                zorder=3, label="observed")
axes[1].scatter(xs, dg["threshold"], marker="_", s=300, color="k",
                zorder=3, label="threshold")
axes[1].set_xticks(xs, feat)
axes[1].set_ylabel("split proportion")
axes[1].set_title("bart: observed use vs. permutation null")
axes[1].legend()
fig.tight_layout()
plt.show()
"""),
    md("""
Orange marks a kept feature. The gate model drives the noise gates to zero. BART
uses `x0` and `x1` in about twice as many splits as the permutation null would
explain; the other features sit inside their null distributions.

## When to use which

| | `bart` (faithful) | `diff` (differentiable) |
|---|---|---|
| Fidelity | high; matches R iBART | lower on hard, high-dimensional screens |
| Speed | seconds to minutes (CPU) | seconds |
| Differentiable | no (discrete MCMC) | yes, end to end |
| Use for | final and validated results | fast exploration, gradient-based work |

On the real catalysis screen (see the [catalysis notebook](02_catalysis.ipynb)
and [the benchmark](../benchmark.md)), `diff` is much faster but finds a less
accurate model.
"""),
)

# --------------------------------------------------------------------------- #
# 04 - Custom operators and units
# --------------------------------------------------------------------------- #
custom = nb(
    md("""
# Custom operators and units

iBART can only find formulas its operators can build. This notebook shows how
operators and units work, how much the unit filter prunes, and how to add an
operator for your own domain.

An operator bundles four things that move in lockstep:

- a numeric function on JAX arrays,
- a SymPy function, so the descriptor stays symbolic,
- a validity predicate (for example `log` needs positive inputs),
- a unit rule (multiplying adds unit exponents; adding needs equal units).
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax_ibart import IBart, engineer
from jax_ibart.operators import Operator, BINARY, UNARY, default_unary, default_binary

print("default unary :", [op.name for op in default_unary()])
print("default binary:", [op.name for op in default_binary()])
print("also available:", sorted(set(UNARY) - {op.name for op in default_unary()}))
"""),
    md("""
## Units: a small example

Units are integer exponent vectors over base dimensions. Here `x` is a length and
`z` is a time, so `x + z` is rejected while `x*z` and `x/z` are allowed.
"""),
    code("""
Xs = np.random.default_rng(0).uniform(1, 2, size=(12, 2))
units = np.array([[1, 0],   # x: [length, time] exponents
                  [0, 1]])  # z
pool = engineer.from_primary(Xs, ["x", "z"], units=units)

add = engineer.apply_binary(pool, [BINARY["add"]])
mul = engineer.apply_binary(pool, [BINARY["mul"]])
print("x + z generated:", add.n_descriptors, "(rejected: mismatched units)")
print("x * z generated:", mul.n_descriptors, "-> unit", mul.units[0].tolist())
"""),
    md("""
## How much the unit filter prunes on real data

On the 59 catalysis features, one round of binary operators builds the
following number of descriptors without units and with them:
"""),
    code("""
from jax_ibart.datasets import load_metal_support

Xc, yc, cnames, cunits = load_metal_support()
n_plain = engineer.apply_binary(engineer.from_primary(Xc, cnames), default_binary()).n_descriptors
n_units = engineer.apply_binary(engineer.from_primary(Xc, cnames, units=cunits),
                                default_binary()).n_descriptors
print(f"without units: {n_plain:,}   with units: {n_units:,}")

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(["no units", "unit filter"], [n_plain, n_units], color=["C0", "C1"])
for i, v in enumerate([n_plain, n_units]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_ylabel("descriptors after one binary round")
fig.tight_layout()
plt.show()
"""),
    md("""
The filter removes about two thirds of the candidates before any screening, and
every one it removes is physically meaningless. That makes the screen's job
easier and the result more trustworthy.

One caveat: as in the R iBART package, `sqrt`, `log`, `exp`, `sin` and `cos`
return a unitless result. So the filter mainly constrains addition and
subtraction, and a descriptor like `sqrt(a) - sqrt(M)` can pass even when `a`
and `M` have different units.
"""),
    md("""
## Adding your own operator: the Stefan-Boltzmann law

Radiated power goes as `T^4`. The default unary operators include `x^2` but not
`x^4`, so in one unary round iBART cannot build `T^4`. We generate data with two
distractor features and fit with the default operators first.
"""),
    code("""
rng = np.random.default_rng(0)
n = 80
T = rng.uniform(300, 1500, n)     # temperature, K
P = rng.uniform(1, 10, n)         # pressure (irrelevant)
d = rng.uniform(0, 1, n)          # size (irrelevant)
j = 5.670e-8 * T**4 * (1 + rng.normal(0, 0.01, n))   # W/m^2, 1% noise
Xsb = np.column_stack([T, P, d])
sb_names = ["T", "P", "d"]

def r2(m):
    p = m.predict(Xsb)
    return 1 - ((j - p) ** 2).sum() / ((j - j.mean()) ** 2).sum()

default = IBart(backend="diff", schedule=("u",), max_dim=1, seed=0,
                steps=1500).fit(Xsb, j, sb_names)
print("default operators:", default.descriptors_, f"R^2 = {r2(default):.5f}")
"""),
    md("""
Now define a `pow4` operator and pass it in with `unary_ops`:
"""),
    code("""
pow4 = Operator(
    name="pow4", arity=1,
    fn=lambda x: x ** 4,                          # numeric (JAX)
    sym=lambda e: e ** 4,                         # symbolic (SymPy)
    valid=lambda x: jnp.ones(x.shape[-1], bool),  # valid everywhere
    unit=lambda u: 4 * u,                         # K -> K^4
)
custom = IBart(backend="diff", schedule=("u",), max_dim=1, seed=0, steps=1500,
               unary_ops=default_unary() + [pow4]).fit(Xsb, j, sb_names)
print("with pow4        :", custom.descriptors_, f"R^2 = {r2(custom):.5f}")
print(f"fitted constant  : {custom.coef_[1]:.4g}  (Stefan-Boltzmann 5.670e-8)")
"""),
    code("""
Tg = np.linspace(300, 1500, 200)
Xg = np.column_stack([Tg, np.full_like(Tg, 5), np.full_like(Tg, 0.5)])
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.scatter(T, j, s=10, color="k", alpha=0.5, label="data")
ax.plot(Tg, default.predict(Xg), label=f"default ops: {default.descriptors_[0]}")
ax.plot(Tg, custom.predict(Xg), "--", label=f"with pow4: {custom.descriptors_[0]}")
ax.set_xlabel("T (K)")
ax.set_ylabel("radiated power (W/m$^2$)")
ax.legend()
fig.tight_layout()
plt.show()
"""),
    md("""
With the default operators the best iBART can do is a quadratic, which misses
at both ends. With `pow4` it finds `T^4` and recovers the Stefan-Boltzmann
constant. `binary_ops` works the same way for two-argument operators.

The lesson generalizes: put the functional forms you expect from physics into
the operator set, and let iBART decide which variables they act on.
"""),
)

# --------------------------------------------------------------------------- #
# 05 - Dynamical systems (SINDy / DAE foundation)
# --------------------------------------------------------------------------- #
dynamics_nb = nb(
    md("""
# Discovering dynamical systems (SINDy / DAEs)

`jax_ibart.dynamics` applies the same sparse-regression idea to governing
equations (SINDy; Brunton, Proctor & Kutz 2016). For an ODE it regresses each
state's time derivative on a library of candidate terms and keeps only a few.

Here the library is a fixed polynomial basis (degree up to 2), not iBART's
grow-and-prune descriptor search. The package supplies the symbolic bookkeeping,
the l0 best-subset fit and an optional BART screen.

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
    md(
        "Derivatives are estimated from the trajectory by finite differences, then each equation is recovered by sparse regression:"
    ),
    code("""
system = dynamics.discover_ode(X, t, ["x", "y", "z"], degree=2, max_terms=3)
print(system)
print("\\ntrue: dx/dt = 10(y - x);  dy/dt = 28x - y - xz;  dz/dt = xy - 2.667 z")
"""),
    md("""
The Lorenz equations come back with the right terms and coefficients, with
R^2 = 1.0, directly from the trajectory.
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
parity plots below compare the finite-difference derivative (the target) with
the fitted right-hand side, one equation at a time.
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

The strongest test is to integrate the *discovered* equations from the same
initial condition and compare with the true trajectory.
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
The discovered system tracks the true trajectory for several oscillations. The
two eventually separate because Lorenz is chaotic: tiny differences in the
coefficients grow exponentially. That is a property of the system, not an error
in the recovered equations.
"""),
    md("""
## Toward DAEs: implicit algebraic constraints

A differential-algebraic system also has algebraic constraints `0 = g(x, z)`.
Finding those is an *implicit* (null-space) problem, not a regression on a
target, so it needs different machinery. `implicit_constraints` looks for a
combination of library terms that is zero on every sample. Here it recovers
the constraint satisfied by points on a unit circle.
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
It recovers `x^2 + y^2 - 1 = 0` exactly. A production DAE module would add
sparse null-space selection and a weak (integral) formulation (Mangan et al.
2016; Kaheman et al. 2020; Messenger & Bortz 2021), and would separate
differential from algebraic relations. The descriptor generation, unit filter,
screening and uncertainty tools all carry over.
"""),
)

# --------------------------------------------------------------------------- #
# 06 - Reaction kinetics: discover the mole-balance RHS
# --------------------------------------------------------------------------- #
reaction = nb(
    md("""
# Reaction kinetics: discovering rate laws from batch data

Given noisy concentration-versus-time data from a batch reactor, can we recover
the rate laws? This is SINDy applied to chemical kinetics. The main lesson of
this notebook is that **the derivative estimate matters as much as the
regression**.

The dataset (`scripts/make_reaction_data.py`) is a liquid-phase batch reactor
run from six initial conditions, with two reactions:

- `A + B -> C`, rate `r1 = k1 C_A C_B` (bimolecular, k1 = 0.5)
- `C -> D`, rate `r2 = k2 C_C` (first order, k2 = 0.2)

so the true mole balances are

- `dCA/dt = dCB/dt = -k1 CA CB`
- `dCC/dt = k1 CA CB - k2 CC`
- `dCD/dt = k2 CC`
"""),
    code("""
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from jax_ibart.datasets import DATA_DIR, load_reaction
from jax_ibart.dynamics import discover_ode

df = pd.read_csv(DATA_DIR / "reaction_kinetics.csv").sort_values(["batch", "t"])
species = ["CA", "CB", "CC", "CD"]
batches = sorted(df.batch.unique())
print(df.shape, "rows;", len(batches), "batches of", len(df) // len(batches), "time points")
df.head()
"""),
    code("""
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey=True)
for ax, b in zip(axes.flat, batches):
    g = df[df.batch == b]
    for s in species:
        ax.plot(g.t, g[s], "o", ms=2.5, label=s)
    ax.set_title(f"batch {b}: CA0={g.CA.iloc[0]:.1f}, CB0={g.CB.iloc[0]:.1f}", fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("t")
for ax in axes[:, 0]:
    ax.set_ylabel("concentration")
axes[0, 0].legend(ncol=2, fontsize=8)
fig.suptitle("Noisy batch-reactor data")
fig.tight_layout()
plt.show()
"""),
    md("""
## Attempt 1: finite-difference derivatives

`load_reaction` returns the concentrations and central finite-difference
derivatives computed within each batch. We fit each mole balance with at most
two library terms, and use the BART screen (`screen="bart"`) to prune the
library first.
"""),
    code("""
C, dCdt_fd, _, batch = load_reaction()
fd = discover_ode(C, t=None, feature_names=species, dXdt=dCdt_fd,
                  degree=2, max_terms=2, screen="bart", seed=0)
print(fd)
"""),
    md("""
`dCA/dt` and `dCB/dt` come out right (about `-0.48 CA CB`), and `dCD/dt` has
the right term (`0.19 CC`) but a poor R^2. `dCC/dt` is wrong: it has the
`CA CB` term but not `-0.2 CC`, with a constant offset in its place.

The plot below shows why. Finite differences amplify the measurement noise: with
a time step of 0.1 and noise of about 0.02, the derivative noise is about 0.13,
as large as the `0.2 CC` term we are trying to see.
"""),
    md("""
## Attempt 2: smoothed derivatives

A Savitzky-Golay filter fits a local cubic polynomial in a sliding window and
differentiates it. That is much less noisy than a finite difference.
"""),
    code("""
from scipy.signal import savgol_filter

parts = []
for b in batches:
    g = df[df.batch == b]
    dt = g.t.iloc[1] - g.t.iloc[0]
    parts.append(savgol_filter(g[species].to_numpy(), 11, 3, deriv=1, delta=dt,
                               axis=0, mode="interp"))
dCdt_sg = np.vstack(parts)

# true derivatives for batch 0, from the known model, for reference
def true_rhs(t, c, k1=0.5, k2=0.2):
    r1, r2 = k1 * c[0] * c[1], k2 * c[2]
    return [-r1, -r1, r1 - r2, r2]

g0 = df[df.batch == batches[0]]
t0 = g0.t.to_numpy()
ref = solve_ivp(true_rhs, [t0[0], t0[-1]], [2.0, 2.0, 0.0, 0.0], t_eval=t0, rtol=1e-8)
true_d = np.array([true_rhs(0, c) for c in ref.y.T])

m0 = batch == batches[0]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, i in zip(axes, [2, 3]):
    ax.plot(t0, dCdt_fd[m0, i], "o", ms=3, alpha=0.6, label="finite difference")
    ax.plot(t0, dCdt_sg[m0, i], "s", ms=3, alpha=0.8, label="Savitzky-Golay")
    ax.plot(t0, true_d[:, i], "k-", lw=1.5, label="true")
    ax.set_xlabel("t")
    ax.set_ylabel(f"d{species[i]}/dt")
    ax.set_title(f"d{species[i]}/dt, batch {batches[0]}")
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()
"""),
    code("""
sg = discover_ode(C, t=None, feature_names=species, dXdt=dCdt_sg,
                  degree=2, max_terms=2, seed=0)
print(sg)
print("\\ntrue: dCA/dt = -0.5 CA CB ; dCB/dt = -0.5 CA CB ; "
      "dCC/dt = 0.5 CA CB - 0.2 CC ; dCD/dt = 0.2 CC")
"""),
    md("""
With smoothed derivatives all four rate laws have the right terms, and the rate
constants are within about 10% of the true values. (The small extra terms in
`dCA/dt` and `dCB/dt` have coefficients below 0.02.)

We left out the BART screen this time. Here is what happens with it:
"""),
    code("""
sg_bart = discover_ode(C, t=None, feature_names=species, dXdt=dCdt_sg,
                       degree=2, max_terms=2, screen="bart", seed=0)
print(sg_bart)
"""),
    md("""
The screen drops `CC` from the `dCC/dt` library: `-0.2 CC` is a small
contribution next to `0.5 CA CB`, and a term the screen misses cannot be
recovered by the l0 fit afterwards. For a library this small (15 terms) the
screen is unnecessary; it earns its keep when the library is large. Either way,
check the result against the data, as below.
"""),
    md("""
## Do the discovered rate laws reproduce the data?

The real test is to integrate the discovered ODEs from each batch's first
measurement and compare the trajectories with the data.
"""),
    code("""
syms = sp.symbols("CA CB CC CD")
funcs = [sp.lambdify(syms, e, "numpy") for e in sg.rhs]

def learned(t, c):
    return [float(f(*c)) for f in funcs]

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey=True)
for ax, b in zip(axes.flat, batches):
    g = df[df.batch == b]
    tb = g.t.to_numpy()
    sim = solve_ivp(learned, [tb[0], tb[-1]], g[species].iloc[0].to_numpy(),
                    t_eval=tb, rtol=1e-8)
    for k, s in enumerate(species):
        ax.plot(tb, g[s], "o", ms=2.5, color=f"C{k}", alpha=0.6)
        ax.plot(tb, sim.y[k], "-", color=f"C{k}", lw=1.5, label=s)
    ax.set_title(f"batch {b}", fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("t")
for ax in axes[:, 0]:
    ax.set_ylabel("concentration")
axes[0, 0].legend(ncol=2, fontsize=8)
fig.suptitle("Data (points) vs. integrated discovered rate laws (lines)")
fig.tight_layout()
plt.show()
"""),
    md("""
The discovered model reproduces all six batches, including the rise and fall of
the intermediate `C`.

Further steps for real kinetic data:

- a weak (integral) formulation avoids differentiating noisy data at all
  (Messenger & Bortz 2021, see [references](../references.md));
- passing `units=...` restricts the library to terms with consistent units;
- the [uncertainty tools](08_uncertainty.ipynb) show how stable the selected
  terms are under resampling.
"""),
)

# --------------------------------------------------------------------------- #
# 07 - What iBART can and cannot find (nonlinearity guide)
# --------------------------------------------------------------------------- #
nonlinear = nb(
    md("""
# What iBART can (and can't) find

iBART discovers **nonlinear equations**. It builds nonlinear descriptors by
composing operators (`square, sqrt, log, exp, sin, cos, abs, *, /, |a-b|`) on
the inputs. The final model is *linear in the coefficients on a nonlinear basis*
(like SINDy or polynomial regression): the discovered equation is nonlinear, but
the fit is convex.

This notebook shows what it recovers, its one real limitation, and two
workarounds.
"""),
    md("## 1. Nonlinear expressions it recovers exactly"),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from jax_ibart import IBart

rng = np.random.default_rng(0)
n = 200
x1, x2, x3 = (rng.uniform(0.1, 1, n) for _ in range(3))
X = np.column_stack([x1, x2, x3]); names = ["x1", "x2", "x3"]

def fit_and_score(y, schedule):
    m = IBart(backend="diff", schedule=schedule, max_dim=1, seed=0, steps=1500).fit(X, y, names)
    pred = m.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return m, pred, r2

cases = [
    ("sin(pi*x1*x2)", np.sin(np.pi * x1 * x2), ("b", "u")),
    ("(exp(x1)-exp(x2))^2", (np.exp(x1) - np.exp(x2)) ** 2, ("u", "b", "u")),
]
results = []
for label, y, sched in cases:
    m, pred, r2 = fit_and_score(y, sched)
    results.append((label, y, pred, m, r2))
    print(f"{label:>22}  ->  {m.descriptors_}   R2={r2:.4f}")
"""),
    md("Parity plots: the recovered nonlinear models land on the diagonal."),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (label, y, pred, m, r2) in zip(axes, results):
    ax.scatter(y, pred, s=14, alpha=0.6)
    lo, hi = float(y.min()), float(y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"y = {label}\\nR$^2$={r2:.4f}")
    ax.set_xlabel("true y"); ax.set_ylabel("predicted"); ax.set_aspect("equal")
fig.tight_layout(); plt.show()
"""),
    md("""
Both are recovered exactly (R^2 = 1.0). Transcendental, rational, product and
composed nonlinearities are all in scope, as long as the operators can build them.
"""),
    md("""
## 2. The one real limitation: free parameters *inside* a nonlinearity

iBART fits **linear coefficients** on an operator-generated library. It cannot
fit a *continuous parameter inside* a term unless that exact term is generated.
A power law with a non-integer exponent is the classic case: there is no
arbitrary-power operator, so `x^2.3` can only be *approximated* (here by `x^2`).
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
    md(
        "Plotting the fit against `x` shows the approximation drifting from the true curve:"
    ),
    code("""
order = np.argsort(xp)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(xp[order], y[order], lw=2, label="true  x$^{2.3}$")
axes[0].plot(xp[order], pr[order], "--", lw=2, label=f"iBART  {m.descriptors_[0]}")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].legend()
axes[0].set_title("best the operator library can do")
axes[1].plot(xp[order], (pr - y)[order], color="C3")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("x"); axes[1].set_ylabel("residual (fit - true)")
axes[1].set_title("structured residual = missing exponent")
fig.tight_layout(); plt.show()
"""),
    md("""
The same goes for **Arrhenius** `k = A exp(-E/RT)`, with a fitted activation
energy `E` inside the exponential, and **Michaelis-Menten** `v = Vmax S/(Km+S)`,
with a fitted `Km` in the denominator. Both are nonlinear *in the parameters*.
There are two standard fixes.
"""),
    md("""
### Workaround A: linearize (Arrhenius)

Taking logs gives `ln k = ln A - (E/R)(1/T)`, which is linear in the descriptor
`1/T`. So we give iBART `ln k` as the target, add the `inv` operator (it is not
in the default unary set), and include two irrelevant features to make it a real
search.
"""),
    code("""
from jax_ibart.operators import UNARY, default_unary

R = 8.314
T = rng.uniform(300, 600, n)          # K
P = rng.uniform(1, 10, n)             # irrelevant
xA = rng.uniform(0, 1, n)             # irrelevant
k = 1e3 * np.exp(-50000 / (R * T)) * np.exp(rng.normal(0, 0.02, n))   # 2% noise

arr = IBart(backend="diff", schedule=("u",), max_dim=1, seed=0, steps=1500,
            unary_ops=default_unary() + [UNARY["inv"]])
arr.fit(np.column_stack([T, P, xA]), np.log(k), ["T", "P", "x_A"])
print("descriptor:", arr.descriptors_)
E_hat, A_hat = -arr.coef_[1] * R, np.exp(arr.coef_[0])
print(f"recovered E = {E_hat:,.0f} J/mol   (true 50,000)")
print(f"recovered A = {A_hat:,.0f}          (true 1,000)")
"""),
    code("""
o = np.argsort(T)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].scatter(T, k, s=10, alpha=0.6)
axes[0].set_xlabel("T (K)"); axes[0].set_ylabel("k")
axes[0].set_title("k vs T: nonlinear in E")
axes[1].scatter(1 / T, np.log(k), s=10, alpha=0.6, label="data")
axes[1].plot(1 / T[o], arr.coef_[0] + arr.coef_[1] / T[o], "C3", lw=2,
             label=f"iBART: {arr.descriptors_[0]}")
axes[1].set_xlabel("1 / T"); axes[1].set_ylabel("ln k")
axes[1].set_title(f"Arrhenius plot: E = {E_hat:,.0f} J/mol")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("### Workaround B: implicit, null-space form (Michaelis-Menten)"),
    md("""
**What "null space" means.** Usual regression fits `y = f(x)`: one column is the
target and we predict it from the others. A constraint like `x^2 + y^2 = 1` has
no target. Instead, some weighted sum of candidate terms is **always zero**.

So build a table whose columns are candidate terms (here `S, v, S^2, v^2, S*v`
and a column of ones). We want weights `w` such that

    w1*S + w2*v + ... + w6*1  =  0   for every row.

That weight vector is a **null-space vector** of the table: multiplying the table
by it gives all zeros. Ordinary regression cannot find it, because there is no
`y`. A singular value decomposition (SVD) can: the direction along which the data
never vary is exactly the constraint. The smallest singular value measures how
close to zero it is (0 means an exact law).

For Michaelis-Menten, rearranging `v = Vmax*S/(Km+S)` gives
`Vmax*S - Km*v - S*v = 0`, a sum of terms that is always zero. That is what we
recover:
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
    md("Reconstructing `v(S)` from the recovered coefficients reproduces the data:"),
    code("""
# constraint:  cS*S + cv*v + cSv*S*v + ... = 0  ->  solve for v(S)
# coef vector aligns with res['terms']; pull the S, v, S*v weights
terms = res["terms"]; w = res["coef"]
def wof(name): return w[terms.index(name)]
cS, cv, cSv = wof("S"), wof("v"), wof("S*v")
# cS*S + cv*v + cSv*S*v = 0  ->  v = -cS*S / (cv + cSv*S)
Sg = np.linspace(0.1, 10, 100)
v_fit = -cS * Sg / (cv + cSv * Sg)
Vmax_hat, Km_hat = -cS / cSv, cv / cSv

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(S, v, s=14, alpha=0.5, label="data")
ax.plot(Sg, v_fit, "C3", lw=2, label=f"recovered (Vmax={Vmax_hat:.2f}, Km={Km_hat:.2f})")
ax.set_xlabel("S"); ax.set_ylabel("v"); ax.set_title("Michaelis-Menten via null-space form")
ax.legend(); fig.tight_layout(); plt.show()
"""),
    md("""
The implicit (null-space) formulation recovers the Michaelis-Menten law
**exactly** as an algebraic constraint, and `Vmax` and `Km` can be read off the
coefficients (here Vmax = 5, Km = 2). This is the same machinery that underlies
the DAE foundation (`implicit_constraints`) and SINDy-PI (see
[references](../references.md)).

**Summary:** nonlinear *expressions* are what iBART is for. Free parameters
*inside* a nonlinearity need an operator that builds the term, a linearizing
transform, or the implicit null-space form.
"""),
)

# --------------------------------------------------------------------------- #
# 08 - Uncertainty quantification
# --------------------------------------------------------------------------- #
uq = nb(
    md("""
# Uncertainty quantification

iBART is a *selection* method, so its biggest uncertainty is **which descriptors
get chosen**, not just the scatter around a fixed model. `jax_ibart.uncertainty`
has tools for both. We use a noisy synthetic problem in which only `a*b` drives
the response (`c` and `d` are irrelevant).
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt
from jax_ibart import IBart, uncertainty

rng = np.random.default_rng(0)
n = 240
a, b, c, d = (rng.uniform(0.5, 2.0, n) for _ in range(4))
X = np.column_stack([a, b, c, d]); names = ["a", "b", "c", "d"]
y = 5.0 * (a * b) + rng.normal(0, 0.4, n)        # noisy

# a factory makes a fresh, identically configured estimator on demand
factory = lambda: IBart(backend="diff", schedule=("b",), max_dim=1, seed=0, steps=900)
"""),
    md("""
## 1. Conformal prediction intervals

Split conformal prediction wraps the **entire fit**, selection included, and
calibrates on held-out data. That gives a finite-sample coverage guarantee even
after selection. We check the *realized* coverage on a fresh test set.
"""),
    code("""
idx = rng.permutation(n); te, tr = idx[:80], idx[80:]
conf = uncertainty.conformal_intervals(factory, X[tr], y[tr], names,
                                       alpha=0.10, cal_frac=0.4, seed=0)
cov = uncertainty.empirical_coverage(conf, X[te], y[te])
se = np.sqrt(0.9 * 0.1 / len(te))
print(f"target coverage: 0.90   realized: {cov:.2f}   half-width: {conf.half_width:.3f}")
print(f"sampling noise on a coverage estimate from {len(te)} test points: +/- {se:.2f} (1 s.e.)")
"""),
    code("""
lo, point, hi = conf.interval(X[te])
order = np.argsort(point)
inside = (y[te] >= lo) & (y[te] <= hi)
fig, ax = plt.subplots(figsize=(7, 3.6))
xs = np.arange(len(order))
ax.fill_between(xs, lo[order], hi[order], alpha=0.25, label="90% conformal band")
ax.plot(xs, point[order], lw=1, label="prediction")
ax.scatter(xs[inside[order]], y[te][order][inside[order]], s=12, color="k",
           label="true (inside)", zorder=3)
ax.scatter(xs[~inside[order]], y[te][order][~inside[order]], s=18, color="C3",
           label="true (outside)", zorder=3)
ax.set_xlabel("test point (sorted by prediction)"); ax.set_ylabel("y")
ax.legend(fontsize=8); fig.tight_layout(); plt.show()
"""),
    md("""
The realized coverage is within sampling noise of the 90% target: with 80 test
points, one standard error on a coverage estimate is about 0.03.

## 2. Selection stability

The most informative check for a selection method is how often each feature and
each descriptor is chosen across bootstrap resamples. The true drivers `a` and
`b` should dominate, and `c` and `d` should rarely or never appear.
"""),
    code("""
stab = uncertainty.selection_stability(factory, X, y, names, n_boot=20, seed=0)
ff = {nm: stab["feature_freq"].get(nm, 0.0) for nm in names}

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(list(ff), list(ff.values()))
ax.set_ylabel("inclusion frequency"); ax.set_ylim(0, 1.05)
ax.set_title(f"Feature selection stability ({stab['n_boot']} bootstrap fits)")
fig.tight_layout(); plt.show()
print("features   :", ff)
print("descriptors:", stab["descriptor_freq"])
"""),
    md("""
Here the answer is clear-cut: every bootstrap fit chose `a*b`. On a real
problem, a descriptor chosen in only a fraction of resamples is a warning that
the data do not pin it down.

## 3. Out-of-sample RMSE distribution

Following the papers' protocol, we report the RMSE over many random train/test
splits as a distribution rather than a single number. The noise level is 0.4, so
that is the best RMSE any model can achieve here.
"""),
    code("""
cv = uncertainty.repeated_split_rmse(factory, X, y, names, n_splits=20, seed=0)
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(cv["rmse"], bins=12)
ax.axvline(cv["mean"], color="k", ls="--", label=f"mean {cv['mean']:.2f}")
ax.axvline(0.4, color="C3", ls=":", label="noise level 0.4")
ax.set_xlabel("out-of-sample RMSE"); ax.set_ylabel("count"); ax.legend()
fig.tight_layout(); plt.show()
print(f"RMSE = {cv['mean']:.3f} +/- {cv['std']:.3f}  (90% CI [{cv['q05']:.2f}, {cv['q95']:.2f}])")
"""),
    md("""
## 4. Coefficient intervals (classical, optimistic)

Classical OLS confidence intervals on the final model are fast but *optimistic*:
they condition on the selected descriptors and ignore the selection step. Use the
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
**Takeaway:** report the conformal band for honest prediction intervals, and the
selection-stability frequencies to show which descriptors are trustworthy.
Together they capture the uncertainty a single point estimate hides. See also
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
