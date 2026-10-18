# R-iBART oracle cross-check

The strongest validation: compare this port against the reference R `iBART`
implementation on the identical catalysis dataset.

## The oracle

A full R `iBART` install needs Java + rJava + bartMachine. Fortunately the R
package **ships the authors' own precomputed result**,
`data/iBART_real_data.rda` (the object the `real_data` vignette loads). That is
the authoritative reference output, produced by the package authors on this exact
data. We extracted it to `src/jax_ibart/data/oracle_ibart.json` (no Java needed to read an
`.rda`).

### What the oracle selected

The interpretable l0 best-k models (directly comparable to our `max_dim`):

| k | descriptors                                                       | in-sample RMSE (eV) |
|---|-------------------------------------------------------------------|---------------------|
| 1 | `abs(Hfo/Oxv)`                                                     | 0.929 |
| 2 | `(Hfo*s_EA)/(m_N_val/CMS)`, `abs(Hfo/Oxv)`                         | 0.664 |
| 3 | `s_EA*Hf`, `abs(Hfo/Oxv)`, `abs((m_n13/m_N_val)/Oxv)`             | 0.624 |
| 5 | `m_n13/s_phi`, `Hf/Oxv`, ..., `abs(Hfo/Oxv)`                       | 0.533 |

The full LASSO model (18 descriptors) has in-sample RMSE 0.466. The
**canonical single descriptor is `abs(Hfo/Oxv)`** = |dH_ox,fm / O_vac|, a ratio
of oxide formation enthalpy to oxygen vacancy energy.

The oracle's per-iteration generated/selected sizes are `gen=[59,333,128,830]`,
`sel=[12,9,2,18]`.

## What the cross-check surfaced

Comparing the port head-to-head exposed three concrete things, which is exactly
the point of an oracle check:

1. **R iBART screens the primary features first.** `gen=[59,333,...]` means it
   selects ~12 of 59 primaries *before* the first binary expansion (333 ~ binary
   on 12 features), whereas our pipeline expanded all 59 first. We added an
   opt-in leading screen (`IBart(screen_primaries=True)`).
2. **A units-propagation bug:** the per-iteration pool reduction dropped the unit
   vectors, silently disabling the dimensional-analysis filter after the first
   selection. Fixed (units now carried through every screening step).
3. **The leading screen is budget-sensitive.** At our small CPU MCMC budget it
   committed a PAN false negative (dropped `Oxv` at the primary stage), so it is
   off by default and recommended only with a larger budget.

## The honest verdict

With the default (over-conservative) screening threshold, the port recovered the
right physical drivers and the right *order* of accuracy, but **not** the exact
`abs(Hfo/Oxv)` descriptor. The next section shows why and how that gap closes:

- **Recovers the right physical drivers** (`Hfo`, and `Oxv` when not over-
  screened).
- **With a permissive screening threshold it selects descriptors built on
  `|Hfo/Oxv|`** (see "Narrowing the gap" below).

This is expected and not alarming: BART-G.SE screening is stochastic and the
default screening threshold was too conservative (see below).

## Narrowing the gap: it was the threshold, not the budget

We tested the obvious hypothesis first, that a larger MCMC budget would close the
gap. **It did not.** With a large budget (`n_perm=50, n_tree=50, 1000 draws`) and
the R-iBART-style leading primary screen, the pipeline actually did *worse*
(k=3 RMSE 1.06): the Global-SE threshold pruned to just 8 primaries -> 3 final
descriptors, so `abs(Hfo/Oxv)` was screened out before l0 ever saw it.

The diagnosis: the default `alpha=0.05` Global-SE threshold **over-prunes**,
committing exactly the PAN false negatives the algorithm forbids. The reference
iBART keeps far more descriptors per round (`sel=[12,9,2,18]`) than our
conservative default did (`[8,7,3,3]`). The PAN criterion explicitly says
intermediate screening should *tolerate false positives and avoid false
negatives*, i.e. be permissive.

Making the screen permissive (`alpha=0.25`, no leading screen, and a modest
budget `n_perm=25, n_tree=40, n_draw=400, n_burn=400, n_keep=15`) narrows the
gap:

| k | PORT descriptor(s)                                   | PORT RMSE | ORACLE RMSE |
|---|------------------------------------------------------|-----------|-------------|
| 1 | `exp(BVS/Hfo) * abs(Hfo/Oxv)`                         | 1.012     | 0.929 |
| 2 | `exp(BVS/Hfo) * abs(Hfo/Oxv)`, ...                    | 0.930     | 0.664 |
| 3 | ... `sin(pi*Oxv*s_phi)*abs(Hfo/Oxv)`, ...            | 0.822     | 0.624 |

The selected models are now built on `|Hfo/Oxv|` as a factor, and k=3 RMSE
improves from 1.06 to 0.82, though it is still above the oracle's at every k.
The bare ratio `abs(Hfo/Oxv)` is generated in round 1 but does not survive to
the final pool, because each binary round replaces its inputs; set
`carry_selected=True` to keep every round's survivors, as R iBART does. The run
takes about 25 s on a laptop CPU (permissiveness is not a budget cost).

On clean, well-separated problems alpha is irrelevant (even `alpha=0.25` keeps
only the true drivers with zero false positives); it matters only on collinear,
high-dimensional screens like this one. We therefore changed the default to
`alpha=0.10` (more PAN-aligned) and recommend `alpha~0.2-0.25` for the most
faithful reproduction. **Most of the gap was screening permissiveness in the
default, not a budget or hardware limit.**

## bartz version

`jax-ibart` pins `bartz>=0.7,<0.11`. bartz 0.11 changed its input layout, and
0.11-0.13 change the BART-G.SE screen enough that the results on this page no
longer reproduce. Re-validate against this page before lifting the pin.

## Reproduce

Run the [catalysis notebook](examples/02_catalysis.ipynb), which compares
with the oracle's l0 models, or `pytest -m slow`
(`test_permissive_screen_recovers_hfo_oxv_coupling`). The oracle facts are in
`src/jax_ibart/data/oracle_ibart.json`:

```python
import json
from jax_ibart.datasets import DATA_DIR

print(json.loads((DATA_DIR / "oracle_ibart.json").read_text())["lzero_1"])
```
