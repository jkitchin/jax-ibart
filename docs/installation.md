# Installation

`jax-ibart` is pure Python and needs no JVM. It is not on PyPI yet; install it
from GitHub:

```bash
pip install "jax-ibart[bart] @ git+https://github.com/jkitchin/jax-ibart"   # recommended
pip install "jax-ibart[diff] @ git+https://github.com/jkitchin/jax-ibart"   # differentiable backend only
pip install "jax-ibart[bart,diff] @ git+https://github.com/jkitchin/jax-ibart"
```

The default `IBart(backend="bart")` needs the `bart` extra; `backend="diff"`
needs the `diff` extra. The bundled datasets (`jax_ibart.datasets`) ship with
the package.

From a clone, for development:

```bash
git clone https://github.com/jkitchin/jax-ibart
cd jax-ibart
uv sync --extra all               # or: python -m venv .venv && pip install -e ".[all]"
uv run pytest                     # fast suite (slow validation: pytest -m slow)
```

## Backends and extras

| Extra  | Adds | Needed for |
|--------|------|------------|
| `bart` | `bartz>=0.7,<0.11` | the faithful `IBart(backend="bart")` selection path |
| `diff` | `optax` | the differentiable `IBart(backend="diff")` path |
| `dev`  | `pytest`, `matplotlib` | running the tests and notebooks |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter | building this site |
| `all`  | `bart` + `diff` + `dev` | development |

### bartz version

`bartz` is pinned below 0.11. bartz 0.11 changed its input layout, and
0.11-0.13 change the BART-G.SE screen enough that the validated catalysis
results no longer reproduce (see the [oracle cross-check](oracle-crosscheck.md)).

## Precision note

The package runs in **float32** and does **not** enable JAX's global
`jax_enable_x64`: the `bartz` MCMC is incompatible with x64, and a library
should not flip that process-wide flag. Precision-critical linear algebra (the
final LASSO/OLS, collinearity checks) runs host-side in NumPy float64
regardless. You do not need to configure anything.

## GPU

`bartz` and the feature-engineering engine are JAX and run on GPU when a
GPU-enabled `jaxlib` is installed. The CPU path needs no special setup.

## Optional: building the docs

```bash
pip install -e ".[docs]"
mkdocs serve      # http://127.0.0.1:8000
```
