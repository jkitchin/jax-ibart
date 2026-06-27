# Installation

`jax-ibart` is pure Python and needs no JVM.

```bash
pip install jax-ibart            # core
pip install "jax-ibart[bart]"    # + faithful BART-G.SE backend (bartz)
pip install "jax-ibart[diff]"    # + differentiable backend (optax)
pip install "jax-ibart[all]"     # everything, incl. test/dev extras
```

From a clone:

```bash
git clone https://github.com/jkitchin/jax-ibart
cd jax-ibart
uv pip install -e ".[all]"
pytest                            # fast suite (slow validation: pytest -m slow)
```

## Backends and extras

| Extra  | Adds | Needed for |
|--------|------|------------|
| `bart` | `bartz` | the faithful `IBart(backend="bart")` selection path |
| `diff` | `optax` | the differentiable `IBart(backend="diff")` path |
| `dev`  | `pytest`, `matplotlib` | running the tests |

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
pip install mkdocs-material "mkdocstrings[python]" mkdocs-jupyter
mkdocs serve      # http://127.0.0.1:8000
```
