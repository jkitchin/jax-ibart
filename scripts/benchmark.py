"""Benchmark harness for jax-ibart.

Reports feature-engineering throughput and end-to-end backend timings on the
catalysis dataset. Run:

    PYTHONPATH=src python scripts/benchmark.py         # FE + diff backend
    PYTHONPATH=src python scripts/benchmark.py --bart  # also time the bart backend (~minutes)

The feature-engineering engine is vectorized (batched JAX ops over column pairs,
structural SymPy dedup), so it benefits from GPU when one is available; these CPU
numbers are the floor, not the ceiling.
"""

from __future__ import annotations

import sys
import time

from jax_ibart import IBart, engineer
from jax_ibart.datasets import load_metal_support
from jax_ibart.operators import default_binary


def _time(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def main(do_bart: bool):
    X, y, names, units = load_metal_support()
    print(f"catalysis: X={X.shape}, {len(names)} features\n")

    # FE throughput on the largest raw-feature step (round-1 binary expansion).
    # Later rounds run on a *selected* (small) pool, so this is the FE worst case.
    pool = engineer.from_primary(X, names, units=units)
    b1, t = _time(lambda: engineer.apply_binary(pool, default_binary()))
    print(
        f"FE  apply_binary (59 features -> pairs) : {t:6.2f}s  -> {b1.n_descriptors} descriptors"
    )

    _, t = _time(
        lambda: IBart(backend="diff", schedule=("b", "u", "b"), seed=0, steps=1000).fit(
            X, y, names, units=units
        )
    )
    print(f"\nend-to-end  diff backend   : {t:6.1f}s")

    if do_bart:
        _, t = _time(
            lambda: IBart(
                backend="bart",
                schedule=("b", "u", "b"),
                seed=0,
                n_perm=10,
                n_tree=30,
                n_draw=200,
                n_burn=200,
            ).fit(X, y, names, units=units)
        )
        print(f"end-to-end  bart backend   : {t:6.1f}s  (BART-G.SE dominates)")


if __name__ == "__main__":
    main(do_bart="--bart" in sys.argv)
