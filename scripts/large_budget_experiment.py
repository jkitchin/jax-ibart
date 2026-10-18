"""Test whether a larger MCMC budget closes the exact-descriptor gap.

Runs the faithful pipeline on catalysis with a large BART-G.SE budget and the
R-iBART-style leading primary screen, screens ONCE, then compares l0 models at
k=1,2,3,5 against the published oracle. CPU-only here (no GPU backend); this is
slow but directly tests the budget hypothesis.
"""

import json
import time
from pathlib import Path

import jax
import numpy as np

from jax_ibart import engineer, sparsify
from jax_ibart.datasets import load_metal_support
from jax_ibart.operators import default_binary, default_unary
from jax_ibart.selection import BartGSESelector

ROOT = Path(__file__).resolve().parents[1]


def has_ratio(expr_str):
    return "Hfo" in expr_str and "Oxv" in expr_str


def main():
    X, y, names, units = load_metal_support()
    oracle = json.loads((ROOT / "src" / "jax_ibart" / "data" / "oracle_ibart.json").read_text())

    # Large, bartMachine-like budget.
    sel = BartGSESelector(n_perm=50, alpha=0.05, n_tree=50, n_draw=1000, n_burn=1000)
    key = jax.random.PRNGKey(0)
    unary, binary = default_unary(), default_binary()

    def screen(pool, k):
        keep = np.asarray(sel.select(pool.X, y, key=k))
        if not keep.any():
            return pool
        return engineer.Pool(
            X=pool.X[:, keep],
            exprs=[e for e, kp in zip(pool.exprs, keep) if bool(kp)],
            units=None if pool.units is None else pool.units[keep])

    t0 = time.time()
    pool = engineer.from_primary(X, names, units=units)

    # Leading screen on primaries (matches R iBART).
    key, sk = jax.random.split(key)
    pool = screen(pool, sk)
    print(f"[{time.time()-t0:6.1f}s] leading screen -> {pool.n_descriptors} primaries kept",
          flush=True)

    for step in ("b", "u", "b"):
        pool = (engineer.apply_unary(pool, unary) if step == "u"
                else engineer.apply_binary(pool, binary))
        gen = pool.n_descriptors
        key, sk = jax.random.split(key)
        pool = screen(pool, sk)
        print(f"[{time.time()-t0:6.1f}s] {step}: generated {gen} -> selected {pool.n_descriptors}",
              flush=True)

    # Does the Hfo/Oxv ratio survive screening into the final pool?
    final_strs = [str(e) for e in pool.exprs]
    ratio_in_pool = [s for s in final_strs if has_ratio(s)]
    print(f"\nfinal pool: {pool.n_descriptors} descriptors; "
          f"{len(ratio_in_pool)} couple Hfo & Oxv", flush=True)
    for s in ratio_in_pool[:8]:
        print("   ", s, flush=True)

    # l0 models at k=1,2,3,5 on the SAME final pool.
    Xp = np.asarray(pool.X, dtype=np.float64)
    cand = sparsify.lasso_screen(Xp, y, n_keep=12)
    print("\n=== l0 models vs oracle ===", flush=True)
    for K in (1, 2, 3, 5):
        idx, coef = sparsify.l0_best_subset(Xp[:, cand], y, max_dim=K, score="bic")
        descs = [pool.exprs[cand[i]] for i in idx]
        A = np.column_stack([np.ones(len(y))] + [np.asarray(pool.X[:, cand[i]]) for i in idx])
        rmse = float(np.sqrt(((y - A @ coef) ** 2).mean()))
        orc = oracle.get(f"lzero_{K}")
        orc_rmse = oracle["lzero_in_sample_rmse"][K - 1]
        print(f"\nk={K}: PORT RMSE {rmse:.3f}  | ORACLE RMSE {orc_rmse:.3f}", flush=True)
        print(f"   PORT  : {[str(d) for d in descs]}", flush=True)
        print(f"   ORACLE: {orc}", flush=True)

    print(f"\ntotal runtime: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
