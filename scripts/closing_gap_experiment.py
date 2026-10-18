import json, time
from pathlib import Path
import jax, numpy as np
from jax_ibart import engineer, sparsify
from jax_ibart.datasets import load_metal_support
from jax_ibart.operators import default_binary, default_unary
from jax_ibart.selection import BartGSESelector
ROOT = Path(__file__).resolve().parents[1]
X, y, names, units = load_metal_support()
oracle = json.loads((ROOT/"src"/"jax_ibart"/"data"/"oracle_ibart.json").read_text())
# PERMISSIVE threshold (alpha=0.25), NO leading screen so the Hfo/Oxv ratio
# forms from all 59 primaries in round 1. Moderate budget.
sel = BartGSESelector(n_perm=25, alpha=0.25, n_tree=40, n_draw=400, n_burn=400)
key = jax.random.PRNGKey(0)
un, bi = default_unary(), default_binary()
def screen(pool, k):
    keep = np.asarray(sel.select(pool.X, y, key=k))
    if not keep.any(): return pool
    return engineer.Pool(X=pool.X[:,keep],
        exprs=[e for e,kp in zip(pool.exprs,keep) if bool(kp)],
        units=None if pool.units is None else pool.units[keep])
t0=time.time()
pool = engineer.from_primary(X, names, units=units)
for step in ("b","u","b"):
    pool = engineer.apply_unary(pool,un) if step=="u" else engineer.apply_binary(pool,bi)
    gen=pool.n_descriptors
    key,sk=jax.random.split(key); pool=screen(pool,sk)
    print(f"[{time.time()-t0:6.1f}s] {step}: {gen} -> {pool.n_descriptors}", flush=True)
strs=[str(e) for e in pool.exprs]
ratio=[s for s in strs if "Hfo" in s and "Oxv" in s]
print(f"final pool {pool.n_descriptors}; {len(ratio)} couple Hfo&Oxv; "
      f"exact abs(Hfo/Oxv) present: {any(s in ('Abs(Hfo/Oxv)','Abs(Hfo)/Abs(Oxv)') for s in strs) or 'Hfo/Oxv' in ' '.join(strs)}", flush=True)
Xp=np.asarray(pool.X,np.float64); cand=sparsify.lasso_screen(Xp,y,n_keep=15)
for K in (1,3):
    idx,coef=sparsify.l0_best_subset(Xp[:,cand],y,max_dim=K,score="bic")
    descs=[str(pool.exprs[cand[i]]) for i in idx]
    A=np.column_stack([np.ones(len(y))]+[np.asarray(pool.X[:,cand[i]]) for i in idx])
    rmse=float(np.sqrt(((y-A@coef)**2).mean()))
    print(f"k={K}: PORT RMSE {rmse:.3f} (oracle {oracle['lzero_in_sample_rmse'][K-1]:.3f}) -> {descs}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
