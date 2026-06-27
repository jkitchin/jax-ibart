# M5: vectorization and benchmarks

Goal: make the engine fast and GPU-ready, then measure honestly where the time
goes. Numbers below are single-core CPU (macOS/arm64, JAX CPU backend) on the
91 x 59 catalysis dataset.

## What was optimized

1. **Vectorized feature engineering.** `apply_unary` / `apply_binary` now evaluate
   each operator over *all* columns / column-pairs in one batched JAX call
   (broadcasting over index arrays), instead of a Python loop dispatching one
   JAX op per column. This is the O(d^2) blowup the design always flagged as the
   prime GPU target; it is now expressed as batched array ops that a GPU
   parallelizes directly.
2. **Cheap structural dedup.** `clean()` previously called `sp.simplify` on every
   engineered descriptor, which dominated FE time. SymPy already canonicalizes
   commutative args on construction, so `srepr` catches reorderings for ~100x
   less cost; deeper algebraic identities are left to the numeric collinearity
   pass (which caught them anyway).
3. **Bounded collinearity dedup.** The greedy correlation pass is O(d^2 n); it is
   skipped above `corr_max_d` (default 10k) descriptors, where it is both
   expensive and unnecessary (pre-selection, with BART screening to follow).

## Results

| Step                              | Before  | After   | Speedup |
|-----------------------------------|---------|---------|---------|
| FE: round-1 binary (59 features)  | 6.4 s   | 1.2 s   | ~5.3x   |
| of which `clean()` (`simplify`)   | 5.35 s  | ~0.05 s | ~100x   |
| End-to-end, `diff` backend        | 10.1 s  | 5.0 s   | ~2x     |
| End-to-end, `bart` backend        | 132 s   | ~125 s  | ~1.05x  |

Correctness preserved: identical descriptor counts and unchanged fit results
(diff R^2 = 0.833, bart recovers Hfo/Oxv) before and after.

## Where the time actually goes (the honest part)

End-to-end with the faithful `bart` backend is **~85% BART-G.SE**, ~12% feature
engineering, ~3% everything else. Vectorizing FE gives a real but modest
end-to-end win there, because the cost is the 33 `bartz` MCMC fits (11 per round
x 3 rounds: one observed + 10 permutation-null), not the array plumbing.

The remaining large levers are therefore in the *selection* step, and they are
hardware/parallelism, not algorithm:

- **Run `bartz` on a real GPU.** `bartz` is already JAX and reports up to ~200x
  over a single CPU core on GPU. This machine is CPU-only (macOS/arm64), so that
  speedup is unrealized here but available on GPU hardware with no code change.
- **Parallelize the permutation null across devices.** The 10 permutation refits
  per round are independent and map cleanly onto `pmap` / multiple devices. On a
  single CPU device there is nothing to overlap, so it is a no-op here but a
  direct multi-GPU win.

The `diff` backend, which is FE-bound rather than MCMC-bound, benefits most from
the vectorization (2x) and is ~25x faster than `bart` end-to-end -- the
speed/fidelity trade documented in `validation-catalysis.md`.

## Reproduce

```bash
PYTHONPATH=src python scripts/benchmark.py          # FE + diff backend
PYTHONPATH=src python scripts/benchmark.py --bart   # also the bart backend (~minutes)
```
