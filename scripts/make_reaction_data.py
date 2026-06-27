"""Generate an example reaction-kinetics dataset for RHS discovery.

Liquid-phase isothermal batch reactor with two reactions:

    R1:  A + B -> C     r1 = k1 * C_A * C_B   (bimolecular,  k1 = 0.5)
    R2:      C -> D     r2 = k2 * C_C         (first order,  k2 = 0.2)

Mole balances (the RHS we want to rediscover):

    dC_A/dt = -k1 C_A C_B
    dC_B/dt = -k1 C_A C_B
    dC_C/dt =  k1 C_A C_B - k2 C_C
    dC_D/dt =  k2 C_C

Several initial conditions (batches) are integrated to cover concentration
space (needed for identifiability), with small measurement noise added. Writes
data/reaction_kinetics.csv with columns: batch, t, CA, CB, CC, CD.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

K1, K2 = 0.5, 0.2
ROOT = Path(__file__).resolve().parents[1]


def rhs(t, C):
    CA, CB, CC, CD = C
    r1 = K1 * CA * CB
    r2 = K2 * CC
    return [-r1, -r1, r1 - r2, r2]


def main(noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, 60)
    ICs = [(2.0, 2.0), (1.5, 3.0), (3.0, 1.0), (2.5, 2.5), (1.0, 1.0), (3.0, 3.0)]

    rows = []
    for b, (a0, b0) in enumerate(ICs):
        sol = solve_ivp(
            rhs, [t[0], t[-1]], [a0, b0, 0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-10
        )
        C = sol.y.T + rng.normal(0.0, noise, sol.y.T.shape)  # measurement noise
        for ti, ci in zip(t, C):
            rows.append([b, float(ti), *map(float, ci)])

    df = pd.DataFrame(rows, columns=["batch", "t", "CA", "CB", "CC", "CD"])
    out = ROOT / "data" / "reaction_kinetics.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {df.batch.nunique()} batches)")
    print(f"true: dCA/dt=-{K1}*CA*CB; dCC/dt={K1}*CA*CB-{K2}*CC; dCD/dt={K2}*CC")


if __name__ == "__main__":
    main()
