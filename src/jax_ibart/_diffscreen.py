"""Differentiable feature screening via hard-concrete L0 gates.

Implements the screening signal for the ``diff`` selection backend. Each
descriptor column gets a learnable stochastic gate z_j in [0, 1] from the
hard-concrete distribution (Louizos, Welling & Kingma, "Learning Sparse Neural
Networks through L0 Regularization", ICLR 2018). A small MLP predicts y from the
gated, standardized features, and training minimizes

    MSE(y, f(z * X)) + l0 * E[ ||z||_0 ]

where the expected L0 norm is the sum of gate-open probabilities. Everything is
differentiable end to end: gradients flow through the reparameterized gates,
which is the defining property of this backend. After training, each gate's
open-probability is the per-feature importance.

Pure JAX + optax, float32 (global x64 stays off; see PLAN precision policy).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

# Hard-concrete hyperparameters (Louizos et al. defaults).
_BETA = 0.5       # temperature
_GAMMA = -0.1     # stretch lower bound
_ZETA = 1.1       # stretch upper bound


def _open_prob(log_alpha):
    """P(z > 0): the gate-open probability (penalty term and importance)."""
    return jax.nn.sigmoid(log_alpha - _BETA * jnp.log(-_GAMMA / _ZETA))


def _sample_gate(log_alpha, key):
    """Reparameterized hard-concrete sample z in [0, 1] (training)."""
    u = jax.random.uniform(key, log_alpha.shape, minval=1e-6, maxval=1 - 1e-6)
    s = jax.nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + log_alpha) / _BETA)
    s_bar = s * (_ZETA - _GAMMA) + _GAMMA
    return jnp.clip(s_bar, 0.0, 1.0)


def _init_params(key, d, hidden):
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(d)
    return {
        "log_alpha": jnp.full((d,), 1.0),  # start gates mostly open (PAN-safe)
        "W1": jax.random.normal(k1, (d, hidden)) * scale,
        "b1": jnp.zeros((hidden,)),
        "W2": jax.random.normal(k2, (hidden, 1)) * (1.0 / jnp.sqrt(hidden)),
        "b2": jnp.zeros((1,)),
    }


def _mlp(params, Xg):
    h = jnp.tanh(Xg @ params["W1"] + params["b1"])
    return (h @ params["W2"] + params["b2"]).ravel()


def screen_importance(X, y, key, *, steps=1500, lr=1e-2, l0=0.03, hidden=16):
    """Train the gated MLP and return per-feature importances in [0, 1].

    Importances are the hard-concrete gate-open probabilities after training:
    high for predictive features, near zero for irrelevant ones.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    yc = (y - y.mean()) / (y.std() + 1e-6)
    d = Xs.shape[1]

    key, ik = jax.random.split(key)
    params = _init_params(ik, d, hidden)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(params, key):
        z = _sample_gate(params["log_alpha"], key)
        pred = _mlp(params, Xs * z)
        mse = jnp.mean((yc - pred) ** 2)
        # Expected L0 = sum of gate-open probabilities. Penalize the *count*
        # (not the mean): keeping a useless gate open must cost ~l0 regardless
        # of d, so noise gates close while predictive gates stay open.
        l0_pen = jnp.sum(_open_prob(params["log_alpha"]))
        return mse + l0 * l0_pen

    @jax.jit
    def step(params, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for _ in range(steps):
        key, sk = jax.random.split(key)
        params, opt_state, _ = step(params, opt_state, sk)

    return _open_prob(params["log_alpha"])
