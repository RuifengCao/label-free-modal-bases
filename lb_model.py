"""Network-generated basis with a per-step Galerkin solve, trained on Newmark incremental potentials (JAX, CPU, float64).

The training code reads only K, M, the Rayleigh coefficients, the load patterns and the time functions.
It never reads reference solutions or eigenvectors.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

jax.config.update("jax_enable_x64", True)

L, H = 10.0, 1.0
BETA, GAMMA = 0.25, 0.5


def features(pts, params):
    xc, d, rho = params
    x, y = pts[:, 0], pts[:, 1]
    cols = [x / L, y / H]
    for k in range(1, 7):
        cols += [np.sin(k * np.pi * x / L), np.cos(k * np.pi * x / L)]
    for s in (rho, 0.3):
        xi, eta = (x - xc) / s, (y - (H - d)) / s
        r = np.sqrt(xi ** 2 + eta ** 2)
        cols += [xi / (1 + r), eta / (1 + r), np.exp(-r)]
    return np.stack(cols, axis=1)                                   # (n_nodes, 20)


def init_mlp(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k1 = jax.random.split(key)
        params.append((jax.random.normal(k1, (a, b)) * np.sqrt(1.0 / a), jnp.zeros(b)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class Problem:
    """Static data of one geometry, as JAX arrays."""

    def __init__(self, inst, a_r, b_r, dt, m):
        self.m = m
        self.n = inst["n_free"]
        self.feats = jnp.asarray(features(inst["pts"], inst["params"]))
        self.ixf = jnp.asarray(inst["ixf"]); self.iyf = jnp.asarray(inst["iyf"])
        self.K = jsparse.BCOO.from_scipy_sparse(inst["K"].tocoo())
        self.M = jsparse.BCOO.from_scipy_sparse(inst["M"].tocoo())
        self.P = jnp.asarray(inst["P"])
        self.a_r, self.b_r, self.dt = a_r, b_r, dt
        a0 = 1 / (BETA * dt ** 2); a1 = GAMMA / (BETA * dt); a2 = 1 / (BETA * dt)
        a3 = 1 / (2 * BETA) - 1; a4 = GAMMA / BETA - 1; a5 = dt / 2 * (GAMMA / BETA - 2)
        self.nc = (a0, a1, a2, a3, a4, a5)


def raw_basis(params, pr):
    out = mlp(params, pr.feats)                                     # (n_nodes, 2m)
    m = pr.m
    Phi = jnp.zeros((pr.n, m))
    Phi = Phi.at[pr.ixf].set(out[:, :m]).at[pr.iyf].set(out[:, m:])
    return Phi


def orthonormal_basis(params, pr):
    Phi = raw_basis(params, pr)
    G = Phi.T @ (pr.M @ Phi)
    Lc = jnp.linalg.cholesky(0.5 * (G + G.T))
    return jax.scipy.linalg.solve_triangular(Lc, Phi.T, lower=True).T    # Phi_o^T M Phi_o = I


def reduced_ops(Phi, pr):
    """Reduced matrices with the full gradient (for the step operator and the loads)."""
    KPhi = pr.K @ Phi; MPhi = pr.M @ Phi
    Kr = Phi.T @ KPhi; Mr = Phi.T @ MPhi
    Cr = pr.a_r * Mr + pr.b_r * Kr
    a0, a1 = pr.nc[0], pr.nc[1]
    Ar = Kr + a0 * Mr + a1 * Cr
    Pr = Phi.T @ pr.P
    return Kr, Mr, Cr, Ar, Pr


def loss_fn(params, pr, W, G):
    """W: (B, 3) load weights; G: (B, N+1) time functions. Mean incremental potential over cases and steps."""
    Phi = orthonormal_basis(params, pr)
    Kr, Mr, Cr, Ar, Pr = reduced_ops(Phi, pr)
    Phid = jax.lax.stop_gradient(Phi)
    Mh = Phi.T @ (pr.M @ Phid)                                      # history operators: gradient only through the left factor
    Kh = Phi.T @ (pr.K @ Phid)
    Ch = pr.a_r * Mh + pr.b_r * Kh
    a0, a1, a2, a3, a4, a5 = pr.nc
    cf = jax.scipy.linalg.cho_factor(0.5 * (Ar + Ar.T))
    fr = W @ Pr.T                                                   # (B, m) reduced load patterns
    Mr_d = jax.lax.stop_gradient(Mr)
    acc0 = jnp.linalg.solve(Mr_d, (fr * G[:, :1]).T).T              # initial acceleration in reduced coordinates
    B = W.shape[0]
    c0 = jnp.zeros((B, pr.m)); v0 = jnp.zeros((B, pr.m))

    def step(carry, g_next):
        c, cd, cdd = carry
        b = fr * g_next[:, None] + (a0 * c + a2 * cd + a3 * cdd) @ Mh.T + (a1 * c + a4 * cd + a5 * cdd) @ Ch.T
        cn = jax.scipy.linalg.cho_solve(cf, b.T).T
        pot = 0.5 * jnp.sum(cn * (cn @ Ar.T), axis=1) - jnp.sum(b * cn, axis=1)     # Pi_n(Phi c_n+1), up to the history constant
        cdd_n = a0 * (cn - c) - a2 * cd - a3 * cdd
        cd_n = cd + pr.dt * ((1 - GAMMA) * cdd + GAMMA * cdd_n)
        new = jax.lax.stop_gradient((cn, cd_n, cdd_n))
        return new, pot

    _, pots = jax.lax.scan(step, (c0, v0, jax.lax.stop_gradient(acc0)), G[:, 1:].T)
    return jnp.mean(pots)


def train(pr, W, G, m, seed, iters=3000, lr0=1e-3, lr1=1e-5, log_every=250, log=print):
    sizes = [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * m]
    params = init_mlp(jax.random.PRNGKey(seed), sizes)
    vg = jax.jit(jax.value_and_grad(lambda p: loss_fn(p, pr, W, G)))
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def update(params, mo, ve, g, t, lr):
        mo = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, mo, g)
        ve = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, ve, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), mo)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), ve)
        params = jax.tree_util.tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh)
        return params, mo, ve

    hist = []
    for t in range(1, iters + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = vg(params)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % log_every == 0 or t == 1:
            hist.append((t, float(val)))
            log("  iter %d loss %.8e" % (t, float(val)))
    return params, hist


def basis_numpy(params, pr):
    return np.asarray(orthonormal_basis(params, pr))
