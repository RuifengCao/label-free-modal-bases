"""Timing and memory benchmark for the planned Unified 2027 runs (not an experiment: no result is read from it).

  python -I bench.py build        # mesh + assembly, 16 eigenpairs, one full Newmark reference, for several geometries
  python -I bench.py o4|o2|o3     # one iteration of log-det / trace / sequential Rayleigh quotient (G0, m = 8)
  python -I bench.py o1           # one iteration of the per-step incremental potential (16 cases, 640 steps)
  python -I bench.py fam4         # one family iteration: four geometries per step, each with its own compiled function
Run with one thread: taskset -c 0 and the XLA / BLAS thread variables set by the caller.
"""
import sys, os, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
UDC = _HERE if os.path.exists(os.path.join(_HERE, "fem_cantilever.py")) else os.path.dirname(_HERE)   # scripts next to it or one level up
sys.path.insert(0, UDC)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc6 as p6
import jax
import jax.numpy as jnp

M_MODES = 8


def rss_mb():
    """Peak resident memory of this process in MB (Linux: ru_maxrss; Windows: peak working set via psutil)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2 ** 20


N_TIME = int(os.environ.get("BENCH_N_TIME", "0"))      # optional override of the number of timed iterations


def adam_update():
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def update(params, mo, ve, g, t, lr):
        mo = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, mo, g)
        ve = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, ve, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), mo)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), ve)
        return jax.tree_util.tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh), mo, ve
    return update


def logdet_loss(p, pr):
    Phi = lb.orthonormal_basis(p, pr)
    Kr = Phi.T @ (pr.K @ Phi)
    Lc = jnp.linalg.cholesky(0.5 * (Kr + Kr.T))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))


def trace_loss(p, pr):
    Phi = lb.orthonormal_basis(p, pr)
    return jnp.trace(Phi.T @ (pr.K @ Phi))


def seqrq_loss(p, pr):
    """Columns M-orthonormalised one by one against the stopped earlier columns; sum of their Rayleigh quotients."""
    Phi = lb.raw_basis(p, pr)
    cols = []
    tot = 0.0
    for k in range(pr.m):
        v = Phi[:, k]
        for q in cols:
            qs = jax.lax.stop_gradient(q)
            v = v - (qs @ (pr.M @ v)) * qs
        v = v / jnp.sqrt(v @ (pr.M @ v))
        tot = tot + v @ (pr.K @ v)
        cols.append(v)
    return tot


def time_loss(loss, pr, n_warm, n_time, seed=0):
    sizes = [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * pr.m]
    params = lb.init_mlp(jax.random.PRNGKey(seed), sizes)
    vg = jax.jit(jax.value_and_grad(loss))
    update = adam_update()
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    t0 = time.perf_counter()
    val, g = vg(params); params, mo, ve = update(params, mo, ve, g, 1, 1e-3)
    jax.block_until_ready(params)
    t_compile = time.perf_counter() - t0
    for t in range(2, n_warm + 2):
        val, g = vg(params); params, mo, ve = update(params, mo, ve, g, t, 1e-3)
    jax.block_until_ready(params)
    t0 = time.perf_counter()
    for t in range(n_warm + 2, n_warm + 2 + n_time):
        val, g = vg(params); params, mo, ve = update(params, mo, ve, g, t, 1e-3)
    jax.block_until_ready(params)
    per_it = (time.perf_counter() - t0) / n_time
    return t_compile, per_it


def setup_g0():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    return inst, pr, dt, train_cases


GEOMS = [(2.0, 0.3), (1.3, 0.15), (3.1, 0.37), (4.6, 0.22), (2.7, 0.11)]


def cmd_build():
    out = []
    for xc, d in GEOMS:
        t0 = time.perf_counter(); inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2); t_build = time.perf_counter() - t0
        t0 = time.perf_counter(); w, V = fc.eig(inst["K"], inst["M"], 16); t_eig = time.perf_counter() - t0
        a_r, b_r, dt = fc.time_setup(w[0], w[2])
        C = (a_r * inst["M"] + b_r * inst["K"]).tocsc()
        case = dict(w=[1.0, 0.0, 0.0], kind="pulse", tau=float(np.pi / w[1]))
        t0 = time.perf_counter(); fc.newmark_full(inst["K"], inst["M"], C, inst["P"], case, dt, rc.N_STEPS); t_ref = time.perf_counter() - t0
        rec = dict(xc=xc, d=d, n_free=inst["n_free"], t_build=t_build, t_eig16=t_eig, t_ref_one_case=t_ref)
        out.append(rec); print(json.dumps(rec), flush=True)
    print(json.dumps(dict(rss_mb=rss_mb())))


def cmd_obj(which):
    inst, pr, dt, train_cases = setup_g0()
    if which == "o1":
        W, G = rc.W_G(train_cases, dt)
        Wj, Gj = jnp.asarray(W), jnp.asarray(G)
        loss = lambda p: lb.loss_fn(p, pr, Wj, Gj)
        n_warm, n_time = 3, (N_TIME or 40)
    else:
        f = dict(o4=logdet_loss, o2=trace_loss, o3=seqrq_loss)[which]
        loss = lambda p: f(p, pr)
        n_warm, n_time = 20, (N_TIME or 300)
    t_compile, per_it = time_loss(loss, pr, n_warm, n_time)
    print(json.dumps(dict(which=which, n_free=inst["n_free"], t_compile_s=t_compile, per_iter_ms=1e3 * per_it,
                          rss_mb=rss_mb())), flush=True)


def cmd_fam4():
    a_r, b_r = 3.439e-4, 0.2554
    prs, t_builds = [], []
    for xc, d in GEOMS[1:]:
        t0 = time.perf_counter()
        inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        w, V = fc.eig(inst["K"], inst["M"], 3)
        _, _, dt = fc.time_setup(w[0], w[2])
        pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
        extra = np.tile(np.array([[(xc - 3.0) / 2.0, (d - 0.25) / 0.15]]), (pr.feats.shape[0], 1))
        pr.feats = jnp.concatenate([pr.feats, jnp.asarray(extra)], axis=1)      # family inputs: normalised x_c and d
        prs.append(pr); t_builds.append(time.perf_counter() - t0)
    sizes = [int(prs[0].feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES]
    params = lb.init_mlp(jax.random.PRNGKey(0), sizes)
    vgs = []
    t0 = time.perf_counter()
    for pr in prs:
        vg = jax.jit(jax.value_and_grad(lambda p, pr=pr: logdet_loss(p, pr)))
        jax.block_until_ready(vg(params)); vgs.append(vg)
    t_compile = (time.perf_counter() - t0) / len(prs)
    update = adam_update()
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)

    def step(params, mo, ve, t):
        tot, gsum = 0.0, None
        for vg in vgs:
            v, g = vg(params)
            tot = tot + v
            gsum = g if gsum is None else jax.tree_util.tree_map(jnp.add, gsum, g)
        return update(params, mo, ve, gsum, t, 1e-3)

    for t in range(1, 21):
        params, mo, ve = step(params, mo, ve, t)
    jax.block_until_ready(params)
    n = N_TIME or 200
    t0 = time.perf_counter()
    for t in range(21, 21 + n):
        params, mo, ve = step(params, mo, ve, t)
    jax.block_until_ready(params)
    per_it = (time.perf_counter() - t0) / n
    print(json.dumps(dict(which="fam4", n_free=[int(p.n) for p in prs], t_build_and_setup_s=t_builds,
                          t_compile_per_geometry_s=t_compile, per_iter_ms=1e3 * per_it, rss_mb=rss_mb())), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build":
        cmd_build()
    elif c == "fam4":
        cmd_fam4()
    else:
        cmd_obj(c)
