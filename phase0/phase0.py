"""Phase 0 (exploratory) of the Unified 2027 dynamics paper (see PHASE0_PLAN.md).

  python -I phase0.py checks                          # A1 to A4
  python -I phase0.py train OBJ SEED ITERS VARIANT    # (a), (c): OBJ in o1, o2, o3, o4; VARIANT hard_angle or hard
  python -I phase0.py long                            # (b): O4, 60,000 Adam iterations, then L-BFGS
  python -I phase0.py gap                             # (d); also writes geometries.json
  python -I phase0.py sens                            # (e)
  python -I phase0.py fam SEED ITERS                  # (f): family pilot
  python -I phase0.py evalfam                         # (f) and (i): evaluation on the 12 pilot geometries
  python -I phase0.py evalg0                          # evaluation of every G0 basis
  python -I phase0.py cost                            # (h)

Environment: P0_OUT replaces the output folder (smoke tests); P0_SMOKE=1 shortens the evaluations;
P0_LOG sets the monitoring interval (default 1,000 iterations).
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import scipy.optimize as sopt

HERE = os.path.dirname(os.path.abspath(__file__))
UDC = os.path.dirname(HERE)
sys.path.insert(0, UDC)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc2 as p2
import posthoc3 as p3
import posthoc4 as p4
import posthoc6 as p6          # importing it installs the exact-clamp dispatch of lb.raw_basis
import posthoc7 as p7
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

OUT = os.environ.get("P0_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("P0_SMOKE", "0") == "1"
LOG_EVERY = int(os.environ.get("P0_LOG", "1000"))
M_MODES = 8
N = rc.N_STEPS
OLD_OUT = os.path.join(UDC, "out")


def _js(o):
    return o.item() if hasattr(o, "item") else str(o)


def jdump(obj, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1, default=_js)


def save(fn, params, **kw):
    with open(os.path.join(OUT, fn), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], **kw), f)


# ----------------------------------------------------------------------------- geometries

def sobol_geoms(seed, n):
    from scipy.stats import qmc
    k = int(np.ceil(np.log2(n)))
    u = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(k)[:n]
    return [(round(1.0 + 4.0 * float(a), 4), round(0.1 + 0.3 * float(b), 4)) for a, b in u]


GEOM_SETS = dict(G123=(11, 3), fam_train=(12, 24), fam_held=(13, 16), pilot_train=(21, 8), pilot_held=(22, 4))


def geometries():
    g = {"G0": [(2.0, 0.3)]}
    for k, (s, n) in GEOM_SETS.items():
        g[k] = sobol_geoms(s, n)
    return g


def geom_setup(xc, d, n_eig=16):
    """rc.setup() for any notch position and depth (same mesh rules, load-case recipe and seeds)."""
    inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
    w, V = fc.eig(inst["K"], inst["M"], n_eig)
    a_r, b_r, dt = fc.time_setup(w[0], w[2])
    C = (a_r * inst["M"] + b_r * inst["K"]).tocsc()
    T1, T2, T3 = (2 * np.pi / w[k] for k in range(3))

    def cases(seed, n):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            wv = rng.uniform(-1, 1, 3)
            tau = float(np.exp(rng.uniform(np.log(0.5 * T3), np.log(0.5 * T1))))
            out.append(dict(w=wv.tolist(), kind="pulse", tau=tau))
        return out

    train_cases = cases(2026, 16)
    eval_cases = cases(2027, 8) + [dict(w=[1.0, 0.0, 0.0], kind="pulse", tau=float(T2 / 2), name="tip_pulse"),
                                   dict(w=[1.0, 0.0, 0.0], kind="step", name="tip_step")]
    return inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases


def fam_problem(inst, a_r=0.0, b_r=0.0, dt=1.0):
    """Family inputs: 'hard_angle' plus the normalised notch position and depth. For O4 training the damping and
    time-step entries are unused placeholders (no eigen solve is needed to build a training problem)."""
    xc, d, _ = inst["params"]
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    extra = np.tile(np.array([[(xc - 3.0) / 2.0, (d - 0.25) / 0.15]]), (pr.feats.shape[0], 1))
    pr.feats = jnp.concatenate([pr.feats, jnp.asarray(extra)], axis=1)
    return pr


# ----------------------------------------------------------------------------- objectives (JAX)

def _orth(Phi, Mop):
    G = Phi.T @ (Mop @ Phi)
    Lc = jnp.linalg.cholesky(0.5 * (G + G.T))
    return jax.scipy.linalg.solve_triangular(Lc, Phi.T, lower=True).T


def o2_loss(p, pr):
    Phi = lb.orthonormal_basis(p, pr)
    return jnp.trace(Phi.T @ (pr.K @ Phi))


def o3_loss(p, pr):
    """Sequential Rayleigh quotients: column k is M-orthogonalised against the stopped earlier columns."""
    Phi = lb.raw_basis(p, pr)
    Q = jax.lax.stop_gradient(_orth(Phi, pr.M))
    U = jnp.triu(jnp.ones((pr.m, pr.m)), 1)
    R = Phi - Q @ (U * (Q.T @ (pr.M @ Phi)))
    return jnp.sum(jnp.sum(R * (pr.K @ R), axis=0) / jnp.sum(R * (pr.M @ R), axis=0))


def o3_loop(p, pr):
    """Loop form, as bench.py seqrq_loss (for check A1 only)."""
    Phi = lb.raw_basis(p, pr)
    cols, tot = [], 0.0
    for k in range(pr.m):
        v = Phi[:, k]
        for q in cols:
            qs = jax.lax.stop_gradient(q)
            v = v - (qs @ (pr.M @ v)) * qs
        v = v / jnp.sqrt(v @ (pr.M @ v))
        tot = tot + v @ (pr.K @ v)
        cols.append(v)
    return tot


def o4_loss(p, pr):
    Phi = lb.orthonormal_basis(p, pr)
    Kr = Phi.T @ (pr.K @ Phi)
    Lc = jnp.linalg.cholesky(0.5 * (Kr + Kr.T))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))


def make_loss(obj, pr, cases=None, dt=None):
    if obj == "o1":
        W, G = rc.W_G(cases, dt)
        Wj, Gj = jnp.asarray(W), jnp.asarray(G)
        return lambda p: lb.loss_fn(p, pr, Wj, Gj)
    f = dict(o2=o2_loss, o3=o3_loss, o4=o4_loss)[obj]
    return lambda p: f(p, pr)


# ----------------------------------------------------------------------------- basis quantities (NumPy)

def m_orth_np(Phi, M):
    G = Phi.T @ (M @ Phi)
    Lc = np.linalg.cholesky(0.5 * (G + G.T))
    return sla.solve_triangular(Lc, Phi.T, lower=True).T


def sin2(Phi, inst, V, k=8):
    """sin^2 of the M-angle between each exact eigenvector v_1..v_k and span(Phi)."""
    Q = m_orth_np(Phi, inst["M"])
    P = Q.T @ (inst["M"] @ V[:, :k])
    return [float(x) for x in 1.0 - np.sum(P ** 2, axis=0)]


def objectives_np(Phi, inst):
    """O2, O3 (Gram-Schmidt column by column) and O4 from a basis, independently of the JAX code."""
    K, M = inst["K"], inst["M"]
    Q = m_orth_np(Phi, M)
    Kr = Q.T @ (K @ Q)
    th = np.linalg.eigvalsh(0.5 * (Kr + Kr.T))
    cols, o3 = [], 0.0
    for k in range(Phi.shape[1]):
        v = Phi[:, k].copy()
        for q in cols:
            v = v - (q @ (M @ v)) * q
        v = v / np.sqrt(v @ (M @ v))
        o3 += float(v @ (K @ v))
        cols.append(v)
    return dict(o2=float(np.sum(th)), o3=o3, o4=float(np.sum(np.log(th))))


def make_cb(pr, inst, w, V):
    bfun = jax.jit(lambda p: lb.orthonormal_basis(p, pr))

    def cb(p):
        Phi = np.asarray(bfun(p))
        return dict(ritz=p6.ritz_errors(Phi, inst, w, 8), sin2=sin2(Phi, inst, V))
    return cb


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


def fam_value_grad(params, vgs, sub):
    tot, gsum = 0.0, None
    for i in sub:
        v, g = vgs[i](params)
        tot = tot + v
        gsum = g if gsum is None else jax.tree_util.tree_map(jnp.add, gsum, g)
    n = float(len(sub))
    return tot / n, jax.tree_util.tree_map(lambda a: a / n, gsum)


# ----------------------------------------------------------------------------- A: code checks

def cmd_checks():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    sizes = [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES]
    p_init = lb.init_mlp(jax.random.PRNGKey(0), sizes)
    p_f1, _ = p3.load_params(os.path.join(OLD_OUT, "posthoc6_F1.pkl"))
    res, ok = {}, True
    for name, p in (("init", p_init), ("F1", p_f1)):
        v1, g1 = jax.value_and_grad(lambda q: o3_loss(q, pr))(p)
        v2, g2 = jax.value_and_grad(lambda q: o3_loop(q, pr))(p)
        f1, _ = ravel_pytree(g1); f2, _ = ravel_pytree(g2)
        r = dict(value_rel=abs(float(v1) - float(v2)) / abs(float(v2)),
                 grad_rel=float(jnp.linalg.norm(f1 - f2) / jnp.linalg.norm(f2)))
        r["pass"] = r["value_rel"] < 1e-10 and r["grad_rel"] < 1e-10
        ok &= r["pass"]; res["A1_" + name] = r
    Phi_raw = np.asarray(lb.raw_basis(p_f1, pr))
    ref = objectives_np(Phi_raw, inst)
    jv = dict(o2=float(o2_loss(p_f1, pr)), o3=float(o3_loss(p_f1, pr)), o4=float(o4_loss(p_f1, pr)))
    a2 = {k: abs(jv[k] - ref[k]) / abs(ref[k]) for k in jv}
    v8 = objectives_np(V[:, :8], inst)
    lam = w[:8] ** 2
    a2["V8_o2_rel"] = abs(v8["o2"] - lam.sum()) / lam.sum()
    a2["V8_o3_rel"] = abs(v8["o3"] - lam.sum()) / lam.sum()
    a2["V8_o4_rel"] = abs(v8["o4"] - np.log(lam).sum()) / abs(np.log(lam).sum())
    a2["pass"] = all(a2[k] < 1e-9 for k in ("o2", "o3", "o4")) and all(a2[k] < 1e-10 for k in ("V8_o2_rel", "V8_o3_rel", "V8_o4_rel"))
    ok &= a2["pass"]; res["A2"] = dict(a2, jax=jv, numpy=ref)
    geos = [(1.3, 0.15), (3.1, 0.37), (4.6, 0.22), (2.7, 0.11)]
    prs = [fam_problem(fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)) for xc, d in geos]
    pf = lb.init_mlp(jax.random.PRNGKey(0), [int(prs[0].feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES])
    vgs = [jax.jit(jax.value_and_grad(lambda q, pr_=pr_: o4_loss(q, pr_))) for pr_ in prs]
    v_loop, g_loop = fam_value_grad(pf, vgs, [0, 1, 2, 3])
    v_ref, g_ref = jax.value_and_grad(lambda q: jnp.mean(jnp.stack([o4_loss(q, pr_) for pr_ in prs])))(pf)
    fl, _ = ravel_pytree(g_loop); fr, _ = ravel_pytree(g_ref)
    a3 = dict(value_rel=abs(float(v_loop) - float(v_ref)) / abs(float(v_ref)),
              grad_rel=float(jnp.linalg.norm(fl - fr) / jnp.linalg.norm(fr)))
    a3["pass"] = a3["value_rel"] < 1e-12 and a3["grad_rel"] < 1e-12
    ok &= a3["pass"]; res["A3"] = a3
    g = geom_setup(2.0, 0.3)          # ARPACK starts from a random vector: compare to a relative 1e-10 (DEVIATIONS_PHASE0.md, 1)
    cs_a, cs_b = train_cases + eval_cases, g[7] + g[8]
    a4 = dict(n_free_equal=bool(g[0]["n_free"] == inst["n_free"]), w_rel=float(np.max(np.abs(g[1] - w) / w)),
              dt_rel=float(abs(g[5] - dt) / dt),
              weights_names_equal=bool([(c["w"], c["kind"], c.get("name")) for c in cs_a] == [(c["w"], c["kind"], c.get("name")) for c in cs_b]),
              tau_rel=float(max(abs(a["tau"] - b["tau"]) / a["tau"] for a, b in zip(cs_a, cs_b) if "tau" in a)))
    a4["pass"] = bool(a4["n_free_equal"] and a4["weights_names_equal"] and max(a4["w_rel"], a4["dt_rel"], a4["tau_rel"]) < 1e-10)
    ok &= a4["pass"]; res["A4"] = a4
    res["all_pass"] = bool(ok)
    print(json.dumps(res, indent=1, default=_js), flush=True)
    jdump(res, "checks.json")


# ----------------------------------------------------------------------------- (a), (c): training on G0

def cmd_train(obj, seed, iters, variant):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, variant)
    loss = make_loss(obj, pr, train_cases, dt)
    t0 = time.time()
    params, hist, logs = p6.train_logged(loss, pr, M_MODES, seed=seed, iters=iters, lr0=1e-3, lr1=1e-5,
                                         log_every=LOG_EVERY, callback=make_cb(pr, inst, w, V))
    t_train = time.time() - t0
    save("g0_%s_s%d_%s_%d.pkl" % (obj, seed, variant, iters), params, logs=logs, t_train=t_train,
         obj=obj, seed=seed, variant=variant, iters=iters)
    print("done %s seed %d %s %d: %.0f s" % (obj, seed, variant, iters, t_train), flush=True)


# ----------------------------------------------------------------------------- (b): O4 for longer, then L-BFGS

def cmd_long(iters=60000):
    if SMOKE:
        iters = 200
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    cb = make_cb(pr, inst, w, V)
    t0 = time.time()
    params, hist, logs = p6.train_logged(make_loss("o4", pr), pr, M_MODES, seed=0, iters=iters, lr0=1e-3, lr1=1e-5,
                                         log_every=2 * LOG_EVERY, callback=cb)
    t_adam = time.time() - t0
    save("g0_o4_long_adam_%d.pkl" % iters, params, logs=logs, t_train=t_adam, obj="o4", seed=0,
         variant="hard_angle", iters=iters)
    x0, unravel = ravel_pytree(params)
    vg = jax.jit(jax.value_and_grad(lambda x: o4_loss(unravel(x), pr)))
    nfev, it, lb_logs = [0], [0], []

    def fun(x):
        v, g = vg(jnp.asarray(x))
        nfev[0] += 1
        return float(v), np.asarray(g, dtype=np.float64)

    def cbk(xk):
        it[0] += 1
        if it[0] % (25 if SMOKE else 250) == 0:
            rec = dict(it=it[0], nfev=nfev[0], loss=float(vg(jnp.asarray(xk))[0]), **cb(unravel(jnp.asarray(xk))))
            lb_logs.append(rec)
            print(json.dumps(rec), flush=True)

    t0 = time.time()
    r = sopt.minimize(fun, np.asarray(x0, dtype=np.float64), jac=True, method="L-BFGS-B", callback=cbk,
                      options=dict(maxiter=50 if SMOKE else 5000, maxcor=20, ftol=0.0, gtol=1e-14,
                                   maxfun=80 if SMOKE else 8000))
    t_lbfgs = time.time() - t0
    p2_ = unravel(jnp.asarray(r.x))
    final = dict(loss=float(r.fun), **cb(p2_))
    print(json.dumps(dict(lbfgs_nit=int(r.nit), nfev=int(r.nfev), message=str(r.message), t=t_lbfgs, **final)), flush=True)
    save("g0_o4_long_lbfgs.pkl", p2_, logs=lb_logs, final=final, t_train=t_adam + t_lbfgs, t_lbfgs=t_lbfgs,
         nit=int(r.nit), nfev=int(r.nfev), message=str(r.message), obj="o4", seed=0, variant="hard_angle", iters=iters)


# ----------------------------------------------------------------------------- (d): gap and mode order

def cmd_gap():
    geo = geometries()
    jdump(geo, "geometries.json")
    w_plain = float(p2.plain_cantilever_w1()[0])
    rows = []
    for group, lst in geo.items():
        for (xc, d) in (lst[:2] if SMOKE else lst):
            t0 = time.time()
            inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
            w, V = fc.eig(inst["K"], inst["M"], 16)
            lam = w ** 2
            mask_x = np.zeros(inst["n_free"]); mask_x[inst["ixf"]] = 1.0
            Mv = inst["M"] @ V[:, :9]
            xshare = np.sum(V[:, :9] * mask_x[:, None] * Mv, axis=0) / np.sum(V[:, :9] * Mv, axis=0)
            axial = [i + 1 for i in range(9) if xshare[i] > 0.5]
            rows.append(dict(group=group, xc=xc, d=d, n_free=int(inst["n_free"]), w=[float(x) for x in w],
                             gap_9_8=float(lam[8] / lam[7]), gap_4_3=float(lam[3] / lam[2]),
                             xshare=[float(x) for x in xshare], axial_modes=axial,
                             w1_drop=float((w_plain - w[0]) / w_plain), t=time.time() - t0))
            print(json.dumps({k: rows[-1][k] for k in ("group", "xc", "d", "n_free", "gap_9_8", "gap_4_3", "axial_modes", "w1_drop")}), flush=True)
    jdump(dict(w_plain=w_plain, rows=rows), "gap.json")


# ----------------------------------------------------------------------------- (e): sensitivity tables

def w2_table(xc, d):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = geom_setup(xc, d)
    if SMOKE:
        train_cases = train_cases[:2]
    lam = w ** 2
    A = p6.step_operator(inst, C, dt)
    luA = spla.splu(A)
    V8 = V[:, :8].copy()
    rng = np.random.default_rng(0)
    c = rng.standard_normal(8); c /= np.linalg.norm(c)
    z = V[:, 8:16] @ c
    lam_z = float(np.sum(c ** 2 * lam[8:16]))
    r = 1.01 ** 2 - 1.0
    base = p6.S_split(V8, inst, C, A, luA, train_cases, dt)
    th0 = p7.ritz_values(V8, inst)
    out = [dict(mode=0, S_over_T=base["S_over_T"]["base"], T=base["T"],
                ritz_freq_err=[float(x) for x in np.sqrt(th0) / w[:8] - 1])]
    for i in range(8):
        t2 = r * lam[i] / (lam_z - lam[i] - r * lam[i])
        t = np.sqrt(t2)
        Phi = V8.copy()
        Phi[:, i] = (V8[:, i] + t * z) / np.sqrt(1.0 + t2)
        th = p7.ritz_values(Phi, inst)
        s = p6.S_split(Phi, inst, C, A, luA, train_cases, dt)
        out.append(dict(mode=i + 1, t=float(t), ritz_freq_err=[float(x) for x in np.sqrt(th) / w[:8] - 1],
                        trace_excess=float(np.sum(th) - np.sum(th0)),
                        logdet_excess=float(np.sum(np.log(th)) - np.sum(np.log(th0))),
                        S_over_T=s["S_over_T"]["base"], dS_over_T=s["S_over_T"]["base"] - base["S_over_T"]["base"]))
        print(json.dumps(dict(xc=xc, d=d, **{k: out[-1][k] for k in ("mode", "trace_excess", "logdet_excess", "dS_over_T")})), flush=True)
    d1 = out[1]["dS_over_T"]
    for rec in out[1:]:
        rec["mode1_over_this"] = float(d1 / rec["dS_over_T"]) if rec["dS_over_T"] != 0 else None
    return dict(xc=xc, d=d, lam=[float(x) for x in lam], lam_z=lam_z, records=out)


def cmd_sens():
    geo = geometries()
    res = [w2_table(xc, d) for (xc, d) in (geo["G123"][:1] if SMOKE else geo["G123"])]
    jdump(res, "sens.json")


# ----------------------------------------------------------------------------- (f): family pilot

def cmd_fam(seed, iters):
    geo = geometries()
    tr_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["pilot_train"]]
    ho_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["pilot_held"]]
    tr = [fam_problem(i) for i in tr_insts]
    mon = []                                                      # monitoring only: finite element frequencies
    for inst, pr in zip(tr_insts + ho_insts, tr + [fam_problem(i) for i in ho_insts]):
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        mon.append((inst, w, jax.jit(lambda p, pr=pr: lb.orthonormal_basis(p, pr))))
    sizes = [int(tr[0].feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES]
    params = lb.init_mlp(jax.random.PRNGKey(seed), sizes)
    vgs = [jax.jit(jax.value_and_grad(lambda p, pr=pr: o4_loss(p, pr))) for pr in tr]
    update = adam_update()
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    log_every = max(1, 5 * LOG_EVERY)
    logs, perm = [], None
    t0 = time.time()
    for t in range(1, iters + 1):
        if (t - 1) % 2 == 0:
            perm = rng.permutation(len(tr))
            sub = perm[:4]
        else:
            sub = perm[4:]
        lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = fam_value_grad(params, vgs, sub)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % log_every == 0 or t == 1:
            rec = dict(t=t, loss=float(val), elapsed=time.time() - t0,
                       ritz=[p6.ritz_errors(np.asarray(bf(params)), inst, w, 8) for (inst, w, bf) in mon])
            logs.append(rec)
            print(json.dumps(dict(t=t, loss=rec["loss"], elapsed=rec["elapsed"],
                                  max_ritz_train=max(max(r) for r in rec["ritz"][:8]),
                                  max_ritz_held=max(max(r) for r in rec["ritz"][8:]))), flush=True)
    save("fam_pilot_s%d_%d.pkl" % (seed, iters), params, logs=logs, t_train=time.time() - t0, seed=seed, iters=iters,
         train=geo["pilot_train"], held=geo["pilot_held"])
    print("done fam %.0f s" % (time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- evaluation

def eval_basis(Phi, inst, w, V, C, dt, A, refs, eval_cases):
    rows = [rc.metrics(Phi, inst, C, case, dt, ref, w, A) for case, ref in zip(eval_cases, refs)]
    out = dict(ritz=p6.ritz_errors(Phi, inst, w, 8), sin2=sin2(Phi, inst, V))
    for q in ("e1_tip_disp", "e1_tip_acc", "e2_energy", "e3_median", "e3_max"):
        vals = [r[q] for r in rows]
        out[q] = dict(median=float(np.median(vals)), max=float(np.max(vals)), per_case=[float(v) for v in vals])
    return out


def interp_basis(src_inst, src_V, dst_inst):
    """Nearest-geometry control: source eigenvectors interpolated onto the destination P2 nodes."""
    from scipy.interpolate import LinearNDInterpolator
    yc = np.linspace(0.0, fc.H, 81)
    clamp_pts = np.stack([np.zeros_like(yc), yc], axis=1)        # clamped edge, zero displacement
    pts = np.vstack([src_inst["pts"], clamp_pts])
    ux = np.vstack([src_V[src_inst["ixf"], :], np.zeros((len(yc), src_V.shape[1]))])
    uy = np.vstack([src_V[src_inst["iyf"], :], np.zeros((len(yc), src_V.shape[1]))])
    fx, fy = LinearNDInterpolator(pts, ux), LinearNDInterpolator(pts, uy)
    q = dst_inst["pts"]
    Phi = np.zeros((dst_inst["n_free"], src_V.shape[1]))
    Phi[dst_inst["ixf"]] = fx(q); Phi[dst_inst["iyf"]] = fy(q)
    assert np.all(np.isfinite(Phi))
    return Phi


def cmd_evalfam(seed=0, iters=60000):
    geo = geometries()
    fn = [f for f in os.listdir(OUT) if f.startswith("fam_pilot_s%d_" % seed) and f.endswith(".pkl")]
    params, meta = p3.load_params(os.path.join(OUT, sorted(fn)[-1]))
    w_plain = float(p2.plain_cantilever_w1()[0])
    src = []
    for (xc, d) in geo["pilot_train"]:
        inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        w, V = fc.eig(inst["K"], inst["M"], 8)
        src.append(((xc, d), inst, V))
    res = dict(w_plain=w_plain, source_file=sorted(fn)[-1], geoms={})
    for group in ("pilot_train", "pilot_held"):
        for (xc, d) in geo[group]:
            t0 = time.time()
            inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = geom_setup(xc, d)
            if SMOKE:
                eval_cases = eval_cases[:2]
            pr = fam_problem(inst, a_r, b_r, dt)
            A = p6.step_operator(inst, C, dt)
            refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, N) for c in eval_cases]
            bases = {"net": np.asarray(lb.orthonormal_basis(params, pr)), "modal_m8": V[:, :8]}
            if group == "pilot_held":
                dist = [np.hypot((xc - s[0][0]) / 2.0, (d - s[0][1]) / 0.15) for s in src]
                j = int(np.argmin(dist))
                bases["nn_interp"] = interp_basis(src[j][1], src[j][2], inst)
            rec = dict(group=group, xc=xc, d=d, n_free=int(inst["n_free"]), w=[float(x) for x in w[:8]])
            for k, B in bases.items():
                rec[k] = eval_basis(B, inst, w, V, C, dt, A, refs, eval_cases)
            if group == "pilot_held":
                rec["nn_interp"]["source"] = list(src[j][0])
            w1_net = w[0] * (1.0 + rec["net"]["ritz"][0])
            rec["w1_drop_fe"] = float((w_plain - w[0]) / w_plain)
            rec["w1_drop_reproduced"] = float((w_plain - w1_net) / (w_plain - w[0]))
            if group == "pilot_held":
                w1_nn = w[0] * (1.0 + rec["nn_interp"]["ritz"][0])
                rec["w1_drop_reproduced_nn_interp"] = float((w_plain - w1_nn) / (w_plain - w[0]))
            rec["t_eval"] = time.time() - t0
            res["geoms"]["%s_%.4f_%.4f" % (group, xc, d)] = rec
            print(json.dumps(dict(group=group, xc=xc, d=d, net_ritz_max=max(rec["net"]["ritz"]),
                                  net_ritz_5=rec["net"]["ritz"][:5], drop_rep=rec["w1_drop_reproduced"],
                                  net_e2=rec["net"]["e2_energy"]["median"], modal_e2=rec["modal_m8"]["e2_energy"]["median"],
                                  nn=(max(rec["nn_interp"]["ritz"]) if "nn_interp" in rec else None))), flush=True)
    jdump(res, "evalfam.json")


def cmd_evalg0():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    if SMOKE:
        eval_cases, train_cases = eval_cases[:2], train_cases[:2]
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    A = p6.step_operator(inst, C, dt)
    luA = spla.splu(A)
    prs = {v: p6.problem(inst, a_r, b_r, dt, M_MODES, v) for v in ("hard_angle", "hard")}
    refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, N) for c in eval_cases]
    runs = {"modal_m8": dict(Phi=V[:, :8])}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("g0_") and fn.endswith(".pkl"):
            params, d = p3.load_params(os.path.join(OUT, fn))
            pr = prs[d["variant"]]
            runs[fn[:-4]] = dict(Phi=lb.basis_numpy(params, pr), params=params, pr=pr, meta=d)
    for tag, fn in (("set6_D1_o1_50k", "posthoc6_D_D1.pkl"), ("set6_D2_o1_50k", "posthoc6_D_D2.pkl"),
                    ("set6_F1_o4_20k", "posthoc6_F1.pkl")):
        params, d = p3.load_params(os.path.join(OLD_OUT, fn))
        runs[tag] = dict(Phi=lb.basis_numpy(params, prs["hard_angle"]), params=params, pr=prs["hard_angle"], meta=d)
    res = {}
    for k, r in runs.items():
        t0 = time.time()
        Phi = r["Phi"]
        s = eval_basis(Phi, inst, w, V, C, dt, A, refs, eval_cases)
        s["S_over_T"] = p6.S_split(Phi, inst, C, A, luA, train_cases, dt)["S_over_T"]["base"]
        s["objectives"] = dict(o1=float(p3.loss_from_basis(jnp.asarray(Phi), prs["hard_angle"], Wj, Gj)), **objectives_np(Phi, inst))
        if "params" in r:
            s["conditioning"] = p6.conditioning(r["params"], r["pr"], inst)
            m = r["meta"]
            s["t_train_s"] = m.get("t_train")
            s["logs"] = m.get("logs")
            for key in ("final", "nit", "nfev", "message", "t_lbfgs"):
                if key in m:
                    s[key] = m[key]
        s["t_eval"] = time.time() - t0
        res[k] = s
        print(json.dumps(dict(run=k, ritz=s["ritz"], e1_acc=s["e1_tip_acc"]["median"], e2=s["e2_energy"]["median"],
                              S_over_T=s["S_over_T"])), flush=True)
    jdump(res, "evalg0.json")


# ----------------------------------------------------------------------------- (h): cost table

def _median_time(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cmd_cost():
    rows = []
    for h in ((0.2,) if SMOKE else (0.2, 0.14, 0.1)):
        rec = dict(h_max=h)
        rec["t_mesh_assembly"] = _median_time(lambda: fc.build(xc=2.0, d=0.3, rho=0.05, h_max=h), 3 if h < 0.15 else 5)
        inst = fc.build(xc=2.0, d=0.3, rho=0.05, h_max=h)
        rec["n_free"] = int(inst["n_free"])
        rec["t_eig16"] = _median_time(lambda: fc.eig(inst["K"], inst["M"], 16))
        w, V = fc.eig(inst["K"], inst["M"], 16)
        a_r, b_r, dt = fc.time_setup(w[0], w[2])
        C = (a_r * inst["M"] + b_r * inst["K"]).tocsc()
        rec["t_inputs"] = _median_time(lambda: (lb.features(inst["pts"], inst["params"]), p4.angle_feats(inst["pts"], inst["params"])))
        pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
        params = lb.init_mlp(jax.random.PRNGKey(0), [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES])
        bf = jax.jit(lambda p: lb.orthonormal_basis(p, pr))
        t0 = time.perf_counter(); jax.block_until_ready(bf(params)); rec["t_net_compile"] = time.perf_counter() - t0
        rec["t_net_eval"] = _median_time(lambda: jax.block_until_ready(bf(params)))
        Phi = np.asarray(bf(params))
        rec["t_reduced_operators"] = _median_time(lambda: (Phi.T @ (inst["K"] @ Phi), Phi.T @ (inst["M"] @ Phi), Phi.T @ inst["P"]))
        a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
        Kr = Phi.T @ (inst["K"] @ Phi); Mr = Phi.T @ (inst["M"] @ Phi); Cr = a_r * Mr + b_r * Kr
        cf = sla.cho_factor(Kr + a0 * Mr + a1 * Cr)
        c = np.zeros(M_MODES); cd = np.zeros(M_MODES); cdd = np.zeros(M_MODES); fr = np.ones(M_MODES)

        def red_steps():
            for _ in range(1000):
                b = fr + Mr @ (a0 * c + a2 * cd + a3 * cdd) + Cr @ (a1 * c + a4 * cd + a5 * cdd)
                sla.cho_solve(cf, b)
        rec["t_reduced_step"] = _median_time(red_steps) / 1000
        Af = (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()
        rec["t_full_factorisation"] = _median_time(lambda: spla.splu(Af))
        lu = spla.splu(Af)
        u = np.zeros(inst["n_free"]); v = np.zeros_like(u); acc = np.zeros_like(u); f = inst["P"][:, 0]

        def full_steps():
            for _ in range(20):
                rhs = f + inst["M"] @ (a0 * u + a2 * v + a3 * acc) + C @ (a1 * u + a4 * v + a5 * acc)
                lu.solve(rhs)
        rec["t_full_step"] = _median_time(full_steps) / 20
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    jdump(rows, "cost.json")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "train":
        cmd_train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif c == "fam":
        cmd_fam(int(sys.argv[2]), int(sys.argv[3]))
    else:
        dict(checks=cmd_checks, long=cmd_long, gap=cmd_gap, sens=cmd_sens, evalfam=cmd_evalfam,
             evalg0=cmd_evalg0, cost=cmd_cost)[c]()
