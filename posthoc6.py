"""Post hoc diagnostics, sixth set: re-examination (see POSTHOC6.md).

  python -I posthoc6.py d TAG LR0 [ITERS]    # D: long training from scratch (exact clamp, notch-angle inputs, m = 8, seed 0)
  python -I posthoc6.py a                    # A1 to A3: code checks
  python -I posthoc6.py e                    # E1 to E3 for fixed and earlier bases
  python -I posthoc6.py b1                   # B1: capacity fit to the eight lowest eigenvectors
  python -I posthoc6.py b2                   # B2: one field, planned inputs, exact clamp, supervised, 30,000 iterations
  python -I posthoc6.py c TAG LR0 LR1        # C: warm start from B1 with the planned label-free objective
  python -I posthoc6.py e4                   # E4 (addendum): what the remaining per-step error consists of
  python -I posthoc6.py f1                   # F1 (second addendum): label-free log-determinant objective
  python -I posthoc6.py f2                   # F2 (second addendum): per-step objective from F1
  python -I posthoc6.py eval                 # E1 to E4, objective, S / T and conditioning for B1, C, D, F
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc3 as p3
import posthoc4 as p4
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

OUT = os.path.join(HERE, "out")
N = rc.N_STEPS

# Raw fields with an optional exact clamp, chosen per problem through pr.clamp (planned problems have none).
_raw_planned = lb.raw_basis


def _raw_dispatch(params, pr):
    Phi = _raw_planned(params, pr)
    clamp = getattr(pr, "clamp", None)
    return Phi if clamp is None else Phi * clamp


lb.raw_basis = _raw_dispatch          # lb.orthonormal_basis and lb.loss_fn look the name up at call time


def problem(inst, a_r, b_r, dt, m, variant):
    """variant: 'planned' (as PLAN.md), 'hard' (exact clamp), 'hard_angle' (exact clamp and notch-angle inputs, as F4b)."""
    pr = lb.Problem(inst, a_r, b_r, dt, m)
    if variant in ("hard", "hard_angle"):
        pr.clamp = jnp.asarray(p3.dof_x(inst) / fc.L)[:, None]
    if variant == "hard_angle":
        pr.feats = jnp.concatenate([pr.feats, jnp.asarray(p4.angle_feats(inst["pts"], inst["params"]))], axis=1)
    return pr


def ritz_errors(Phi, inst, w, k):
    K, M = inst["K"], inst["M"]
    Kr = Phi.T @ (K @ Phi); Mr = Phi.T @ (M @ Phi)
    lam = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), eigvals_only=True)
    r = np.sqrt(np.maximum(lam, 0.0))
    kk = min(k, len(r))
    return [float(x) for x in (r[:kk] / w[:kk] - 1)]


def save(fn, params, **kw):
    with open(os.path.join(OUT, fn), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], **kw), f)


def train_logged(loss, pr, m, seed=0, iters=3000, lr0=1e-3, lr1=1e-5, log_every=500, params0=None, callback=None):
    """Same optimiser as posthoc.train_generic (Adam, cosine decay), with optional start point and logging callback."""
    sizes = [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * m]
    params = params0 if params0 is not None else lb.init_mlp(jax.random.PRNGKey(seed), sizes)
    vg = jax.jit(jax.value_and_grad(loss))
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def update(params, mo, ve, g, t, lr):
        mo = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, mo, g)
        ve = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, ve, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), mo)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), ve)
        return jax.tree_util.tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh), mo, ve

    hist, logs = [], []
    for t in range(1, iters + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = vg(params)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % log_every == 0 or t == 1:
            rec = dict(t=t, loss=float(val))
            if callback is not None:
                rec.update(callback(params))
            hist.append((t, float(val))); logs.append(rec)
            print(json.dumps(rec), flush=True)
    return params, hist, logs


# ----------------------------------------------------------------------------- full-space quantities (numpy)

def step_operator(inst, C, dt):
    a0, a1 = fc.newmark_consts(dt)[:2]
    return (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()


def full_history(Phi, inst, C, case, dt):
    """Reduced rollout of Phi lifted to full space: U[:, k] = Phi c_k (k = 0..N); F[:, k] = effective load of step k."""
    Cs, Ds, As, _ = rc.reduced_rollout(Phi, inst, C, case, dt)
    U = Phi @ Cs.T; Vv = Phi @ Ds.T; Aa = Phi @ As.T
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    g = fc.load_history(case, dt, N)
    F = (np.outer(inst["P"] @ np.asarray(case["w"], float), g[1:])
         + inst["M"] @ (a0 * U[:, :-1] + a2 * Vv[:, :-1] + a3 * Aa[:, :-1])
         + C @ (a1 * U[:, :-1] + a4 * Vv[:, :-1] + a5 * Aa[:, :-1]))
    for k in (0, 100, N - 1):                                   # vectorised loads against the reference routine
        Fk = fc.effective_load(inst["M"], C, inst["P"], case, dt, N, U[:, k], Vv[:, k], Aa[:, k], k)
        assert np.linalg.norm(F[:, k] - Fk) <= 1e-12 * max(np.linalg.norm(Fk), 1e-300)
    return U, F


def objective_fullspace(Phi, inst, C, A, cases, dt):
    tot = 0.0
    for case in cases:
        U, F = full_history(Phi, inst, C, case, dt)
        Uh = U[:, 1:]
        tot += float(np.sum(0.5 * np.einsum("ij,ij->j", Uh, A @ Uh) - np.einsum("ij,ij->j", F, Uh)))
    return tot / (len(cases) * N)


def a_orthonormal(S_, A):
    G = S_.T @ (A @ S_)
    lam, Q = np.linalg.eigh(0.5 * (G + G.T))
    keep = lam > 1e-13 * lam.max()
    return S_ @ (Q[:, keep] / np.sqrt(lam[keep]))


def S_split(Phi, inst, C, A, luA, cases, dt, extra=None):
    """T and S for span(Phi) along its own rollout; S for span([Phi, Z]) with the same targets, for each Z in extra."""
    spans = {"base": a_orthonormal(Phi, A)}
    for k, Z in (extra or {}).items():
        spans[k] = a_orthonormal(np.hstack([Phi, Z]), A)
    T, S, S_direct = 0.0, {k: 0.0 for k in spans}, 0.0
    for case in cases:
        U, F = full_history(Phi, inst, C, case, dt)
        Ustar = luA.solve(F)
        tgt = np.einsum("ij,ij->j", F, Ustar)                   # ||u*_k||_A^2 (A u* = F)
        T += float(tgt.sum())
        E = U[:, 1:] - Ustar                                    # Galerkin step minus exact step
        S_direct += float(np.sum(np.einsum("ij,ij->j", E, A @ E)))
        for k, Psi in spans.items():
            PF = Psi.T @ F
            S[k] += float(np.sum(tgt - np.einsum("ij,ij->j", PF, PF)))
    return dict(T=T, S=S, S_base_direct=S_direct, S_over_T={k: v / T for k, v in S.items()})


def load_responses(inst, luA):
    return luA.solve(np.asarray(inst["P"]))                       # columns: A^-1 P_ty, A^-1 P_tx, A^-1 P_top


def load_residuals(Phi, inst, A, Z):
    Psi = a_orthonormal(Phi, A)
    P = np.asarray(inst["P"])
    out = []
    for j in range(Z.shape[1]):
        nz = float(P[:, j] @ Z[:, j])                           # ||Z_j||_A^2
        out.append(float(1.0 - np.sum((Psi.T @ P[:, j]) ** 2) / nz))
    return out


def conditioning(params, pr, inst):
    Phi = np.asarray(lb.raw_basis(params, pr))
    G = Phi.T @ (inst["M"] @ Phi)
    lam = np.linalg.eigvalsh(0.5 * (G + G.T))
    return dict(eig_min=float(lam[0]), eig_max=float(lam[-1]), cond=float(lam[-1] / lam[0]))


# ----------------------------------------------------------------------------- A: code checks

def frozen_functional(params0, pr, W, G):
    """Mean per-step potential of the Galerkin steps of span(Phi(params)), with the history frozen at the rollout of params0."""
    Phi0 = lb.orthonormal_basis(params0, pr)
    Kr, Mr, Cr, Ar, Pr = lb.reduced_ops(Phi0, pr)
    a0, a1, a2, a3, a4, a5 = pr.nc
    cf0 = jax.scipy.linalg.cho_factor(0.5 * (Ar + Ar.T))
    fr0 = W @ Pr.T
    acc0 = jnp.linalg.solve(Mr, (fr0 * G[:, :1]).T).T
    B, m = W.shape[0], pr.m

    def step(carry, g_next):
        c, cd, cdd = carry
        b = fr0 * g_next[:, None] + (a0 * c + a2 * cd + a3 * cdd) @ Mr.T + (a1 * c + a4 * cd + a5 * cdd) @ Cr.T
        cn = jax.scipy.linalg.cho_solve(cf0, b.T).T
        cdd_n = a0 * (cn - c) - a2 * cd - a3 * cdd
        cd_n = cd + pr.dt * ((1 - fc.GAMMA) * cdd + fc.GAMMA * cdd_n)
        return (cn, cd_n, cdd_n), (c, cd, cdd)

    _, (Cst, Dst, Ast) = jax.lax.scan(step, (jnp.zeros((B, m)), jnp.zeros((B, m)), acc0), G[:, 1:].T)
    Cst, Dst, Ast, Phi0 = (jax.lax.stop_gradient(x) for x in (Cst, Dst, Ast, Phi0))
    hM = a0 * Cst + a2 * Dst + a3 * Ast                              # (N, B, m)
    hC = a1 * Cst + a4 * Dst + a5 * Ast
    gn = G[:, 1:].T                                                  # (N, B)

    def Lf(params):
        Phi = lb.orthonormal_basis(params, pr)
        Kr, Mr, Cr, Ar, Pr = lb.reduced_ops(Phi, pr)
        Mh = Phi.T @ (pr.M @ Phi0); Kh = Phi.T @ (pr.K @ Phi0)
        Ch = pr.a_r * Mh + pr.b_r * Kh
        b = (W @ Pr.T)[None] * gn[:, :, None] + hM @ Mh.T + hC @ Ch.T
        bb = b.reshape(-1, m)
        cf = jax.scipy.linalg.cho_factor(0.5 * (Ar + Ar.T))
        cn = jax.scipy.linalg.cho_solve(cf, bb.T).T
        return jnp.mean(0.5 * jnp.sum(cn * (cn @ Ar.T), axis=1) - jnp.sum(bb * cn, axis=1))

    return Lf


def grad_check(p0, pr, W, G):
    l0, g_semi = jax.value_and_grad(lambda p: lb.loss_fn(p, pr, W, G))(p0)
    Lf = frozen_functional(p0, pr, W, G)
    lf0, g_frozen = jax.value_and_grad(Lf)(p0)
    fs, _ = ravel_pytree(g_semi); ff, _ = ravel_pytree(g_frozen)
    fs, ff = np.asarray(fs), np.asarray(ff)
    out = dict(loss=float(l0), value_rel=abs(float(l0) - float(lf0)) / abs(float(l0)),
               grad_rel=float(np.linalg.norm(fs - ff) / np.linalg.norm(fs)), grad_norm=float(np.linalg.norm(fs)))
    x0, unravel = ravel_pytree(p0)
    Lf_flat = jax.jit(lambda x: Lf(unravel(x)))
    L_flat = jax.jit(lambda x: lb.loss_fn(unravel(x), pr, W, G))
    rng = np.random.default_rng(3)
    ghat = ff / np.linalg.norm(ff)
    dirs = [ghat]
    for _ in range(3):
        r = rng.standard_normal(len(ghat)); r /= np.linalg.norm(r)
        d = ghat + r; dirs.append(d / np.linalg.norm(d))
    rows = []
    for d in dirs:
        dj = jnp.asarray(d)
        ad = float(ff @ d)
        fd, fd_true = {}, {}
        for h in (1e-2, 1e-3, 1e-4, 1e-5):
            fd[h] = float((Lf_flat(x0 + h * dj) - Lf_flat(x0 - h * dj)) / (2 * h))
            fd_true[h] = float((L_flat(x0 + h * dj) - L_flat(x0 - h * dj)) / (2 * h))
        best = min(abs(v - ad) / abs(ad) for v in fd.values())
        rows.append(dict(directional_derivative=ad, fd_frozen={str(h): v for h, v in fd.items()}, best_rel=best,
                         fd_loss_history_moving={str(h): v for h, v in fd_true.items()}))
    out["directions"] = rows
    out["best_rel_worst_direction"] = max(r["best_rel"] for r in rows)
    return out


def cmd_a():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    A = step_operator(inst, C, dt)
    res = {}
    for name, fn, variant in (("planned_m8_s0", "params_m8_s0.pkl", "planned"), ("F4b", "posthoc4_F4b_m8.pkl", "hard_angle")):
        pr = problem(inst, a_r, b_r, dt, 8, variant)
        params, _ = p3.load_params(os.path.join(OUT, fn))
        l_jax = float(lb.loss_fn(params, pr, Wj, Gj))
        l_np = objective_fullspace(lb.basis_numpy(params, pr), inst, C, A, train_cases, dt)
        res["A1_" + name] = dict(jax=l_jax, fullspace=l_np, rel_diff=abs(l_jax - l_np) / abs(l_np))
        print("A1", name, json.dumps(res["A1_" + name]), flush=True)
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    p_init = lb.init_mlp(jax.random.PRNGKey(0), [int(pr.feats.shape[1]), 64, 64, 64, 64, 16])
    p_f4b, _ = p3.load_params(os.path.join(OUT, "posthoc4_F4b_m8.pkl"))
    for name, p0 in (("init", p_init), ("F4b", p_f4b)):
        r = grad_check(p0, pr, Wj, Gj)
        res["A2_" + name] = r
        print("A2", name, json.dumps({k: r[k] for k in ("value_rel", "grad_rel", "grad_norm", "best_rel_worst_direction")}), flush=True)
    a3 = {}
    for m in (8, 16):
        for s in (0, 1):
            pr = problem(inst, a_r, b_r, dt, m, "planned")
            params, _ = p3.load_params(os.path.join(OUT, "params_m%d_s%d.pkl" % (m, s)))
            a3["planned_m%d_s%d" % (m, s)] = conditioning(params, pr, inst)
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    for name, fn in (("F4b", "posthoc4_F4b_m8.pkl"), ("F5b", "posthoc5_F5b_m8.pkl")):
        params, _ = p3.load_params(os.path.join(OUT, fn))
        a3[name] = conditioning(params, pr, inst)
    res["A3"] = a3
    print("A3", json.dumps(a3), flush=True)
    with open(os.path.join(OUT, "posthoc6_a.json"), "w") as f:
        json.dump(res, f, indent=1)


# ----------------------------------------------------------------------------- E: what the training gradient rewards

def cmd_e():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    A = step_operator(inst, C, dt)
    luA = spla.splu(A)
    Z = load_responses(inst, luA)
    res = {}
    pr_h = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    pr_p = problem(inst, a_r, b_r, dt, 8, "planned")
    bases = {"modal_m8": V[:, :8]}
    params, _ = p3.load_params(os.path.join(OUT, "params_m8_s0.pkl")); bases["planned_m8_s0"] = lb.basis_numpy(params, pr_p)
    params, _ = p3.load_params(os.path.join(OUT, "posthoc4_F4b_m8.pkl")); bases["F4b"] = lb.basis_numpy(params, pr_h)
    params, _ = p3.load_params(os.path.join(OUT, "posthoc5_F5b_m8.pkl")); bases["F5b"] = lb.basis_numpy(params, pr_h)
    t0 = time.time()
    res["E1_F5b"] = S_split(bases["F5b"], inst, C, A, luA, train_cases, dt,
                            extra={"plus_modes2to5": V[:, 1:5], "plus_load3": Z, "plus_both": np.hstack([V[:, 1:5], Z])})
    print("E1 F5b", json.dumps(res["E1_F5b"]), "%.0f s" % (time.time() - t0), flush=True)
    res["E1_modal_m8"] = S_split(bases["modal_m8"], inst, C, A, luA, train_cases, dt, extra={"plus_load3": Z})
    print("E1 modal_m8", json.dumps(res["E1_modal_m8"]), flush=True)
    res["E2"] = {k: load_residuals(bases[k], inst, A, Z) for k in ("planned_m8_s0", "F4b", "F5b", "modal_m8")}
    print("E2", json.dumps(res["E2"]), flush=True)
    res["E3"] = {"modal_m8": res["E1_modal_m8"]["S_over_T"]["base"], "F5b": res["E1_F5b"]["S_over_T"]["base"]}
    for k in ("planned_m8_s0", "F4b"):
        res["E3"][k] = S_split(bases[k], inst, C, A, luA, train_cases, dt)["S_over_T"]["base"]
    print("E3", json.dumps(res["E3"]), flush=True)
    with open(os.path.join(OUT, "posthoc6_e.json"), "w") as f:
        json.dump(res, f, indent=1)


# ----------------------------------------------------------------------------- B, C, D: training runs

def cmd_b1(iters=20000):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    KV8 = jnp.asarray(inst["K"] @ V[:, :8]); vKv = jnp.asarray(w[:8] ** 2)

    def cap_loss(p):
        Phi = lb.orthonormal_basis(p, pr)
        Kr = Phi.T @ (pr.K @ Phi)
        Bm = Phi.T @ KV8
        sol = jnp.linalg.solve(0.5 * (Kr + Kr.T), Bm)
        return jnp.mean(1.0 - jnp.sum(Bm * sol, axis=0) / vKv)

    t0 = time.time()
    params, hist, logs = train_logged(cap_loss, pr, 8, seed=0, iters=iters, lr0=1e-3, lr1=1e-5, log_every=2000,
                                      callback=lambda p: dict(ritz=ritz_errors(lb.basis_numpy(p, pr), inst, w, 8)))
    save("posthoc6_B1.pkl", params, hist=hist, logs=logs, t_train=time.time() - t0)


def cmd_b2(iters=30000):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr1 = problem(inst, a_r, b_r, dt, 1, "hard")
    v1 = jnp.asarray(V[:, 0])
    Kv1 = (pr1.K @ v1[:, None])[:, 0]
    v1Kv1 = float(v1 @ Kv1)

    def sup_loss(p):
        phi = lb.raw_basis(p, pr1)[:, 0]
        Kphi = (pr1.K @ phi[:, None])[:, 0]
        return 1.0 - (phi @ Kv1) ** 2 / ((phi @ Kphi) * v1Kv1)

    def cb(p):
        phi = np.asarray(lb.raw_basis(p, pr1))[:, 0]
        return dict(ritz1=float(np.sqrt(phi @ (inst["K"] @ phi) / (phi @ (inst["M"] @ phi))) / w[0] - 1))

    t0 = time.time()
    params, hist, logs = train_logged(sup_loss, pr1, 1, seed=0, iters=iters, lr0=1e-3, lr1=1e-5, log_every=5000, callback=cb)
    save("posthoc6_B2.pkl", params, hist=hist, logs=logs, t_train=time.time() - t0)


def cmd_c(tag, lr0, lr1, iters=3000):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    params0, _ = p3.load_params(os.path.join(OUT, "posthoc6_B1.pkl"))
    t0 = time.time()
    params, hist, logs = train_logged(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, 8, iters=iters, lr0=lr0, lr1=lr1,
                                      log_every=500, params0=params0,
                                      callback=lambda p: dict(ritz=ritz_errors(lb.basis_numpy(p, pr), inst, w, 5)))
    save("posthoc6_C_%s.pkl" % tag, params, hist=hist, logs=logs, t_train=time.time() - t0, lr0=lr0, lr1=lr1)


def cmd_d(tag, lr0, iters=50000):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    t0 = time.time()
    params, hist, logs = train_logged(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, 8, seed=0, iters=iters, lr0=lr0, lr1=1e-6,
                                      log_every=min(5000, iters),
                                      callback=lambda p: dict(ritz=ritz_errors(lb.basis_numpy(p, pr), inst, w, 5)))
    save("posthoc6_D_%s.pkl" % tag, params, hist=hist, logs=logs, t_train=time.time() - t0, lr0=lr0, iters=iters)
    print("done %.0f s" % (time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- evaluation of the new runs

def cmd_eval():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    A = step_operator(inst, C, dt)
    luA = spla.splu(A)
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    runs = {"modal_m8": (None, V[:, :8])}
    for tag, fn in (("B1", "posthoc6_B1.pkl"), ("C1", "posthoc6_C_C1.pkl"), ("C2", "posthoc6_C_C2.pkl"),
                    ("D1", "posthoc6_D_D1.pkl"), ("D2", "posthoc6_D_D2.pkl"),
                    ("F1", "posthoc6_F1.pkl"), ("F2", "posthoc6_F2.pkl")):
        path = os.path.join(OUT, fn)
        if os.path.exists(path):
            params, d = p3.load_params(path)
            runs[tag] = (d, lb.basis_numpy(params, pr), params)
    rows = {k: [] for k in runs}
    names = [c.get("name", "case%d" % i) for i, c in enumerate(eval_cases)]
    for case in eval_cases:                                       # references recomputed case by case (as in run_check prep)
        ref = fc.newmark_full(inst["K"], inst["M"], C, inst["P"], case, dt, N)
        for k, v in runs.items():
            rows[k].append(rc.metrics(v[1], inst, C, case, dt, ref, w, A))
    keys = ["e1_tip_disp", "e1_tip_acc", "e2_energy", "e3_median", "e3_p90", "e3_max"]
    res = {}
    for k, v in runs.items():
        s = {q: dict(median=float(np.median([r[q] for r in rows[k]])), max=float(np.max([r[q] for r in rows[k]]))) for q in keys}
        s["e4_rel_freq"] = rows[k][0]["e4_rel_freq"]
        s["ritz_m8"] = ritz_errors(v[1], inst, w, 8)
        s["objective_value"] = float(p3.loss_from_basis(jnp.asarray(v[1]), pr, Wj, Gj))
        s["S_over_T"] = S_split(v[1], inst, C, A, luA, train_cases, dt)["S_over_T"]["base"]
        s["per_case_e2"] = dict(zip(names, [r["e2_energy"] for r in rows[k]]))
        if v[0] is not None:
            s["conditioning"] = conditioning(v[2], pr, inst)
            s["t_train_s"] = v[0]["t_train"]; s["logs"] = v[0]["logs"]
        res[k] = s
        print("eval", k, json.dumps({q: s[q] for q in keys + ["ritz_m8", "objective_value", "S_over_T"]}), flush=True)
    path = os.path.join(OUT, "posthoc6_B2.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            res["B2"] = dict(logs=pickle.load(f)["logs"])
    with open(os.path.join(OUT, "posthoc6_eval.json"), "w") as f:
        json.dump(res, f, indent=1)




def cmd_e4():
    """E4 (POSTHOC6.md addendum): what the remaining per-step error of the trained bases consists of."""
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    A = step_operator(inst, C, dt)
    luA = spla.splu(A)
    wf, Vf = fc.eig(inst["K"], inst["M"], 16)
    pr_h = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    pr_p = problem(inst, a_r, b_r, dt, 8, "planned")
    res = {}
    for name, fn, pr in (("F5b", "posthoc5_F5b_m8.pkl", pr_h), ("F4b", "posthoc4_F4b_m8.pkl", pr_h),
                         ("planned_m8_s0", "params_m8_s0.pkl", pr_p)):
        params, _ = p3.load_params(os.path.join(OUT, fn))
        r = S_split(lb.basis_numpy(params, pr), inst, C, A, luA, train_cases, dt,
                    extra={"plus_v1": Vf[:, :1], "plus_v1_to_v8": Vf[:, :8], "plus_v1_to_v16": Vf[:, :16]})
        r["fraction_removed"] = {k: 1.0 - v / r["S"]["base"] for k, v in r["S"].items() if k != "base"}
        res[name] = r
        print("E4", name, json.dumps(r["fraction_removed"]), flush=True)
    with open(os.path.join(OUT, "posthoc6_e4.json"), "w") as f:
        json.dump(res, f, indent=1)


def cmd_f1(iters=20000):
    """F1 (second addendum): label-free log-determinant Rayleigh-Ritz objective, m = 8."""
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")

    def logdet_loss(p):
        Phi = lb.orthonormal_basis(p, pr)                        # M-orthonormal, so log det(Phi^T M Phi) = 0
        Kr = Phi.T @ (pr.K @ Phi)
        Lc = jnp.linalg.cholesky(0.5 * (Kr + Kr.T))
        return 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))

    t0 = time.time()
    params, hist, logs = train_logged(logdet_loss, pr, 8, seed=0, iters=iters, lr0=1e-3, lr1=1e-5, log_every=2000,
                                      callback=lambda p: dict(ritz=ritz_errors(lb.basis_numpy(p, pr), inst, w, 8)))
    save("posthoc6_F1.pkl", params, hist=hist, logs=logs, t_train=time.time() - t0)


def cmd_f2(iters=3000):
    """F2 (second addendum): the planned per-step objective from F1, small learning rate."""
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = problem(inst, a_r, b_r, dt, 8, "hard_angle")
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    params0, _ = p3.load_params(os.path.join(OUT, "posthoc6_F1.pkl"))
    t0 = time.time()
    params, hist, logs = train_logged(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, 8, iters=iters, lr0=1e-4, lr1=1e-6,
                                      log_every=500, params0=params0,
                                      callback=lambda p: dict(ritz=ritz_errors(lb.basis_numpy(p, pr), inst, w, 5)))
    save("posthoc6_F2.pkl", params, hist=hist, logs=logs, t_train=time.time() - t0)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "d":
        cmd_d(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 50000)
    elif cmd == "c":
        cmd_c(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    else:
        dict(a=cmd_a, e=cmd_e, b1=cmd_b1, b2=cmd_b2, e4=cmd_e4, f1=cmd_f1, f2=cmd_f2, eval=cmd_eval)[cmd]()
