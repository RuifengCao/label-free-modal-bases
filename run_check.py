"""Pipeline check driver (see PLAN.md).

  python -I run_check.py prep            # geometry, eigen, load cases, references, identity checks I1 and I2 (random basis)
  python -I run_check.py train M SEED    # one training run; saves params_mM_sSEED.npz
  python -I run_check.py eval            # E1 to E5 for trained bases and baselines; I2 and I3 on trained bases; results.json
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc

N_STEPS = 640
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- shared set-up

def setup():
    inst = fc.build(xc=2.0, d=0.3, rho=0.05, h_max=0.2)
    w, V = fc.eig(inst["K"], inst["M"], 16)
    a_r, b_r, dt = fc.time_setup(w[0], w[2])
    C = (a_r * inst["M"] + b_r * inst["K"]).tocsc()
    T1, T2, T3 = 2 * np.pi / w[0], 2 * np.pi / w[1], 2 * np.pi / w[2]

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


def W_G(cases, dt):
    W = np.array([c["w"] for c in cases])
    G = np.stack([fc.load_history(c, dt, N_STEPS) for c in cases])
    return W, G


# ----------------------------------------------------------------------------- numpy reduced model

def reduced_rollout(Phi, inst, C, case, dt):
    K, M, P = inst["K"], inst["M"], inst["P"]
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    Kr = Phi.T @ (K @ Phi); Mr = Phi.T @ (M @ Phi); Cr = Phi.T @ (C @ Phi)
    Ar = Kr + a0 * Mr + a1 * Cr
    cf = sla.cho_factor(0.5 * (Ar + Ar.T))
    fr = Phi.T @ (P @ np.asarray(case["w"], float))
    g = fc.load_history(case, dt, N_STEPS)
    m = Phi.shape[1]
    c = np.zeros(m); cd = np.zeros(m); cdd = np.linalg.solve(Mr, fr * g[0])
    Cs = np.zeros((N_STEPS + 1, m)); Ds = np.zeros_like(Cs); As = np.zeros_like(Cs)
    Cs[0], Ds[0], As[0] = c, cd, cdd
    for k in range(N_STEPS):
        b = fr * g[k + 1] + Mr @ (a0 * c + a2 * cd + a3 * cdd) + Cr @ (a1 * c + a4 * cd + a5 * cdd)
        cn = sla.cho_solve(cf, b)
        cdd_n = a0 * (cn - c) - a2 * cd - a3 * cdd
        cd_n = cd + dt * ((1 - fc.GAMMA) * cdd + fc.GAMMA * cdd_n)
        c, cd, cdd = cn, cd_n, cdd_n
        Cs[k + 1], Ds[k + 1], As[k + 1] = c, cd, cdd
    return Cs, Ds, As, (Kr, Mr, Cr, Ar)


def metrics(Phi, inst, C, case, dt, ref, w_fe, A):
    """E1 to E4 for one basis and one evaluation case. ref = (U, V, Acc) full reference."""
    U, Vf, Acc = ref
    Cs, Ds, As, (Kr, Mr, Cr, Ar) = reduced_rollout(Phi, inst, C, case, dt)
    tip = inst["tip_dof"]
    u_tip = Cs @ Phi[tip]; a_tip = As @ Phi[tip]
    e1_u = np.linalg.norm(u_tip - U[:, tip]) / np.linalg.norm(U[:, tip])
    e1_a = np.linalg.norm(a_tip - Acc[:, tip]) / np.linalg.norm(Acc[:, tip])
    E = (Phi @ Cs.T) - U.T                                          # (n, N+1)
    K = inst["K"]
    e2 = np.sqrt(np.sum(E * (K @ E)) / np.sum(U.T * (K @ U.T)))
    # E3: A-norm distance from the subspace to each reference step U_{n+1}
    Ut = U[1:].T                                                    # (n, N)
    AU = A @ Ut
    PAP = Phi.T @ (A @ Phi)
    cst = np.linalg.solve(PAP, Phi.T @ AU)
    R = Ut - Phi @ cst
    num = np.sqrt(np.maximum(np.sum(R * (A @ R), axis=0), 0.0))
    den = np.sqrt(np.sum(Ut * AU, axis=0))
    rel3 = num / den
    # E4: Ritz frequencies
    ritz = np.sqrt(np.maximum(sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), eigvals_only=True), 0.0))
    e4 = (ritz[:5] - w_fe[:5]) / w_fe[:5]
    return dict(e1_tip_disp=float(e1_u), e1_tip_acc=float(e1_a), e2_energy=float(e2),
                e3_median=float(np.median(rel3)), e3_p90=float(np.quantile(rel3, 0.9)), e3_max=float(np.max(rel3)),
                e4_rel_freq=[float(x) for x in e4])


def pod_basis(inst, refs, m, stride=4):
    S = np.concatenate([U[1::stride].T for (U, _, _) in refs], axis=1)   # (n, n_snap)
    MS = inst["M"] @ S
    Cm = S.T @ MS
    lam, Q = np.linalg.eigh(0.5 * (Cm + Cm.T))
    idx = np.argsort(lam)[::-1][:m]
    Phi = S @ Q[:, idx] / np.sqrt(lam[idx])
    return Phi


# ----------------------------------------------------------------------------- identity checks

def identity_checks_I1_I2(inst, C, case, dt, ref, Phi, rng):
    """I1 on random vectors; I2 for the basis Phi, at step k = 100 of the reference trajectory."""
    U, Vf, Acc = ref
    k = 100
    K, M, P = inst["K"], inst["M"], inst["P"]
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    A = (K + a0 * M + a1 * C).tocsc()
    F = fc.effective_load(M, C, P, case, dt, N_STEPS, U[k], Vf[k], Acc[k], k)
    Ustar = spla.splu(A).solve(F)
    Pi = lambda u: 0.5 * u @ (A @ u) - F @ u
    out = {}
    worst_val, worst_grad = 0.0, 0.0
    for _ in range(5):
        u = Ustar + rng.standard_normal(len(F)) * np.linalg.norm(Ustar) / np.sqrt(len(F))
        e = u - Ustar
        rhs = 0.5 * e @ (A @ e)
        worst_val = max(worst_val, abs(Pi(u) - Pi(Ustar) - rhs) / rhs)
        gdiff = (A @ u - F) - A @ e
        worst_grad = max(worst_grad, np.linalg.norm(gdiff) / np.linalg.norm(A @ e))
    out["I1_value_rel"] = float(worst_val); out["I1_grad_rel"] = float(worst_grad)
    out["step_vs_reference_rel"] = float(np.linalg.norm(Ustar - U[k + 1]) / np.linalg.norm(U[k + 1]))
    PAP = Phi.T @ (A @ Phi)
    c = np.linalg.solve(PAP, Phi.T @ F)
    uh = Phi @ c
    lhs = Pi(uh) - Pi(Ustar)
    e = uh - Ustar
    rhs = 0.5 * e @ (A @ e)
    # independent best approximation: minimise the A-norm distance by normal equations with A Ustar
    c2 = np.linalg.solve(PAP, Phi.T @ (A @ Ustar))
    out["I2_rel"] = float(abs(lhs - rhs) / rhs)
    out["I2_coeff_rel"] = float(np.linalg.norm(c - c2) / np.linalg.norm(c2))
    return out


def identity_check_I3(Phi, inst, C, case, dt):
    """Along the reduced rollout: actual A-norm error of each Galerkin step against the exact step from the
    visited state, and the bound beta dt^2 g^T M^-1 g."""
    K, M, P = inst["K"], inst["M"], inst["P"]
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    A = (K + a0 * M + a1 * C).tocsc()
    luA = spla.splu(A); luM = spla.splu(M.tocsc())
    Cs, Ds, As, _ = reduced_rollout(Phi, inst, C, case, dt)
    ratios, over = [], []
    for k in range(N_STEPS):
        u, v, acc = Phi @ Cs[k], Phi @ Ds[k], Phi @ As[k]
        F = fc.effective_load(M, C, P, case, dt, N_STEPS, u, v, acc, k)
        Ustar = luA.solve(F)
        uh = Phi @ Cs[k + 1]
        e = uh - Ustar
        actual = e @ (A @ e)
        g = F - A @ uh
        bound = fc.BETA * dt ** 2 * g @ luM.solve(g)
        if actual > 0:
            ratios.append(actual / bound); over.append(np.sqrt(bound / actual))
    ratios = np.array(ratios); over = np.array(over)
    return dict(I3_max_actual_over_bound=float(ratios.max()), I3_violations=int(np.sum(ratios > 1 + 1e-9)),
                I3_median_overestimate=float(np.median(over)), I3_p90_overestimate=float(np.quantile(over, 0.9)))


# ----------------------------------------------------------------------------- commands

def cmd_prep():
    t0 = time.time()
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = setup()
    log("dofs %d, elements %d, tip dof %d at %s" % (inst["n_free"], inst["n_elem"], inst["tip_dof"], inst["tip_xy"]))
    log("omega (first 6): " + ", ".join("%.6f" % x for x in w[:6]))
    log("a_r %.3e b_r %.3e dt %.5f; T1/dt %.1f" % (a_r, b_r, dt, 2 * np.pi / w[0] / dt))
    t1 = time.time()
    train_refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, N_STEPS) for c in train_cases]
    t_train_refs = time.time() - t1
    t1 = time.time()
    eval_refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, N_STEPS) for c in eval_cases]
    t_eval_refs = time.time() - t1
    log("references: training %.1f s, evaluation %.1f s" % (t_train_refs, t_eval_refs))
    rng = np.random.default_rng(7)
    Phi_rand = np.linalg.qr(rng.standard_normal((inst["n_free"], 16)))[0]
    idc = identity_checks_I1_I2(inst, C, eval_cases[-2], dt, eval_refs[-2], Phi_rand, rng)
    log("identity checks (random basis): " + json.dumps(idc))
    with open(os.path.join(OUT, "prep.pkl"), "wb") as f:
        pickle.dump(dict(w=w, V=V, a_r=a_r, b_r=b_r, dt=dt, train_cases=train_cases, eval_cases=eval_cases,
                         train_refs=train_refs, eval_refs=eval_refs, idc_random=idc,
                         t_train_refs=t_train_refs, t_eval_refs=t_eval_refs), f)
    log("prep done in %.1f s" % (time.time() - t0))


def cmd_train(m, seed, iters=3000):
    import lb_model as lb
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = setup()
    pr = lb.Problem(inst, a_r, b_r, dt, m)
    W, G = W_G(train_cases, dt)
    t0 = time.time()
    params, hist = lb.train(pr, np.asarray(W), np.asarray(G), m, seed, iters=iters, log=log)
    t_train = time.time() - t0
    log("training m=%d seed=%d: %.1f s" % (m, seed, t_train))
    with open(os.path.join(OUT, "params_m%d_s%d.pkl" % (m, seed)), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], hist=hist,
                         t_train=t_train, iters=iters), f)


def cmd_eval():
    import jax.numpy as jnp
    import lb_model as lb
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = setup()
    with open(os.path.join(OUT, "prep.pkl"), "rb") as f:
        prep = pickle.load(f)
    eval_refs, train_refs = prep["eval_refs"], prep["train_refs"]
    a0, a1 = fc.newmark_consts(dt)[:2]
    A = (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()
    res = dict(n_dof=inst["n_free"], n_elem=inst["n_elem"], omega_fe=[float(x) for x in w[:8]], dt=dt,
               N=N_STEPS, a_r=a_r, b_r=b_r, identity_random=prep["idc_random"], runs={}, baselines={})
    names = [c.get("name", "case%d" % i) for i, c in enumerate(eval_cases)]
    res["eval_case_names"] = names

    def summarise(Phi):
        rows = [metrics(Phi, inst, C, c, dt, r, w, A) for c, r in zip(eval_cases, eval_refs)]
        keys = ["e1_tip_disp", "e1_tip_acc", "e2_energy", "e3_median", "e3_p90", "e3_max"]
        summ = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows])))
                for k in keys}
        summ["e4_rel_freq"] = rows[0]["e4_rel_freq"]
        summ["per_case"] = dict(zip(names, rows))
        return summ

    for m in (8, 16):
        res["baselines"]["modal_m%d" % m] = summarise(V[:, :m])
        res["baselines"]["pod_m%d" % m] = summarise(pod_basis(inst, train_refs, m))
        for seed in (0, 1):
            fn = os.path.join(OUT, "params_m%d_s%d.pkl" % (m, seed))
            if not os.path.exists(fn):
                continue
            with open(fn, "rb") as f:
                d = pickle.load(f)
            pr = lb.Problem(inst, a_r, b_r, dt, m)
            params = [(jnp.asarray(Wt), jnp.asarray(b)) for (Wt, b) in d["params"]]
            Phi = lb.basis_numpy(params, pr)
            s = summarise(Phi)
            rng = np.random.default_rng(11)
            s["identity_I2"] = identity_checks_I1_I2(inst, C, eval_cases[-2], dt, eval_refs[-2], Phi, rng)
            s["identity_I3_tip_pulse"] = identity_check_I3(Phi, inst, C, eval_cases[-2], dt)
            s["identity_I3_case0"] = identity_check_I3(Phi, inst, C, eval_cases[0], dt)
            s["t_train_s"] = d["t_train"]; s["loss_history"] = d["hist"]
            res["runs"]["m%d_s%d" % (m, seed)] = s
            log("evaluated m=%d seed=%d" % (m, seed))
    res["baselines"]["modal_m16"]["identity_I3_tip_pulse"] = identity_check_I3(V[:, :16], inst, C, eval_cases[-2], dt)
    # E5 timings
    tim = {}
    t0 = time.perf_counter(); spla.splu(A); tim["factorise_A_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); fc.eig(inst["K"], inst["M"], 16); tim["eigsh16_s"] = time.perf_counter() - t0
    for m in (8, 16):
        fn = os.path.join(OUT, "params_m%d_s0.pkl" % m)
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                d = pickle.load(f)
            pr = lb.Problem(inst, a_r, b_r, dt, m)
            params = [(jnp.asarray(Wt), jnp.asarray(b)) for (Wt, b) in d["params"]]
            import jax
            fb = jax.jit(lambda p: lb.reduced_ops(lb.orthonormal_basis(p, pr), pr)[3])
            fb(params).block_until_ready()
            ts = []
            for _ in range(20):
                t0 = time.perf_counter(); fb(params).block_until_ready(); ts.append(time.perf_counter() - t0)
            tim["setup_network_and_projections_m%d_s" % m] = float(np.median(ts))
    # per-step costs: the reduced Newmark rollout itself (N_STEPS steps), timed and divided by N_STEPS
    for m in (8, 16):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); reduced_rollout(V[:, :m], inst, C, eval_cases[-2], dt); ts.append(time.perf_counter() - t0)
        tim["reduced_rollout_per_step_incl_setup_m%d_s" % m] = float(np.median(ts)) / N_STEPS
    lu = spla.splu(A); u = np.zeros(inst["n_free"]); ts = []
    for _ in range(200):
        t0 = time.perf_counter(); rhs = inst["M"] @ u + C @ u + 1e-3; u = lu.solve(rhs) * 1e-3; ts.append(time.perf_counter() - t0)
    tim["full_step_s"] = float(np.median(ts[20:]))
    res["timings"] = tim
    res["t_refs_s"] = dict(train=prep["t_train_refs"], eval=prep["t_eval_refs"])
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("results written")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        cmd_prep()
    elif cmd == "train":
        cmd_train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
    elif cmd == "eval":
        cmd_eval()
