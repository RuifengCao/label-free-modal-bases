"""Exploratory runs after the formal runs of the Unified 2027 dynamics paper (EXPLORE_PLAN.md).

  python -I explore.py acc SHARD NSHARD     # X1a: Part II networks and controls on refined meshes (h_max 0.2, 0.1, 0.05)
  python -I explore.py fine K               # X1a: the same on the K-th FINE geometry at h_max 0.025
  python -I explore.py timing               # X1b: online cost on the FINE geometries (run alone, single-threaded)
  python -I explore.py long GEOM OBJ SEED   # X2: shared network, O2 or O3, G1 to G3, seed 10 or 11, 100,000 iterations

Environment: EXP_OUT (output folder; default results/ next to this file, results_smoke/ with EXP_SMOKE=1);
EXP_NETS (folder with the formal Part II networks and e2_shard*.json; default ../formal/results);
EXP_SMOKE=1 (small meshes, two evaluation cases, 40 iterations; smoke outputs are not used).
The stamped formal code is imported from ../formal and not modified. Normally started through run_explore.py.
"""
import os, sys, json, time, glob
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAL = os.path.join(os.path.dirname(HERE), "formal")
SMOKE = os.environ.get("EXP_SMOKE", "0") == "1"
OUT = os.environ.get("EXP_OUT", os.path.join(HERE, "results_smoke" if SMOKE else "results"))
NETS = os.environ.get("EXP_NETS", os.path.join(FORMAL, "results"))
os.makedirs(OUT, exist_ok=True)
# The formal modules create output folders when imported; keep them inside OUT so that nothing is written into formal/.
for _v, _d in (("P0_OUT", "_p0"), ("P0B_OUT", "_p0b"), ("P0C_OUT", "_p0c"), ("P0D_OUT", "_p0d"), ("P0E_OUT", "_p0e"),
               ("P1_OUT", "_p1")):
    os.environ[_v] = os.path.join(OUT, _d)
os.environ["P0_SMOKE"] = "0"
os.environ["P1_SMOKE"] = "0"
sys.path.insert(0, FORMAL)
import phase1 as F                                   # stamped; imported, not modified
from phase1 import P, PB, PD, PE, fc, rc, lb, p2, p3, p6, jax, jnp
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
p4 = P.p4

M_MODES = F.M_MODES
H_LIST = (0.2, 0.14) if SMOKE else (0.2, 0.1, 0.05)
H_FINE = 0.1 if SMOKE else 0.025
H_TIMING = (0.2, 0.14) if SMOKE else (0.2, 0.1, 0.05, 0.025)
FINE = ("1.1698_0.2501", "2.6263_0.3995", "3.5026_0.2488", "4.0451_0.1017")    # EXPLORE_PLAN.md, X1
ITERS_LONG, LOG_LONG = (40, 10) if SMOKE else (100000, 5000)
REPS = 3                                            # X1b: median of three repeats
N_FULL_STEPS, N_RED_STEPS = 20, 1000                # X1b: steps timed in each repeat


def key(g):
    return "%.4f_%.4f" % (g[0], g[1])


def jdump(obj, fn):
    tmp = os.path.join(OUT, fn + ".partial")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=P._js)
    os.replace(tmp, os.path.join(OUT, fn))            # the final name appears only when the file is complete


# ----------------------------------------------------------------------------- X1a accuracy

def geom_setup_h(xc, d, h, n_eig=16):
    """phase0.geom_setup with the mesh size as a parameter; otherwise the same code."""
    inst = fc.build(xc=xc, d=d, rho=0.05, h_max=h)
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


def stream_metrics(bases, inst, C, case, dt, lu, lu_m):
    """E1 and E2 of several bases on one evaluation case, without storing the full trajectory.
    Full recursion as fem_cantilever.newmark_full; reduced rollouts by run_check.reduced_rollout; E1 and E2 as
    run_check.metrics. The error e_k = Phi c_k - u_k is formed at each step and e_k^T K e_k is computed as
    e_k^T (K Phi c_k - K u_k); expanding it into c^T Kr c - 2 c^T (K Phi)^T u + u^T K u instead loses up to
    seven digits to cancellation (sandbox test of 7 November 2026). lu and lu_m: the factors of the step operator
    K + a0 M + a1 C and of M (in newmark_full they are recomputed for every case; the factors are the same)."""
    K, M, Pm = inst["K"], inst["M"], inst["P"]
    N = rc.N_STEPS
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    tip = inst["tip_dof"]
    names = list(bases)
    red = []
    for name in names:
        Phi = bases[name]
        Cs, Ds, As, _ = rc.reduced_rollout(Phi, inst, C, case, dt)
        red.append(dict(Cs=Cs, As=As, Phi=Phi, KPhi=K @ Phi, tip=Phi[tip]))
    num = np.zeros(len(names)); den = 0.0
    w = np.asarray(case["w"], float)
    g = fc.load_history(case, dt, N)
    f_pat = Pm @ w
    n = K.shape[0]
    u = np.zeros(n); v = np.zeros(n)
    acc = lu_m.solve(f_pat * g[0])
    ut = np.zeros(N + 1); at = np.zeros(N + 1)
    ut[0], at[0] = u[tip], acc[tip]
    for k in range(N + 1):
        if k > 0:
            rhs = f_pat * g[k] + M @ (a0 * u + a2 * v + a3 * acc) + C @ (a1 * u + a4 * v + a5 * acc)
            un = lu.solve(rhs)
            an = a0 * (un - u) - a2 * v - a3 * acc
            vn = v + dt * ((1 - fc.GAMMA) * acc + fc.GAMMA * an)
            u, v, acc = un, vn, an
            ut[k], at[k] = u[tip], acc[tip]
        Ku = K @ u
        den += float(u @ Ku)
        for b, r in enumerate(red):
            c = r["Cs"][k]
            num[b] += float((r["Phi"] @ c - u) @ (r["KPhi"] @ c - Ku))
    out = {}
    for b, name in enumerate(names):
        r = red[b]
        u_tip = r["Cs"] @ r["tip"]; a_tip = r["As"] @ r["tip"]
        out[name] = dict(e1_tip_disp=float(np.linalg.norm(u_tip - ut) / np.linalg.norm(ut)),
                         e1_tip_acc=float(np.linalg.norm(a_tip - at) / np.linalg.norm(at)),
                         e2_energy=float(np.sqrt(max(num[b], 0.0) / den)))
    return out


def load_nets():
    """Every formal Part II network in NETS: name -> function giving its M-orthonormal basis on a problem
    (the same names, files and functions as phase1._p2_nets)."""
    nets = {}
    for arch, objs, pre in (("shared", F.OBJ_P2, "p2_"), ("sep", F.OBJ_P2_SEP, "p2sep_")):
        bfun = PE.orth_sepv if arch == "sep" else lb.orthonormal_basis
        for obj in objs:
            for seed in F.SEEDS_P2:
                fn = os.path.join(NETS, "%s%s_s%d.pkl" % (pre, obj, seed))
                if os.path.exists(fn):
                    p = p3.load_params(fn)[0]
                    nets["%s%s_s%d" % ("" if arch == "shared" else "sep_", obj, seed)] = (lambda pr, p=p, bf=bfun: bf(p, pr))
    assert len(nets) == 15, sorted(nets)
    return nets


def common():
    """Training-geometry data as in phase1.cmd_e2, and the formal held-out records for the checks at h_max = 0.2."""
    geo = F.geometries()
    srcs = PD.sources(geo["train"])                    # classical controls: training eigen-solutions at h_max = 0.2
    assert [g for g, _, _ in srcs] == list(geo["train"])
    tr_xy = np.array([PB.norm_xy(*g) for g in geo["train"]])
    hull = Delaunay(tr_xy)
    W8 = np.array([w8 for _, _, w8 in srcs])
    rbf = [RBFInterpolator(tr_xy, W8[:, i], kernel="thin_plate_spline", degree=1, smoothing=0.0) for i in range(M_MODES)]
    formal = {}
    for fn in sorted(glob.glob(os.path.join(NETS, "e2_shard*.json"))):
        with open(fn) as f:
            formal.update(json.load(f)["geoms"])
    return geo, srcs, tr_xy, hull, rbf, formal


def _maxrel(a, b, floor=0.0):
    return max(abs(x - y) / max(abs(y), floor) for x, y in zip(a, b))


def evaluate(g, h, nets, srcs, tr_xy, hull, rbf, formal, w_plain):
    xc, d = g
    t0 = time.time()
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = geom_setup_h(xc, d, h)
    if SMOKE:
        eval_cases = eval_cases[:2]
    pr = P.fam_problem(inst, a_r, b_r, dt)
    rec = dict(xc=xc, d=d, h_max=h, n_free=int(inst["n_free"]), n_elem=int(inst["n_elem"]), w=[float(x) for x in w[:8]],
               w_plain1=w_plain, w1_drop_fe=float((w_plain - w[0]) / w_plain), dt=float(dt),
               dist_nearest=float(np.min(np.linalg.norm(tr_xy - PB.norm_xy(xc, d), axis=1))),
               inside_hull=bool(hull.find_simplex(PB.norm_xy(xc, d)[None, :])[0] >= 0),
               t_setup=time.time() - t0)
    bases = dict(modal_m8=V[:, :M_MODES])
    t1 = time.time()
    for k, f in nets.items():
        bases[k] = np.asarray(f(pr))
    rec["t_nets"] = time.time() - t1
    t1 = time.time()
    B, info = PD.rbm_basis([(gg, s) for gg, s, _ in srcs], inst, (xc, d), m=M_MODES, tol=F.RBM_TOL)
    rec["t_rbm"] = time.time() - t1
    bases["rbm_all"] = B
    per = {k: [] for k in bases}
    t1 = time.time()
    a0, a1 = fc.newmark_consts(dt)[:2]
    lu = spla.splu((inst["K"] + a0 * inst["M"] + a1 * C).tocsc())
    lu_m = spla.splu(inst["M"].tocsc())
    for case in eval_cases:
        r = stream_metrics(bases, inst, C, case, dt, lu, lu_m)
        for k in bases:
            per[k].append(r[k])
    del lu, lu_m
    rec["t_metrics"] = time.time() - t1
    for k, Bk in bases.items():
        s = dict(ritz=p6.ritz_errors(Bk, inst, w, 8), sin2=P.sin2(Bk, inst, V))
        for q in ("e1_tip_disp", "e1_tip_acc", "e2_energy"):
            vals = [x[q] for x in per[k]]
            s[q] = dict(median=float(np.median(vals)), max=float(np.max(vals)), per_case=[float(v) for v in vals])
        s["drop_reproduced"] = F._drop(w_plain, w[0], s["ritz"][0])
        if k == "rbm_all":
            s["rb_info"] = info
        rec[k] = s
    q = PB.norm_xy(xc, d)[None, :]                      # frequency regression: frequencies only
    e = [float(x) for x in np.array([float(f(q)[0]) for f in rbf]) / w[:M_MODES] - 1]
    rec["freq_rbf"] = dict(ritz=e, drop_reproduced=F._drop(w_plain, w[0], e[0]))
    if abs(h - 0.2) < 1e-12:
        # Checks at the formal mesh size (EXPLORE_PLAN.md, X1a). (i) The streaming E1 and E2 against phase0.eval_basis
        # on the same bases and cases. (ii) The Ritz errors against the formal outputs: the largest absolute difference
        # (a difference of relative frequency errors, i.e. the relative difference of the Ritz frequencies), and the
        # largest relative difference of the errors themselves where the formal error exceeds 1e-8 in size.
        # (iii) The finite element frequencies and E1, E2 against the formal outputs.
        A = p6.step_operator(inst, C, dt)
        refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, rc.N_STEPS) for c in eval_cases]
        chk = {}
        for k, Bk in bases.items():
            s0 = P.eval_basis(Bk, inst, w, V, C, dt, A, refs, eval_cases)
            chk[k] = max(_maxrel(rec[k][qq]["per_case"], s0[qq]["per_case"]) for qq in ("e1_tip_disp", "e1_tip_acc", "e2_energy"))
        rec["check_stream_vs_eval_basis"] = chk
        del refs
        fr = formal.get(key(g))
        if fr is not None:
            ks = [k for k in list(bases) + ["freq_rbf"] if k in fr]
            rec["check_ritz_vs_formal_abs"] = {k: max(abs(a - b) for a, b in zip(rec[k]["ritz"], fr[k]["ritz"])) for k in ks}
            rec["check_ritz_vs_formal_rel"] = {k: max([abs(a - b) / abs(b) for a, b in zip(rec[k]["ritz"], fr[k]["ritz"])
                                                       if abs(b) > 1e-8] or [0.0]) for k in ks}
            rec["check_w_vs_formal"] = _maxrel(rec["w"], fr["w"])
            rec["check_metrics_vs_formal"] = {k: max(_maxrel(rec[k][qq]["per_case"], fr[k][qq]["per_case"])
                                                     for qq in ("e1_tip_disp", "e1_tip_acc", "e2_energy")) for k in bases if k in fr}
            rec["check_keys_missing_in_formal"] = [k for k in list(bases) + ["freq_rbf"] if k not in fr]
    rec["t_eval"] = time.time() - t0
    print("%s h=%g n=%d: %.0f s (set-up %.0f, nets %.0f, rbm %.0f, metrics %.0f)" % (
        key(g), h, rec["n_free"], rec["t_eval"], rec["t_setup"], rec["t_nets"], rec["t_rbm"], rec["t_metrics"]), flush=True)
    return rec


def cmd_acc(shard, nshard):
    geo, srcs, tr_xy, hull, rbf, formal = common()
    nets = load_nets()
    held = geo["held"][shard::nshard]
    w_plain = {h: float(p2.plain_cantilever_w1(h_max=h)[0]) for h in H_LIST}
    res = {}
    for g in (held[:1] if SMOKE else held):
        for h in H_LIST:
            res["%s_h%g" % (key(g), h)] = evaluate(g, h, nets, srcs, tr_xy, hull, rbf, formal, w_plain[h])
    jdump(dict(nets=sorted(nets), h_list=list(H_LIST), w_plain={str(h): v for h, v in w_plain.items()}, geoms=res),
          "x1_acc_shard%d.json" % shard)


def cmd_fine(k):
    geo, srcs, tr_xy, hull, rbf, formal = common()
    nets = load_nets()
    g = [x for x in geo["held"] if key(x) == FINE[k]]
    assert len(g) == 1, FINE[k]
    w_plain = float(p2.plain_cantilever_w1(h_max=H_FINE)[0])
    rec = evaluate(g[0], H_FINE, nets, srcs, tr_xy, hull, rbf, formal, w_plain)
    jdump(dict(nets=sorted(nets), h_max=H_FINE, w_plain=w_plain, geoms={"%s_h%g" % (FINE[k], H_FINE): rec}), "x1_fine%d.json" % k)


# ----------------------------------------------------------------------------- X1b timing

def _median_time(f, n=REPS):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _step_time(step, n_steps, n=REPS):
    """Median over n repeats of the median time of one step over n_steps consecutive steps."""
    meds = []
    for _ in range(n):
        ts = []
        for _ in range(n_steps):
            t0 = time.perf_counter(); step(); ts.append(time.perf_counter() - t0)
        meds.append(float(np.median(ts)))
    return float(np.median(meds))


def newmark_stepper(solve, Mop, Cop, f, g, dt):
    """One Newmark step per call (average acceleration, the recursion of fem_cantilever.newmark_full and
    run_check.reduced_rollout), from rest; the load history repeats if more than len(g) - 1 steps are taken."""
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    n = f.shape[0]
    st = dict(u=np.zeros(n), v=np.zeros(n), acc=np.zeros(n), k=0)
    N = len(g) - 1

    def step():
        u, v, acc, k = st["u"], st["v"], st["acc"], st["k"]
        rhs = f * g[k % N + 1] + Mop @ (a0 * u + a2 * v + a3 * acc) + Cop @ (a1 * u + a4 * v + a5 * acc)
        un = solve(rhs)
        an = a0 * (un - u) - a2 * v - a3 * acc
        vn = v + dt * ((1 - fc.GAMMA) * acc + fc.GAMMA * an)
        st.update(u=un, v=vn, acc=an, k=k + 1)
    return step


@jax.jit
def basis_args(params, feats, clamp, ixf, iyf, M):
    """lb.orthonormal_basis with the exact clamp (posthoc6), with the mesh data as arguments of the compiled function
    rather than constants inside it."""
    out = lb.mlp(params, feats)
    m = out.shape[1] // 2
    Phi = jnp.zeros((clamp.shape[0], m)).at[ixf].set(out[:, :m]).at[iyf].set(out[:, m:]) * clamp
    G = Phi.T @ (M @ Phi)
    Lc = jnp.linalg.cholesky(0.5 * (G + G.T))
    return jax.scipy.linalg.solve_triangular(Lc, Phi.T, lower=True).T


def basis_numpy(params_np, inst):
    """The same basis in NumPy, without compilation: inputs (phase0.fam_problem), network, exact clamp,
    M-orthonormalisation."""
    xc, d, _ = inst["params"]
    x = np.hstack([lb.features(inst["pts"], inst["params"]), p4.angle_feats(inst["pts"], inst["params"]),
                   np.tile([[(xc - 3.0) / 2.0, (d - 0.25) / 0.15]], (inst["pts"].shape[0], 1))])
    for W, b in params_np[:-1]:
        x = np.tanh(x @ W + b)
    W, b = params_np[-1]
    out = x @ W + b
    Phi = np.zeros((inst["n_free"], M_MODES))
    Phi[inst["ixf"]] = out[:, :M_MODES]; Phi[inst["iyf"]] = out[:, M_MODES:]
    Phi *= (p3.dof_x(inst) / fc.L)[:, None]
    G = Phi.T @ (inst["M"] @ Phi)
    Lc = np.linalg.cholesky(0.5 * (G + G.T))
    return sla.solve_triangular(Lc, Phi.T, lower=True).T


def _relmax(A, B):
    return float(np.max(np.abs(A - B)) / np.max(np.abs(B)))


def cmd_timing():
    geo = F.geometries()
    fine = [g for k in FINE for g in geo["held"] if key(g) == k]
    assert len(fine) == 4
    params = p3.load_params(os.path.join(NETS, "p2_o4_s0.pkl"))[0]
    params_np = [(np.asarray(W), np.asarray(b)) for W, b in params]
    t0 = time.perf_counter()
    srcs = PD.sources(geo["train"])
    t_sources = time.perf_counter() - t0
    src_list = [(g, s) for g, s, _ in srcs]
    rows = []
    for (xc, d) in (fine[:1] if SMOKE else fine):
        for h in H_TIMING:
            rec = dict(geom=key((xc, d)), h_max=h)
            rec["t_mesh"] = _median_time(lambda: fc.make_mesh(xc, d, 0.05, h))
            rec["t_build"] = _median_time(lambda: fc.build(xc=xc, d=d, rho=0.05, h_max=h))
            inst = fc.build(xc=xc, d=d, rho=0.05, h_max=h)
            rec["n_free"] = int(inst["n_free"])
            rec["t_eig8"] = _median_time(lambda: fc.eig(inst["K"], inst["M"], 8))
            w, V = fc.eig(inst["K"], inst["M"], 8)
            a_r, b_r, dt = fc.time_setup(w[0], w[2])
            C = (a_r * inst["M"] + b_r * inst["K"]).tocsc()
            rec["t_problem"] = _median_time(lambda: P.fam_problem(inst, a_r, b_r, dt))
            pr = P.fam_problem(inst, a_r, b_r, dt)
            args = (pr.feats, pr.clamp, pr.ixf, pr.iyf, pr.M)
            t1 = time.perf_counter()
            Phi = np.asarray(jax.block_until_ready(basis_args(params, *args)))
            rec["t_net_first"] = time.perf_counter() - t1                # one measurement; compilation for this mesh included
            rec["t_net"] = _median_time(lambda: jax.block_until_ready(basis_args(params, *args)))
            ref = np.asarray(lb.orthonormal_basis(params, pr))
            rec["net_args_vs_lb"] = _relmax(Phi, ref)
            rec["t_net_numpy"] = _median_time(lambda: basis_numpy(params_np, inst))
            rec["net_numpy_vs_lb"] = _relmax(basis_numpy(params_np, inst), ref)
            rec["t_reduced_ops"] = _median_time(lambda: (Phi.T @ (inst["K"] @ Phi), Phi.T @ (inst["M"] @ Phi), Phi.T @ inst["P"]))
            rec["t_rbm"] = _median_time(lambda: PD.rbm_basis(src_list, inst, (xc, d), m=M_MODES, tol=F.RBM_TOL))
            a0, a1 = fc.newmark_consts(dt)[:2]
            Af = (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()
            rec["t_full_factorisation"] = _median_time(lambda: spla.splu(Af))
            lu = spla.splu(Af)
            case = dict(w=[1.0, 0.0, 0.0], kind="pulse", tau=float(np.pi / w[1]))     # the tip pulse case
            g = fc.load_history(case, dt, rc.N_STEPS)
            rec["t_full_step"] = _step_time(newmark_stepper(lu.solve, inst["M"], C, inst["P"] @ np.asarray(case["w"]), g, dt),
                                            N_FULL_STEPS)
            Kr = Phi.T @ (inst["K"] @ Phi); Mr = Phi.T @ (inst["M"] @ Phi); Cr = Phi.T @ (C @ Phi)
            Ar = Kr + a0 * Mr + a1 * Cr
            cf = sla.cho_factor(0.5 * (Ar + Ar.T))
            rec["t_reduced_step"] = _step_time(newmark_stepper(lambda b: sla.cho_solve(cf, b), Mr, Cr,
                                                               Phi.T @ (inst["P"] @ np.asarray(case["w"])), g, dt), N_RED_STEPS)
            rec["t_full_sim_640"] = rec["t_full_factorisation"] + 640 * rec["t_full_step"]
            rec["t_reduced_sim_640"] = 640 * rec["t_reduced_step"]
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    jdump(dict(t_sources_offline=t_sources, reps=REPS, n_full_steps=N_FULL_STEPS, n_reduced_steps=N_RED_STEPS,
               network="p2_o4_s0", rows=rows), "x1_timing.json")


# ----------------------------------------------------------------------------- X2 longer training

def cmd_long(gname, obj, seed):
    assert gname in ("G1", "G2", "G3") and obj in ("o2", "o3") and seed in (10, 11)
    F.OUT, F.ITERS_P1, F.LOG_P1, F.SMOKE = OUT, ITERS_LONG, LOG_LONG, SMOKE    # same code path as the formal Part I runs
    F.cmd_p1(gname, obj, seed, "shared")
    for ext in (".pkl", ".json"):                    # the .json last: its presence marks the job as finished
        os.replace(os.path.join(OUT, "p1_%s_%s_s%d%s" % (gname, obj, seed, ext)),
                   os.path.join(OUT, "x2_%s_%s_s%d%s" % (gname, obj, seed, ext)))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "acc":
        cmd_acc(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "fine":
        cmd_fine(int(sys.argv[2]))
    elif c == "timing":
        cmd_timing()
    elif c == "long":
        cmd_long(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit("unknown command " + c)
