# Exploratory runs X1 to X3: code

Bundled on 8 November 2026 for the project record; the files themselves are in unified2027/explore on the laptop and in the sandbox. explore.py and run_explore.py are the versions that ran on the laptop (hashes as in results/environment.json).

| File | sha256 |
|---|---|
| explore.py | 372becc51ffb4935d15f80376a97a72d8bf234d710abeaa18bf6ea21aca447f9 |
| run_explore.py | acc9fa0d701cfff769f2688adef8e2c6c0db1116f87ab41724a78407fc2d145a |
| run_explore.bat | cd00189ced92c33656afbf01089437dfd05e05b95d902d7300c13a38412aa3db |
| run_explore_smoke.bat | 92a439f5df1635d265c464d06afc2a0c14427d43b81000c8f0e42d575e044b66 |
| x3_analysis.py | 778487c3d90f39d2700bf2b2d5384118fe1fa0dc099b2998ed2486fbb3592f2d |
| x3_figures.py | 5c027f08703d20d4975709ce140d698b8aa340cf75d3a9bb9ea6f24ab9aa2016 |
| x12_report.py | cb84acf287cb95f1ecf6e135afe3c63d8ba781162625e37b9e12dc4f4cc3d29e |
| check_h02/repeat_sources.py | 743759659ab2e6d06afb85740551f8d59b9bbc95899a6f206eb6301f04d47abc |
| sbx_tests/test_launcher.py | 9d9602bb0f0a18414b13726113e9d46a003d7e7bc6c0ad9cdee2324bbea7bb31 |

## explore.py

```python
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
```

## run_explore.py

```python
"""Start the exploratory runs X1 and X2 of the Unified 2027 dynamics paper (EXPLORE_PLAN.md) on this computer.

  python run_explore.py --smoke     every job type on small meshes with a few iterations (6 jobs, a few minutes)
  python run_explore.py             the exploratory runs (21 jobs, about 3 to 4 hours)

Before starting, it checks that the formal code it imports still matches formal/STAMP.txt and that the formal networks
and evaluation records it reads still match formal/results/SHA256SUMS_formal_results.txt; it refuses to start otherwise.
Each job is a separate Python process pinned to one logical CPU (default 2, 4, 6, 8). The four FINE-mesh jobs run one
after another; the timing job runs last, alone, on the first CPU. Progress is written to results/progress.txt
(results_smoke/ for --smoke). Run it again after an interruption: finished jobs are skipped; a job that failed or was
cut off is run once more and the rerun is recorded. While it runs, Windows is asked not to sleep.
"""
import os, sys, json, time, hashlib, subprocess, argparse, platform

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAL = os.path.join(os.path.dirname(HERE), "formal")
OWN = ["EXPLORE_PLAN.md", "explore.py", "run_explore.py"]
NET_FILES = (["p2_%s_s%d.pkl" % (o, s) for o in ("o4", "o3s", "sup") for s in (0, 1, 2)]
             + ["p2sep_%s_s%d.pkl" % (o, s) for o in ("o4", "o3s") for s in (0, 1, 2)]
             + ["e2_shard%d.json" % k for k in range(4)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(fn):
    """'sha256 relpath' lines (STAMP.txt and SHA256SUMS files); other lines are ignored."""
    out = {}
    with open(fn) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                out[parts[1]] = parts[0]
    return out


def check_inputs(nets_dir, sums_fn):
    """The formal code must match formal/STAMP.txt; the formal networks and records read here must match the
    checksum list written on this computer after the formal run."""
    bad, hashes = [], {}
    stamp = os.path.join(FORMAL, "STAMP.txt")
    if not os.path.exists(stamp):
        return ["formal/STAMP.txt not found"], hashes
    for rel, h in read_sums(stamp).items():
        p = os.path.join(FORMAL, rel)
        cur = sha256(p) if os.path.exists(p) else "missing"
        hashes["formal/" + rel] = cur
        if cur != h:
            bad.append("formal/%s changed since the stamp" % rel)
    if not os.path.exists(sums_fn):
        return bad + ["%s not found" % sums_fn], hashes
    sums = read_sums(sums_fn)
    for rel in NET_FILES:
        p = os.path.join(nets_dir, rel)
        cur = sha256(p) if os.path.exists(p) else "missing"
        hashes["nets/" + rel] = cur
        if sums.get(rel) != cur:
            bad.append("%s: %s" % (rel, "missing" if cur == "missing" else "not in the checksum list" if rel not in sums
                                   else "differs from the checksum list"))
    return bad, hashes


class Job:
    def __init__(self, name, args, output, after=(), cpu=None):
        self.name, self.args, self.output, self.after, self.cpu_req = name, args, output, set(after), cpu
        self.attempts, self.status, self.cpu, self.t0, self.proc, self.log = 0, "pending", None, None, None, None


def job_list(smoke, cpus):
    if smoke:
        fine = [Job("fine_0", ["fine", "0"], "x1_fine0.json")]
        acc = [Job("acc_%d" % k, ["acc", str(k), "4"], "x1_acc_shard%d.json" % k) for k in (0, 1)]
        long_ = [Job("x2_%s_%s_s%d" % (g, o, s), ["long", g, o, str(s)], "x2_%s_%s_s%d.json" % (g, o, s))
                 for (g, o, s) in (("G1", "o2", 10), ("G2", "o3", 11))]
    else:
        fine = [Job("fine_%d" % k, ["fine", str(k)], "x1_fine%d.json" % k, ["fine_%d" % (k - 1)] if k else [])
                for k in range(4)]                                # one at a time (memory)
        acc = [Job("acc_%d" % k, ["acc", str(k), "4"], "x1_acc_shard%d.json" % k) for k in range(4)]
        long_ = [Job("x2_%s_%s_s%d" % (g, o, s), ["long", g, o, str(s)], "x2_%s_%s_s%d.json" % (g, o, s))
                 for o in ("o3", "o2") for g in ("G1", "G2", "G3") for s in (10, 11)]
    others = fine + acc + long_
    timing = Job("timing", ["timing"], "x1_timing.json", [j.name for j in others], cpu=cpus[0])   # alone, at the end
    return others + [timing]


def keep_awake(on):
    if platform.system() != "Windows":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpus", default="2,4,6,8")
    ap.add_argument("--nets", default=os.path.join(FORMAL, "results"))
    ap.add_argument("--sums", default=None, help="default: SHA256SUMS_formal_results.txt in the --nets folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sums_fn = a.sums or os.path.join(a.nets, "SHA256SUMS_formal_results.txt")
    out = a.out or os.path.join(HERE, "results_smoke" if a.smoke else "results")
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    progress_txt, state_fn = os.path.join(out, "progress.txt"), os.path.join(out, "state.json")

    def note(msg):
        line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(progress_txt, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    bad, in_hashes = check_inputs(a.nets, sums_fn)
    if bad:
        note("NOT STARTED. " + "; ".join(bad))
        sys.exit(2)
    with open(os.path.join(out, "environment.json"), "w") as f:
        import numpy, scipy, jax
        json.dump(dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                       numpy=numpy.__version__, scipy=scipy.__version__, jax=jax.__version__,
                       logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                       memory_gb=round(psutil.virtual_memory().total / 2 ** 30, 1),
                       file_hashes={rel: sha256(os.path.join(HERE, rel)) for rel in OWN}, inputs=in_hashes,
                       nets=os.path.abspath(a.nets), smoke=a.smoke, started=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=1)
    cpus = [int(x) for x in a.cpus.split(",")]
    jobs = job_list(a.smoke, cpus)
    by_name = {j.name: j for j in jobs}
    prev = json.load(open(state_fn)) if os.path.exists(state_fn) else {}
    for j in jobs:
        if os.path.exists(os.path.join(out, j.output)):
            j.status = "done"
        elif j.name in prev and prev[j.name]["attempts"] > 0:     # failed or cut off in an earlier start
            j.attempts = prev[j.name]["attempts"]
            if j.attempts >= 2:
                j.status = "failed"; note("%s: failed twice in an earlier start; not run again" % j.name)
            else:
                note("%s: earlier attempt did not finish (%s); it will be run once more" % (j.name, prev[j.name]["status"]))
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               EXP_OUT=out, EXP_NETS=os.path.abspath(a.nets), EXP_SMOKE="1" if a.smoke else "0")

    def save_state():
        with open(state_fn, "w") as f:
            json.dump({j.name: dict(status=j.status, attempts=j.attempts, cpu=j.cpu) for j in jobs}, f, indent=1)

    note("START %s: %d jobs, %d already done; CPUs %s" % ("smoke test" if a.smoke else "exploratory runs", len(jobs),
                                                          sum(j.status == "done" for j in jobs), cpus))
    keep_awake(True)
    free = list(cpus)
    try:
        while True:
            for j in jobs:                                         # collect finished jobs
                if j.status == "running" and j.proc.poll() is not None:
                    rc_, el = j.proc.returncode, time.time() - j.t0
                    j.log.close(); free.append(j.cpu)
                    if rc_ == 0 and os.path.exists(os.path.join(out, j.output)):
                        j.status = "done"; note("%s: done in %.0f s (CPU %d)" % (j.name, el, j.cpu))
                    elif j.attempts < 2:
                        j.status = "pending"; note("%s: FAILED (exit %s) after %.0f s; it will be run once more" % (j.name, rc_, el))
                    else:
                        j.status = "failed"; note("%s: FAILED again (exit %s); not run again" % (j.name, rc_))
                    save_state()
            for j in jobs:                                         # the order is for memory and timing only, so a
                if j.status == "pending":                          # failed job does not hold back the jobs after it
                    j.after = {x for x in j.after if by_name[x].status != "failed"}
            for j in jobs:                                         # start what can start
                if not free:
                    break
                if j.status == "pending" and all(by_name[x].status == "done" for x in j.after):
                    if j.cpu_req is not None:
                        if j.cpu_req not in free:
                            continue
                        cpu = j.cpu_req; free.remove(cpu)
                    else:
                        cpu = free.pop(0)
                    j.attempts += 1; j.cpu = cpu; j.t0 = time.time(); j.status = "running"
                    j.log = open(os.path.join(logdir, j.name + ".log"), "a")
                    j.log.write("\n=== attempt %d, %s ===\n" % (j.attempts, time.strftime("%Y-%m-%d %H:%M:%S"))); j.log.flush()
                    j.proc = subprocess.Popen([sys.executable, "-I", "-B", os.path.join(HERE, "explore.py")] + j.args, cwd=HERE,
                                              env=env, stdout=j.log, stderr=subprocess.STDOUT)
                    try:
                        psutil.Process(j.proc.pid).cpu_affinity([cpu])
                    except Exception as e:                       # noqa: BLE001
                        note("%s: could not pin to CPU %d (%s)" % (j.name, cpu, e))
                    note("%s: started on CPU %d (attempt %d)" % (j.name, cpu, j.attempts))
                    save_state()
            if all(j.status in ("done", "failed") for j in jobs):
                break
            time.sleep(2 if a.smoke else 10)
    finally:
        keep_awake(False)
        for j in jobs:
            if j.status == "running" and j.proc.poll() is None:
                j.proc.terminate()
        save_state()
    n_done = sum(j.status == "done" for j in jobs)
    note("END: %d of %d jobs done%s" % (n_done, len(jobs), "" if n_done == len(jobs) else "; see FAILED lines above"))


if __name__ == "__main__":
    main()
```

## run_explore.bat

```bat
@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" run_explore.py
echo.
echo Finished. See results\progress.txt
pause
```

## run_explore_smoke.bat

```bat
@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" run_explore.py --smoke
echo.
echo Smoke test finished. See results_smoke\progress.txt
pause
```

## x3_analysis.py

```python
"""X3 (EXPLORE_PLAN.md and DEVIATIONS_EXPLORE.md, items 10 to 14): shape errors of the Ritz vectors against frequency errors,
the E2 prediction from them, and the data for figures 3 and 4. Runs in the sandbox on the saved formal networks.

  python -I -B x3_analysis.py [fig4|notch]   # writes x3/x3_shape.json, x3_fig3.json, x3_fig4.npz and x3_notch.json
                                             # (fig4 or notch: only that part)

Environment: EXP_NETS (folder with the formal networks and e2_shard*.json), EXP_OUT (scratch folder for the formal
modules' output folders; default x3/_scratch).
"""
import os, sys, json, time
import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
X3 = os.path.join(HERE, "x3")
os.makedirs(X3, exist_ok=True)
os.environ.setdefault("EXP_OUT", os.path.join(X3, "_scratch"))
os.environ["EXP_SMOKE"] = "0"
sys.path.insert(0, HERE)
import explore as X                                  # the same imports and network loading as the X1 runs
F, P, PD, fc, rc, lb, p6 = X.F, X.P, X.PD, X.fc, X.rc, X.lb, X.p6

O4 = ("o4_s0", "o4_s1", "o4_s2")                     # planned
EXTRA = ("sup_s0", "sup_s1", "sup_s2", "rbm_all")    # added (DEVIATIONS_EXPLORE.md, item 10)
FIG3_GEOM = "3.1654_0.2769"


def ritz_pairs(Phi, inst, k=8):
    K, M = inst["K"], inst["M"]
    Kr = Phi.T @ (K @ Phi); Mr = Phi.T @ (M @ Phi)
    th, Y = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    return th[:k], Phi @ Y[:, :k]                    # Ritz values; Ritz vectors, M-orthonormal


def shape_errors(Phi, inst, w, V, k=8):
    """delta_i = theta_i / lambda_i - 1 and eps_i = min over s = +-1 of ||vt_i - s v_i||_K / ||v_i||_K, by order."""
    K = inst["K"]
    th, Vt = ritz_pairs(Phi, inst, k)
    lam = w[:k] ** 2
    eps, sgn = [], []
    for i in range(k):
        best = None
        for s in (1.0, -1.0):
            dvec = Vt[:, i] - s * V[:, i]
            e = float(np.sqrt(max(dvec @ (K @ dvec), 0.0) / lam[i]))
            if best is None or e < best[0]:
                best = (e, s)
        eps.append(best[0]); sgn.append(best[1])
    return dict(theta=[float(x) for x in th], lam=[float(x) for x in lam], delta=[float(x) for x in th / lam - 1],
                eps=eps, sign=sgn), Vt


def full_energy(inst, C, case, dt, lu, lu_m):
    """Energy of the whole reference response summed over the time steps, sum_k u_k^T K u_k, by the recursion of
    fem_cantilever.newmark_full (DEVIATIONS_EXPLORE.md, item 11)."""
    K, M = inst["K"], inst["M"]
    a0, a1, a2, a3, a4, a5 = fc.newmark_consts(dt)
    g = fc.load_history(case, dt, rc.N_STEPS)
    f_pat = inst["P"] @ np.asarray(case["w"], float)
    u = np.zeros(K.shape[0]); v = np.zeros_like(u); acc = lu_m.solve(f_pat * g[0])
    tot = 0.0
    for k in range(rc.N_STEPS):
        rhs = f_pat * g[k + 1] + M @ (a0 * u + a2 * v + a3 * acc) + C @ (a1 * u + a4 * v + a5 * acc)
        un = lu.solve(rhs)
        an = a0 * (un - u) - a2 * v - a3 * acc
        vn = v + dt * ((1 - fc.GAMMA) * acc + fc.GAMMA * an)
        u, v, acc = un, vn, an
        tot += float(u @ (K @ u))
    return tot


def modal_energies(inst, V16, C, case, dt):
    """Energy (summed over the time steps) in each of the first 16 exact modes. The 16-mode modal rollout equals the
    projection of the full Newmark response on these modes (Rayleigh damping)."""
    Cs = rc.reduced_rollout(V16, inst, C, case, dt)[0]
    lam = np.einsum("ij,ij->j", V16, inst["K"] @ V16)
    return lam * np.sum(Cs ** 2, axis=0)


def modal_shares(inst, V16, C, case, dt):
    """Share of the reference response's energy (summed over the time steps) in each of the first 16 exact modes.
    The 16-mode modal rollout equals the projection of the full Newmark response on these modes (Rayleigh damping)."""
    Cs = rc.reduced_rollout(V16, inst, C, case, dt)[0]
    lam = np.einsum("ij,ij->j", V16, inst["K"] @ V16)
    en = lam * np.sum(Cs ** 2, axis=0)
    return en / en.sum()


def main():
    t0 = time.time()
    geo, srcs, tr_xy, hull, rbf, formal = X.common()
    nets = X.load_nets()
    out = dict(bases_planned=list(O4), bases_added=list(EXTRA), geoms={})
    fig3 = None
    for g in geo["held"]:
        kg = X.key(g)
        fr = formal[kg]
        inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(*g)
        pr = P.fam_problem(inst, a_r, b_r, dt)
        V16 = V[:, :16]
        shares = [modal_shares(inst, V16, C, c, dt) for c in eval_cases]          # item 10: first 16 modes
        a0, a1 = fc.newmark_consts(dt)[:2]
        lu = X.spla.splu((inst["K"] + a0 * inst["M"] + a1 * C).tocsc()); lu_m = X.spla.splu(inst["M"].tocsc())
        Q = [rc.reduced_rollout(V16, inst, C, c, dt)[0] for c in eval_cases]       # exact modal amplitudes (item 12)
        tot = [full_energy(inst, C, c, dt, lu, lu_m) for c in eval_cases]
        pf = [modal_energies(inst, V16, C, c, dt) / tt for c, tt in zip(eval_cases, tot)]   # item 11
        rec = dict(w=[float(x) for x in w[:16]], inside_hull=fr["inside_hull"], dist_nearest=fr["dist_nearest"],
                   p16=[[float(x) for x in s] for s in shares], p=[[float(x) for x in s] for s in pf],
                   modal_check=dict(E2_formal=fr["modal_m8"]["e2_energy"]["per_case"],
                                    E2_pred16=[float(np.sqrt(np.sum(s[8:]))) for s in shares],
                                    E2_pred=[float(np.sqrt(max(1.0 - np.sum(s[:8]), 0.0))) for s in pf]))
        bases = {k: np.asarray(nets[k](pr)) for k in O4 + EXTRA[:3]}
        bases["rbm_all"] = PD.rbm_basis([(gg, s) for gg, s, _ in srcs], inst, g, m=X.M_MODES, tol=F.RBM_TOL)[0]
        for k, B in bases.items():
            se, Vt = shape_errors(B, inst, w, V)
            se["e_formal"] = fr[k]["ritz"]
            se["e_here"] = [float(np.sqrt(t / l) - 1) for t, l in zip(se["theta"], se["lam"])]
            eps2 = np.array(se["eps"]) ** 2
            se["E2_formal"] = fr[k]["e2_energy"]["per_case"]
            se["E2_pred16"] = [float(np.sqrt(np.sum(s[:8] * eps2) + np.sum(s[8:]))) for s in shares]
            se["E2_pred"] = [float(np.sqrt(np.sum(s[:8] * eps2) + max(1.0 - np.sum(s[:8]), 0.0))) for s in pf]
            # item 12: amplitude and phase errors of the modal amplitudes, from the reduced rollout on the Ritz vectors
            sg = np.array(se["sign"]); lam8 = np.array(se["lam"])
            Dm = Vt - V[:, :8] * sg[None, :]
            DKv = np.einsum("ij,ij->j", Dm, inst["K"] @ V[:, :8])
            parts, pred2 = [], []
            for c, q, tt, s in zip(eval_cases, Q, tot, pf):
                qt = rc.reduced_rollout(Vt, inst, C, c, dt)[0]                  # (steps, 8): Ritz amplitudes
                dq = qt * sg[None, :] - q[:, :8]
                A = float(np.sum(lam8 * np.sum(dq ** 2, axis=0)))
                Bsh = float(np.sum(lam8 * eps2 * np.sum(qt ** 2, axis=0)))
                Cx = float(np.sum(2.0 * DKv * np.sum(dq * qt, axis=0)))
                tr_ = max(1.0 - float(np.sum(s[:8])), 0.0) * tt
                pred2.append(float(np.sqrt(max(A + Bsh + Cx + tr_, 0.0) / tt)))
                parts.append(dict(amplitude_phase=A / tt, shape=Bsh / tt, cross=Cx / tt, above_mode_8=tr_ / tt))
            se["E2_pred2"] = pred2
            se["E2_pred2_parts"] = parts
            rec[k] = se
        out["geoms"][kg] = rec
        if kg == FIG3_GEOM:                          # figure 3: tip displacement histories
            tip = inst["tip_dof"]
            fig3 = dict(geom=kg, dt=float(dt), tip_xy=[float(x) for x in inst["tip_xy"]], cases={})
            for name, case in (("tip_pulse", eval_cases[8]), ("random_1", eval_cases[0])):
                U = fc.newmark_full(inst["K"], inst["M"], C, inst["P"], case, dt, rc.N_STEPS)[0]
                cr = dict(case=case, full=[float(x) for x in U[:, tip]])
                for k in ("o4_s0", "rbm_all"):
                    Cs = rc.reduced_rollout(bases[k], inst, C, case, dt)[0]
                    cr[k] = [float(x) for x in Cs @ bases[k][tip]]
                fig3["cases"][name] = cr
        print("%s: %.0f s" % (kg, time.time() - t0), flush=True)
    with open(os.path.join(X3, "x3_shape.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(X3, "x3_fig3.json"), "w") as f:
        json.dump(fig3, f, indent=1)
    print("done %.0f s" % (time.time() - t0), flush=True)


def energy_density(vec, ib, free):
    """Element means of sigma : epsilon for the free-dof field vec (plane stress, E = 1, nu = 0.3), and the total
    sum over elements of its integral (equal to vec^T K vec)."""
    full = np.zeros(ib.N); full[free] = vec
    Gr = ib.interpolate(full).grad                                   # (2, 2, elements, quadrature points)
    eps = 0.5 * (Gr + Gr.transpose(1, 0, 2, 3))
    dens = fc.LAM_PS * (eps[0, 0] + eps[1, 1]) ** 2 + 2 * fc.MU * np.sum(eps ** 2, axis=(0, 1))
    dx = ib.dx
    return np.sum(dens * dx, axis=1) / np.sum(dx, axis=1), float(np.sum(dens * dx))


def main_fig4():
    """Figure 4: the first four modes of G1, finite element against the Part I shared O4 network (seed 10)."""
    geo = F.geometries()
    xc, d = geo["G"]["G1"]
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
    pr = p6.problem(inst, a_r, b_r, dt, X.M_MODES, "hard_angle")
    params = X.p3.load_params(os.path.join(X.NETS, "p1_G1_o4_s10.pkl"))[0]
    B = np.asarray(lb.orthonormal_basis(params, pr))
    se, Vt = shape_errors(B, inst, w, V)
    with open(os.path.join(X.NETS, "p1_G1_o4_s10.json")) as f:
        fr = json.load(f)
    m = inst["mesh"]
    from skfem import Basis, ElementTriP2, ElementVector
    ib = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    n_all = ib.N
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten()
    free = np.setdiff1d(np.arange(n_all), D)
    assert len(free) == inst["n_free"]

    def vertex_field(vec):
        full = np.zeros(n_all); full[free] = vec
        return np.stack([full[ib.nodal_dofs[0]], full[ib.nodal_dofs[1]]], axis=1)
    fe = np.stack([vertex_field(V[:, i]) for i in range(4)])
    nn = np.stack([vertex_field(se["sign"][i] * Vt[:, i]) for i in range(4)])
    dens_fe, dens_diff, chk = [], [], []
    for i in range(4):
        dv = se["sign"][i] * Vt[:, i] - V[:, i]
        a, tot_fe = energy_density(V[:, i], ib, free)
        b, tot_d = energy_density(dv, ib, free)
        dens_fe.append(a); dens_diff.append(b)
        chk.append(dict(energy_fe=tot_fe, lam=float(w[i] ** 2), energy_diff=tot_d, dKd=float(dv @ (inst["K"] @ dv)),
                        eps_from_density=float(np.sqrt(tot_d / w[i] ** 2)), eps=float(se["eps"][i])))
    np.savez(os.path.join(X3, "x3_fig4.npz"), p=m.p.T, t=m.t.T, fe=fe, net=nn, w=w[:4], theta=np.array(se["theta"][:4]),
             eps=np.array(se["eps"][:4]), delta=np.array(se["delta"][:4]), e_formal=np.array(fr["net"]["ritz"][:4]),
             dens_fe=np.array(dens_fe), dens_diff=np.array(dens_diff), xc=xc, d=d)
    print("G1 modes: eps", [round(x, 4) for x in se["eps"][:4]], "e here", [float(np.sqrt(t / l) - 1) for t, l in
          zip(se["theta"][:4], se["lam"][:4])], "e formal", fr["net"]["ritz"][:4], flush=True)
    print("energy checks", json.dumps(chk), flush=True)


def main_notch():
    """Item 13: share of the energy of (Ritz vector - mode), modes 1 to 8, within 0.3 of the notch root, for the Part II
    bases of item 10 on the 16 held-out geometries (h_max = 0.2)."""
    from skfem import Basis, ElementTriP2, ElementVector
    t0 = time.time()
    geo, srcs, tr_xy, hull, rbf, formal = X.common()
    nets = X.load_nets()
    out = {}
    worst_check = 0.0
    for g in geo["held"]:
        inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(*g)
        pr = P.fam_problem(inst, a_r, b_r, dt)
        ib = Basis(inst["mesh"], ElementVector(ElementTriP2()), intorder=4)
        free = np.setdiff1d(np.arange(ib.N), ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten())
        assert len(free) == inst["n_free"]
        m = inst["mesh"]
        cen = m.p[:, m.t].mean(axis=1).T
        near = np.linalg.norm(cen - np.array([g[0], 1.0 - g[1]]), axis=1) < 0.3
        dx = ib.dx.sum(axis=1)
        bases = {k: np.asarray(nets[k](pr)) for k in O4 + EXTRA[:3]}
        bases["rbm_all"] = PD.rbm_basis([(gg, s_) for gg, s_, _ in srcs], inst, g, m=X.M_MODES, tol=F.RBM_TOL)[0]
        rec = dict(area_share_near=float(dx[near].sum() / dx.sum()))
        for k, B in bases.items():
            se, Vt = shape_errors(B, inst, w, V)
            shares, bands, clamp = [], [], []
            band = np.minimum((cen[:, 0] // 1.0).astype(int), 9)
            for i in range(8):
                D = Vt[:, i] - se["sign"][i] * V[:, i]
                el, tot = energy_density(D, ib, free)
                worst_check = max(worst_check, abs(tot / float(D @ (inst["K"] @ D)) - 1))
                shares.append(float(np.sum(el[near] * dx[near]) / np.sum(el * dx)))
                tot_e = np.sum(el * dx)                                       # item 14: along the beam
                bands.append([float(np.sum((el * dx)[band == b]) / tot_e) for b in range(10)])
                clamp.append(float(np.sum((el * dx)[cen[:, 0] < 0.3]) / tot_e))
            rec[k] = dict(share_near=shares, eps=se["eps"], share_bands=bands, share_clamp=clamp)
        out[X.key(g)] = rec
        print("%s: %.0f s" % (X.key(g), time.time() - t0), flush=True)
    with open(os.path.join(X3, "x3_notch.json"), "w") as f:
        json.dump(dict(radius=0.3, check_energy_sum_max_rel=worst_check, geoms=out), f, indent=1)
    print("energy-sum check, largest relative difference: %.2e" % worst_check, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig4":
        main_fig4()
    elif len(sys.argv) > 1 and sys.argv[1] == "notch":
        main_notch()
    else:
        main()
        main_fig4()
        main_notch()
```

## x3_figures.py

```python
"""X3 figures (EXPLORE_PLAN.md, X3; DEVIATIONS_EXPLORE.md, items 10 and 11). Reads the formal outputs and the outputs of
x3_analysis.py; writes x3/figs/fig*.pdf and .png.

  python -I -B x3_figures.py [FORMAL_RESULTS_DIR]

Colours: the categorical slots of the reference palette (validated on white, light mode, adjacent and all pairs for
the first three slots); objectives and methods keep the same colour in every figure.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
X3 = os.path.join(HERE, "x3")
FIGS = os.path.join(HERE, "figs")                     # DEVIATIONS_EXPLORE.md, item 17 (first versions: x3/figs)
os.makedirs(FIGS, exist_ok=True)
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "formal", "results")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
# One colour per objective or method in every figure (item 17)
C_O1, C_O2, C_O3, C_O4 = YELLOW, AQUA, ORANGE, BLUE
C_SUP, C_RB, C_FREQ, C_EIG = RED, GREEN, MAGENTA, VIOLET
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fde3d6", "#f9c1a6", "#f39a75", "#eb6834", "#c4501f", "#9a3d17", "#6e2b10"]
OBJ_COL = dict(o1=C_O1, o2=C_O2, o3=C_O3, o4=C_O4)
OBJ_MK = dict(o1="o", o2="s", o3="^", o4="D")
OBJ_NAME = dict(o1="O1 incremental potential", o2="O2 trace", o3="O3 sequential Rayleigh quotients", o4="O4 log-determinant")
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "text.color": INK, "axes.titlecolor": INK, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def jload(fn):
    with open(fn) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- figure 1: Part I learning curves

def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), sharey=True)
    table = {}
    for ax, (arch, objs, title) in zip(axs, (("p1", ("o1", "o2", "o3", "o4"), "Shared network"),
                                             ("p1sep", ("o2", "o3", "o4"), "One network per mode"))):
        for o in objs:
            runs = [jload(os.path.join(RES, "%s_%s_%s_s%d.json" % (arch, g, o, s))) for g in ("G1", "G2", "G3") for s in (10, 11)]
            t = np.array([l["t"] for l in runs[0]["logs"]])
            E = np.array([[l["ritz"][0] for l in r["logs"]] for r in runs])          # (6 runs, logs)
            med = np.median(E, axis=0)
            ax.fill_between(t, E.min(axis=0), E.max(axis=0), color=OBJ_COL[o], alpha=0.12, linewidth=0)
            ax.plot(t, med, color=OBJ_COL[o], label=OBJ_NAME[o])
            ax.plot(t[-1], med[-1], OBJ_MK[o], color=OBJ_COL[o], ms=5, mec="white", mew=0.8, zorder=5, clip_on=False)
            table["%s_%s" % (arch, o)] = dict(t=t.tolist(), median=med.tolist(), min=E.min(axis=0).tolist(), max=E.max(axis=0).tolist())
        ax.set_yscale("log")
        ax.set_xlim(0, 20000)
        ax.set_xticks([0, 5000, 10000, 15000, 20000]); ax.set_xticklabels(["0", "5,000", "10,000", "15,000", "20,000"])
        ax.set_xlabel("Adam iteration")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("Error of the first frequency, $e_1$")
    fig.subplots_adjust(bottom=0.3, wspace=0.12)
    handles = [plt.Line2D([], [], color=OBJ_COL[o], marker=OBJ_MK[o], ms=5, mec="white", mew=0.8) for o in OBJ_COL]
    fig.legend(handles, [OBJ_NAME[o] for o in OBJ_COL], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.06),
               handlelength=2.2, columnspacing=1.6)
    fig.text(0.5, 0.0, "Lines: median over the six geometry-seed runs (G1 to G3, seeds 10 and 11); bands: smallest to largest.",
             ha="center", color=INK2, fontsize=7)
    save(fig, "fig1_partI_learning_curves")
    return table


# ----------------------------------------------------------------------------- figure 2: Part II per held-out geometry

def fig2():
    d = {}
    for k in range(4):
        d.update(jload(os.path.join(RES, "e2_shard%d.json" % k))["geoms"])
    inside = sorted([g for g in d if d[g]["inside_hull"]], key=lambda g: d[g]["dist_nearest"])
    outside = sorted([g for g in d if not d[g]["inside_hull"]], key=lambda g: d[g]["dist_nearest"])
    order = inside + outside
    xs = np.array([i + (1.0 if i >= len(inside) else 0.0) for i in range(len(order))])        # a gap between groups
    mx = lambda r: max(abs(x) for x in r["ritz"])
    series = [("Shared network, O4 (three seeds)", C_O4, "o", lambda r: [mx(r["o4_s%d" % s]) for s in range(3)]),
              ("Supervised network (three seeds)", C_SUP, "s", lambda r: [mx(r["sup_s%d" % s]) for s in range(3)]),
              ("Mapped reduced basis", C_RB, "^", lambda r: [mx(r["rbm_all"])]),
              ("Frequency regression", C_FREQ, "D", lambda r: [mx(r["freq_rbf"])])]
    fig, ax = plt.subplots(figsize=(180 * MM, 70 * MM))
    off = np.array([-0.27, -0.09, 0.09, 0.27])
    table = {g: {} for g in order}
    for j, (name, col, mk, f) in enumerate(series):
        for x, g in zip(xs, order):
            v = np.array(f(d[g]))
            table[g][name] = v.tolist()
            if len(v) > 1:
                ax.plot([x + off[j]] * 2, [v.min(), v.max()], color=col, lw=1.2, solid_capstyle="round")
            ax.plot(x + off[j], np.median(v), mk, color=col, ms=5.6 if mk == "^" else 4.8, mec="white", mew=0.7, zorder=5,
                    label=name if g == order[0] else None)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.2f, %.2f" % tuple(map(float, g.split("_"))) for g in order], rotation=50, ha="right")
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Largest frequency error,\n" + r"$|e_i|$ over modes 1 to 8")
    ax.set_xlabel(r"Held-out geometry (notch position $x_c$, depth $d$)")
    y_top = ax.get_ylim()[1]
    ax.text(np.mean(xs[:len(inside)]), y_top, "Inside the hull of the training geometries", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.text(np.mean(xs[len(inside):]), y_top, "Outside", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.axvline(xs[len(inside)] - 1.0, color=AXIS, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4, handlelength=1.2, columnspacing=1.4)
    fig.text(0.5, -0.36, "Networks: marker at the median of the three seeds, line from the smallest to the largest.\n"
             "Geometries in each group sorted by the distance to the nearest training geometry; $h_{max}$ = 0.2.",
             ha="center", va="top", color=INK2, fontsize=7, transform=fig.transFigure)
    save(fig, "fig2_partII_heldout_frequency_errors")
    return table


# ----------------------------------------------------------------------------- figure 3: tip displacement histories

def fig3():
    f3 = jload(os.path.join(X3, "x3_fig3.json"))
    w1 = jload(os.path.join(X3, "x3_shape.json"))["geoms"][f3["geom"]]["w"][0]
    T1 = 2 * np.pi / w1
    fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 95 * MM), sharex=True, gridspec_kw=dict(height_ratios=(3, 2)))
    titles = dict(tip_pulse="Tip pulse", random_1="First random case")
    table = {}
    for c, name in enumerate(("tip_pulse", "random_1")):
        cr = f3["cases"][name]
        full = np.array(cr["full"]); t = np.arange(len(full)) * f3["dt"] / T1
        sc = np.max(np.abs(full))
        ax = axs[0, c]
        ax.plot(t, full / sc, color=MUTED, lw=3.0, label="Full model", zorder=2)
        ax.plot(t, np.array(cr["o4_s0"]) / sc, color=C_O4, lw=0.9, label="Shared network, O4, seed 0", zorder=4)
        ax.plot(t, np.array(cr["rbm_all"]) / sc, color=C_RB, lw=1.4, label="Mapped reduced basis", zorder=3)
        ax.set_title(titles[name], loc="left")
        ax2 = axs[1, c]
        e_net = (np.array(cr["o4_s0"]) - full) / sc; e_rb = (np.array(cr["rbm_all"]) - full) / sc
        ax2.plot(t, e_net, color=C_O4, lw=1.1)
        ax2.plot(t, e_rb, color=C_RB, lw=1.1)
        ax2.axhline(0, color=AXIS, lw=0.6)
        ax2.set_xlabel(r"Time / first period, $t / T_1$")
        table[name] = dict(scale=float(sc), max_abs_err_net=float(np.max(np.abs(e_net))), max_abs_err_rb=float(np.max(np.abs(e_rb))),
                           e1_net=float(np.linalg.norm(e_net) / np.linalg.norm(full / sc)),
                           e1_rb=float(np.linalg.norm(e_rb) / np.linalg.norm(full / sc)))
    axs[0, 0].set_ylabel("Tip displacement\n(scaled by its maximum)")
    axs[1, 0].set_ylabel("Reduced minus full\n(same scale)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), handlelength=2.2)
    xc, d = map(float, f3["geom"].split("_"))
    fig.text(0.5, -0.07, "Held-out geometry $x_c$ = %.4f, $d$ = %.4f (inside the training hull); vertical displacement at "
             "the middle of the free end; $h_{max}$ = 0.2.\nAt the scale of the upper panels the three curves coincide; "
             "the lower panels show the differences." % (xc, d), ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig3_tip_displacement_history")
    return table


# ----------------------------------------------------------------------------- figure 4: first four modes of G1

def fig4():
    z = np.load(os.path.join(X3, "x3_fig4.npz"))
    p, t, fe, nn = z["p"], z["t"], z["fe"], z["net"]
    tri = Triangulation(p[:, 0], p[:, 1], t)
    cen = p[t].mean(axis=1)
    root = np.array([float(z["xc"]), 1.0 - float(z["d"])])
    near = np.linalg.norm(cen - root, axis=1) < 0.3
    cm_b = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cm_o = LinearSegmentedColormap.from_list("seq_orange", SEQ_ORANGE)
    fig, axs = plt.subplots(4, 3, figsize=(180 * MM, 82 * MM),
                            gridspec_kw=dict(wspace=0.08, hspace=0.38, width_ratios=(10.6, 10.6, 3.0)))
    table = []
    for i in range(4):
        amp = np.max(np.linalg.norm(fe[i], axis=1))
        s = 0.5 / amp
        for c, U in enumerate((fe[i], nn[i])):
            ax = axs[i, c]
            tr = Triangulation(p[:, 0] + s * U[:, 0], p[:, 1] + s * U[:, 1], t)
            ax.tripcolor(tr, np.linalg.norm(U, axis=1) / amp, cmap=cm_b, vmin=0, vmax=1, shading="gouraud", rasterized=True)
        rel = z["dens_diff"][i] / z["dens_fe"][i].max()
        share = float(np.sum(z["dens_diff"][i][near] * 1.0) / np.sum(z["dens_diff"][i]))
        im = axs[i, 2].tripcolor(tri, facecolors=np.clip(rel, 1e-6, None), cmap=cm_o,
                                 norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1), rasterized=True)
        e = float(np.sqrt(z["theta"][i]) / z["w"][i] - 1)
        axs[i, 0].set_title("Mode %d" % (i + 1), loc="left", fontsize=7.5, pad=2)
        axs[i, 1].set_title(r"$e_{%d}$ = %.1e, $\varepsilon_{%d}$ = %.3f" % (i + 1, e, i + 1, z["eps"][i]), loc="left",
                            fontsize=7.5, color=INK2, pad=2)
        axs[i, 2].set_title("%.0f%% inside" % (100 * share), loc="left", fontsize=7.5, color=INK2, pad=2)
        axs[i, 2].add_patch(matplotlib.patches.Circle(root, 0.3, fill=False, ec=INK2, lw=0.6, zorder=5))
        table.append(dict(mode=i + 1, e=e, eps=float(z["eps"][i]), delta=float(z["delta"][i]), share_near_notch=share,
                          max_rel_density=float(rel.max())))
        for c, ax in enumerate(axs[i]):
            ax.set_aspect("equal")
            if c < 2:
                ax.set_xlim(-0.2, 10.4); ax.set_ylim(-0.55, 1.55)
            else:                                              # zoom on the notch
                ax.set_xlim(root[0] - 0.75, root[0] + 0.75); ax.set_ylim(0.0, 1.05)
            ax.axis("off")
    for c, name in enumerate(("Finite element", "Network (Ritz vector)", "Difference, near the notch")):
        axs[0, c].text(0.5, 1.45, name, transform=axs[0, c].transAxes, ha="center", va="bottom", fontsize=8)
    cax = fig.add_axes([0.945, 0.18, 0.01, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("strain energy density of the difference,\nrelative to the largest value of the mode", color=INK2, fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    fig.text(0.5, 0.02, "G1 ($x_c$ = %.4f, $d$ = %.4f); Part I shared network, O4, seed 10; $h_{max}$ = 0.2. Columns 1 and 2: "
             "deformed shape (largest displacement drawn as 0.5),\nshaded by displacement magnitude (0 to the largest). "
             "Column 3: element mean of the strain energy density of\n(Ritz vector minus mode), log scale, "
             "near the notch; circle: radius 0.3 around the notch root, with the share of the difference's energy inside it."
             % (float(z["xc"]), float(z["d"])), ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig4_G1_modes")
    return table


# ----------------------------------------------------------------------------- figure 5: shape errors and the E2 prediction

def fig5():
    d = jload(os.path.join(X3, "x3_shape.json"))
    G = d["geoms"]
    groups = (("Shared network, O4 (planned)", C_O4, "o", d["bases_planned"]),
              ("Supervised network (added)", C_SUP, "s", ["sup_s0", "sup_s1", "sup_s2"]),
              ("Mapped reduced basis (added)", C_RB, "^", ["rbm_all"]))
    fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 118 * MM), gridspec_kw=dict(hspace=0.45, wspace=0.28))
    table = {}
    for c, (name, col, mk, ks) in enumerate(groups):
        sd = np.array([np.sqrt(max(x, 1e-300)) for g in G.values() for k in ks for x in g[k]["delta"]])
        ep = np.array([x for g in G.values() for k in ks for x in g[k]["eps"]])
        ax = axs[0, c]
        lo, hi = 1e-3, 3e-1
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
        ax.plot(sd, ep, mk, color=col, ms=3.2, mec="white", mew=0.4, alpha=0.9, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(r"$\sqrt{\delta_i}$, $\delta_i = \theta_i / \lambda_i - 1$")
        if c == 0:
            ax.set_ylabel(r"Shape error $\varepsilon_i$ (energy norm)")
        ax.set_title(name, loc="left")
        r = ep / sd
        ax.text(0.04, 0.96, "median ratio %.3f\n90%% below %.2f" % (np.median(r), np.quantile(r, 0.9)), transform=ax.transAxes,
                va="top", fontsize=6.5, color=INK2)
        E2 = np.array([x for g in G.values() for k in ks for x in g[k]["E2_formal"]])
        Ep = np.array([x for g in G.values() for k in ks for x in g[k]["E2_pred"]])
        Ep2 = np.array([x for g in G.values() for k in ks for x in g[k]["E2_pred2"]])
        ax = axs[1, c]
        lo2, hi2 = 0.0, 0.16
        ax.plot([lo2, hi2], [lo2, hi2], color=MUTED, lw=0.8, zorder=1)
        ax.plot(E2, Ep, mk, color=col, mfc="white", ms=3.4, mew=0.7, alpha=0.9, zorder=3)
        ax.plot(E2, Ep2, mk, color=col, ms=3.2, mec="white", mew=0.4, alpha=0.9, zorder=4)
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2); ax.set_aspect("equal")
        ax.set_xlabel("E2, formal evaluation")
        if c == 0:
            ax.set_ylabel("E2 predicted mode by mode")
        rr, rr2 = Ep / E2, Ep2 / E2
        ax.set_xticks([0, 0.05, 0.10, 0.15]); ax.set_yticks([0, 0.05, 0.10, 0.15])
        ax.text(0.04, 0.96, "median ratio\nopen %.3f\nfilled %.3f" % (np.median(rr), np.median(rr2)),
                transform=ax.transAxes, va="top", fontsize=6.2, color=INK2, linespacing=1.3)
        table[name] = dict(n_modes=int(len(ep)), eps_over_sqrt_delta=dict(median=float(np.median(r)), q10=float(np.quantile(r, 0.1)),
                           q90=float(np.quantile(r, 0.9)), min=float(r.min()), max=float(r.max())),
                           n_cases=int(len(E2)), E2pred_over_E2=dict(median=float(np.median(rr)), q10=float(np.quantile(rr, 0.1)),
                           q90=float(np.quantile(rr, 0.9)), min=float(rr.min()), max=float(rr.max())),
                           E2pred2_over_E2=dict(median=float(np.median(rr2)), q10=float(np.quantile(rr2, 0.1)),
                           q90=float(np.quantile(rr2, 0.9)), min=float(rr2.min()), max=float(rr2.max())))
    hd = [plt.Line2D([], [], ls="", marker="o", color=MUTED, mfc="white", ms=4, mew=0.8),
          plt.Line2D([], [], ls="", marker="o", color=MUTED, ms=4, mec="white", mew=0.5)]
    fig.subplots_adjust(bottom=0.16)
    fig.legend(hd, ["Open: shape errors only", "Filled: shape, amplitude and phase errors"], loc="upper center",
               bbox_to_anchor=(0.5, 0.075), ncol=2, columnspacing=2.0)
    fig.text(0.5, -0.04, "Top: each point one mode (i = 1 to 8) of one basis on one of the 16 held-out geometries; grey line: equality. "
             "Bottom: each point one evaluation case.\nOpen: E2 from the shape errors alone, "
             r"$\sqrt{\sum_{i \leq 8} p_i \varepsilon_i^2 + \sum_{i > 8} p_i}$, $p_i$ the share of the reference response's "
             r"energy in exact mode $i$." "\nFilled: also the errors of the modal amplitudes over time (amplitude and phase), "
             "from the reduced model's amplitudes on its Ritz vectors. Both ignore coupling between different modes.",
             ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig5_shape_errors_and_E2")
    return table


if __name__ == "__main__":
    out = dict(fig1=fig1(), fig2=fig2(), fig3=fig3(), fig4=fig4(), fig5=fig5())
    with open(os.path.join(FIGS, "figure_data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(fig3=out["fig3"], fig4=out["fig4"], fig5=out["fig5"]), indent=1))
```

## x12_report.py

```python
"""Tables and figures 6 to 8 for X1 and X2 (EXPLORE_PLAN.md; DEVIATIONS_EXPLORE.md, items 16 and 17).

  python -I -B x12_report.py EXPLORE_RESULTS_DIR FORMAL_RESULTS_DIR

Writes TABLES_EXPLORE.md next to this file and figs/fig6 to fig8 (PDF and PNG), with figs/figure_data_x12.json.
"""
import os, sys, json
import numpy as np
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x3_figures as XF                              # style, colours and save()
plt = XF.plt
matplotlib = XF.matplotlib
RES, FRES = sys.argv[1], sys.argv[2]
MM = XF.MM


def jload(fn):
    with open(fn) as f:
        return json.load(f)


ACC = {}
for s in range(4):
    ACC.update(jload(os.path.join(RES, "x1_acc_shard%d.json" % s))["geoms"])
FINE = {}
for k in range(4):
    FINE.update(jload(os.path.join(RES, "x1_fine%d.json" % k))["geoms"])
TIM = jload(os.path.join(RES, "x1_timing.json"))
GEOMS = sorted(set(k.rsplit("_h", 1)[0] for k in ACC))
FINE_G = sorted(set(k.rsplit("_h", 1)[0] for k in FINE))
H3 = ("0.2", "0.1", "0.05")
NETS = [p + "%s_s%d" % (o, s) for p, objs in (("", ("o4", "o3s", "sup")), ("sep_", ("o4", "o3s"))) for o in objs for s in range(3)]
KINDS = [("Shared O4", ["o4_s0", "o4_s1", "o4_s2"]), ("Shared O3s", ["o3s_s0", "o3s_s1", "o3s_s2"]),
         ("Supervised", ["sup_s0", "sup_s1", "sup_s2"]), ("Per mode O4", ["sep_o4_s0", "sep_o4_s1", "sep_o4_s2"]),
         ("Per mode O3s", ["sep_o3s_s0", "sep_o3s_s1", "sep_o3s_s2"]), ("Modal truncation", ["modal_m8"]),
         ("Mapped reduced basis", ["rbm_all"]), ("Frequency regression", ["freq_rbf"])]
mx = lambda r: max(abs(x) for x in r["ritz"])
L = []                                               # lines of TABLES_EXPLORE.md
out = {}


def fmt(x, p=2):
    return ("%%.%de" % p) % x


# ----------------------------------------------------------------------------- X1a tables
L += ["# Exploratory runs: tables for X1 and X2", "",
      "Written by x12_report.py from the laptop outputs (explore/results, SHA256SUMS_explore_results.txt). "
      "Medians are over the 16 held-out geometries (and, for the networks, the three seeds) unless stated.", "",
      "## X1a.1 Largest frequency error, max over modes 1 to 8 of |e_i|, against mesh size", "",
      "| Basis | h 0.2 | h 0.1 | h 0.05 | ratio 0.05 / 0.2: median (range) |", "|---|---|---|---|---|"]
tab1 = {}
for name, ks in KINDS:
    v = {h: np.array([mx(ACC[g + "_h" + h][k]) for g in GEOMS for k in ks]) for h in H3}
    rat = v["0.05"] / v["0.2"]
    tab1[name] = {h: float(np.median(v[h])) for h in H3}
    tab1[name]["ratio"] = [float(np.median(rat)), float(rat.min()), float(rat.max())]
    L.append("| %s | %s | %s | %s | %.2f (%.2f to %.2f) |" % (name, fmt(np.median(v["0.2"])), fmt(np.median(v["0.1"])),
                                                           fmt(np.median(v["0.05"])), np.median(rat), rat.min(), rat.max()))
dw = np.array([ACC[g + "_h0.05"]["w"][0] / ACC[g + "_h0.2"]["w"][0] - 1 for g in GEOMS])
dw8 = np.array([max(abs(ACC[g + "_h0.05"]["w"][i] / ACC[g + "_h0.2"]["w"][i] - 1) for i in range(8)) for g in GEOMS])
nfree = {h: int(np.median([ACC[g + "_h" + h]["n_free"] for g in GEOMS])) for h in H3}
L += ["", "Degrees of freedom (median): %s. Finite element first frequency, h 0.05 against 0.2: median change %s (range %s to %s); "
      "largest change over modes 1 to 8: median %s, largest %s." % (", ".join("%s at h %s" % ("{:,}".format(nfree[h]), h) for h in H3),
      fmt(np.median(dw)), fmt(dw.min()), fmt(dw.max()), fmt(np.median(dw8)), fmt(dw8.max())), ""]
out["x1a_table1"] = tab1

L += ["## X1a.2 E1 (tip displacement), E2 and the largest sin^2, against mesh size", "",
      "Median over the evaluation cases, then over geometries (and seeds).", "",
      "| Basis | E1 h 0.2 | E1 h 0.1 | E1 h 0.05 | E2 h 0.2 | E2 h 0.1 | E2 h 0.05 | max sin^2 h 0.2 | h 0.05 | drop reproduced h 0.2 | h 0.05 |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
tab2 = {}
for name, ks in KINDS:
    if name == "Frequency regression":
        dr = {h: np.median([ACC[g + "_h" + h][k]["drop_reproduced"] for g in GEOMS for k in ks]) for h in H3}
        L.append("| %s | | | | | | | | | %.3f | %.3f |" % (name, dr["0.2"], dr["0.05"]))
        tab2[name] = dict(drop={h: float(dr[h]) for h in H3})
        continue
    e1 = {h: np.median([ACC[g + "_h" + h][k]["e1_tip_disp"]["median"] for g in GEOMS for k in ks]) for h in H3}
    e2 = {h: np.median([ACC[g + "_h" + h][k]["e2_energy"]["median"] for g in GEOMS for k in ks]) for h in H3}
    s2 = {h: np.median([max(ACC[g + "_h" + h][k]["sin2"]) for g in GEOMS for k in ks]) for h in H3}
    dr = {h: np.median([ACC[g + "_h" + h][k]["drop_reproduced"] for g in GEOMS for k in ks]) for h in H3}
    tab2[name] = dict(e1={h: float(e1[h]) for h in H3}, e2={h: float(e2[h]) for h in H3}, sin2={h: float(s2[h]) for h in H3},
                      drop={h: float(dr[h]) for h in H3})
    L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %.3f | %.3f |" % (
        name, e1["0.2"], e1["0.1"], e1["0.05"], e2["0.2"], e2["0.1"], e2["0.05"], fmt(s2["0.2"]), fmt(s2["0.05"]), dr["0.2"], dr["0.05"]))
L += ["", "Drop reproduced: the share of the finite element drop of the first frequency (notched against plain cantilever on the same mesh) "
      "that the basis reproduces, as in the formal evaluation.", ""]
out["x1a_table2"] = tab2

L += ["## X1a.3 The four FINE geometries down to h_max 0.025", "",
      "| Geometry | DOFs at 0.025 | Basis | max e_i at 0.2 | 0.1 | 0.05 | 0.025 | E2 at 0.2 | 0.025 |", "|---|---|---|---|---|---|---|---|---|"]
tab3 = {}
for g in FINE_G:
    r25 = FINE[g + "_h0.025"]
    for k, nm in (("o4_s0", "Shared O4 s0"), ("sup_s0", "Supervised s0"), ("sep_o4_s0", "Per mode O4 s0"), ("rbm_all", "Mapped reduced basis"),
                  ("freq_rbf", "Frequency regression")):
        vals = [mx(ACC[g + "_h" + h][k]) for h in H3] + [mx(r25[k])]
        e2a = ACC[g + "_h0.2"][k]["e2_energy"]["median"] if k != "freq_rbf" else None
        e2b = r25[k]["e2_energy"]["median"] if k != "freq_rbf" else None
        tab3["%s %s" % (g, k)] = dict(max_e=vals, e2=[e2a, e2b])
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (g, "{:,}".format(r25["n_free"]), nm, *[fmt(v) for v in vals],
                                                               "%.4f" % e2a if e2a is not None else "", "%.4f" % e2b if e2b is not None else ""))
    L.append("| %s | | finite element w_1 change 0.2 to 0.025 | %s | | | | | |" % (g, fmt(r25["w"][0] / ACC[g + "_h0.2"]["w"][0] - 1)))
out["x1a_table3"] = tab3

L += ["", "## X1a.4 Checks at h_max 0.2 against the formal evaluation (same laptop)", "",
      "| Geometry | streaming vs eval_basis | Ritz, absolute | Ritz, relative (basis) | FE frequencies | E1, E2 vs formal |", "|---|---|---|---|---|---|"]
for g in GEOMS:
    r = ACC[g + "_h0.2"]
    kr = max(r["check_ritz_vs_formal_rel"], key=r["check_ritz_vs_formal_rel"].get)
    L.append("| %s | %s | %s | %s (%s) | %s | %s |" % (g, fmt(max(r["check_stream_vs_eval_basis"].values()), 1), fmt(max(r["check_ritz_vs_formal_abs"].values()), 1),
             fmt(r["check_ritz_vs_formal_rel"][kr], 1), kr, fmt(r["check_w_vs_formal"], 1), fmt(max(r["check_metrics_vs_formal"].values()), 1)))

# ----------------------------------------------------------------------------- X1b table
rows = TIM["rows"]
HT = sorted(set(r["h_max"] for r in rows), reverse=True)
comp = [("t_mesh", "mesh"), ("t_build", "mesh and assembly"), ("t_eig8", "8 eigenpairs"), ("t_problem", "network inputs"),
        ("t_net_first", "network, first call"), ("t_net", "network, compiled"), ("t_net_numpy", "network, NumPy"),
        ("t_reduced_ops", "reduced operators"), ("t_rbm", "mapped reduced basis"), ("t_full_factorisation", "factorisation"),
        ("t_full_step", "full step"), ("t_reduced_step", "reduced step"), ("t_full_sim_640", "full simulation, 640 steps"),
        ("t_reduced_sim_640", "reduced simulation, 640 steps")]
med = {h: {c: float(np.median([r[c] for r in rows if r["h_max"] == h])) for c, _ in comp + [("n_free", "")]} for h in HT}
L += ["", "## X1b Online cost on one core (seconds; median over the four FINE geometries of the median of three repeats)", "",
      "| Component | " + " | ".join("h %s (%s DOFs)" % (h, "{:,}".format(int(med[h]["n_free"]))) for h in HT) + " | growth exponent |",
      "|---|" + "---|" * (len(HT) + 1)]
n = np.array([med[h]["n_free"] for h in HT])
slopes = {}
for c, nm in comp:
    y = np.array([med[h][c] for h in HT])
    sl = float(np.polyfit(np.log(n), np.log(y), 1)[0])
    slopes[c] = sl
    L.append("| %s | %s | %.2f |" % (nm, " | ".join("%.4g" % med[h][c] for h in HT), sl))
route = {h: med[h]["t_problem"] + med[h]["t_net"] + med[h]["t_reduced_ops"] for h in HT}
L += ["", "Network route = network inputs + compiled network + reduced operators: " +
      ", ".join("%.3g s at h %s" % (route[h], h) for h in HT) + ". Growth exponent: slope of log time against log DOFs over the four meshes. "
      "Offline preparation of the 24 sources of the mapped reduced basis: %.2f s. Largest differences of the timed bases from lb.orthonormal_basis: "
      "compiled with mesh arguments %s, NumPy %s." % (TIM["t_sources_offline"], fmt(max(r["net_args_vs_lb"] for r in rows), 1),
                                                     fmt(max(r["net_numpy_vs_lb"] for r in rows), 1)), ""]
out["x1b"] = dict(median=med, slopes=slopes, network_route=route)

# ----------------------------------------------------------------------------- X2 table
L += ["## X2 Shared O2 and O3, 100,000 iterations, next to the formal 20,000-iteration runs", "",
      "| Objective | Geometry | Seed | formal: e_1 | formal: max e_3..e_8 | 100k run: e_1 at 20k | 50k | 100k | max e_3..e_8 at 100k | E2 formal | E2 100k | train s, formal | 100k |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
x2 = {}
for o in ("o2", "o3"):
    for g in ("G1", "G2", "G3"):
        for s in (10, 11):
            x = jload(os.path.join(RES, "x2_%s_%s_s%d.json" % (g, o, s)))
            f = jload(os.path.join(FRES, "p1_%s_%s_s%d.json" % (g, o, s)))
            Lg = {l["t"]: l for l in x["logs"]}
            rec = dict(formal_e1=f["net"]["ritz"][0], formal_hi=max(abs(v) for v in f["net"]["ritz"][2:]),
                       e1=[[l["t"], l["ritz"][0]] for l in x["logs"]], hi=[[l["t"], max(abs(v) for v in l["ritz"][2:])] for l in x["logs"]],
                       final_e1=x["net"]["ritz"][0], E2=x["net"]["e2_energy"]["median"], E2_formal=f["net"]["e2_energy"]["median"],
                       t_train=x["t_train"], t_formal=f["t_train"], formal_logs=[[l["t"], l["ritz"][0]] for l in f["logs"]])
            x2["%s_%s_s%d" % (o, g, s)] = rec
            L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.0f | %.0f |" % (
                o.upper(), g, s, fmt(rec["formal_e1"]), fmt(rec["formal_hi"]), fmt(Lg[20000]["ritz"][0]), fmt(Lg[50000]["ritz"][0]),
                fmt(Lg[100000]["ritz"][0]), fmt(max(abs(v) for v in Lg[100000]["ritz"][2:])), rec["E2_formal"], rec["E2"], rec["t_formal"], rec["t_train"]))
for o in ("o2", "o3"):
    rr = [v for k, v in x2.items() if k.startswith(o)]
    L.append("")
    L.append("%s: median e_1, formal %s, 100k run %s; below 0.5%%: %d of 6, below 1%%: %d of 6; median E2 %.3f (formal %.3f)." % (
        o.upper(), fmt(np.median([r["formal_e1"] for r in rr])), fmt(np.median([r["final_e1"] for r in rr])),
        sum(r["final_e1"] < 0.005 for r in rr), sum(r["final_e1"] < 0.01 for r in rr), np.median([r["E2"] for r in rr]),
        np.median([r["E2_formal"] for r in rr])))
o4 = [jload(os.path.join(FRES, "p1_%s_o4_s%d.json" % (g, s)))["net"] for g in ("G1", "G2", "G3") for s in (10, 11)]
o1 = [jload(os.path.join(FRES, "p1_%s_o1_s%d.json" % (g, s)))["net"] for g in ("G1", "G2", "G3") for s in (10, 11)]
ref = dict(o4_e1=float(np.median([n_["ritz"][0] for n_ in o4])), o4_E2=float(np.median([n_["e2_energy"]["median"] for n_ in o4])),
           o1_e1=float(np.median([n_["ritz"][0] for n_ in o1])), o1_E2=float(np.median([n_["e2_energy"]["median"] for n_ in o1])))
L += ["", "For reference, the formal 20,000-iteration runs of the same geometries and seeds: O4 median e_1 %s (E2 %.3f), O1 %s (E2 %.3f)." % (
      fmt(ref["o4_e1"]), ref["o4_E2"], fmt(ref["o1_e1"]), ref["o1_E2"]), ""]
out["x2"] = x2
out["x2_ref"] = ref
with open(os.path.join(HERE, "TABLES_EXPLORE.md"), "w") as f:
    f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- figure 6: online cost
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 86 * MM), gridspec_kw=dict(wspace=0.55))
    nn = np.array([med[h]["n_free"] for h in HT])
    ax = axs[0]
    ser = [("Mesh and assembly", XF.MUTED, "s", [med[h]["t_build"] for h in HT], "-"),
           ("8 eigenpairs (shift-invert)", XF.C_EIG, "D", [med[h]["t_eig8"] for h in HT], "-"),
           ("Mapped reduced basis", XF.C_RB, "^", [med[h]["t_rbm"] for h in HT], "-"),
           ("Network O4: inputs, compiled network, reduced operators", XF.C_O4, "o", [route[h] for h in HT], "-"),
           ("Network O4: first call, compilation included", XF.C_O4, "v", [med[h]["t_net_first"] for h in HT], "--")]
    for nm, col, mk, y, ls in ser:
        ax.plot(nn, y, ls, color=col, lw=1.5, marker=mk, ms=4.5, mec=col if ls == "--" else "white", mew=0.9 if ls == "--" else 0.7,
                mfc="white" if ls == "--" else col, label=nm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Degrees of freedom"); ax.set_ylabel("Time on one core (s)")
    ax.set_title("A reduced model for a new geometry", loc="left")
    ax.set_xlim(nn[0] / 1.5, nn[-1] * 1.5)
    ax = axs[1]
    for nm, col, y, lw in (("Full model: factorisation\nand 640 Newmark steps", XF.MUTED, [med[h]["t_full_sim_640"] for h in HT], 2.6),
                           ("Reduced model, 8 coordinates:\n640 steps", XF.INK2, [med[h]["t_reduced_sim_640"] for h in HT], 1.5)):
        ax.plot(nn, y, "-", color=col, lw=lw, marker="o", ms=4.5, mec="white", mew=0.7)
        ax.annotate(nm, (nn[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5, color=XF.INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Degrees of freedom")
    ax.set_title("One simulation", loc="left")
    ax.set_xlim(nn[0] / 1.5, nn[-1] * 1.5)
    fig.subplots_adjust(bottom=0.32)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03), handlelength=2.4, columnspacing=1.6, title="Left panel",
               title_fontsize=7)
    fig.text(0.5, -0.07, "Laptop, one core, single-threaded libraries; median over the four FINE geometries of the median of three repeats; "
             "h_max 0.2, 0.1, 0.05 and 0.025.", ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig6_online_cost")


# ----------------------------------------------------------------------------- figure 7: accuracy against mesh size
def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), sharey=True, gridspec_kw=dict(wspace=0.08))
    series = [("Supervised network", XF.C_SUP, "s", "-", ["sup_s0", "sup_s1", "sup_s2"]),
              ("Shared network, O4", XF.C_O4, "o", "-", ["o4_s0", "o4_s1", "o4_s2"]),
              ("One network per mode, O4", XF.C_O4, "o", "--", ["sep_o4_s0", "sep_o4_s1", "sep_o4_s2"]),
              ("Mapped reduced basis (sources at h_max 0.2)", XF.C_RB, "^", "-", ["rbm_all"]),
              ("Frequency regression", XF.C_FREQ, "D", "-", ["freq_rbf"])]
    data = {}
    for nm, col, mk, ls, ks in series:
        y = [np.median([mx(ACC[g + "_h" + h][k]) for g in GEOMS for k in ks]) for h in H3]
        x = [nfree[h] for h in H3]
        kw = dict(color=col, marker=mk, ms=4.5, mec=col if ls == "--" else "white", mew=0.9 if ls == "--" else 0.7,
                  mfc="white" if ls == "--" else col)
        axs[0].plot(x, y, ls, label=nm, **kw)
        H4 = H3 + ("0.025",)
        yf = [np.median([mx((FINE if h == "0.025" else ACC)[g + "_h" + h][k]) for g in FINE_G for k in ks]) for h in H4]
        xf = [np.median([(FINE if h == "0.025" else ACC)[g + "_h" + h]["n_free"] for g in FINE_G]) for h in H4]
        axs[1].plot(xf, yf, ls, **kw)
        data[nm + " xf"] = [float(v) for v in xf]
        data[nm] = dict(held_out=[float(v) for v in y], fine=[float(v) for v in yf])
    xt0 = [nfree[h] for h in H3]
    xt1 = data["Shared network, O4 xf"]
    for ax, t, xt, hs in zip(axs, ("16 held-out geometries", "4 FINE geometries"), (xt0, xt1), (H3, H3 + ("0.025",))):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("h_max (degrees of freedom, median)")
        ax.set_title(t, loc="left")
        ax.set_xticks(xt); ax.set_xticklabels(["%s\n(%s)" % (h, "{:,}".format(int(round(v)))) for h, v in zip(hs, xt)])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(xt[0] / 1.4, xt[-1] * 1.4)
    axs[0].set_ylabel("Largest frequency error,\nmedian over geometries")
    fig.subplots_adjust(bottom=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), handlelength=2.4, columnspacing=1.8)
    fig.text(0.5, -0.12, "All networks trained, and all sources of the mapped reduced basis computed, at h_max 0.2; evaluated against the "
             "finite element solution on each mesh.\nNetworks: median over geometries and three seeds. On the right the shared O4 and the "
             "supervised network nearly coincide.",
             ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig7_accuracy_mesh_refinement")
    return data


# ----------------------------------------------------------------------------- figure 8: longer training
def fig8():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), sharey=True, gridspec_kw=dict(wspace=0.16))
    for ax, o, col, nm in ((axs[0], "o2", XF.C_O2, "O2 trace"), (axs[1], "o3", XF.C_O3, "O3 sequential Rayleigh quotients")):
        runs = [v for k, v in x2.items() if k.startswith(o)]
        t = np.array([a for a, _ in runs[0]["e1"]])
        E = np.array([[b for _, b in r["e1"]] for r in runs])
        ax.fill_between(t, E.min(0), E.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(t, np.median(E, 0), "-", color=col, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.5, label="100,000 iterations (X2)")
        tf = np.array([a for a, _ in runs[0]["formal_logs"]])
        Ef = np.array([[b for _, b in r["formal_logs"]] for r in runs])
        ax.plot(tf, np.median(Ef, 0), "--", color=col, lw=1.2, label="20,000 iterations (formal)")
        ax.axhline(ref["o4_e1"], color=XF.C_O4, lw=1.0, ls=":", label="O4 after 20,000 iterations (formal)")
        ax.set_yscale("log"); ax.set_xlim(0, 100000)
        ax.set_xticks([0, 25000, 50000, 75000, 100000]); ax.set_xticklabels(["0", "25,000", "50,000", "75,000", "100,000"])
        ax.set_xlabel("Adam iteration"); ax.set_title("Shared network, " + nm, loc="left")
    axs[0].set_ylabel("Error of the first frequency, $e_1$")
    fig.subplots_adjust(bottom=0.28)
    h, l = axs[0].get_legend_handles_labels()
    hd = [plt.Line2D([], [], color=XF.MUTED, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.5),
          plt.Line2D([], [], color=XF.MUTED, lw=1.2, ls="--"), h[2]]
    fig.legend(hd, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    fig.text(0.5, -0.07, "Median over the six geometry-seed runs (G1 to G3, seeds 10 and 11); band: smallest to largest of the 100,000-iteration runs. "
             "The learning rate falls by a cosine\nfrom 1e-3 to 1e-5 over each run's length, so the 100,000-iteration run is not comparable "
             "with the formal run at 20,000 iterations.", ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig8_longer_training_O2_O3")


fig6()
out["fig7"] = fig7()
fig8()
with open(os.path.join(XF.FIGS, "figure_data_x12.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("tables and figures written")
```

## check_h02/repeat_sources.py

```python
"""Explanation check for the X1a check at h_max = 0.2 (EXPLORE_PLAN.md, X1a, Check): repeat, twice in the same process,
the parts that use ARPACK with a random start vector (the 24 source eigen-solves of the mapped reduced basis and the
frequency regression; the eigen-solve of the modal basis), and compare the Ritz errors of the two repeats as the
check compares the exploratory run with the formal one."""
import os, sys, json
os.environ.update(EXP_SMOKE="0", EXP_NETS="/home/claude/udc/formal_results/results", EXP_OUT="/home/claude/udc/explore/check_h02/_scratch")
sys.path.insert(0, "/home/claude/udc/explore")
import explore as X
import numpy as np
from scipy.interpolate import RBFInterpolator
F, P, PB, PD, fc, p6 = X.F, X.P, X.PB, X.PD, X.fc, X.p6
geo = F.geometries()
out = {}
for gk in ("3.1654_0.2769", "1.5060_0.2220", "3.3505_0.1723"):
    g = [x for x in geo["held"] if X.key(x) == gk][0]
    inst = fc.build(xc=g[0], d=g[1], rho=0.05, h_max=0.2)
    reps = []
    for rep in range(2):
        w, V = fc.eig(inst["K"], inst["M"], 16)                     # random start vector each call
        srcs = PD.sources(geo["train"])                             # 24 eigen-solves, random starts
        B = PD.rbm_basis([(gg, s) for gg, s, _ in srcs], inst, g, m=8, tol=F.RBM_TOL)[0]
        tr_xy = np.array([PB.norm_xy(*x) for x in geo["train"]]); W8 = np.array([w8 for _, _, w8 in srcs])
        q = PB.norm_xy(*g)[None, :]
        fr = [float(RBFInterpolator(tr_xy, W8[:, i], kernel="thin_plate_spline", degree=1, smoothing=0.0)(q)[0]) / w[i] - 1 for i in range(8)]
        reps.append(dict(modal=p6.ritz_errors(V[:, :8], inst, w, 8), rbm=p6.ritz_errors(B, inst, w, 8), freq=fr))
    res = {}
    for k in ("modal", "rbm", "freq"):
        a, b = np.array(reps[0][k]), np.array(reps[1][k])
        big = np.abs(b) > 1e-8
        res[k] = dict(abs=float(np.max(np.abs(a - b))), rel=float(np.max(np.abs(a - b)[big] / np.abs(b[big]))) if big.any() else 0.0)
    out[gk] = res
    print(gk, json.dumps(res), flush=True)
json.dump(out, open("/home/claude/udc/explore/check_h02/repeat_sources.json", "w"), indent=1)
```

## sbx_tests/test_launcher.py

```python
"""Sandbox tests of run_explore.py paths not exercised by a clean smoke run: refusal on changed inputs, a failing job
(run once more, then given up, and the jobs ordered after it still run), restart after completion, restart after an
interruption (attempts carried over)."""
import os, sys, json, shutil, subprocess
EXP = "/home/claude/udc/explore"
sys.path.insert(0, EXP)
import run_explore as R

NETS = "/home/claude/udc/formal_results/results"
SUMS = "/home/claude/udc/formal_results/SHA256SUMS_formal_results.txt"
T = os.path.join(EXP, "sbx_tests")


def run(args, out):
    p = subprocess.run([sys.executable, os.path.join(EXP, "run_explore.py")] + args + ["--out", out, "--cpus", "0,1"],
                       capture_output=True, text=True)
    return p.returncode, p.stdout


# 1. refusal: a checksum list in which one network has a wrong hash
bad_sums = os.path.join(T, "bad_sums.txt")
with open(SUMS) as f, open(bad_sums, "w") as g:
    for line in f:
        g.write(line.replace(line.split()[0], "0" * 64) if line.strip().endswith(" p2_o4_s1.pkl") else line)
out1 = os.path.join(T, "o1"); shutil.rmtree(out1, ignore_errors=True)
rc, so = run(["--smoke", "--nets", NETS, "--sums", bad_sums], out1)
print("1 refusal:", rc, [l for l in so.splitlines() if "NOT STARTED" in l])
assert rc == 2 and "p2_o4_s1.pkl: differs from the checksum list" in so

# 2. a failing job, with a job ordered after it, using a patched job list
out2 = os.path.join(T, "o2"); shutil.rmtree(out2, ignore_errors=True)
code = ("import sys; sys.argv=%r; sys.path.insert(0, %r); import run_explore as R\n"
        "def jl(smoke, cpus):\n"
        "    a = R.Job('bad', ['bogus'], 'never.json')\n"
        "    b = R.Job('x2_G1_o2_s10', ['long', 'G1', 'o2', '10'], 'x2_G1_o2_s10.json', ['bad'])\n"
        "    t = R.Job('timing', ['timing'], 'x1_timing.json', ['bad', 'x2_G1_o2_s10'], cpu=cpus[0])\n"
        "    return [a, b, t]\n"
        "R.job_list = jl\nR.main()\n") % (["run_explore.py", "--smoke", "--nets", NETS, "--sums", SUMS, "--out", out2, "--cpus", "1,0"], EXP)
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("2 failing job:\n" + p.stdout)
assert "bad: FAILED (exit 1)" in p.stdout and "bad: FAILED again" in p.stdout and "x2_G1_o2_s10: done" in p.stdout
assert "timing: started on CPU 1" in p.stdout and "END: 2 of 3 jobs done" in p.stdout
st = json.load(open(os.path.join(out2, "state.json")))
print("   state:", st)

# 3. restart after completion: nothing runs again
rc, so = run(["--smoke", "--nets", NETS, "--sums", SUMS], os.path.join(EXP, "sbx_launch_smoke"))
print("3 restart:", [l for l in so.splitlines() if "START" in l or "END" in l])
assert "6 already done" in so and "END: 6 of 6" in so

# 4. restart after an interruption: attempts carried over from state.json
out4 = os.path.join(T, "o4"); shutil.rmtree(out4, ignore_errors=True); os.makedirs(out4)
for fn in os.listdir(os.path.join(EXP, "sbx_launch_smoke")):
    if fn.endswith(".json") and fn.startswith(("x1_acc", "x1_fine", "x2_")):
        shutil.copy(os.path.join(EXP, "sbx_launch_smoke", fn), out4)
os.remove(os.path.join(out4, "x2_G2_o3_s11.json"))                 # cut off once: one more attempt
os.remove(os.path.join(out4, "x1_acc_shard1.json"))                # failed twice: not run again
json.dump({"x2_G2_o3_s11": dict(status="running", attempts=1, cpu=0), "acc_1": dict(status="failed", attempts=2, cpu=1)},
          open(os.path.join(out4, "state.json"), "w"))
rc, so = run(["--smoke", "--nets", NETS, "--sums", SUMS], out4)
print("4 interruption:\n" + so)
assert "x2_G2_o3_s11: earlier attempt did not finish (running)" in so and "x2_G2_o3_s11: started on CPU 0 (attempt 2)" in so
assert "acc_1: failed twice in an earlier start; not run again" in so and "END: 5 of 6 jobs done" in so
print("all launcher tests passed")
```
