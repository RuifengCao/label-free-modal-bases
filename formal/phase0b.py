"""Phase 0b (exploratory): family thresholds and controls (see PHASE0B_PLAN.md).

  python -I phase0b.py checks                 # A5, A6
  python -I phase0b.py fam OBJ SEED ITERS     # (j) OBJ = o4, (k) OBJ = o3s; pilot-2 geometries
  python -I phase0b.py evalfam2 SHARD NSHARD  # (j), (k), (l) on the pilot-2 held-out geometries, one shard
  python -I phase0b.py merge                  # merge the shards, summary and threshold rule
  python -I phase0b.py rb1                    # (l) on the Phase 0 pilot held-out geometries

Environment: P0B_OUT replaces the output folder; P0_SMOKE=1 shortens the evaluations; P0_LOG as in phase0.py.
"""
import sys, os, json, time, math, pickle
import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P
from phase0 import fc, rc, lb, p2, p3, p6, jax, jnp, ravel_pytree
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

OUT = os.environ.get("P0B_OUT", os.path.join(HERE, "out0b"))
os.makedirs(OUT, exist_ok=True)
SMOKE = P.SMOKE
M_MODES = 8
N = rc.N_STEPS


def jdump(obj, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1, default=P._js)


def pilot2_geoms():
    return dict(train=P.sobol_geoms(31, 24), held=P.sobol_geoms(32, 16))


def norm_xy(xc, d):
    return np.array([(xc - 3.0) / 2.0, (d - 0.25) / 0.15])


# ----------------------------------------------------------------------------- O3 with per-geometry sorting

def o3s_loss(p, pr):
    Phi = lb.raw_basis(p, pr)
    rq = jnp.sum(Phi * (pr.K @ Phi), axis=0) / jnp.sum(Phi * (pr.M @ Phi), axis=0)
    order = jnp.argsort(jax.lax.stop_gradient(rq))
    Phi = Phi[:, order]
    Q = jax.lax.stop_gradient(P._orth(Phi, pr.M))
    U = jnp.triu(jnp.ones((pr.m, pr.m)), 1)
    R = Phi - Q @ (U * (Q.T @ (pr.M @ Phi)))
    return jnp.sum(jnp.sum(R * (pr.K @ R), axis=0) / jnp.sum(R * (pr.M @ R), axis=0))


def o3s_loop(p, pr, order):
    """Loop form with the column order given explicitly (sorted in NumPy outside the trace; check A5 only)."""
    Phi = lb.raw_basis(p, pr)
    cols, tot = [], 0.0
    for k in order:
        v = Phi[:, int(k)]
        for q in cols:
            qs = jax.lax.stop_gradient(q)
            v = v - (qs @ (pr.M @ v)) * qs
        v = v / jnp.sqrt(v @ (pr.M @ v))
        tot = tot + v @ (pr.K @ v)
        cols.append(v)
    return tot


LOSSES = dict(o4=P.o4_loss, o3s=o3s_loss)


# ----------------------------------------------------------------------------- reduced-basis controls

def src_prep(inst, V):
    """Source node cloud (free nodes plus the clamped edge), its triangulation and the eigenvector fields."""
    yc = np.linspace(0.0, fc.H, 81)
    pts = np.vstack([inst["pts"], np.stack([np.zeros_like(yc), yc], axis=1)])
    z = np.zeros((len(yc), V.shape[1]))
    return dict(tri=Delaunay(pts), ux=np.vstack([V[inst["ixf"], :], z]), uy=np.vstack([V[inst["iyf"], :], z]))


def interp_from(src, dst):
    q = dst["pts"]
    Phi = np.zeros((dst["n_free"], src["ux"].shape[1]))
    Phi[dst["ixf"]] = LinearNDInterpolator(src["tri"], src["ux"])(q)
    Phi[dst["iyf"]] = LinearNDInterpolator(src["tri"], src["uy"])(q)
    assert np.all(np.isfinite(Phi))
    return Phi


def rb_basis(srcs, dst, m=M_MODES, tol=1e-10):
    """Rayleigh-Ritz in the span of interpolated source eigenvectors; returns the lowest m Ritz vectors."""
    t0 = time.perf_counter()
    B = np.hstack([interp_from(s, dst) for s in srcs])
    G = B.T @ (dst["M"] @ B)
    lam, U = np.linalg.eigh(0.5 * (G + G.T))
    keep = lam > tol * lam.max()
    Q = B @ (U[:, keep] / np.sqrt(lam[keep]))
    Kr = Q.T @ (dst["K"] @ Q)
    th, Y = np.linalg.eigh(0.5 * (Kr + Kr.T))
    return Q @ Y[:, :m], dict(n_vectors=int(B.shape[1]), rank=int(keep.sum()), t=time.perf_counter() - t0)


# ----------------------------------------------------------------------------- checks

def cmd_checks():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    p_init = lb.init_mlp(jax.random.PRNGKey(0), [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES])
    p_f1, _ = p3.load_params(os.path.join(P.OLD_OUT, "posthoc6_F1.pkl"))
    res, ok = {}, True
    for name, p in (("init", p_init), ("F1", p_f1)):
        Phi = np.asarray(lb.raw_basis(p, pr))
        rq = np.sum(Phi * (inst["K"] @ Phi), axis=0) / np.sum(Phi * (inst["M"] @ Phi), axis=0)
        order = [int(i) for i in np.argsort(rq)]
        v1, g1 = jax.value_and_grad(lambda q: o3s_loss(q, pr))(p)
        v2, g2 = jax.value_and_grad(lambda q: o3s_loop(q, pr, order))(p)
        f1, _ = ravel_pytree(g1); f2, _ = ravel_pytree(g2)
        tr = float(P.o2_loss(p, pr))
        r = dict(value_rel=abs(float(v1) - float(v2)) / abs(float(v2)),
                 grad_rel=float(jnp.linalg.norm(f1 - f2) / jnp.linalg.norm(f2)),
                 trace_rel=abs(float(v1) - tr) / abs(tr), order=order)
        r["pass"] = bool(r["value_rel"] < 1e-10 and r["grad_rel"] < 1e-10 and r["trace_rel"] < 1e-10)
        ok &= r["pass"]; res["A5_" + name] = r
    ev = json.load(open(os.path.join(P.OUT, "evalfam.json")))
    geo = P.geometries()
    srcs = []
    for (xc, d) in geo["pilot_train"]:
        si = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        sw, sV = fc.eig(si["K"], si["M"], 8)
        srcs.append(((xc, d), src_prep(si, sV)))
    a6 = {}
    for (xc, d) in geo["pilot_held"]:
        dst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        w8, _ = fc.eig(dst["K"], dst["M"], 8)
        j = int(np.argmin([np.linalg.norm(norm_xy(xc, d) - norm_xy(*s[0])) for s in srcs]))
        Phi, info = rb_basis([srcs[j][1]], dst)
        r_new = np.array(p6.ritz_errors(Phi, dst, w8, 8))
        r_old = np.array(ev["geoms"]["pilot_held_%.4f_%.4f" % (xc, d)]["nn_interp"]["ritz"])
        a6["%.4f_%.4f" % (xc, d)] = float(np.max(np.abs(r_new - r_old) / np.abs(r_old)))
    res["A6"] = dict(max_rel=a6, pass_=bool(max(a6.values()) < 1e-8))
    ok &= res["A6"]["pass_"]
    res["all_pass"] = bool(ok)
    print(json.dumps(res, indent=1, default=P._js), flush=True)
    jdump(res, "checks0b.json")


# ----------------------------------------------------------------------------- (j), (k): pilot-2 training

def cmd_fam(obj, seed, iters):
    geo = pilot2_geoms()
    tr_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["train"]]
    ho_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["held"]]
    tr = [P.fam_problem(i) for i in tr_insts]
    mon = []                                                      # monitoring only: finite element frequencies
    for inst, pr in zip(tr_insts + ho_insts, tr + [P.fam_problem(i) for i in ho_insts]):
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        mon.append((inst, w, jax.jit(lambda p, pr=pr: lb.orthonormal_basis(p, pr))))
    params = lb.init_mlp(jax.random.PRNGKey(seed), [int(tr[0].feats.shape[1]), 64, 64, 64, 64, 2 * M_MODES])
    vgs = [jax.jit(jax.value_and_grad(lambda p, pr=pr: LOSSES[obj](p, pr))) for pr in tr]
    update = P.adam_update()
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    per, blocks = 4, len(tr) // 4
    log_every = max(1, 5 * P.LOG_EVERY)
    logs, perm = [], None
    t0 = time.time()
    for t in range(1, iters + 1):
        b = (t - 1) % blocks
        if b == 0:
            perm = rng.permutation(len(tr))
        sub = perm[per * b: per * (b + 1)]
        lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = P.fam_value_grad(params, vgs, sub)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % log_every == 0 or t == 1:
            rec = dict(t=t, loss=float(val), elapsed=time.time() - t0,
                       ritz=[p6.ritz_errors(np.asarray(bf(params)), inst, w, 8) for (inst, w, bf) in mon])
            logs.append(rec)
            print(json.dumps(dict(t=t, loss=rec["loss"], elapsed=rec["elapsed"],
                                  max_ritz_train=max(max(r) for r in rec["ritz"][:len(tr)]),
                                  max_ritz_held=max(max(r) for r in rec["ritz"][len(tr):]),
                                  med_e1_held=float(np.median([r[0] for r in rec["ritz"][len(tr):]])))), flush=True)
    with open(os.path.join(OUT, "fam2_%s_s%d_%d.pkl" % (obj, seed, iters)), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(bb)) for (Wt, bb) in params], logs=logs,
                         t_train=time.time() - t0, obj=obj, seed=seed, iters=iters, train=geo["train"], held=geo["held"]), f)
    print("done fam2 %s %.0f s" % (obj, time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- evaluation

def _drop(w_plain, w1_fe, e1):
    w1 = w1_fe * (1.0 + e1)
    return float((w_plain - w1) / (w_plain - w1_fe))


def _eval_geometry(xc, d, nets, srcs, w_plain, group):
    """All bases on one geometry, sharing the finite element references."""
    t0 = time.time()
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
    if SMOKE:
        eval_cases = eval_cases[:2]
    pr = P.fam_problem(inst, a_r, b_r, dt)
    A = p6.step_operator(inst, C, dt)
    refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, N) for c in eval_cases]
    bases = {"modal_m8": (V[:, :8], {})}
    for name, params in nets.items():
        bases[name] = (np.asarray(lb.orthonormal_basis(params, pr)), {})
    if srcs is not None:
        order = np.argsort([np.linalg.norm(norm_xy(xc, d) - norm_xy(*s[0])) for s in srcs])
        for tag, k in (("rb_k1", 1), ("rb_k4", 4), ("rb_all", len(srcs))):
            Phi, info = rb_basis([srcs[i][1] for i in order[:k]], inst)
            bases[tag] = (Phi, info)
    rec = dict(group=group, xc=xc, d=d, n_free=int(inst["n_free"]), w=[float(x) for x in w[:8]],
               w1_drop_fe=float((w_plain - w[0]) / w_plain))
    for k, (B, info) in bases.items():
        s = P.eval_basis(B, inst, w, V, C, dt, A, refs, eval_cases)
        s["drop_reproduced"] = _drop(w_plain, w[0], s["ritz"][0])
        if info:
            s["rb_info"] = info
        rec[k] = s
    rec["t_eval"] = time.time() - t0
    return rec


def _load_nets(fam_prefix="fam2_"):
    nets = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith(fam_prefix) and fn.endswith(".pkl"):
            params, meta = p3.load_params(os.path.join(OUT, fn))
            nets["net_%s" % meta["obj"]] = params
    return nets


def cmd_evalfam2(shard, nshard):
    geo = pilot2_geoms()
    w_plain = float(p2.plain_cantilever_w1()[0])
    nets = _load_nets()
    held = geo["held"][shard::nshard]
    srcs = []
    for (xc, d) in geo["train"]:
        si = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        sw, sV = fc.eig(si["K"], si["M"], 8)
        srcs.append(((xc, d), src_prep(si, sV)))
    res = {}
    for (xc, d) in (held[:1] if SMOKE else held):
        rec = _eval_geometry(xc, d, nets, srcs, w_plain, "pilot2_held")
        tr_xy = np.array([norm_xy(*g) for g in geo["train"]])
        dist = np.linalg.norm(tr_xy - norm_xy(xc, d), axis=1)
        rec["dist_nearest"] = float(dist.min())
        rec["inside_hull"] = bool(Delaunay(tr_xy).find_simplex(norm_xy(xc, d)[None, :])[0] >= 0)
        res["%.4f_%.4f" % (xc, d)] = rec
        print(json.dumps(dict(xc=xc, d=d, dist=rec["dist_nearest"], **{k: dict(max=max(rec[k]["ritz"]), e1=rec[k]["ritz"][0],
              drop=rec[k]["drop_reproduced"], e2=rec[k]["e2_energy"]["median"]) for k in rec if isinstance(rec[k], dict) and "ritz" in rec[k]})),
              flush=True)
    if shard == 0:                                                # training geometries: Ritz errors only
        trn = {}
        for (xc, d) in (geo["train"][:2] if SMOKE else geo["train"]):
            inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
            w, _ = fc.eig(inst["K"], inst["M"], 8)
            pr = P.fam_problem(inst)
            trn["%.4f_%.4f" % (xc, d)] = {k: dict(ritz=p6.ritz_errors(np.asarray(lb.orthonormal_basis(pp, pr)), inst, w, 8),
                                                  drop_reproduced=None) for k, pp in nets.items()}
            for k in nets:
                trn["%.4f_%.4f" % (xc, d)][k]["drop_reproduced"] = _drop(w_plain, w[0], trn["%.4f_%.4f" % (xc, d)][k]["ritz"][0])
        res["_train"] = trn
    jdump(dict(w_plain=w_plain, geoms=res), "evalfam2_shard%d.json" % shard)


def cmd_rb1():
    """(l) on the Phase 0 pilot: the held-out geometries with the reduced-basis controls and the Phase 0 pilot network."""
    geo = P.geometries()
    w_plain = float(p2.plain_cantilever_w1()[0])
    fn = [f for f in os.listdir(P.OUT) if f.startswith("fam_pilot_s0_") and f.endswith(".pkl")]
    params, _ = p3.load_params(os.path.join(P.OUT, sorted(fn)[-1]))
    srcs = []
    for (xc, d) in geo["pilot_train"]:
        si = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        sw, sV = fc.eig(si["K"], si["M"], 8)
        srcs.append(((xc, d), src_prep(si, sV)))
    res = {}
    for (xc, d) in (geo["pilot_held"][:1] if SMOKE else geo["pilot_held"]):
        rec = _eval_geometry(xc, d, {"net_o4": params}, srcs, w_plain, "pilot1_held")
        res["%.4f_%.4f" % (xc, d)] = rec
        print(json.dumps(dict(xc=xc, d=d, **{k: dict(max=max(rec[k]["ritz"]), drop=rec[k]["drop_reproduced"],
              e2=rec[k]["e2_energy"]["median"]) for k in rec if isinstance(rec[k], dict) and "ritz" in rec[k]})), flush=True)
    jdump(dict(w_plain=w_plain, geoms=res), "rb1.json")


# ----------------------------------------------------------------------------- merge, summary, threshold rule

def _ceil_to(x, step):
    return math.ceil(x / step - 1e-9) * step


def _floor_to(x, step):
    return math.floor(x / step + 1e-9) * step


def cmd_merge():
    geoms, w_plain = {}, None
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("evalfam2_shard") and fn.endswith(".json"):
            r = json.load(open(os.path.join(OUT, fn)))
            w_plain = r["w_plain"]
            geoms.update(r["geoms"])
    train = geoms.pop("_train", {})
    held = list(geoms.values())
    bases = [k for k in held[0] if isinstance(held[0][k], dict) and "ritz" in held[0][k]]
    summ = {}
    for b in bases:
        e15 = [float(np.median(g[b]["ritz"][:5])) for g in held]
        drop = [g[b]["drop_reproduced"] for g in held]
        e2 = [g[b]["e2_energy"]["median"] for g in held]
        w1n = np.array([g["w"][0] * (1 + g[b]["ritz"][0]) for g in held]); w1f = np.array([g["w"][0] for g in held])
        slope, icpt = np.polyfit(w1f, w1n, 1)
        r2 = 1 - np.sum((w1n - (slope * w1f + icpt)) ** 2) / np.sum((w1n - w1n.mean()) ** 2)
        big = [g[b]["drop_reproduced"] for g in held if g["w1_drop_fe"] >= 0.01]
        summ[b] = dict(H4=float(np.median(e15)), H5=float(np.median(drop)), H6=float(np.median(e2)),
                       median_e1=float(np.median([g[b]["ritz"][0] for g in held])),
                       median_max_ritz=float(np.median([max(g[b]["ritz"]) for g in held])),
                       max_max_ritz=float(max(max(g[b]["ritz"]) for g in held)),
                       median_e1_tip_acc=float(np.median([g[b]["e1_tip_acc"]["median"] for g in held])),
                       slope=float(slope), r2=float(r2), H5_drop_ge_1pct=float(np.median(big)) if big else None, n_big=len(big))
        if b.startswith("rb_"):
            summ[b]["median_t"] = float(np.median([g[b]["rb_info"]["t"] for g in held]))
    rule = {}
    if "net_o4" in summ:
        a4, a5, a6 = summ["net_o4"]["H4"], summ["net_o4"]["H5"], summ["net_o4"]["H6"]
        rule["H4"] = 0.01 if a4 <= 0.0067 else _ceil_to(1.5 * a4, 0.001)
        rule["H5"] = 0.90 if a5 >= 0.933 else _floor_to(1.0 - 1.5 * (1.0 - a5), 0.01)
        rule["H6"] = 0.10 if a6 <= 0.067 else _ceil_to(1.5 * a6, 0.01)
        rule["kept_from_plan"] = dict(H4=a4 <= 0.0067, H5=a5 >= 0.933, H6=a6 <= 0.067)
        if "net_o3s" in summ:
            rule["H7_direction_seen"] = bool(summ["net_o3s"]["median_e1"] > summ["net_o4"]["median_e1"])
    tr_summ = {}
    for b in (train[next(iter(train))].keys() if train else []):
        tr_summ[b] = dict(median_max_ritz=float(np.median([max(v[b]["ritz"]) for v in train.values()])),
                          max_max_ritz=float(max(max(v[b]["ritz"]) for v in train.values())),
                          median_drop=float(np.median([v[b]["drop_reproduced"] for v in train.values()])))
    out = dict(w_plain=w_plain, held=geoms, train=train, summary_held=summ, summary_train=tr_summ, threshold_rule=rule,
               coverage=dict(dist=[g["dist_nearest"] for g in held], inside=[g["inside_hull"] for g in held]))
    jdump(out, "evalfam2.json")
    print(json.dumps(dict(summary_held=summ, summary_train=tr_summ, threshold_rule=rule), indent=1, default=P._js))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "fam":
        cmd_fam(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif c == "evalfam2":
        cmd_evalfam2(int(sys.argv[2]), int(sys.argv[3]))
    else:
        dict(checks=cmd_checks, merge=cmd_merge, rb1=cmd_rb1)[c]()
