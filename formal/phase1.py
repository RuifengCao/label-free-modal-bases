"""Formal runs of the Unified 2027 dynamics paper: Part I (G1 to G3) and Part II (the notched-cantilever family).

Pre-registration: PREREG.md, stamped together with this file (STAMP.txt). Normally started through run_formal.py.

  python -I phase1.py p1 GEOM OBJ SEED [ARCH]   # Part I: GEOM in G1, G2, G3; SEED in 10, 11; ARCH shared (default;
                                                #   OBJ in o1, o2, o3, o4) or sep (one network per mode; OBJ in o2, o3, o4)
  python -I phase1.py p2 OBJ SEED [ARCH]    # Part II training: SEED in 0, 1, 2; ARCH shared (default; OBJ in o4, o3s,
                                            #   sup; 60,000 iterations) or sep (OBJ in o4, o3s; 30,000 iterations)
  python -I phase1.py e2 SHARD NSHARD       # Part II evaluation on the held-out geometries, one shard
  python -I phase1.py e2train               # Part II training geometries: Ritz errors and drop reproduction
  python -I phase1.py verdicts              # H1 to H10 by the pre-registered rules (writes verdicts.json, verdicts.md)

Environment: P1_OUT (output folder; default results/ next to this file); P1_SMOKE=1 (a few iterations and shortened
evaluations, for testing only; smoke outputs are never used for verdicts).
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P                       # objectives O1 to O4, problem set-up, evaluation (unchanged from Phase 0)
import phase0b as PB                     # O3 with per-geometry sorting (unchanged from Phase 0b)
import phase0c as PC                     # one network per mode: initialisation and objectives on the assembled basis (Phase 0c)
import phase0d as PD                     # classical controls: notch-aligning map, mapped reduced basis (unchanged from Phase 0d)
import phase0e as PE                     # one network per mode, batched form (Phase 0e, check E2)
from phase0 import fc, rc, lb, p2, p3, p6, jax, jnp
from scipy.interpolate import RBFInterpolator

SMOKE = os.environ.get("P1_SMOKE", "0") == "1"
OUT = os.environ.get("P1_OUT", os.path.join(HERE, "results_smoke" if SMOKE else "results"))
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- fixed settings (PREREG.md, section 2)
M_MODES = 8
LR0, LR1 = 1e-3, 1e-5
ITERS_P1, LOG_P1 = (40, 10) if SMOKE else (20000, 1000)
ITERS_P2, LOG_P2 = (24, 12) if SMOKE else (60000, 5000)
ITERS_P2_SEP = 24 if SMOKE else 30000     # one network per mode: about the training time of 60,000 shared iterations
PER_STEP = 4                             # training geometries per iteration in Part II
SEEDS_P1, SEEDS_P2 = (10, 11), (0, 1, 2)
OBJ_P1, OBJ_P2 = ("o1", "o2", "o3", "o4"), ("o4", "o3s", "sup")
OBJ_P1_SEP = ("o2", "o3", "o4")          # Part I with one network per mode (PREREG.md, section 3)
OBJ_P2_SEP = ("o4", "o3s")               # Part II with one network per mode (PREREG.md, section 4)

# Part I thresholds (PREREG.md, section 3)
H1_MAX, H2_MIN, H3_E1, H3B_E1 = 0.005, 0.05, 0.01, 0.005
H8_RATIO, H9_E1, H10_MAX = 3.0, 0.01, 0.005   # one network per mode (PREREG.md, section 3)
# Part II thresholds (PREREG.md, section 4); from Phase 0b pilot 2 by the rule fixed in PHASE0B_PLAN.md (all three kept)
H4_MAX, H5_MIN, H6_MAX = 0.01, 0.90, 0.10
# Part II classical controls (PREREG.md, section 4): Gram truncation of the mapped reduced basis (Phase 0d, deviation 4)
RBM_TOL = 1e-13


def jdump(obj, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1, default=P._js)


def geometries():
    """The stamped geometry lists; recomputed from the Sobol construction and checked against geometries.json."""
    g = P.geometries()
    with open(os.path.join(HERE, "geometries.json")) as f:
        stamped = json.load(f)
    for k in ("G123", "fam_train", "fam_held"):
        assert [list(x) for x in g[k]] == [list(x) for x in stamped[k]], k
    return dict(G={"G%d" % (i + 1): tuple(x) for i, x in enumerate(g["G123"])},
                train=[tuple(x) for x in g["fam_train"]], held=[tuple(x) for x in g["fam_held"]])


def save_params(fn, params, **kw):
    with open(os.path.join(OUT, fn), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], **kw), f)


def _drop(w_plain, w1_fe, e1):
    return float((w_plain - w1_fe * (1.0 + e1)) / (w_plain - w1_fe))


# ----------------------------------------------------------------------------- Part I

def _sep_conditioning(sp, pr, inst):
    Phi = np.asarray(PE.raw_sepv(sp, pr))
    G = Phi.T @ (inst["M"] @ Phi)
    lam = np.linalg.eigvalsh(0.5 * (G + G.T))
    return dict(eig_min=float(lam[0]), eig_max=float(lam[-1]), cond=float(lam[-1] / lam[0]))


def cmd_p1(gname, obj, seed, arch="shared"):
    assert seed in SEEDS_P1 and ((arch == "shared" and obj in OBJ_P1) or (arch == "sep" and obj in OBJ_P1_SEP))
    xc, d = geometries()["G"][gname]
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    if arch == "shared":
        loss, params0, cb = P.make_loss(obj, pr, train_cases, dt), None, P.make_cb(pr, inst, w, V)
        basis = lambda p: lb.basis_numpy(p, pr)
    else:                                                         # one network per mode, batched form
        params0 = PE.stack_sep(PC.init_sep(seed, int(pr.feats.shape[1])))
        loss = lambda p: PE.lossv(obj, p, pr)
        bf = jax.jit(lambda p: PE.orth_sepv(p, pr))
        basis = lambda p: np.asarray(bf(p))
        cb = lambda p: dict(ritz=p6.ritz_errors(basis(p), inst, w, 8), sin2=P.sin2(basis(p), inst, V))
    t0 = time.time()
    params, hist, logs = p6.train_logged(loss, pr, M_MODES, seed=seed, iters=ITERS_P1, lr0=LR0, lr1=LR1,
                                         log_every=LOG_P1, params0=params0, callback=cb)
    t_train = time.time() - t0
    tag = "p1%s_%s_%s_s%d" % ("" if arch == "shared" else "sep", gname, obj, seed)
    save_params(tag + ".pkl", params, logs=logs, t_train=t_train, geom=gname, xc=xc, d=d, obj=obj, seed=seed, arch=arch)
    if SMOKE:
        eval_cases, train_cases = eval_cases[:2], train_cases[:2]
    A = p6.step_operator(inst, C, dt)
    luA = spla.splu(A)
    refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, rc.N_STEPS) for c in eval_cases]
    W, G = rc.W_G(train_cases, dt)
    Phi = basis(params)
    res = dict(geom=gname, xc=xc, d=d, obj=obj, seed=seed, arch=arch, iters=ITERS_P1, t_train=t_train, w=[float(x) for x in w[:8]])
    for name, B in (("net", Phi), ("modal_m8", V[:, :8])):
        s = P.eval_basis(B, inst, w, V, C, dt, A, refs, eval_cases)
        s["S_over_T"] = p6.S_split(B, inst, C, A, luA, train_cases, dt)["S_over_T"]["base"]
        s["objectives"] = dict(o1=float(p3.loss_from_basis(jnp.asarray(B), pr, jnp.asarray(W), jnp.asarray(G))),
                               **P.objectives_np(B, inst))
        res[name] = s
    res["conditioning"] = p6.conditioning(params, pr, inst) if arch == "shared" else _sep_conditioning(params, pr, inst)
    res["logs"] = logs
    jdump(res, tag + ".json")
    print("done %s: train %.0f s, total %.0f s, max Ritz error %.3g" % (tag, t_train, time.time() - t0, max(res["net"]["ritz"])),
          flush=True)


# ----------------------------------------------------------------------------- Part II training

def _sup_loss(pr, KV8, vKv):
    """Supervised control: mean over the eight lowest eigenvectors of the energy share outside span(Phi) (B1 loss)."""
    def f(p):
        Phi = lb.orthonormal_basis(p, pr)
        Kr = Phi.T @ (pr.K @ Phi)
        Bm = Phi.T @ KV8
        sol = jnp.linalg.solve(0.5 * (Kr + Kr.T), Bm)
        return jnp.mean(1.0 - jnp.sum(Bm * sol, axis=0) / vKv)
    return f


def cmd_p2(obj, seed, arch="shared"):
    assert seed in SEEDS_P2 and ((arch == "shared" and obj in OBJ_P2) or (arch == "sep" and obj in OBJ_P2_SEP))
    geo = geometries()
    tr_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["train"]]
    ho_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["held"]]
    tr = [P.fam_problem(i) for i in tr_insts]
    n_in = int(tr[0].feats.shape[1])
    losses = []
    for inst, pr in zip(tr_insts, tr):
        if obj == "sup":                                          # labels: eigen-solves on the training geometries
            w8, V8 = fc.eig(inst["K"], inst["M"], 8)
            losses.append(_sup_loss(pr, jnp.asarray(inst["K"] @ V8), jnp.asarray(w8 ** 2)))
        elif arch == "sep":
            losses.append(lambda p, pr=pr: PE.lossv(obj, p, pr))
        else:
            fobj = dict(o4=P.o4_loss, o3s=PB.o3s_loss)[obj]
            losses.append(lambda p, pr=pr, fobj=fobj: fobj(p, pr))
    bfun = PE.orth_sepv if arch == "sep" else lb.orthonormal_basis
    mon = []                                                      # monitoring only
    for inst, pr in zip(tr_insts + ho_insts, tr + [P.fam_problem(i) for i in ho_insts]):
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        mon.append((inst, w, jax.jit(lambda p, pr=pr: bfun(p, pr))))
    if arch == "sep":
        params, iters = PE.stack_sep(PC.init_sep(seed, n_in)), ITERS_P2_SEP
    else:
        params, iters = lb.init_mlp(jax.random.PRNGKey(seed), [n_in, 64, 64, 64, 64, 2 * M_MODES]), ITERS_P2
    vgs = [jax.jit(jax.value_and_grad(f)) for f in losses]
    update = P.adam_update()
    mo = jax.tree_util.tree_map(jnp.zeros_like, params)
    ve = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    blocks = len(tr) // PER_STEP
    logs, perm = [], None
    t0 = time.time()
    for t in range(1, iters + 1):
        b = (t - 1) % blocks
        if b == 0:
            perm = rng.permutation(len(tr))
        sub = perm[PER_STEP * b: PER_STEP * (b + 1)]
        lr = LR1 + 0.5 * (LR0 - LR1) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = P.fam_value_grad(params, vgs, sub)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % LOG_P2 == 0 or t == 1:
            rec = dict(t=t, loss=float(val), elapsed=time.time() - t0,
                       ritz=[p6.ritz_errors(np.asarray(bf(params)), inst, w, 8) for (inst, w, bf) in mon])
            logs.append(rec)
            print(json.dumps(dict(t=t, loss=rec["loss"], elapsed=rec["elapsed"],
                                  max_ritz_train=max(max(r) for r in rec["ritz"][:len(tr)]),
                                  max_ritz_held=max(max(r) for r in rec["ritz"][len(tr):]))), flush=True)
    tag = "p2%s_%s_s%d" % ("" if arch == "shared" else "sep", obj, seed)
    save_params(tag + ".pkl", params, logs=logs, t_train=time.time() - t0, obj=obj, seed=seed, arch=arch,
                iters=iters, train=geo["train"], held=geo["held"])
    print("done %s: %.0f s" % (tag, time.time() - t0), flush=True)


# ----------------------------------------------------------------------------- Part II evaluation

def _p2_nets():
    """Every Part II network in OUT: name -> function giving its M-orthonormal basis on a problem
    (shared: o4_s0 ...; one network per mode: sep_o4_s0 ...)."""
    nets = {}
    for arch, objs, pre in (("shared", OBJ_P2, "p2_"), ("sep", OBJ_P2_SEP, "p2sep_")):
        bfun = PE.orth_sepv if arch == "sep" else lb.orthonormal_basis
        for obj in objs:
            for seed in SEEDS_P2:
                fn = os.path.join(OUT, "%s%s_s%d.pkl" % (pre, obj, seed))
                if os.path.exists(fn):
                    p = p3.load_params(fn)[0]
                    nets["%s%s_s%d" % ("" if arch == "shared" else "sep_", obj, seed)] = (lambda pr, p=p, bf=bfun: bf(p, pr))
    return nets


def cmd_e2(shard, nshard):
    geo = geometries()
    w_plain = float(p2.plain_cantilever_w1()[0])
    nets = _p2_nets()
    from scipy.spatial import Delaunay
    tr_xy = np.array([PB.norm_xy(*g) for g in geo["train"]])
    hull = Delaunay(tr_xy)
    held = geo["held"][shard::nshard]
    srcs = PD.sources(geo["train"])                               # classical controls: training eigen-solutions
    assert [g for g, _, _ in srcs] == list(geo["train"])
    W8 = np.array([w8 for _, _, w8 in srcs])
    rbf = [RBFInterpolator(tr_xy, W8[:, i], kernel="thin_plate_spline", degree=1, smoothing=0.0) for i in range(M_MODES)]
    res = {}
    for (xc, d) in (held[:1] if SMOKE else held):
        t0 = time.time()
        inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
        if SMOKE:
            eval_cases = eval_cases[:2]
        pr = P.fam_problem(inst, a_r, b_r, dt)
        A = p6.step_operator(inst, C, dt)
        refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, rc.N_STEPS) for c in eval_cases]
        rec = dict(xc=xc, d=d, n_free=int(inst["n_free"]), w=[float(x) for x in w[:8]],
                   w1_drop_fe=float((w_plain - w[0]) / w_plain),
                   dist_nearest=float(np.min(np.linalg.norm(tr_xy - PB.norm_xy(xc, d), axis=1))),
                   inside_hull=bool(hull.find_simplex(PB.norm_xy(xc, d)[None, :])[0] >= 0))
        bases = dict(modal_m8=(V[:, :8], None))
        bases.update({k: (np.asarray(f(pr)), None) for k, f in nets.items()})
        bases["rbm_all"] = PD.rbm_basis([(g, s) for g, s, _ in srcs], inst, (xc, d), m=M_MODES, tol=RBM_TOL)
        for k, (B, info) in bases.items():
            s = P.eval_basis(B, inst, w, V, C, dt, A, refs, eval_cases)
            s["drop_reproduced"] = _drop(w_plain, w[0], s["ritz"][0])
            if info:
                s["rb_info"] = info
            rec[k] = s
        q = PB.norm_xy(xc, d)[None, :]                            # frequency regression: frequencies only
        e = [float(x) for x in np.array([float(f(q)[0]) for f in rbf]) / w[:M_MODES] - 1]
        rec["freq_rbf"] = dict(ritz=e, drop_reproduced=_drop(w_plain, w[0], e[0]))
        rec["t_eval"] = time.time() - t0
        res["%.4f_%.4f" % (xc, d)] = rec
        print("e2 (%.4f, %.4f): %.0f s" % (xc, d, rec["t_eval"]), flush=True)
    jdump(dict(w_plain=w_plain, nets=sorted(nets), geoms=res), "e2_shard%d.json" % shard)


def cmd_e2train():
    geo = geometries()
    w_plain = float(p2.plain_cantilever_w1()[0])
    nets = _p2_nets()
    res = {}
    for (xc, d) in (geo["train"][:2] if SMOKE else geo["train"]):
        inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        pr = P.fam_problem(inst)
        rec = dict(xc=xc, d=d, w=[float(x) for x in w], w1_drop_fe=float((w_plain - w[0]) / w_plain))
        for k, f in nets.items():
            r = p6.ritz_errors(np.asarray(f(pr)), inst, w, 8)
            rec[k] = dict(ritz=r, drop_reproduced=_drop(w_plain, w[0], r[0]))
        res["%.4f_%.4f" % (xc, d)] = rec
    jdump(dict(w_plain=w_plain, nets=sorted(nets), geoms=res), "e2_train.json")


# ----------------------------------------------------------------------------- verdicts (PREREG.md, sections 3 to 5)

def _verdict(n_pass, n_total):
    if n_total == 0:
        return "missing"
    if n_pass == n_total:
        return "supported"
    if (n_total == 6 and n_pass >= 4) or (n_total == 3 and n_pass == 2):
        return "partly supported"
    return "not supported"


def verdicts_from(p1_runs, e2_geoms, thresholds):
    """Pure function of the run outputs, so that it can be tested on made-up inputs (audit)."""
    t4, t5, t6 = thresholds
    out = dict(part1={}, part2={})
    checks = dict(
        H1=("o4", lambda r: max(r) <= H1_MAX),
        H2=("o1", lambda r: max(r[2:8]) > H2_MIN),
        H3=("o2", lambda r: r[0] > H3_E1 and r[0] > max(r[2:8])),
        H3b=("o3", lambda r: r[0] > H3B_E1 and r[0] > max(r[2:8])))
    shared = {k: v for k, v in p1_runs.items() if v.get("arch", "shared") == "shared"}
    sep = {k: v for k, v in p1_runs.items() if v.get("arch") == "sep"}
    checks_sep = dict(
        H9=("o2", lambda r: r[0] > H9_E1 and r[0] > max(r[2:8])),
        H10=("o4", lambda r: max(r) <= H10_MAX))
    for runs, chk, arch in ((shared, checks, "shared"), (sep, checks_sep, "sep")):
        for h, (obj, test) in chk.items():
            rows = [(k, test(v["net"]["ritz"])) for k, v in sorted(runs.items()) if v["obj"] == obj]
            out["part1"][h] = dict(objective=obj, architecture=arch, per_run={k: bool(ok) for k, ok in rows},
                                   n_pass=int(sum(ok for _, ok in rows)), n=len(rows),
                                   verdict=_verdict(int(sum(ok for _, ok in rows)), len(rows)))
    rows = {}                                                     # H8: per-mode O3 against shared O3, same geometry and seed
    for k, v in sorted(sep.items()):
        if v["obj"] != "o3":
            continue
        mate = [u for u in shared.values() if u["obj"] == "o3" and u["geom"] == v["geom"] and u["seed"] == v["seed"]]
        if mate:
            rows[k] = dict(e1_sep=v["net"]["ritz"][0], e1_shared=mate[0]["net"]["ritz"][0],
                           pass_=bool(v["net"]["ritz"][0] < mate[0]["net"]["ritz"][0] / H8_RATIO))
    n_pass = sum(r["pass_"] for r in rows.values())
    out["part1"]["H8"] = dict(objective="o3", architecture="sep against shared", per_run=rows, n_pass=int(n_pass), n=len(rows),
                              verdict=_verdict(int(n_pass), len(rows)))
    out["part1"] = {h: out["part1"][h] for h in ("H1", "H2", "H3", "H3b", "H8", "H9", "H10")}
    held = list(e2_geoms.values())
    for h, thr in (("H4", t4), ("H5", t5), ("H6", t6)):
        per_seed = {}
        for seed in SEEDS_P2:
            k = "o4_s%d" % seed
            if not held or k not in held[0]:
                continue
            if h == "H4":
                v = float(np.median([np.median(g[k]["ritz"][:5]) for g in held])); ok = v <= thr
            elif h == "H5":
                v = float(np.median([g[k]["drop_reproduced"] for g in held])); ok = v >= thr
            else:
                v = float(np.median([g[k]["e2_energy"]["median"] for g in held])); ok = v <= thr
            per_seed[k] = dict(value=v, pass_=bool(ok))
        n_pass = sum(x["pass_"] for x in per_seed.values())
        out["part2"][h] = dict(threshold=thr, per_seed=per_seed, n_pass=int(n_pass), n=len(per_seed),
                               verdict=_verdict(int(n_pass), len(per_seed)))
    per_seed = {}
    for seed in SEEDS_P2:
        a, b = "o3s_s%d" % seed, "o4_s%d" % seed
        if held and a in held[0] and b in held[0]:
            m3 = float(np.median([g[a]["ritz"][0] for g in held])); m4 = float(np.median([g[b]["ritz"][0] for g in held]))
            per_seed[str(seed)] = dict(o3s_median_e1=m3, o4_median_e1=m4, pass_=bool(m3 > m4))
    n_pass = sum(x["pass_"] for x in per_seed.values())
    out["part2"]["H7"] = dict(per_seed=per_seed, n_pass=int(n_pass), n=len(per_seed), verdict=_verdict(int(n_pass), len(per_seed)))
    per_seed = {}
    for seed in SEEDS_P2:
        a, b = "sep_o3s_s%d" % seed, "sep_o4_s%d" % seed
        if held and a in held[0] and b in held[0]:
            m3 = float(np.median([g[a]["ritz"][0] for g in held])); m4 = float(np.median([g[b]["ritz"][0] for g in held]))
            per_seed[str(seed)] = dict(o3s_median_e1=m3, o4_median_e1=m4, pass_=bool(m3 > m4))
    n_pass = sum(x["pass_"] for x in per_seed.values())
    out["part2"]["H7sep"] = dict(per_seed=per_seed, n_pass=int(n_pass), n=len(per_seed), verdict=_verdict(int(n_pass), len(per_seed)))
    return out


def cmd_verdicts():
    p1_runs = {}
    for fn in sorted(os.listdir(OUT)):
        if (fn.startswith("p1_") or fn.startswith("p1sep_")) and fn.endswith(".json"):
            with open(os.path.join(OUT, fn)) as f:
                p1_runs[fn[:-5]] = json.load(f)
    e2 = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("e2_shard") and fn.endswith(".json"):
            with open(os.path.join(OUT, fn)) as f:
                e2.update(json.load(f)["geoms"])
    thr = (H4_MAX, H5_MIN, H6_MAX)
    assert None not in thr, "Part II thresholds are not set"
    v = verdicts_from(p1_runs, e2, thr)
    v["smoke"] = SMOKE
    v["n_p1_runs"], v["n_held"] = len(p1_runs), len(e2)
    jdump(v, "verdicts.json")
    lines = ["# Verdicts (computed by phase1.py verdicts)%s" % (": SMOKE TEST, NOT A RESULT" if SMOKE else ""), ""]
    for part in ("part1", "part2"):
        for h, r in v[part].items():
            lines.append("- %s: %s (%d of %d)" % (h, r["verdict"], r["n_pass"], r["n"]))
    with open(os.path.join(OUT, "verdicts.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "p1":
        cmd_p1(sys.argv[2], sys.argv[3], int(sys.argv[4]), *(sys.argv[5:6]))
    elif c == "p2":
        cmd_p2(sys.argv[2], int(sys.argv[3]), *(sys.argv[4:5]))
    elif c == "e2":
        cmd_e2(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "e2train":
        cmd_e2train()
    elif c == "verdicts":
        cmd_verdicts()
    else:
        raise SystemExit("unknown command " + c)
