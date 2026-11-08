"""Phase 0e (exploratory): one network per mode on the geometry family (see PHASE0E_PLAN.md).

  python -I phase0e.py checks               # E1
  python -I phase0e.py check2               # E2 (deviation 5): batched form equals the loop form; timing
  python -I phase0e.py fam ARCH OBJ SEED ITERS   # ARCH sepv (one network per mode) or shared; OBJ o4 or o3s; pilot 2
  python -I phase0e.py evalfam              # every run in the output folder, pilot-2 held-out and training geometries

Environment: P0E_OUT replaces the output folder; P0_SMOKE=1 shortens the evaluation (2 held-out and 2 training
geometries, 2 load cases); P0_LOG as in phase0.py.
"""
import sys, os, json, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P
import phase0b as PB
import phase0c as PC
from phase0 import fc, lb, p2, p6, jax, jnp
from scipy.spatial import Delaunay

OUT = os.environ.get("P0E_OUT", os.path.join(HERE, "out0e"))
os.makedirs(OUT, exist_ok=True)
SMOKE = P.SMOKE
M_MODES = 8


def jdump(obj, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1, default=P._js)


def loss(obj, p, pr):
    return PC.loss_from_raw(obj, PC.raw_sep(p, pr), pr)


# Batched form of the same eight MLPs (deviation 5): parameters stacked layer by layer.

def stack_sep(params):
    return [(jnp.stack([p[l][0] for p in params]), jnp.stack([p[l][1] for p in params])) for l in range(len(params[0]))]


def unstack_sep(sp):
    return [[(W[i], b[i]) for (W, b) in sp] for i in range(sp[0][0].shape[0])]


def raw_sepv(sp, pr):
    """The fields of phase0c.raw_sep, with the eight MLPs evaluated as one batched network."""
    W, b = sp[0]
    h = jnp.tanh(jnp.einsum("na,mab->mnb", pr.feats, W) + b[:, None, :])
    for W, b in sp[1:-1]:
        h = jnp.tanh(jnp.einsum("mna,mab->mnb", h, W) + b[:, None, :])
    W, b = sp[-1]
    out = jnp.einsum("mna,mab->mnb", h, W) + b[:, None, :]           # (modes, nodes, 2)
    Phi = jnp.zeros((pr.n, out.shape[0])).at[pr.ixf].set(out[:, :, 0].T).at[pr.iyf].set(out[:, :, 1].T)
    return Phi * pr.clamp


def lossv(obj, sp, pr):
    return PC.loss_from_raw(obj, raw_sepv(sp, pr), pr)


def orth_sepv(sp, pr):
    return P._orth(raw_sepv(sp, pr), pr.M)


def cmd_check2():
    """E2 (deviation 5): batched and loop forms agree; timing of one value-and-gradient of each."""
    geo = PB.pilot2_geoms()
    xc, d = geo["train"][0]
    inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
    pr = P.fam_problem(inst)
    params = PC.init_sep(0, int(pr.feats.shape[1]))
    rng = np.random.default_rng(5)
    params = [[(W, b + 0.1 * jnp.asarray(rng.standard_normal(b.shape))) for (W, b) in p] for p in params]
    sp = stack_sep(params)
    A, B = np.asarray(PC.raw_sep(params, pr)), np.asarray(raw_sepv(sp, pr))
    res = dict(geometry=[xc, d], raw_rel=float(np.max(np.abs(A - B)) / np.max(np.abs(A))))
    for obj in ("o4", "o3s"):
        v1, g1 = jax.jit(jax.value_and_grad(lambda p: loss(obj, p, pr)))(params)
        v2, g2 = jax.jit(jax.value_and_grad(lambda q: lossv(obj, q, pr)))(sp)
        f1 = np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(stack_sep(g1))])
        f2 = np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(g2)])
        res[obj] = dict(value_rel=abs(float(v1) - float(v2)) / abs(float(v1)), grad_rel=float(np.linalg.norm(f1 - f2) / np.linalg.norm(f1)))
        for tag, fun, arg in (("loop", jax.jit(jax.value_and_grad(lambda p: loss(obj, p, pr))), params),
                              ("batched", jax.jit(jax.value_and_grad(lambda q: lossv(obj, q, pr))), sp)):
            jax.block_until_ready(fun(arg))
            t0 = time.perf_counter()
            for _ in range(30):
                jax.block_until_ready(fun(arg))
            res[obj]["ms_" + tag] = 1000 * (time.perf_counter() - t0) / 30
    rt = [np.max(np.abs(np.asarray(a) - np.asarray(b))) for a, b in zip(jax.tree_util.tree_leaves(unstack_sep(sp)), jax.tree_util.tree_leaves(params))]
    res["unstack_roundtrip_max"] = float(max(rt))
    res["pass"] = bool(res["raw_rel"] < 1e-13 and res["unstack_roundtrip_max"] == 0.0
                       and all(res[o]["value_rel"] < 1e-12 and res[o]["grad_rel"] < 1e-10 for o in ("o4", "o3s")))
    print(json.dumps(res, indent=1), flush=True)
    jdump(res, "check2_0e.json")


def cmd_checks():
    geo = PB.pilot2_geoms()
    xc, d = geo["train"][0]
    inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
    pr = P.fam_problem(inst)
    params = PC.init_sep(0, int(pr.feats.shape[1]))
    Phi = PC.raw_sep(params, pr)
    ref = P.objectives_np(np.asarray(Phi), inst)
    res = dict(geometry=[xc, d], n_in=int(pr.feats.shape[1]), shape=list(Phi.shape),
               n_params=int(sum(np.size(W) + np.size(b) for p in params for (W, b) in p)))
    for obj in ("o3", "o4"):
        res[obj + "_rel"] = abs(float(loss(obj, params, pr)) - ref[obj]) / abs(ref[obj])
    res["o3s_vs_trace_rel"] = abs(float(loss("o3s", params, pr)) - ref["o2"]) / abs(ref["o2"])
    for obj in ("o4", "o3s"):
        g = jax.grad(lambda p: loss(obj, p, pr))(params)
        res[obj + "_grad_norm_per_mlp"] = [float(np.sqrt(sum(float(jnp.sum(W ** 2) + jnp.sum(b ** 2)) for (W, b) in gi))) for gi in g]
    res["pass"] = bool(res["shape"] == [inst["n_free"], M_MODES]
                       and all(res[k] < 1e-9 for k in ("o3_rel", "o4_rel", "o3s_vs_trace_rel"))
                       and all(x > 0 for k in ("o4", "o3s") for x in res[k + "_grad_norm_per_mlp"]))
    print(json.dumps(res, indent=1), flush=True)
    jdump(res, "checks0e.json")


def cmd_fam(arch, obj, seed, iters):
    """As phase0b.cmd_fam (pilot 2, same schedule and batching); arch "sepv": one network per mode (batched form,
    deviation 5), "shared": the network of Phase 0b."""
    assert arch in ("sepv", "shared") and obj in ("o4", "o3s")
    geo = PB.pilot2_geoms()
    tr_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["train"]]
    ho_insts = [fc.build(xc=xc, d=d, rho=0.05, h_max=0.2) for xc, d in geo["held"]]
    tr = [P.fam_problem(i) for i in tr_insts]
    n_in = int(tr[0].feats.shape[1])
    if arch == "sepv":
        params = stack_sep(PC.init_sep(seed, n_in))
        lf, bf_ = (lambda p, pr: lossv(obj, p, pr)), orth_sepv
    else:
        params = lb.init_mlp(jax.random.PRNGKey(seed), [n_in, 64, 64, 64, 64, 2 * M_MODES])
        lf, bf_ = (lambda p, pr: PB.LOSSES[obj](p, pr)), lb.orthonormal_basis
    mon = []
    for inst, pr in zip(tr_insts + ho_insts, tr + [P.fam_problem(i) for i in ho_insts]):
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        mon.append((inst, w, jax.jit(lambda p, pr=pr: bf_(p, pr))))
    vgs = [jax.jit(jax.value_and_grad(lambda p, pr=pr: lf(p, pr))) for pr in tr]
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
    with open(os.path.join(OUT, "fam2%s_%s_s%d_%d.pkl" % (arch, obj, seed, iters)), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(W), np.asarray(bb)) for (W, bb) in params], logs=logs, arch=arch,
                         t_train=time.time() - t0, obj=obj, seed=seed, iters=iters, train=geo["train"], held=geo["held"]), f)
    print("done fam2%s %s %.0f s" % (arch, obj, time.time() - t0), flush=True)


def _load():
    """Every Phase 0e run in the output folder: name -> function giving its M-orthonormal basis on a problem."""
    nets = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("fam2") and fn.endswith(".pkl"):
            d = pickle.load(open(os.path.join(OUT, fn), "rb"))
            p = [(jnp.asarray(W), jnp.asarray(b)) for (W, b) in d["params"]]
            bf = orth_sepv if d["arch"] == "sepv" else lb.orthonormal_basis
            nets["%s_%s_%d" % (d["arch"], d["obj"], d["iters"])] = (lambda pr, p=p, bf=bf: bf(p, pr))
    return nets


def cmd_evalfam():
    geo = PB.pilot2_geoms()
    w_plain = float(p2.plain_cantilever_w1()[0])
    nets = _load()
    tr_xy = np.array([PB.norm_xy(*g) for g in geo["train"]])
    held = {}
    for (xc, d) in (geo["held"][:2] if SMOKE else geo["held"]):
        t0 = time.time()
        inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
        if SMOKE:
            eval_cases = eval_cases[:2]
        pr = P.fam_problem(inst, a_r, b_r, dt)
        A = p6.step_operator(inst, C, dt)
        refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, PB.N) for c in eval_cases]
        rec = dict(group="pilot2_held", xc=xc, d=d, n_free=int(inst["n_free"]), w=[float(x) for x in w[:8]],
                   w1_drop_fe=float((w_plain - w[0]) / w_plain))
        bases = {"modal_m8": V[:, :8]}
        bases.update({k: np.asarray(f(pr)) for k, f in nets.items()})
        for k, B in bases.items():
            s = P.eval_basis(B, inst, w, V, C, dt, A, refs, eval_cases)
            s["drop_reproduced"] = PB._drop(w_plain, w[0], s["ritz"][0])
            rec[k] = s
        dist = np.linalg.norm(tr_xy - PB.norm_xy(xc, d), axis=1)
        rec["dist_nearest"] = float(dist.min())
        rec["inside_hull"] = bool(Delaunay(tr_xy).find_simplex(PB.norm_xy(xc, d)[None, :])[0] >= 0)
        rec["t_eval"] = time.time() - t0
        held["%.4f_%.4f" % (xc, d)] = rec
        print(json.dumps(dict(xc=xc, d=d, **{k: dict(max=max(rec[k]["ritz"]), e1=rec[k]["ritz"][0], drop=rec[k]["drop_reproduced"],
              e2=rec[k]["e2_energy"]["median"]) for k in bases})), flush=True)
    train = {}
    for (xc, d) in (geo["train"][:2] if SMOKE else geo["train"]):
        inst = fc.build(xc=xc, d=d, rho=0.05, h_max=0.2)
        w, _ = fc.eig(inst["K"], inst["M"], 8)
        pr = P.fam_problem(inst)
        train["%.4f_%.4f" % (xc, d)] = {k: dict(ritz=p6.ritz_errors(np.asarray(f(pr)), inst, w, 8)) for k, f in nets.items()}
        for k in nets:
            train["%.4f_%.4f" % (xc, d)][k]["drop_reproduced"] = PB._drop(w_plain, w[0], train["%.4f_%.4f" % (xc, d)][k]["ritz"][0])
    hl = list(held.values())
    summ = {}
    for b in ["modal_m8"] + list(nets):
        e15 = [float(np.median(g[b]["ritz"][:5])) for g in hl]
        w1n = np.array([g["w"][0] * (1 + g[b]["ritz"][0]) for g in hl]); w1f = np.array([g["w"][0] for g in hl])
        slope, icpt = np.polyfit(w1f, w1n, 1)
        r2 = 1 - np.sum((w1n - (slope * w1f + icpt)) ** 2) / np.sum((w1n - w1n.mean()) ** 2)
        big = [g[b]["drop_reproduced"] for g in hl if g["w1_drop_fe"] >= 0.01]
        summ[b] = dict(H4=float(np.median(e15)), H5=float(np.median([g[b]["drop_reproduced"] for g in hl])),
                       H6=float(np.median([g[b]["e2_energy"]["median"] for g in hl])),
                       median_e1=float(np.median([g[b]["ritz"][0] for g in hl])),
                       median_max_ritz=float(np.median([max(g[b]["ritz"]) for g in hl])),
                       max_max_ritz=float(max(max(g[b]["ritz"]) for g in hl)),
                       median_e1_tip_acc=float(np.median([g[b]["e1_tip_acc"]["median"] for g in hl])),
                       slope=float(slope), r2=float(r2), H5_drop_ge_1pct=float(np.median(big)) if big else None, n_big=len(big))
    tr_summ = {k: dict(median_max_ritz=float(np.median([max(v[k]["ritz"]) for v in train.values()])),
                       max_max_ritz=float(max(max(v[k]["ritz"]) for v in train.values())),
                       median_e1=float(np.median([v[k]["ritz"][0] for v in train.values()])),
                       median_drop=float(np.median([v[k]["drop_reproduced"] for v in train.values()]))) for k in nets}
    jdump(dict(w_plain=w_plain, held=held, train=train, summary_held=summ, summary_train=tr_summ), "evalfam0e.json")
    print(json.dumps(dict(summary_held=summ, summary_train=tr_summ), indent=1, default=P._js))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "fam":
        cmd_fam(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        dict(checks=cmd_checks, check2=cmd_check2, evalfam=cmd_evalfam)[c]()
