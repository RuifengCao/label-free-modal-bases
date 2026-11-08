"""Phase 0c (exploratory): one network per mode (see PHASE0C_PLAN.md).

  python -I phase0c.py checks             # C1
  python -I phase0c.py train OBJ SEED     # OBJ in o2, o3, o3s, o4; G0, 20,000 iterations
  python -I phase0c.py evalall            # evaluation of every run in out0c

Environment: P0C_OUT replaces the output folder; P0_SMOKE=1 and P0_LOG as in phase0.py (smoke: 20 iterations).
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P
import phase0b as PB
from phase0 import fc, rc, lb, p3, p6, jax, jnp

OUT = os.environ.get("P0C_OUT", os.path.join(HERE, "out0c"))
os.makedirs(OUT, exist_ok=True)
M_MODES, WIDTH = 8, 24
ITERS = 20 if P.SMOKE else 20000


def init_sep(seed, n_in):
    keys = jax.random.split(jax.random.PRNGKey(seed), M_MODES)
    return [lb.init_mlp(k, [n_in, WIDTH, WIDTH, WIDTH, WIDTH, 2]) for k in keys]


def raw_sep(params, pr):
    """Column i from MLP i: its two outputs are the x and y displacements; exact clamp as the shared network."""
    cols = []
    for p in params:
        out = lb.mlp(p, pr.feats)                                   # (n_nodes, 2)
        c = jnp.zeros(pr.n).at[pr.ixf].set(out[:, 0]).at[pr.iyf].set(out[:, 1])
        cols.append(c)
    return jnp.stack(cols, axis=1) * pr.clamp


def orth_sep(params, pr):
    return P._orth(raw_sep(params, pr), pr.M)


def loss_from_raw(obj, Phi, pr):
    if obj == "o4":
        Q = P._orth(Phi, pr.M)
        Kr = Q.T @ (pr.K @ Q)
        return 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(0.5 * (Kr + Kr.T)))))
    if obj == "o2":
        Q = P._orth(Phi, pr.M)
        return jnp.trace(Q.T @ (pr.K @ Q))
    if obj == "o3s":
        rq = jnp.sum(Phi * (pr.K @ Phi), axis=0) / jnp.sum(Phi * (pr.M @ Phi), axis=0)
        Phi = Phi[:, jnp.argsort(jax.lax.stop_gradient(rq))]
    assert obj in ("o3", "o3s")
    Q = jax.lax.stop_gradient(P._orth(Phi, pr.M))
    U = jnp.triu(jnp.ones((pr.m, pr.m)), 1)
    R = Phi - Q @ (U * (Q.T @ (pr.M @ Phi)))
    return jnp.sum(jnp.sum(R * (pr.K @ R), axis=0) / jnp.sum(R * (pr.M @ R), axis=0))


def setup():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    pr = p6.problem(inst, a_r, b_r, dt, M_MODES, "hard_angle")
    return inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases, pr


def cmd_checks():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases, pr = setup()
    params = init_sep(0, int(pr.feats.shape[1]))
    Phi = raw_sep(params, pr)
    ref = P.objectives_np(np.asarray(Phi), inst)
    res = dict(shape=list(Phi.shape), n_params=int(sum(np.size(W) + np.size(b) for p in params for (W, b) in p)))
    for obj in ("o2", "o3", "o4"):
        v = float(loss_from_raw(obj, Phi, pr))
        res[obj + "_rel"] = abs(v - ref[obj]) / abs(ref[obj])
    res["o3s_vs_trace_rel"] = abs(float(loss_from_raw("o3s", Phi, pr)) - ref["o2"]) / abs(ref["o2"])
    res["pass"] = bool(res["shape"] == [inst["n_free"], 8] and all(res[k] < 1e-9 for k in ("o2_rel", "o3_rel", "o4_rel", "o3s_vs_trace_rel")))
    print(json.dumps(res, indent=1), flush=True)
    with open(os.path.join(OUT, "checks0c.json"), "w") as f:
        json.dump(res, f, indent=1)


def cmd_train(obj, seed):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases, pr = setup()
    params0 = init_sep(seed, int(pr.feats.shape[1]))
    bfun = jax.jit(lambda p: orth_sep(p, pr))

    def cb(p):
        Phi = np.asarray(bfun(p))
        return dict(ritz=p6.ritz_errors(Phi, inst, w, 8), sin2=P.sin2(Phi, inst, V))
    t0 = time.time()
    params, hist, logs = p6.train_logged(lambda p: loss_from_raw(obj, raw_sep(p, pr), pr), pr, M_MODES, seed=seed,
                                         iters=ITERS, lr0=1e-3, lr1=1e-5, log_every=P.LOG_EVERY, params0=params0, callback=cb)
    with open(os.path.join(OUT, "sep_%s_s%d.pkl" % (obj, seed)), "wb") as f:
        pickle.dump(dict(params=[[(np.asarray(W), np.asarray(b)) for (W, b) in p] for p in params], logs=logs,
                         t_train=time.time() - t0, obj=obj, seed=seed, iters=ITERS), f)
    print("done sep %s s%d %.0f s" % (obj, seed, time.time() - t0), flush=True)


def cmd_evalall():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases, pr = setup()
    if P.SMOKE:
        eval_cases = eval_cases[:2]
    A = p6.step_operator(inst, C, dt)
    refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, rc.N_STEPS) for c in eval_cases]
    res = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("sep_") and fn.endswith(".pkl"):
            d = pickle.load(open(os.path.join(OUT, fn), "rb"))
            params = [[(jnp.asarray(W), jnp.asarray(b)) for (W, b) in p] for p in d["params"]]
            Phi = np.asarray(orth_sep(params, pr))
            s = P.eval_basis(Phi, inst, w, V, C, dt, A, refs, eval_cases)
            s.update(objectives=P.objectives_np(Phi, inst), t_train=d["t_train"], logs=d["logs"])
            res[fn[:-4]] = s
            print(json.dumps(dict(run=fn[:-4], ritz=s["ritz"], e1_acc=s["e1_tip_acc"]["median"], e2=s["e2_energy"]["median"])), flush=True)
    with open(os.path.join(OUT, "eval0c.json"), "w") as f:
        json.dump(res, f, indent=1, default=P._js)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "train":
        cmd_train(sys.argv[2], int(sys.argv[3]))
    else:
        dict(checks=cmd_checks, evalall=cmd_evalall)[c]()
