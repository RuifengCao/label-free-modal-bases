"""Post hoc diagnostics V1 to V3 (see POSTHOC.md).

  python -I posthoc.py train V M      # V in {V1, V2, V3}; seed 0; saves out/posthoc_V_mM.pkl
  python -I posthoc.py eval           # E1 to E4 at the evaluation step; out/posthoc_results.json
"""
import sys, os, json, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import jax
import jax.numpy as jnp

OUT = os.path.join(HERE, "out")


def static_loss(params, pr):
    Phi = lb.orthonormal_basis(params, pr)
    Kr = Phi.T @ (pr.K @ Phi)
    Pr = Phi.T @ pr.P                                          # (m, 3)
    cf = jax.scipy.linalg.cho_factor(0.5 * (Kr + Kr.T))
    c = jax.scipy.linalg.cho_solve(cf, Pr)
    return jnp.mean(-0.5 * jnp.sum(Pr * c, axis=0))             # Pi(Phi c*) = -1/2 P^T Phi Kr^-1 Phi^T P


def trace_loss(params, pr):
    Phi = lb.orthonormal_basis(params, pr)
    return jnp.trace(Phi.T @ (pr.K @ Phi))


def train_generic(loss, pr, m, seed=0, iters=3000, lr0=1e-3, lr1=1e-5, log_every=250):
    sizes = [int(pr.feats.shape[1]), 64, 64, 64, 64, 2 * m]
    params = lb.init_mlp(jax.random.PRNGKey(seed), sizes)
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

    hist = []
    for t in range(1, iters + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * (t - 1) / iters))
        val, g = vg(params)
        params, mo, ve = update(params, mo, ve, g, t, lr)
        if t % log_every == 0 or t == 1:
            hist.append((t, float(val))); print("  iter %d loss %.8e" % (t, float(val)), flush=True)
    return params, hist


def cmd_train(variant, m):
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    t0 = time.time()
    if variant == "V1":
        pr = lb.Problem(inst, a_r, b_r, dt, m)
        params, hist = train_generic(lambda p: static_loss(p, pr), pr, m)
    elif variant == "V2":
        pr = lb.Problem(inst, a_r, b_r, dt, m)
        params, hist = train_generic(lambda p: trace_loss(p, pr), pr, m)
    elif variant == "V3":
        dt_train = (2 * np.pi / w[0]) / 20.0
        pr = lb.Problem(inst, a_r, b_r, dt_train, m)
        W, G = rc.W_G(train_cases, dt_train)
        Wj, Gj = jnp.asarray(W), jnp.asarray(G)
        params, hist = train_generic(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, m)
    t_train = time.time() - t0
    print("trained %s m=%d in %.1f s" % (variant, m, t_train), flush=True)
    with open(os.path.join(OUT, "posthoc_%s_m%d.pkl" % (variant, m)), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], hist=hist, t_train=t_train), f)


def cmd_eval():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    with open(os.path.join(OUT, "prep.pkl"), "rb") as f:
        prep = pickle.load(f)
    a0, a1 = fc.newmark_consts(dt)[:2]
    A = (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()
    res = {}
    for variant in ("V1", "V2", "V3"):
        for m in (8, 16):
            fn = os.path.join(OUT, "posthoc_%s_m%d.pkl" % (variant, m))
            if not os.path.exists(fn):
                continue
            with open(fn, "rb") as f:
                d = pickle.load(f)
            pr = lb.Problem(inst, a_r, b_r, dt, m)
            params = [(jnp.asarray(Wt), jnp.asarray(b)) for (Wt, b) in d["params"]]
            Phi = lb.basis_numpy(params, pr)
            rows = [rc.metrics(Phi, inst, C, c, dt, r, w, A) for c, r in zip(eval_cases, prep["eval_refs"])]
            keys = ["e1_tip_disp", "e1_tip_acc", "e2_energy", "e3_median", "e3_p90", "e3_max"]
            s = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in keys}
            s["e4_rel_freq"] = rows[0]["e4_rel_freq"]; s["t_train_s"] = d["t_train"]; s["loss_history"] = d["hist"]
            res["%s_m%d" % (variant, m)] = s
            print("evaluated %s m=%d" % (variant, m), flush=True)
    with open(os.path.join(OUT, "posthoc_results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        cmd_train(sys.argv[2], int(sys.argv[3]))
    else:
        cmd_eval()
