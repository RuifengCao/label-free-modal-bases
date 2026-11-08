"""Post hoc diagnostics F4a and F4b (see POSTHOC4.md): exact clamp plus three notch-angle inputs.

  python -I posthoc4.py
"""
import sys, os, json, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc as ph
import posthoc3 as p3
import jax
import jax.numpy as jnp

OUT = os.path.join(HERE, "out")


def angle_feats(pts, params):
    """Angle around the centre of the notch root arc, from the downward direction; the branch cut lies in the gap."""
    xc, d, rho = params
    dx = pts[:, 0] - xc
    dy = pts[:, 1] - (fc.H - d + rho)
    gap = (np.abs(dx) < rho - 1e-12) & (dy > 0)
    assert not np.any(gap)                      # no node lies on the branch cut
    th = np.arctan2(dx, -dy)
    return np.stack([th / np.pi, np.sin(th / 2), np.cos(th / 2)], axis=1)


def main():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    xd = p3.dof_x(inst)
    clamp = jnp.asarray(xd / fc.L)[:, None]
    raw_soft = lb.raw_basis

    def raw_basis_hard(params, pr):
        return raw_soft(params, pr) * clamp

    lb.raw_basis = raw_basis_hard          # exact clamp, as in F2
    extra = jnp.asarray(angle_feats(inst["pts"], inst["params"]))

    def with_angle(pr):
        pr.feats = jnp.concatenate([pr.feats, extra], axis=1)
        return pr

    res = {}
    # F4a: one field, supervised to the first eigenvector (as F2b)
    pr1 = with_angle(lb.Problem(inst, a_r, b_r, dt, 1))
    v1 = jnp.asarray(V[:, 0])
    Kv1 = (pr1.K @ v1[:, None])[:, 0]
    v1Kv1 = float(v1 @ Kv1)

    def sup_loss(p):
        phi = lb.raw_basis(p, pr1)[:, 0]
        Kphi = (pr1.K @ phi[:, None])[:, 0]
        return 1.0 - (phi @ Kv1) ** 2 / ((phi @ Kphi) * v1Kv1)

    t0 = time.time()
    params, hist = ph.train_generic(sup_loss, pr1, 1, seed=0, iters=3000, log_every=500)
    phi = np.asarray(lb.raw_basis(params, pr1))[:, 0]
    rq = float(phi @ (inst["K"] @ phi) / (phi @ (inst["M"] @ phi)))
    res["F4a_supervised_hard_clamp_angle"] = dict(ritz_freq_rel_err=float(np.sqrt(rq) / w[0] - 1),
                                                  final_loss=float(hist[-1][1]), t_s=time.time() - t0, hist=hist)
    print("F4a", res["F4a_supervised_hard_clamp_angle"]["ritz_freq_rel_err"], flush=True)
    # F4b: planned per-step objective, m = 8, seed 0 (as F2c)
    m = 8
    pr = with_angle(lb.Problem(inst, a_r, b_r, dt, m))
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    t0 = time.time()
    params, hist = ph.train_generic(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, m, seed=0, iters=3000, log_every=500)
    t_train = time.time() - t0
    with open(os.path.join(OUT, "posthoc4_F4b_m8.pkl"), "wb") as f:
        pickle.dump(dict(params=[(np.asarray(Wt), np.asarray(b)) for (Wt, b) in params], hist=hist, t_train=t_train), f)
    Phi = lb.basis_numpy(params, pr)
    with open(os.path.join(OUT, "prep.pkl"), "rb") as f:
        prep = pickle.load(f)
    a0, a1 = fc.newmark_consts(dt)[:2]
    A = (inst["K"] + a0 * inst["M"] + a1 * C).tocsc()
    rows = [rc.metrics(Phi, inst, C, c, dt, r, w, A) for c, r in zip(eval_cases, prep["eval_refs"])]
    keys = ["e1_tip_disp", "e1_tip_acc", "e2_energy", "e3_median", "e3_p90", "e3_max"]
    s = {k: dict(median=float(np.median([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in keys}
    s["e4_rel_freq"] = rows[0]["e4_rel_freq"]; s["t_train_s"] = t_train; s["loss_history"] = hist
    mats, _ = p3.region_mats(inst)
    s["first_ritz"] = p3.ritz_report(Phi, inst, w, V[:, 0], mats, xd)
    s["objective_value"] = float(p3.loss_from_basis(jnp.asarray(Phi), pr, Wj, Gj))
    res["F4b_per_step_hard_clamp_angle_m8_s0"] = s
    print("F4b", json.dumps({k: s[k] for k in keys + ["e4_rel_freq", "objective_value"]}), flush=True)
    with open(os.path.join(OUT, "posthoc4_results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
