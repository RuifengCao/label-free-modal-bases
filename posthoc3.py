"""Post hoc diagnostics F1 to F3 (see POSTHOC3.md).

  python -I posthoc3.py f1      # region and shear split of the first Ritz vector error, planned bases
  python -I posthoc3.py f3      # planned objective at fixed bases (modal, POD, planned trained bases)
  python -I posthoc3.py f2      # hard clamp: F2a, F2b (one field), F2c (per-step objective, m = 8, seed 0)
"""
import sys, os, json, time, pickle
import numpy as np
import scipy.linalg as sla
from skfem import Basis, ElementTriP2, ElementVector, BilinearForm, asm
from skfem.models.elasticity import linear_elasticity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc as ph
import jax
import jax.numpy as jnp

OUT = os.path.join(HERE, "out")


@BilinearForm
def shear_form(u, v, w):
    """mu * gamma(u) * gamma(v): the shear part of the bilinear form of linear_elasticity."""
    return fc.MU * (u.grad[0, 1] + u.grad[1, 0]) * (v.grad[0, 1] + v.grad[1, 0])


def region_mats(inst):
    mesh = inst["mesh"]
    e = ElementVector(ElementTriP2())
    ib = Basis(mesh, e, intorder=4)
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten()
    free = np.setdiff1d(np.arange(ib.N), D)
    cen = mesh.p[:, mesh.t].mean(axis=1)
    xc, d, rho = inst["params"]
    clamp = np.where(cen[0] < 0.5)[0]
    notch = np.where(np.hypot(cen[0] - xc, cen[1] - (fc.H - d)) < 0.5)[0]
    rest = np.setdiff1d(np.arange(mesh.t.shape[1]), np.union1d(clamp, notch))
    mats = {}
    for name, idx in (("clamp", clamp), ("notch", notch), ("rest", rest)):
        b = Basis(mesh, e, intorder=4, elements=idx)
        Kt = asm(linear_elasticity(fc.LAM_PS, fc.MU), b).tocsr()[free][:, free]
        Ks = asm(shear_form, b).tocsr()[free][:, free]
        mats[name] = (Kt, Ks)
    tot = mats["clamp"][0] + mats["notch"][0] + mats["rest"][0]
    assert abs(tot - inst["K"]).max() < 1e-10 * abs(inst["K"]).max()
    return mats, dict(clamp=int(len(clamp)), notch=int(len(notch)), rest=int(len(rest)))


def split(vec, mats):
    parts = {k: (float(vec @ (Kt @ vec)), float(vec @ (Ks @ vec))) for k, (Kt, Ks) in mats.items()}
    tot = sum(p[0] for p in parts.values())
    out = {k: dict(share=p[0] / tot, shear_fraction_within=p[1] / p[0] if p[0] > 0 else 0.0) for k, p in parts.items()}
    out["shear_share_total"] = sum(p[1] for p in parts.values()) / tot
    return out


def dof_x(inst):
    xd = np.zeros(inst["n_free"])
    xd[inst["ixf"]] = inst["pts"][:, 0]
    xd[inst["iyf"]] = inst["pts"][:, 0]
    assert np.all(xd > 0)
    return xd


def first_ritz(Phi, inst, v1):
    K, M = inst["K"], inst["M"]
    Kr = Phi.T @ (K @ Phi); Mr = Phi.T @ (M @ Phi)
    lam, Q = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    phi = Phi @ Q[:, 0]
    phi = phi / np.sqrt(phi @ (M @ phi))
    c = float(phi @ (M @ v1))
    if c < 0:
        phi, c = -phi, -c
    return phi, c, float(lam[0])


def ritz_report(Phi, inst, w, v1, mats, xd):
    phi, cM, lam = first_ritz(Phi, inst, v1)
    e = phi - v1
    near = xd < 0.15
    tip = inst["tip_dof"]
    return dict(cos_M=cM, ritz_freq_rel_err=float(np.sqrt(lam) / w[0] - 1),
                eKe_over_w1sq=float(e @ (inst["K"] @ e)) / w[0] ** 2, eMe=float(e @ (inst["M"] @ e)),
                near_clamp_max_over_tip=float(np.max(np.abs(phi[near])) / abs(phi[tip])),
                v1_near_clamp_max_over_tip=float(np.max(np.abs(v1[near])) / abs(v1[tip])),
                error_energy=split(e, mats), ritz_vector_energy=split(phi, mats))


def load_params(fn):
    with open(fn, "rb") as f:
        d = pickle.load(f)
    return [(jnp.asarray(Wt), jnp.asarray(b)) for (Wt, b) in d["params"]], d


def cmd_f1():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    mats, counts = region_mats(inst)
    xd = dof_x(inst)
    v1 = V[:, 0]
    res = dict(element_counts=counts, v1_energy=split(v1, mats))
    for m in (8, 16):
        for s in (0, 1):
            params, _ = load_params(os.path.join(OUT, "params_m%d_s%d.pkl" % (m, s)))
            pr = lb.Problem(inst, a_r, b_r, dt, m)
            Phi = lb.basis_numpy(params, pr)
            res["m%d_s%d" % (m, s)] = ritz_report(Phi, inst, w, v1, mats, xd)
            print("F1 m=%d s=%d" % (m, s), json.dumps(res["m%d_s%d" % (m, s)]), flush=True)
    print("F1 v1", json.dumps(res["v1_energy"]), flush=True)
    with open(os.path.join(OUT, "posthoc3_f1.json"), "w") as f:
        json.dump(res, f, indent=1)


def loss_from_basis(Phi, pr, W, G):
    """Value of lb.loss_fn for a fixed basis (same body; the gradient stops do not change the value)."""
    Kr, Mr, Cr, Ar, Pr = lb.reduced_ops(Phi, pr)
    a0, a1, a2, a3, a4, a5 = pr.nc
    cf = jax.scipy.linalg.cho_factor(0.5 * (Ar + Ar.T))
    fr = W @ Pr.T
    acc0 = jnp.linalg.solve(Mr, (fr * G[:, :1]).T).T
    B = W.shape[0]
    c0 = jnp.zeros((B, pr.m)); v0 = jnp.zeros((B, pr.m))

    def step(carry, g_next):
        c, cd, cdd = carry
        b = fr * g_next[:, None] + (a0 * c + a2 * cd + a3 * cdd) @ Mr.T + (a1 * c + a4 * cd + a5 * cdd) @ Cr.T
        cn = jax.scipy.linalg.cho_solve(cf, b.T).T
        pot = 0.5 * jnp.sum(cn * (cn @ Ar.T), axis=1) - jnp.sum(b * cn, axis=1)
        cdd_n = a0 * (cn - c) - a2 * cd - a3 * cdd
        cd_n = cd + pr.dt * ((1 - fc.GAMMA) * cdd + fc.GAMMA * cdd_n)
        return (cn, cd_n, cdd_n), pot

    _, pots = jax.lax.scan(step, (c0, v0, acc0), G[:, 1:].T)
    return jnp.mean(pots)


def cmd_f3():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    with open(os.path.join(OUT, "prep.pkl"), "rb") as f:
        prep = pickle.load(f)
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    res = {}
    params, _ = load_params(os.path.join(OUT, "params_m8_s0.pkl"))
    pr = lb.Problem(inst, a_r, b_r, dt, 8)
    l_net = float(lb.loss_fn(params, pr, Wj, Gj))
    l_fix = float(loss_from_basis(lb.orthonormal_basis(params, pr), pr, Wj, Gj))
    res["consistency_rel"] = abs(l_net - l_fix) / abs(l_net)
    print("F3 consistency", res["consistency_rel"], flush=True)
    for m in (8, 16):
        pr = lb.Problem(inst, a_r, b_r, dt, m)
        bases = dict(modal=V[:, :m], pod=rc.pod_basis(inst, prep["train_refs"], m))
        for s in (0, 1):
            params, _ = load_params(os.path.join(OUT, "params_m%d_s%d.pkl" % (m, s)))
            bases["trained_s%d" % s] = lb.basis_numpy(params, pr)
        for name, Phi in bases.items():
            res["%s_m%d" % (name, m)] = float(loss_from_basis(jnp.asarray(Phi), pr, Wj, Gj))
            print("F3 %s m=%d %.10e" % (name, m, res["%s_m%d" % (name, m)]), flush=True)
    with open(os.path.join(OUT, "posthoc3_f3.json"), "w") as f:
        json.dump(res, f, indent=1)


def cmd_f2():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    xd = dof_x(inst)
    clamp = jnp.asarray(xd / fc.L)[:, None]
    raw_soft = lb.raw_basis

    def raw_basis_hard(params, pr):
        return raw_soft(params, pr) * clamp

    lb.raw_basis = raw_basis_hard          # lb.orthonormal_basis looks the name up at call time
    res = {}
    pr1 = lb.Problem(inst, a_r, b_r, dt, 1)

    def rq_loss(p):
        Phi = lb.raw_basis(p, pr1)
        return (Phi[:, 0] @ (pr1.K @ Phi)[:, 0]) / (Phi[:, 0] @ (pr1.M @ Phi)[:, 0])

    v1 = jnp.asarray(V[:, 0])
    Kv1 = (pr1.K @ v1[:, None])[:, 0]
    v1Kv1 = float(v1 @ Kv1)

    def sup_loss(p):
        phi = lb.raw_basis(p, pr1)[:, 0]
        Kphi = (pr1.K @ phi[:, None])[:, 0]
        return 1.0 - (phi @ Kv1) ** 2 / ((phi @ Kphi) * v1Kv1)

    for name, loss in (("F2a_rayleigh_hard_clamp", rq_loss), ("F2b_supervised_hard_clamp", sup_loss)):
        t0 = time.time()
        params, hist = ph.train_generic(loss, pr1, 1, seed=0, iters=3000, log_every=500)
        phi = np.asarray(lb.raw_basis(params, pr1))[:, 0]
        rq = float(phi @ (inst["K"] @ phi) / (phi @ (inst["M"] @ phi)))
        res[name] = dict(ritz_freq_rel_err=float(np.sqrt(rq) / w[0] - 1), final_loss=float(hist[-1][1]),
                         t_s=time.time() - t0, hist=hist)
        print(name, res[name]["ritz_freq_rel_err"], flush=True)
    # F2c: planned per-step objective, hard clamp, m = 8, seed 0
    m = 8
    pr = lb.Problem(inst, a_r, b_r, dt, m)
    W, G = rc.W_G(train_cases, dt)
    Wj, Gj = jnp.asarray(W), jnp.asarray(G)
    t0 = time.time()
    params, hist = ph.train_generic(lambda p: lb.loss_fn(p, pr, Wj, Gj), pr, m, seed=0, iters=3000, log_every=500)
    t_train = time.time() - t0
    with open(os.path.join(OUT, "posthoc3_F2c_m8.pkl"), "wb") as f:
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
    mats, _ = region_mats(inst)
    s["first_ritz"] = ritz_report(Phi, inst, w, V[:, 0], mats, xd)
    s["objective_value"] = float(loss_from_basis(jnp.asarray(Phi), pr, Wj, Gj))
    res["F2c_per_step_hard_clamp_m8_s0"] = s
    print("F2c", json.dumps({k: s[k] for k in keys + ["e4_rel_freq", "objective_value"]}), flush=True)
    with open(os.path.join(OUT, "posthoc3_f2.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    dict(f1=cmd_f1, f2=cmd_f2, f3=cmd_f3)[sys.argv[1]]()
