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
