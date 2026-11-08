"""X1a control (DEVIATIONS_EXPLORE.md, items 18 to 21): the mapped reduced basis with its 24 sources computed on the evaluation
mesh (rbm_fine), against the X1a results of the laptop.

  python -I -B rbfine.py H PART NPART [CHECK]     # H = 0.1 or 0.05; held-out geometries PART::NPART;
                                                  # CHECK = number of these geometries on which rbm_all (sources at 0.2)
                                                  # is recomputed for the cross-check with the laptop values
Environment: EXP_NETS (formal results folder: geometries come from formal/geometries.json through phase1),
EXP_OUT (scratch for the formal modules), RBF_OUT (output folder; default rbfine/), RBF_SMOKE=1 (h 0.14, 2 geometries,
2 evaluation cases).
"""
import os, sys, json, time
import numpy as np
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("RBF_SMOKE", "0") == "1"
os.environ.setdefault("EXP_OUT", os.path.join(HERE, "rbfine", "_scratch"))
os.environ["EXP_SMOKE"] = "0"
sys.path.insert(0, HERE)
import explore as X                                  # the X1a code: geom_setup_h, stream_metrics, common
F, P, PB, PD, fc, rc, p2, p6 = X.F, X.P, X.PB, X.PD, X.fc, X.rc, X.p2, X.p6
OUT = os.environ.get("RBF_OUT", os.path.join(HERE, "rbfine"))
os.makedirs(OUT, exist_ok=True)


def interp_mapped_safe(src, src_g, dst, dst_g):
    """phase0d.interp_mapped, except that a mapped point with a non-finite interpolated value is moved by 1e-9 towards
    the centre of the beam and interpolated again (DEVIATIONS_EXPLORE.md, item 21); if still non-finite, it takes the
    value of the nearest source node (item 19). Returns also the counts of both kinds and the largest distance to the
    nearest node used."""
    q = PD.map_points(dst["pts"], src_g, dst_g)
    ix = PD.LinearNDInterpolator(src["tri"], src["ux"]); iy = PD.LinearNDInterpolator(src["tri"], src["uy"])
    vx, vy = ix(q), iy(q)
    finite = lambda a, b: np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    bad = ~finite(vx, vy)
    n_nudge, n_near, dist = int(bad.sum()), 0, 0.0
    if bad.any():                                    # item 21: 1e-9 towards the centre of the beam, linear again
        qb = q[bad]
        dvec = np.array([PD.L / 2, PD.H / 2]) - qb
        qn = qb + 1e-9 * dvec / np.linalg.norm(dvec, axis=1)[:, None]
        nx, ny = ix(qn), iy(qn)
        still = ~finite(nx, ny)
        if still.any():                              # item 19 fallback, only if the nudge did not help
            sp = src["tri"].points
            dd = np.linalg.norm(qb[still][:, None, :] - sp[None, :, :], axis=2)
            j = np.argmin(dd, axis=1)
            dist = float(np.max(dd[np.arange(len(j)), j]))
            nx[still] = src["ux"][j]; ny[still] = src["uy"][j]
            n_near = int(still.sum())
        vx[bad] = nx; vy[bad] = ny
    Phi = np.zeros((dst["n_free"], src["ux"].shape[1]))
    Phi[dst["ixf"]] = vx
    Phi[dst["iyf"]] = vy
    assert np.all(np.isfinite(Phi))
    return Phi, float(np.mean(PD.in_void(q, src_g))), (n_nudge, n_near), dist


def rbm_basis_safe(srcs, dst, dst_g, m=8, tol=1e-10):
    """phase0d.rbm_basis with gram="M" (as used), with interp_mapped_safe in place of interp_mapped."""
    t0 = time.perf_counter()
    blocks, voids, counts, dists = zip(*[interp_mapped_safe(s, g, dst, dst_g) for g, s in srcs])
    B = np.hstack(blocks)
    G = B.T @ (dst["M"] @ B)
    lam, U = np.linalg.eigh(0.5 * (G + G.T))
    keep = lam > tol * lam.max()
    Q = B @ (U[:, keep] / np.sqrt(lam[keep]))
    Kr = Q.T @ (dst["K"] @ Q)
    th, Y = np.linalg.eigh(0.5 * (Kr + Kr.T))
    return Q @ Y[:, :m], dict(n_vectors=int(B.shape[1]), rank=int(keep.sum()), t=time.perf_counter() - t0,
                              void_mean=float(np.mean(voids)), void_max=float(np.max(voids)), tol=tol, gram="M",
                              nonfinite_points=int(sum(c[0] for c in counts)), nearest_node_points=int(sum(c[1] for c in counts)),
                              nearest_node_max_dist=float(max(dists)))


def sources_h(geoms, h):
    """phase0d.sources with the mesh size as a parameter; otherwise the same code."""
    out = []
    for (xc, d) in geoms:
        si = fc.build(xc=xc, d=d, rho=PD.RHO, h_max=h)
        sw, sV = fc.eig(si["K"], si["M"], 8)
        out.append(((xc, d), PB.src_prep(si, sV), np.asarray(sw[:8])))
    return out


def evaluate(g, h, bases_fn, w_plain):
    xc, d = g
    t0 = time.time()
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = X.geom_setup_h(xc, d, h)
    if SMOKE:
        eval_cases = eval_cases[:2]
    rec = dict(xc=xc, d=d, h_max=h, n_free=int(inst["n_free"]), w=[float(x) for x in w[:8]], t_setup=time.time() - t0)
    bases, infos = {}, {}
    for k, fn in bases_fn.items():
        t1 = time.time()
        B, info = fn(inst, g)
        rec["t_map_" + k] = time.time() - t1
        bases[k], infos[k] = B, info
    a0, a1 = fc.newmark_consts(dt)[:2]
    lu = spla.splu((inst["K"] + a0 * inst["M"] + a1 * C).tocsc())
    lu_m = spla.splu(inst["M"].tocsc())
    per = {k: [] for k in bases}
    t1 = time.time()
    for case in eval_cases:
        r = X.stream_metrics(bases, inst, C, case, dt, lu, lu_m)
        for k in bases:
            per[k].append(r[k])
    rec["t_metrics"] = time.time() - t1
    for k, B in bases.items():
        s = dict(ritz=p6.ritz_errors(B, inst, w, 8), sin2=P.sin2(B, inst, V), rb_info=infos[k])
        for q in ("e1_tip_disp", "e1_tip_acc", "e2_energy"):
            vals = [x[q] for x in per[k]]
            s[q] = dict(median=float(np.median(vals)), max=float(np.max(vals)), per_case=[float(v) for v in vals])
        s["drop_reproduced"] = F._drop(w_plain, w[0], s["ritz"][0])
        rec[k] = s
    rec["t_eval"] = time.time() - t0
    print("%s h=%g n=%d: %.0f s (maps %s, metrics %.0f); max e: %s" % (
        X.key(g), h, rec["n_free"], rec["t_eval"], ", ".join("%.1f" % rec["t_map_" + k] for k in bases), rec["t_metrics"],
        ", ".join("%s %.2e" % (k, max(abs(v) for v in rec[k]["ritz"])) for k in bases)), flush=True)
    return rec


def main(h, part, npart, ncheck):
    geo = F.geometries()
    timing = {}
    t0 = time.perf_counter(); src02 = PD.sources(geo["train"]); timing["sources_0.2"] = time.perf_counter() - t0
    t0 = time.perf_counter(); srch = sources_h(geo["train"], h); timing["sources_h"] = time.perf_counter() - t0
    assert [g for g, _, _ in srch] == list(geo["train"])
    fine = [(gg, s) for gg, s, _ in srch]
    coarse = [(gg, s) for gg, s, _ in src02]
    held = geo["held"][part::npart]
    if SMOKE:
        held = held[:2]
    w_plain = float(p2.plain_cantilever_w1(h_max=h)[0])
    res = {}
    for i, g in enumerate(held):
        fns = dict(rbm_fine=lambda inst, gg: rbm_basis_safe(fine, inst, gg, m=X.M_MODES, tol=F.RBM_TOL))
        if i < ncheck:
            fns["rbm_all"] = lambda inst, gg: rbm_basis_safe(coarse, inst, gg, m=X.M_MODES, tol=F.RBM_TOL)   # item 20
        res["%s_h%g" % (X.key(g), h)] = evaluate(g, h, fns, w_plain)
    out = dict(h_max=h, part=part, npart=npart, smoke=SMOKE, timing=timing, w_plain=w_plain, geoms=res)
    fn = os.path.join(OUT, "rbfine_h%g_part%d.json" % (h, part))
    with open(fn + ".partial", "w") as f:
        json.dump(out, f, indent=1, default=P._js)
    os.replace(fn + ".partial", fn)
    print("sources: 0.2 %.1f s, %g %.1f s" % (timing["sources_0.2"], h, timing["sources_h"]), flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
