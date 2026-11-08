"""Phase 0d (exploratory): classical controls for the geometry family (see PHASE0D_PLAN.md).

  python -I phase0d.py checks   # B1, B2, B3 (on the check pairs), B4
  python -I phase0d.py b4diag   # deviation 4: B4 with other truncation settings
  python -I phase0d.py eval     # (m) mapped reduced basis and (n) frequency regression, pilot 2 and the Phase 0 pilot

Environment: P0D_OUT replaces the output folder; P0_SMOKE=1 shortens the evaluations.
"""
import sys, os, json, time
import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P
import phase0b as PB
from phase0 import fc, p2, p6
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

OUT = os.environ.get("P0D_OUT", os.path.join(HERE, "out0d"))
os.makedirs(OUT, exist_ok=True)
SMOKE = P.SMOKE
L, H, RHO = fc.L, fc.H, 0.05
XREF, DREF, A_ZONE, B_BLEND = 3.0, 0.25, 0.5, 0.5


def jdump(obj, fn):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1, default=P._js)


# ----------------------------------------------------------------------------- the map

def _w(x, xc):
    return np.clip(1.0 - (np.abs(x - xc) - A_ZONE) / B_BLEND, 0.0, 1.0)


def to_ref(pts, xc, d):
    """Physical points of geometry (xc, d) to the reference geometry."""
    x, y = pts[:, 0], pts[:, 1]
    xr = np.interp(x, [0.0, xc - A_ZONE, xc + A_ZONE, L], [0.0, XREF - A_ZONE, XREF + A_ZONE, L])
    w = _w(x, xc)
    phi = np.interp(y, [0.0, H - d, H], [0.0, H - DREF, H])
    return np.stack([xr, (1.0 - w) * y + w * phi], axis=1)


def from_ref(pts, xc, d):
    """Reference points to the physical points of geometry (xc, d); exact inverse of to_ref."""
    xr, yr = pts[:, 0], pts[:, 1]
    x = np.interp(xr, [0.0, XREF - A_ZONE, XREF + A_ZONE, L], [0.0, xc - A_ZONE, xc + A_ZONE, L])
    w = _w(x, xc)
    s1 = (1.0 - w) + w * (H - DREF) / (H - d)
    s2 = (1.0 - w) + w * DREF / d
    ys = s1 * (H - d)
    y = np.where(yr <= ys, yr / s1, (H - d) + (yr - ys) / s2)
    return np.stack([x, y], axis=1)


def map_points(pts, src_g, dst_g):
    """Nodes of the destination geometry to the matching points of the source geometry, T_src(T_dst^-1(p))."""
    q = from_ref(to_ref(pts, *dst_g), *src_g)
    q[:, 0] = np.clip(q[:, 0], 0.0, L)
    q[:, 1] = np.clip(q[:, 1], 0.0, H)
    return q


def in_void(q, g, tol=1e-9):
    """Points strictly inside the notch void of geometry g (U notch: slot of width 2 rho, semicircular root)."""
    xc, d = g
    yr = H - d + RHO
    slot = (np.abs(q[:, 0] - xc) < RHO - tol) & (q[:, 1] > yr)
    root = ((q[:, 0] - xc) ** 2 + (q[:, 1] - yr) ** 2 < (RHO - tol) ** 2) & (q[:, 1] <= yr)
    return slot | root


def interp_mapped(src, src_g, dst, dst_g):
    q = map_points(dst["pts"], src_g, dst_g)
    Phi = np.zeros((dst["n_free"], src["ux"].shape[1]))
    Phi[dst["ixf"]] = LinearNDInterpolator(src["tri"], src["ux"])(q)
    Phi[dst["iyf"]] = LinearNDInterpolator(src["tri"], src["uy"])(q)
    assert np.all(np.isfinite(Phi))
    return Phi, float(np.mean(in_void(q, src_g)))


def rbm_basis(srcs, dst, dst_g, m=8, tol=1e-10, gram="M"):
    """Rayleigh-Ritz in the span of mapped source eigenvectors. gram="M": same steps as phase0b.rb_basis
    (truncation on the mass Gram matrix). gram="K": truncation on the stiffness Gram matrix, then the
    generalised eigenproblem in the kept span (deviation 4)."""
    t0 = time.perf_counter()
    blocks, voids = zip(*[interp_mapped(s, g, dst, dst_g) for g, s in srcs])
    B = np.hstack(blocks)
    if gram == "M":
        G = B.T @ (dst["M"] @ B)
        lam, U = np.linalg.eigh(0.5 * (G + G.T))
        keep = lam > tol * lam.max()
        Q = B @ (U[:, keep] / np.sqrt(lam[keep]))
        Kr = Q.T @ (dst["K"] @ Q)
        th, Y = np.linalg.eigh(0.5 * (Kr + Kr.T))
    else:
        G = B.T @ (dst["K"] @ B)
        lam, U = np.linalg.eigh(0.5 * (G + G.T))
        keep = lam > tol * lam.max()
        Q = B @ U[:, keep]
        Kr = Q.T @ (dst["K"] @ Q); Mr = Q.T @ (dst["M"] @ Q)
        th, Y = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    return Q @ Y[:, :m], dict(n_vectors=int(B.shape[1]), rank=int(keep.sum()), t=time.perf_counter() - t0,
                              void_mean=float(np.mean(voids)), void_max=float(np.max(voids)), tol=tol, gram=gram)


def sources(geoms):
    out = []
    for (xc, d) in geoms:
        si = fc.build(xc=xc, d=d, rho=RHO, h_max=0.2)
        sw, sV = fc.eig(si["K"], si["M"], 8)
        out.append(((xc, d), PB.src_prep(si, sV), np.asarray(sw[:8])))
    return out


# ----------------------------------------------------------------------------- checks

def cmd_checks():
    geo = PB.pilot2_geoms()
    srcs = sources(geo["train"])
    res, ok = dict(B1={}, B2={}, B4={}), True
    for (g, s, w8) in srcs[:4]:
        inst = fc.build(xc=g[0], d=g[1], rho=RHO, h_max=0.2)
        key = "%.4f_%.4f" % g
        p = inst["pts"]
        res["B1"][key] = float(np.max(np.abs(from_ref(to_ref(p, *g), *g) - p)))
        Phi, info = rbm_basis([(g, s)], inst, g)
        res["B2"][key] = float(np.max(np.abs(p6.ritz_errors(Phi, inst, w8, 8))))
        Phi, info = rbm_basis([(gg, ss) for gg, ss, _ in srcs], inst, g)
        res["B4"][key] = dict(max_abs_ritz=float(np.max(np.abs(p6.ritz_errors(Phi, inst, w8, 8)))), **info)
    res["B1_pass"] = bool(max(res["B1"].values()) < 1e-12)
    res["B2_pass"] = bool(max(res["B2"].values()) < 1e-8)
    res["all_pass"] = res["B1_pass"] and res["B2_pass"]
    print(json.dumps(res, indent=1, default=P._js), flush=True)
    jdump(res, "checks0d.json")


def cmd_b4diag():
    """Deviation 4: B4 with other truncation settings, same 4 training geometries, target among the sources."""
    geo = PB.pilot2_geoms()
    srcs = sources(geo["train"])
    allsrc = [(gg, ss) for gg, ss, _ in srcs]
    res = {}
    for (g, s, w8) in srcs[:4]:
        inst = fc.build(xc=g[0], d=g[1], rho=RHO, h_max=0.2)
        r = {}
        for gram, tol in (("M", 1e-10), ("M", 1e-12), ("M", 1e-13), ("M", 1e-14), ("K", 1e-10), ("K", 1e-12)):
            Phi, info = rbm_basis(allsrc, inst, g, tol=tol, gram=gram)
            e = p6.ritz_errors(Phi, inst, w8, 8)
            r["%s_%g" % (gram, tol)] = dict(max_abs_ritz=float(np.max(np.abs(e))), min_ritz=float(np.min(e)), rank=info["rank"])
        res["%.4f_%.4f" % g] = r
    best = {}
    for v in next(iter(res.values())):
        best[v] = float(max(r[v]["max_abs_ritz"] for r in res.values()))
    res["worst_over_geometries"] = best
    print(json.dumps(res, indent=1, default=P._js), flush=True)
    jdump(res, "b4diag.json")


# ----------------------------------------------------------------------------- evaluation

def _summ_ritz(held, b):
    e15 = [float(np.median(g[b]["ritz"][:5])) for g in held]
    w1n = np.array([g["w"][0] * (1 + g[b]["ritz"][0]) for g in held]); w1f = np.array([g["w"][0] for g in held])
    slope, icpt = np.polyfit(w1f, w1n, 1)
    r2 = 1 - np.sum((w1n - (slope * w1f + icpt)) ** 2) / np.sum((w1n - w1n.mean()) ** 2)
    big = [g[b]["drop_reproduced"] for g in held if g["w1_drop_fe"] >= 0.01]
    out = dict(H4=float(np.median(e15)), H5=float(np.median([g[b]["drop_reproduced"] for g in held])),
               median_e1=float(np.median([g[b]["ritz"][0] for g in held])),
               median_max_ritz=float(np.median([max(np.abs(g[b]["ritz"])) for g in held])),
               max_max_ritz=float(max(max(np.abs(g[b]["ritz"])) for g in held)),
               slope=float(slope), r2=float(r2), H5_drop_ge_1pct=float(np.median(big)) if big else None, n_big=len(big))
    if "e2_energy" in held[0][b]:
        out["H6"] = float(np.median([g[b]["e2_energy"]["median"] for g in held]))
        out["median_e1_tip_acc"] = float(np.median([g[b]["e1_tip_acc"]["median"] for g in held]))
    if "rb_info" in held[0][b]:
        for k in ("t", "rank", "void_mean", "void_max"):
            out["median_" + k] = float(np.median([g[b]["rb_info"][k] for g in held]))
        out["max_void_max"] = float(max(g[b]["rb_info"]["void_max"] for g in held))
    return out


def run_set(train, held, w_plain, group):
    srcs = sources(train)
    trx = np.array([PB.norm_xy(*g) for g, _, _ in srcs])
    W = np.array([w8 for _, _, w8 in srcs])
    rbf = [RBFInterpolator(trx, W[:, i], kernel="thin_plate_spline", degree=1, smoothing=0.0) for i in range(8)]
    res = {}
    for (xc, d) in (held[:1] if SMOKE else held):
        t0 = time.time()
        inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.geom_setup(xc, d)
        if SMOKE:
            eval_cases = eval_cases[:2]
        A = p6.step_operator(inst, C, dt)
        refs = [fc.newmark_full(inst["K"], inst["M"], C, inst["P"], c, dt, PB.N) for c in eval_cases]
        dist = np.linalg.norm(trx - PB.norm_xy(xc, d), axis=1)
        order = np.argsort(dist)
        rec = dict(group=group, xc=xc, d=d, w=[float(x) for x in w[:8]], w1_drop_fe=float((w_plain - w[0]) / w_plain),
                   dist_nearest=float(dist.min()))
        for tag, k, tol in (("rbm_k4", 4, 1e-10), ("rbm_all", len(srcs), 1e-10),          # planned
                            ("rbm_k4_t13", 4, 1e-13), ("rbm_all_t13", len(srcs), 1e-13)):  # added after B4 (deviation 4)
            Phi, info = rbm_basis([(srcs[i][0], srcs[i][1]) for i in order[:k]], inst, (xc, d), tol=tol)
            s = P.eval_basis(Phi, inst, w, V, C, dt, A, refs, eval_cases)
            s["drop_reproduced"] = PB._drop(w_plain, w[0], s["ritz"][0])
            s["rb_info"] = info
            rec[tag] = s
        q = PB.norm_xy(xc, d)[None, :]
        for tag, wp in (("freq_rbf", np.array([float(f(q)[0]) for f in rbf])), ("freq_nearest", W[order[0]])):
            e = [float(x) for x in wp / w[:8] - 1]
            rec[tag] = dict(ritz=e, drop_reproduced=PB._drop(w_plain, w[0], e[0]))
        rec["t_eval"] = time.time() - t0
        res["%.4f_%.4f" % (xc, d)] = rec
        print(json.dumps(dict(xc=xc, d=d, **{k: dict(max=max(np.abs(rec[k]["ritz"])), e1=rec[k]["ritz"][0],
              drop=rec[k]["drop_reproduced"], e2=rec[k].get("e2_energy", {}).get("median")) for k in rec
              if isinstance(rec[k], dict) and "ritz" in rec[k]})), flush=True)
    held_l = list(res.values())
    summ = {}
    for b in ("rbm_k4", "rbm_all", "rbm_k4_t13", "rbm_all_t13"):
        summ[b] = _summ_ritz(held_l, b)
    for b in ("freq_rbf", "freq_nearest"):                       # signed errors kept; H4 from absolute errors
        s = _summ_ritz(held_l, b)
        s["H4_abs"] = float(np.median([np.median(np.abs(g[b]["ritz"][:5])) for g in held_l]))
        s["median_abs_e1"] = float(np.median([abs(g[b]["ritz"][0]) for g in held_l]))
        summ[b] = s
    return res, summ


def cmd_eval():
    w_plain = float(p2.plain_cantilever_w1()[0])
    out = dict(w_plain=w_plain, map=dict(XREF=XREF, DREF=DREF, A_ZONE=A_ZONE, B_BLEND=B_BLEND))
    geo2 = PB.pilot2_geoms()
    out["pilot2"], out["summary_pilot2"] = run_set(geo2["train"], geo2["held"], w_plain, "pilot2_held")
    jdump(out, "eval0d.json")
    geo1 = P.geometries()
    out["pilot1"], out["summary_pilot1"] = run_set(geo1["pilot_train"], geo1["pilot_held"], w_plain, "pilot1_held")
    jdump(out, "eval0d.json")
    print(json.dumps(dict(summary_pilot2=out["summary_pilot2"], summary_pilot1=out["summary_pilot1"]), indent=1, default=P._js))


if __name__ == "__main__":
    dict(checks=cmd_checks, b4diag=cmd_b4diag, eval=cmd_eval)[sys.argv[1]]()
