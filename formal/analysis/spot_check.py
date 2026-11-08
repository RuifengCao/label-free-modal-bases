"""Spot check of the formal outputs: rebuild six saved networks' bases in the sandbox and recompute their Ritz errors
with an eigen-solve written here (Cholesky M-orthonormalisation, then the reduced stiffness), against finite element
frequencies from an independent shift-invert solve. Compares with the numbers the laptop stored.

Usage: P1_OUT=/tmp/p1_spot python3 spot_check.py <laptop_results_dir>
"""
import json
import os
import pickle
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

sys.path.insert(0, "/home/claude/udc/formal")
import phase1 as F                          # the stamped code, used only to rebuild problems and bases
from phase1 import P, PE, p2, p6, lb, jnp

R = sys.argv[1]


def load(fn):
    with open(os.path.join(R, fn), "rb") as f:
        d = pickle.load(f)
    return [(jnp.asarray(Wt), jnp.asarray(b)) for (Wt, b) in d["params"]], d


def fe_freqs(inst, k=8):
    lam = spla.eigsh(inst["K"], k=k, M=inst["M"], sigma=0.0, which="LM", return_eigenvectors=False)
    return np.sqrt(np.sort(lam))


def ritz_here(Phi, inst, w):
    Phi = np.asarray(Phi, dtype=float)
    Mr = Phi.T @ (inst["M"] @ Phi)
    L = np.linalg.cholesky(0.5 * (Mr + Mr.T))
    Q = sla.solve_triangular(L, Phi.T, lower=True).T             # M-orthonormal span of Phi
    Kr = Q.T @ (inst["K"] @ Q)
    r = np.sqrt(np.maximum(np.linalg.eigvalsh(0.5 * (Kr + Kr.T)), 0.0))
    return r[:8] / w[:8] - 1.0


rows = []
geo = F.geometries()
# Part I: two shared networks and one network per mode
for fn, arch in (("p1_G1_o4_s10", "shared"), ("p1_G3_o3_s10", "shared"), ("p1sep_G2_o3_s11", "sep")):
    params, meta = load(fn + ".pkl")
    with open(os.path.join(R, fn + ".json")) as f:
        stored = json.load(f)
    xc, d = geo["G"][meta["geom"]]
    inst, w, V, a_r, b_r, dt, C, tr, ev = P.geom_setup(xc, d)
    pr = p6.problem(inst, a_r, b_r, dt, F.M_MODES, "hard_angle")
    Phi = lb.basis_numpy(params, pr) if arch == "shared" else np.asarray(PE.orth_sepv(params, pr))
    w_here = fe_freqs(inst)
    e = ritz_here(Phi, inst, w_here)
    rows.append((fn, np.max(np.abs(w_here / np.array(stored["w"]) - 1)), np.max(np.abs(e - np.array(stored["net"]["ritz"]))),
                 float(np.max(np.abs(e)))))
# Part II: three networks on the first held-out geometry of shard 0
with open(os.path.join(R, "e2_shard0.json")) as f:
    sh = json.load(f)
key = sorted(sh["geoms"])[0]
g = sh["geoms"][key]
inst, w, V, a_r, b_r, dt, C, tr, ev = P.geom_setup(g["xc"], g["d"])
pr = P.fam_problem(inst, a_r, b_r, dt)
w_here = fe_freqs(inst)
w_plain_here = float(p2.plain_cantilever_w1()[0])
for fn, name, bfun in (("p2_o4_s0", "o4_s0", lb.orthonormal_basis), ("p2_o3s_s1", "o3s_s1", lb.orthonormal_basis),
                       ("p2sep_o3s_s1", "sep_o3s_s1", PE.orth_sepv)):
    params, meta = load(fn + ".pkl")
    e = ritz_here(np.asarray(bfun(params, pr)), inst, w_here)
    rows.append(("%s on held-out %s" % (fn, key), np.max(np.abs(w_here / np.array(g["w"]) - 1)),
                 np.max(np.abs(e - np.array(g[name]["ritz"]))), float(np.max(np.abs(e)))))
print("%-40s %-22s %-24s %s" % ("network", "FE freq. rel. diff.", "Ritz error abs. diff.", "max Ritz error (scale)"))
for r in rows:
    print("%-40s %-22.1e %-24.1e %.3g" % r)
print("plain cantilever omega_1: stored %.12g, here %.12g, rel. diff. %.1e"
      % (sh["w_plain"], w_plain_here, abs(w_plain_here / sh["w_plain"] - 1)))
