"""Post hoc (DEVIATIONS_PHASE0.md, 3): sin^2 of the K-angle between each exact eigenvector and span(Phi)."""
import sys, os, json
import numpy as np
import scipy.linalg as sla
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0 as P

inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = P.rc.setup()
K = inst["K"]
prs = {v: P.p6.problem(inst, a_r, b_r, dt, 8, v) for v in ("hard_angle",)}


def sin2_K(Phi, k=8):
    G = Phi.T @ (K @ Phi)
    Lc = np.linalg.cholesky(0.5 * (G + G.T))
    Q = sla.solve_triangular(Lc, Phi.T, lower=True).T             # K-orthonormal
    Vk = V[:, :k]
    P_ = Q.T @ (K @ Vk)
    return [float(x) for x in 1.0 - np.sum(P_ ** 2, axis=0) / np.einsum("ij,ij->j", Vk, K @ Vk)]


res = {"modal_m8": sin2_K(V[:, :8])}
for fn in ("g0_o1_s0_hard_angle_20000", "g0_o2_s0_hard_angle_20000", "g0_o3_s0_hard_angle_20000",
           "g0_o4_s0_hard_angle_20000", "g0_o4_long_adam_60000", "g0_o4_long_lbfgs"):
    params, d = P.p3.load_params(os.path.join(P.OUT, fn + ".pkl"))
    res[fn] = sin2_K(P.lb.basis_numpy(params, prs["hard_angle"]))
for k, v in res.items():
    print("| %s | %s |" % (k, " | ".join("%.2e" % x for x in v)))
P.jdump(res, "posthoc_k.json")
