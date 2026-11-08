"""Explanation check for the X1a check at h_max = 0.2 (EXPLORE_PLAN.md, X1a, Check): repeat, twice in the same process,
the parts that use ARPACK with a random start vector (the 24 source eigen-solves of the mapped reduced basis and the
frequency regression; the eigen-solve of the modal basis), and compare the Ritz errors of the two repeats as the
check compares the exploratory run with the formal one."""
import os, sys, json
os.environ.update(EXP_SMOKE="0", EXP_NETS="/home/claude/udc/formal_results/results", EXP_OUT="/home/claude/udc/explore/check_h02/_scratch")
sys.path.insert(0, "/home/claude/udc/explore")
import explore as X
import numpy as np
from scipy.interpolate import RBFInterpolator
F, P, PB, PD, fc, p6 = X.F, X.P, X.PB, X.PD, X.fc, X.p6
geo = F.geometries()
out = {}
for gk in ("3.1654_0.2769", "1.5060_0.2220", "3.3505_0.1723"):
    g = [x for x in geo["held"] if X.key(x) == gk][0]
    inst = fc.build(xc=g[0], d=g[1], rho=0.05, h_max=0.2)
    reps = []
    for rep in range(2):
        w, V = fc.eig(inst["K"], inst["M"], 16)                     # random start vector each call
        srcs = PD.sources(geo["train"])                             # 24 eigen-solves, random starts
        B = PD.rbm_basis([(gg, s) for gg, s, _ in srcs], inst, g, m=8, tol=F.RBM_TOL)[0]
        tr_xy = np.array([PB.norm_xy(*x) for x in geo["train"]]); W8 = np.array([w8 for _, _, w8 in srcs])
        q = PB.norm_xy(*g)[None, :]
        fr = [float(RBFInterpolator(tr_xy, W8[:, i], kernel="thin_plate_spline", degree=1, smoothing=0.0)(q)[0]) / w[i] - 1 for i in range(8)]
        reps.append(dict(modal=p6.ritz_errors(V[:, :8], inst, w, 8), rbm=p6.ritz_errors(B, inst, w, 8), freq=fr))
    res = {}
    for k in ("modal", "rbm", "freq"):
        a, b = np.array(reps[0][k]), np.array(reps[1][k])
        big = np.abs(b) > 1e-8
        res[k] = dict(abs=float(np.max(np.abs(a - b))), rel=float(np.max(np.abs(a - b)[big] / np.abs(b[big]))) if big.any() else 0.0)
    out[gk] = res
    print(gk, json.dumps(res), flush=True)
json.dump(out, open("/home/claude/udc/explore/check_h02/repeat_sources.json", "w"), indent=1)
