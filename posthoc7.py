"""Post hoc diagnostics, seventh set (see POSTHOC7.md): spectral weighting of the label-free objectives.

  python -I posthoc7.py w1      # second-order formulas for the trace and log-determinant excess
  python -I posthoc7.py w2      # per-mode sensitivity: 1% frequency error in mode i, i = 1..8
"""
import sys, os, json
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_check as rc
import posthoc6 as p6

OUT = os.path.join(HERE, "out")


def ritz_values(Phi, inst):
    K, M = inst["K"], inst["M"]
    Kr = Phi.T @ (K @ Phi); Mr = Phi.T @ (M @ Phi)
    return np.sort(sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T), eigvals_only=True))


def cmd_w1():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    lam = w ** 2
    V8, W = V[:, :8], V[:, 8:16]
    li, lj = lam[:8], lam[8:16]
    rng = np.random.default_rng(0)
    out = []
    for eps in (1e-2, 1e-3, 1e-4):
        for d in range(5):
            E = eps * rng.standard_normal((8, 8))           # rows: modes 9..16, columns: modes 1..8
            th = ritz_values(V8 + W @ E, inst)
            gap = lj[:, None] - li[None, :]
            pred_tr = float(np.sum(E ** 2 * gap))
            pred_ld = float(np.sum(E ** 2 * gap / li[None, :]))
            ex_tr = float(np.sum(th) - np.sum(li))
            ex_ld = float(np.sum(np.log(th)) - np.sum(np.log(li)))
            rec = dict(eps=eps, draw=d, ratio_trace=ex_tr / pred_tr, ratio_logdet=ex_ld / pred_ld)
            out.append(rec)
            print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, "posthoc7_w1.json"), "w") as f:
        json.dump(out, f, indent=1)


def cmd_w2():
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    lam = w ** 2
    A = p6.step_operator(inst, C, dt)
    luA = spla.splu(A)
    V8 = V[:, :8].copy()
    rng = np.random.default_rng(0)
    c = rng.standard_normal(8); c /= np.linalg.norm(c)
    z = V[:, 8:16] @ c
    lam_z = float(np.sum(c ** 2 * lam[8:16]))
    r = 1.01 ** 2 - 1.0                                       # relative Ritz-value error for +1% frequency
    base = p6.S_split(V8, inst, C, A, luA, train_cases, dt)
    th0 = ritz_values(V8, inst)
    rec0 = dict(mode=0, S_over_T=base["S_over_T"]["base"], T=base["T"],
                ritz_freq_err=[float(x) for x in np.sqrt(th0) / w[:8] - 1])
    print(json.dumps(rec0), flush=True)
    out = [rec0]
    for i in range(8):
        t2 = r * lam[i] / (lam_z - lam[i] - r * lam[i])
        t = np.sqrt(t2)
        Phi = V8.copy()
        Phi[:, i] = (V8[:, i] + t * z) / np.sqrt(1.0 + t2)
        th = ritz_values(Phi, inst)
        s = p6.S_split(Phi, inst, C, A, luA, train_cases, dt)
        rec = dict(mode=i + 1, t=float(t),
                   ritz_freq_err=[float(x) for x in np.sqrt(th) / w[:8] - 1],
                   trace_excess=float(np.sum(th) - np.sum(th0)),
                   logdet_excess=float(np.sum(np.log(th)) - np.sum(np.log(th0))),
                   inv_excess=float(np.sum(-1.0 / th) - np.sum(-1.0 / th0)),
                   S_over_T=s["S_over_T"]["base"],
                   dS_over_T=s["S_over_T"]["base"] - base["S_over_T"]["base"])
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, "posthoc7_w2.json"), "w") as f:
        json.dump(dict(lam=[float(x) for x in lam[:16]], lam_z=lam_z, records=out), f, indent=1)


if __name__ == "__main__":
    {"w1": cmd_w1, "w2": cmd_w2}[sys.argv[1]]()
