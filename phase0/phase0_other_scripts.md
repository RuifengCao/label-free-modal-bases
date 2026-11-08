# Phase 0: other scripts (bundled for the project)

Bundled copies of the shell and Python scripts used in Phase 0 besides phase0.py. Each section is one file, verbatim.

## queue.sh

```bash
#!/bin/bash
# Phase 0 run queues (PHASE0_PLAN.md, "Compute"); usage: queue.sh CORE {A|B1|B2|E}
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c $1 python -I phase0.py"
run() { local name=$1; shift; local t0=$(date +%s); $PY "$@" > logs/$name.log 2>&1; echo "$(date -u '+%H:%M:%S') $name exit $? $(( $(date +%s) - t0 )) s"; }
case "$2" in
  A)  run fam fam 0 60000; run long long; run evalfam evalfam ;;
  B1) for obj in o4 o2 o3; do for s in 0 1 2; do run train_${obj}_s${s} train $obj $s 20000 hard_angle; done; done ;;
  B2) for s in 0 1 2; do run train_o1_s${s} train o1 $s 20000 hard_angle; done
      run train_o4_s0_hard train o4 0 20000 hard; run gap gap; run sens sens; run cost cost ;;
  E)  run evalg0 evalg0 ;;
esac
echo "$(date -u '+%H:%M:%S') QUEUE $2 DONE"
```

## smoke.sh

```bash
#!/bin/bash
# Smoke test: few iterations, separate folder; usage: smoke.sh CORE PART
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P0_OUT=/home/claude/udc/phase0/smoke P0_SMOKE=1 P0_LOG=10
PY="taskset -c $1 python -I phase0.py"
run() { local name=$1; shift; $PY "$@" > smoke/logs/$name.log 2>&1; echo "$name exit $?"; }
if [ "$2" = "A" ]; then
  run train_o1 train o1 0 20 hard_angle
  run train_o2 train o2 0 20 hard_angle
  run train_o3 train o3 0 20 hard_angle
  run train_o4 train o4 0 20 hard_angle
  run train_o4_hard train o4 0 20 hard
  run long long
  run evalg0 evalg0
else
  run fam fam 0 20
  run evalfam evalfam
  run gap gap
  run sens sens
  run cost cost
fi
```

## analyse.py

```python
"""Tables for RESULTS_PHASE0.md from the Phase 0 outputs (reads phase0/out only)."""
import json, os, sys
import numpy as np

OUT = os.environ.get("P0_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


def load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, nd=2):
    return ("%." + str(nd) + "f%%") % (100 * x)


def sci(x):
    return "%.2e" % x


def first_below(logs, thr, key="ritz"):
    for r in logs:
        if max(r[key]) <= thr:
            return r.get("t", r.get("it"))
    return None


def table_g0(ev):
    print("\n### G0 runs: final Ritz frequency errors (modes 1 to 8) and response errors\n")
    print("| Run | " + " | ".join("m%d" % i for i in range(1, 9)) + " | max | E1 tip acc (median) | E2 (median) | S/T | t train (s) |")
    print("|---" * 14 + "|")
    for k, s in ev.items():
        r = s["ritz"]
        print("| %s | %s | %s | %s | %s | %s | %s |" % (
            k, " | ".join(pct(x) for x in r), pct(max(r)), pct(s["e1_tip_acc"]["median"]), pct(s["e2_energy"]["median"]),
            sci(s["S_over_T"]), ("%.0f" % s["t_train_s"]) if s.get("t_train_s") else "-"))


def table_sin2(ev):
    print("\n### G0 runs: sin^2 of the M-angle between v_i and span(Phi)\n")
    print("| Run | " + " | ".join("v%d" % i for i in range(1, 9)) + " |")
    print("|---" * 9 + "|")
    for k, s in ev.items():
        print("| %s | %s |" % (k, " | ".join(sci(x) for x in s["sin2"])))


def table_speed(ev):
    print("\n### G0 runs: first logged iteration with all eight Ritz errors at or below a threshold\n")
    print("| Run | 5% | 1% | 0.5% | 0.1% |")
    print("|---|---|---|---|---|")
    for k, s in ev.items():
        logs = s.get("logs")
        if not logs or "ritz" not in logs[0] or len(logs[0]["ritz"]) < 8:
            continue
        print("| %s | %s |" % (k, " | ".join(str(first_below(logs, th)) for th in (0.05, 0.01, 0.005, 0.001))))


def table_objectives(ev):
    print("\n### G0 runs: values of the four objectives at each final basis\n")
    print("| Run | O1 | O2 (trace) | O3 | O4 (log-det) |")
    print("|---|---|---|---|---|")
    for k, s in ev.items():
        o = s["objectives"]
        print("| %s | %.10e | %.8f | %.8f | %.8f |" % (k, o["o1"], o["o2"], o["o3"], o["o4"]))


def table_gap(g):
    rows = g["rows"]
    print("\n### Gap and mode-order check (%d geometries; plain cantilever omega_1 = %.6f)\n" % (len(rows), g["w_plain"]))
    print("| Group | n | lambda9/lambda8 min | median | max | lambda4/lambda3 min | axial modes (index: count) | omega_1 drop min | max |")
    print("|---|---|---|---|---|---|---|---|---|")
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    for gname in groups:
        rr = [r for r in rows if r["group"] == gname]
        gap = [r["gap_9_8"] for r in rr]
        g43 = [r["gap_4_3"] for r in rr]
        ax = {}
        for r in rr:
            key = ",".join(str(a) for a in r["axial_modes"])
            ax[key] = ax.get(key, 0) + 1
        drop = [r["w1_drop"] for r in rr]
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |" % (
            gname, len(rr), min(gap), float(np.median(gap)), max(gap), min(g43),
            "; ".join("%s: %d" % (k, v) for k, v in sorted(ax.items())), pct(min(drop)), pct(max(drop))))
    worst = sorted(rows, key=lambda r: r["gap_9_8"])[:5]
    print("\nSmallest lambda9/lambda8: " + "; ".join("%s (%.4f, %.4f): %.4f" % (r["group"], r["xc"], r["d"], r["gap_9_8"]) for r in worst))


def table_sens(sens):
    print("\n### Sensitivity tables (1% frequency error in mode i) on G1 to G3\n")
    for g in sens:
        print("\nGeometry x_c = %.4f, d = %.4f; lambda_8/lambda_1 = %.0f; S/T of V_8 = %.3e\n" % (
            g["xc"], g["d"], g["lam"][7] / g["lam"][0], g["records"][0]["S_over_T"]))
        print("| Mode | trace excess | log-det excess | Delta S/T | mode 1 / this |")
        print("|---|---|---|---|---|")
        for r in g["records"][1:]:
            m1 = r.get("mode1_over_this")
            print("| %d | %.3e | %.4f | %.3e | %s |" % (r["mode"], r["trace_excess"], r["logdet_excess"], r["dS_over_T"],
                                                        ("%.0f" % m1) if m1 is not None and m1 > 0 else ("negative" if m1 is not None else "-")))


def table_fam(ef):
    print("\n### Family pilot (O4, seed 0, 60,000 iterations)\n")
    print("| Group | x_c | d | net max Ritz | net median Ritz m1-5 | omega_1 drop (FE) | drop reproduced | net E1 acc | modal E1 acc | net E2 | modal E2 | NN-interp max Ritz | NN-interp E2 | NN-interp drop |")
    print("|---" * 14 + "|")
    for k, r in ef["geoms"].items():
        net, mod = r["net"], r["modal_m8"]
        nn = r.get("nn_interp")
        print("| %s | %.4f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["group"].replace("pilot_", ""), r["xc"], r["d"], pct(max(net["ritz"])), pct(float(np.median(net["ritz"][:5]))),
            pct(r["w1_drop_fe"]), pct(r["w1_drop_reproduced"], 1), pct(net["e1_tip_acc"]["median"]), pct(mod["e1_tip_acc"]["median"]),
            pct(net["e2_energy"]["median"]), pct(mod["e2_energy"]["median"]),
            pct(max(nn["ritz"])) if nn else "-", pct(nn["e2_energy"]["median"]) if nn else "-",
            pct(r["w1_drop_reproduced_nn_interp"], 1) if nn else "-"))


def table_cost(c):
    print("\n### Cost (one sandbox core, median of repeats, seconds)\n")
    keys = ["n_free", "t_mesh_assembly", "t_eig16", "t_inputs", "t_net_compile", "t_net_eval", "t_reduced_operators",
            "t_reduced_step", "t_full_factorisation", "t_full_step"]
    print("| h_max | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for r in c:
        print("| %.2f | %s |" % (r["h_max"], " | ".join(("%d" % r[k]) if k == "n_free" else ("%.3g" % r[k]) for k in keys)))


if __name__ == "__main__":
    what = sys.argv[1:] or ["g0", "gap", "sens", "fam", "cost"]
    ev = load("evalg0.json")
    if "g0" in what and ev:
        table_g0(ev); table_sin2(ev); table_speed(ev); table_objectives(ev)
    if "gap" in what and load("gap.json"):
        table_gap(load("gap.json"))
    if "sens" in what and load("sens.json"):
        table_sens(load("sens.json"))
    if "fam" in what and load("evalfam.json"):
        table_fam(load("evalfam.json"))
    if "cost" in what and load("cost.json"):
        table_cost(load("cost.json"))
```

## fig_learning.py

```python
"""Figure: Ritz frequency error of modes 1 to 8 against iteration, one panel per objective (G0, seed 0)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("P0_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
SEED = int(os.environ.get("P0_FIG_SEED", "0"))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # validated, light
SURFACE, TEXT1, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TITLES = dict(o1="O1  per-step incremental potential", o2="O2  trace (sum of Ritz values)",
              o3="O3  sequential Rayleigh quotients", o4="O4  log-det (sum of log Ritz values)")


def main(fn_out):
    ev = json.load(open(os.path.join(OUT, "evalg0.json")))
    runs = {}
    for k, s in ev.items():
        for obj in ("o1", "o2", "o3", "o4"):
            if k.startswith("g0_%s_s%d_hard_angle_" % (obj, SEED)):
                runs[obj] = s["logs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, obj in zip(axes.ravel(), ("o1", "o2", "o3", "o4")):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.axhline(5e-3, color=TEXT2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.text(300, 5e-3 * 0.42, "0.5%", color=TEXT2, fontsize=8,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
        ax.set_title(TITLES[obj], loc="left", color=TEXT1, fontsize=10)
        logs = runs.get(obj)
        if not logs:
            continue
        t = np.array([r["t"] for r in logs])
        R = np.maximum(np.array([r["ritz"] for r in logs]), 1e-7)      # (n_logs, 8); floor only for the log scale
        ends = []
        for i in range(8):
            ax.plot(t, R[:, i], color=SERIES[i], linewidth=1.6, solid_capstyle="round", label="mode %d" % (i + 1))
            ends.append([R[-1, i], i])
        ends.sort()                                                   # de-overlap end labels in log space
        ys = [np.log10(e[0]) for e in ends]
        gap = 0.24
        for _ in range(50):                                           # push apart, keep each cluster centred
            moved = False
            for j in range(1, len(ys)):
                if ys[j] - ys[j - 1] < gap - 1e-9:
                    mid = 0.5 * (ys[j] + ys[j - 1])
                    ys[j - 1], ys[j] = mid - gap / 2, mid + gap / 2
                    moved = True
            if not moved:
                break
        for (val, i), y in zip(ends, ys):
            ax.text(t[-1] * 1.015, 10 ** y, str(i + 1), color=TEXT1, fontsize=8, va="center")
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ritz frequency error (relative)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(1e-6, 1e2)
    axes[0, 0].set_xlim(0, 21500)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0),
               labelcolor=TEXT1, handlelength=1.6)
    fig.text(0.01, 0.006, "Notched cantilever G0 (x_c = 2.0, d = 0.3), m = 8, seed %d; same network, initialisation and Adam schedule "
             "(1e-3 cosine to 1e-5).\nErrors of the sorted Ritz frequencies against the finite element frequencies; "
             "logged every 1,000 iterations." % SEED, color=TEXT2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(fn_out, dpi=200, facecolor=SURFACE)
    print("saved", fn_out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "fig_learning_seed%d.png" % SEED))
```

## posthoc_k.py

```python
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
```

## make_compact_p0.py

```python
"""Compact, rounded copy of the Phase 0 outputs for the project (out/phase0_compact.json)."""
import json, os

OUT = os.environ.get("P0_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))


def r4(x):
    if isinstance(x, float):
        return float("%.4g" % x)
    if isinstance(x, list):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    return x


def load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else None


res = dict(note="Rounded to 4 significant digits. Full outputs: phase0/out/*.json in the sandbox and the laptop copy.")
res["checks"] = load("checks.json")
ev = load("evalg0.json") or {}
g0 = {}
for k, s in ev.items():
    d = dict(ritz=s["ritz"], sin2=s["sin2"], S_over_T=s["S_over_T"], objectives=s["objectives"],
             e1_tip_acc=[s["e1_tip_acc"]["median"], s["e1_tip_acc"]["max"]],
             e1_tip_disp=[s["e1_tip_disp"]["median"], s["e1_tip_disp"]["max"]],
             e2=[s["e2_energy"]["median"], s["e2_energy"]["max"]])
    for key in ("t_train_s", "conditioning", "final", "nit", "nfev", "message", "t_lbfgs"):
        if s.get(key) is not None:
            d[key] = s[key]
    if s.get("logs") and len(s["logs"][0].get("ritz", [])) == 8 and ("_s0_" in k or "long" in k):
        d["curve_t"] = [l.get("t", l.get("it")) for l in s["logs"]]
        d["curve_ritz"] = [l["ritz"] for l in s["logs"]]
    g0[k] = d
res["g0_runs"] = g0
gap = load("gap.json")
if gap:
    res["gap"] = dict(w_plain=gap["w_plain"], rows=[{k: r[k] for k in ("group", "xc", "d", "n_free", "gap_9_8", "gap_4_3",
                                                                       "axial_modes", "w1_drop")} for r in gap["rows"]])
res["sens"] = load("sens.json")
ef = load("evalfam.json")
if ef:
    fam = {}
    for k, r in ef["geoms"].items():
        d = {q: r[q] for q in ("group", "xc", "d", "n_free", "w1_drop_fe", "w1_drop_reproduced") if q in r}
        for b in ("net", "modal_m8", "nn_interp"):
            if b in r:
                d[b] = dict(ritz=r[b]["ritz"], sin2=r[b]["sin2"], e1_tip_acc=r[b]["e1_tip_acc"]["median"],
                            e2=r[b]["e2_energy"]["median"], e2_max=r[b]["e2_energy"]["max"])
        if "w1_drop_reproduced_nn_interp" in r:
            d["w1_drop_reproduced_nn_interp"] = r["w1_drop_reproduced_nn_interp"]
            d["nn_interp"]["source"] = r["nn_interp"]["source"]
        fam[k] = d
    res["family_pilot"] = dict(w_plain=ef["w_plain"], geoms=fam)
res["cost"] = load("cost.json")
res["geometries"] = load("geometries.json")
with open(os.path.join(OUT, "phase0_compact.json"), "w") as f:
    json.dump(r4(res), f, separators=(",", ":"))
print(os.path.getsize(os.path.join(OUT, "phase0_compact.json")), "bytes")
```
