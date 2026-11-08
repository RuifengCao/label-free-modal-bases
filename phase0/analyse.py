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
