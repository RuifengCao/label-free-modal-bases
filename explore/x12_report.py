"""Tables and figures 6 to 8 for X1 and X2 (EXPLORE_PLAN.md; DEVIATIONS_EXPLORE.md, items 16 and 17).

  python -I -B x12_report.py EXPLORE_RESULTS_DIR FORMAL_RESULTS_DIR

Writes TABLES_EXPLORE.md next to this file and figs/fig6 to fig8 (PDF and PNG), with figs/figure_data_x12.json.
"""
import os, sys, json
import numpy as np
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x3_figures as XF                              # style, colours and save()
plt = XF.plt
matplotlib = XF.matplotlib
RES, FRES = sys.argv[1], sys.argv[2]
MM = XF.MM


def jload(fn):
    with open(fn) as f:
        return json.load(f)


ACC = {}
for s in range(4):
    ACC.update(jload(os.path.join(RES, "x1_acc_shard%d.json" % s))["geoms"])
FINE = {}
for k in range(4):
    FINE.update(jload(os.path.join(RES, "x1_fine%d.json" % k))["geoms"])
TIM = jload(os.path.join(RES, "x1_timing.json"))
GEOMS = sorted(set(k.rsplit("_h", 1)[0] for k in ACC))
FINE_G = sorted(set(k.rsplit("_h", 1)[0] for k in FINE))
H3 = ("0.2", "0.1", "0.05")
NETS = [p + "%s_s%d" % (o, s) for p, objs in (("", ("o4", "o3s", "sup")), ("sep_", ("o4", "o3s"))) for o in objs for s in range(3)]
KINDS = [("Shared O4", ["o4_s0", "o4_s1", "o4_s2"]), ("Shared O3s", ["o3s_s0", "o3s_s1", "o3s_s2"]),
         ("Supervised", ["sup_s0", "sup_s1", "sup_s2"]), ("Per mode O4", ["sep_o4_s0", "sep_o4_s1", "sep_o4_s2"]),
         ("Per mode O3s", ["sep_o3s_s0", "sep_o3s_s1", "sep_o3s_s2"]), ("Modal truncation", ["modal_m8"]),
         ("Mapped reduced basis", ["rbm_all"]), ("Frequency regression", ["freq_rbf"])]
mx = lambda r: max(abs(x) for x in r["ritz"])
L = []                                               # lines of TABLES_EXPLORE.md
out = {}


def fmt(x, p=2):
    return ("%%.%de" % p) % x


# ----------------------------------------------------------------------------- X1a tables
L += ["# Exploratory runs: tables for X1 and X2", "",
      "Written by x12_report.py from the laptop outputs (explore/results, SHA256SUMS_explore_results.txt). "
      "Medians are over the 16 held-out geometries (and, for the networks, the three seeds) unless stated.", "",
      "## X1a.1 Largest frequency error, max over modes 1 to 8 of |e_i|, against mesh size", "",
      "| Basis | h 0.2 | h 0.1 | h 0.05 | ratio 0.05 / 0.2: median (range) |", "|---|---|---|---|---|"]
tab1 = {}
for name, ks in KINDS:
    v = {h: np.array([mx(ACC[g + "_h" + h][k]) for g in GEOMS for k in ks]) for h in H3}
    rat = v["0.05"] / v["0.2"]
    tab1[name] = {h: float(np.median(v[h])) for h in H3}
    tab1[name]["ratio"] = [float(np.median(rat)), float(rat.min()), float(rat.max())]
    L.append("| %s | %s | %s | %s | %.2f (%.2f to %.2f) |" % (name, fmt(np.median(v["0.2"])), fmt(np.median(v["0.1"])),
                                                           fmt(np.median(v["0.05"])), np.median(rat), rat.min(), rat.max()))
dw = np.array([ACC[g + "_h0.05"]["w"][0] / ACC[g + "_h0.2"]["w"][0] - 1 for g in GEOMS])
dw8 = np.array([max(abs(ACC[g + "_h0.05"]["w"][i] / ACC[g + "_h0.2"]["w"][i] - 1) for i in range(8)) for g in GEOMS])
nfree = {h: int(np.median([ACC[g + "_h" + h]["n_free"] for g in GEOMS])) for h in H3}
L += ["", "Degrees of freedom (median): %s. Finite element first frequency, h 0.05 against 0.2: median change %s (range %s to %s); "
      "largest change over modes 1 to 8: median %s, largest %s." % (", ".join("%s at h %s" % ("{:,}".format(nfree[h]), h) for h in H3),
      fmt(np.median(dw)), fmt(dw.min()), fmt(dw.max()), fmt(np.median(dw8)), fmt(dw8.max())), ""]
out["x1a_table1"] = tab1

L += ["## X1a.2 E1 (tip displacement), E2 and the largest sin^2, against mesh size", "",
      "Median over the evaluation cases, then over geometries (and seeds).", "",
      "| Basis | E1 h 0.2 | E1 h 0.1 | E1 h 0.05 | E2 h 0.2 | E2 h 0.1 | E2 h 0.05 | max sin^2 h 0.2 | h 0.05 | drop reproduced h 0.2 | h 0.05 |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
tab2 = {}
for name, ks in KINDS:
    if name == "Frequency regression":
        dr = {h: np.median([ACC[g + "_h" + h][k]["drop_reproduced"] for g in GEOMS for k in ks]) for h in H3}
        L.append("| %s | | | | | | | | | %.3f | %.3f |" % (name, dr["0.2"], dr["0.05"]))
        tab2[name] = dict(drop={h: float(dr[h]) for h in H3})
        continue
    e1 = {h: np.median([ACC[g + "_h" + h][k]["e1_tip_disp"]["median"] for g in GEOMS for k in ks]) for h in H3}
    e2 = {h: np.median([ACC[g + "_h" + h][k]["e2_energy"]["median"] for g in GEOMS for k in ks]) for h in H3}
    s2 = {h: np.median([max(ACC[g + "_h" + h][k]["sin2"]) for g in GEOMS for k in ks]) for h in H3}
    dr = {h: np.median([ACC[g + "_h" + h][k]["drop_reproduced"] for g in GEOMS for k in ks]) for h in H3}
    tab2[name] = dict(e1={h: float(e1[h]) for h in H3}, e2={h: float(e2[h]) for h in H3}, sin2={h: float(s2[h]) for h in H3},
                      drop={h: float(dr[h]) for h in H3})
    L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %.3f | %.3f |" % (
        name, e1["0.2"], e1["0.1"], e1["0.05"], e2["0.2"], e2["0.1"], e2["0.05"], fmt(s2["0.2"]), fmt(s2["0.05"]), dr["0.2"], dr["0.05"]))
L += ["", "Drop reproduced: the share of the finite element drop of the first frequency (notched against plain cantilever on the same mesh) "
      "that the basis reproduces, as in the formal evaluation.", ""]
out["x1a_table2"] = tab2

L += ["## X1a.3 The four FINE geometries down to h_max 0.025", "",
      "| Geometry | DOFs at 0.025 | Basis | max e_i at 0.2 | 0.1 | 0.05 | 0.025 | E2 at 0.2 | 0.025 |", "|---|---|---|---|---|---|---|---|---|"]
tab3 = {}
for g in FINE_G:
    r25 = FINE[g + "_h0.025"]
    for k, nm in (("o4_s0", "Shared O4 s0"), ("sup_s0", "Supervised s0"), ("sep_o4_s0", "Per mode O4 s0"), ("rbm_all", "Mapped reduced basis"),
                  ("freq_rbf", "Frequency regression")):
        vals = [mx(ACC[g + "_h" + h][k]) for h in H3] + [mx(r25[k])]
        e2a = ACC[g + "_h0.2"][k]["e2_energy"]["median"] if k != "freq_rbf" else None
        e2b = r25[k]["e2_energy"]["median"] if k != "freq_rbf" else None
        tab3["%s %s" % (g, k)] = dict(max_e=vals, e2=[e2a, e2b])
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (g, "{:,}".format(r25["n_free"]), nm, *[fmt(v) for v in vals],
                                                               "%.4f" % e2a if e2a is not None else "", "%.4f" % e2b if e2b is not None else ""))
    L.append("| %s | | finite element w_1 change 0.2 to 0.025 | %s | | | | | |" % (g, fmt(r25["w"][0] / ACC[g + "_h0.2"]["w"][0] - 1)))
out["x1a_table3"] = tab3

L += ["", "## X1a.4 Checks at h_max 0.2 against the formal evaluation (same laptop)", "",
      "| Geometry | streaming vs eval_basis | Ritz, absolute | Ritz, relative (basis) | FE frequencies | E1, E2 vs formal |", "|---|---|---|---|---|---|"]
for g in GEOMS:
    r = ACC[g + "_h0.2"]
    kr = max(r["check_ritz_vs_formal_rel"], key=r["check_ritz_vs_formal_rel"].get)
    L.append("| %s | %s | %s | %s (%s) | %s | %s |" % (g, fmt(max(r["check_stream_vs_eval_basis"].values()), 1), fmt(max(r["check_ritz_vs_formal_abs"].values()), 1),
             fmt(r["check_ritz_vs_formal_rel"][kr], 1), kr, fmt(r["check_w_vs_formal"], 1), fmt(max(r["check_metrics_vs_formal"].values()), 1)))

# ----------------------------------------------------------------------------- X1b table
rows = TIM["rows"]
HT = sorted(set(r["h_max"] for r in rows), reverse=True)
comp = [("t_mesh", "mesh"), ("t_build", "mesh and assembly"), ("t_eig8", "8 eigenpairs"), ("t_problem", "network inputs"),
        ("t_net_first", "network, first call"), ("t_net", "network, compiled"), ("t_net_numpy", "network, NumPy"),
        ("t_reduced_ops", "reduced operators"), ("t_rbm", "mapped reduced basis"), ("t_full_factorisation", "factorisation"),
        ("t_full_step", "full step"), ("t_reduced_step", "reduced step"), ("t_full_sim_640", "full simulation, 640 steps"),
        ("t_reduced_sim_640", "reduced simulation, 640 steps")]
med = {h: {c: float(np.median([r[c] for r in rows if r["h_max"] == h])) for c, _ in comp + [("n_free", "")]} for h in HT}
L += ["", "## X1b Online cost on one core (seconds; median over the four FINE geometries of the median of three repeats)", "",
      "| Component | " + " | ".join("h %s (%s DOFs)" % (h, "{:,}".format(int(med[h]["n_free"]))) for h in HT) + " | growth exponent |",
      "|---|" + "---|" * (len(HT) + 1)]
n = np.array([med[h]["n_free"] for h in HT])
slopes = {}
for c, nm in comp:
    y = np.array([med[h][c] for h in HT])
    sl = float(np.polyfit(np.log(n), np.log(y), 1)[0])
    slopes[c] = sl
    L.append("| %s | %s | %.2f |" % (nm, " | ".join("%.4g" % med[h][c] for h in HT), sl))
route = {h: med[h]["t_problem"] + med[h]["t_net"] + med[h]["t_reduced_ops"] for h in HT}
L += ["", "Network route = network inputs + compiled network + reduced operators: " +
      ", ".join("%.3g s at h %s" % (route[h], h) for h in HT) + ". Growth exponent: slope of log time against log DOFs over the four meshes. "
      "Offline preparation of the 24 sources of the mapped reduced basis: %.2f s. Largest differences of the timed bases from lb.orthonormal_basis: "
      "compiled with mesh arguments %s, NumPy %s." % (TIM["t_sources_offline"], fmt(max(r["net_args_vs_lb"] for r in rows), 1),
                                                     fmt(max(r["net_numpy_vs_lb"] for r in rows), 1)), ""]
out["x1b"] = dict(median=med, slopes=slopes, network_route=route)

# ----------------------------------------------------------------------------- X2 table
L += ["## X2 Shared O2 and O3, 100,000 iterations, next to the formal 20,000-iteration runs", "",
      "| Objective | Geometry | Seed | formal: e_1 | formal: max e_3..e_8 | 100k run: e_1 at 20k | 50k | 100k | max e_3..e_8 at 100k | E2 formal | E2 100k | train s, formal | 100k |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
x2 = {}
for o in ("o2", "o3"):
    for g in ("G1", "G2", "G3"):
        for s in (10, 11):
            x = jload(os.path.join(RES, "x2_%s_%s_s%d.json" % (g, o, s)))
            f = jload(os.path.join(FRES, "p1_%s_%s_s%d.json" % (g, o, s)))
            Lg = {l["t"]: l for l in x["logs"]}
            rec = dict(formal_e1=f["net"]["ritz"][0], formal_hi=max(abs(v) for v in f["net"]["ritz"][2:]),
                       e1=[[l["t"], l["ritz"][0]] for l in x["logs"]], hi=[[l["t"], max(abs(v) for v in l["ritz"][2:])] for l in x["logs"]],
                       final_e1=x["net"]["ritz"][0], E2=x["net"]["e2_energy"]["median"], E2_formal=f["net"]["e2_energy"]["median"],
                       t_train=x["t_train"], t_formal=f["t_train"], formal_logs=[[l["t"], l["ritz"][0]] for l in f["logs"]])
            x2["%s_%s_s%d" % (o, g, s)] = rec
            L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.0f | %.0f |" % (
                o.upper(), g, s, fmt(rec["formal_e1"]), fmt(rec["formal_hi"]), fmt(Lg[20000]["ritz"][0]), fmt(Lg[50000]["ritz"][0]),
                fmt(Lg[100000]["ritz"][0]), fmt(max(abs(v) for v in Lg[100000]["ritz"][2:])), rec["E2_formal"], rec["E2"], rec["t_formal"], rec["t_train"]))
for o in ("o2", "o3"):
    rr = [v for k, v in x2.items() if k.startswith(o)]
    L.append("")
    L.append("%s: median e_1, formal %s, 100k run %s; below 0.5%%: %d of 6, below 1%%: %d of 6; median E2 %.3f (formal %.3f)." % (
        o.upper(), fmt(np.median([r["formal_e1"] for r in rr])), fmt(np.median([r["final_e1"] for r in rr])),
        sum(r["final_e1"] < 0.005 for r in rr), sum(r["final_e1"] < 0.01 for r in rr), np.median([r["E2"] for r in rr]),
        np.median([r["E2_formal"] for r in rr])))
o4 = [jload(os.path.join(FRES, "p1_%s_o4_s%d.json" % (g, s)))["net"] for g in ("G1", "G2", "G3") for s in (10, 11)]
o1 = [jload(os.path.join(FRES, "p1_%s_o1_s%d.json" % (g, s)))["net"] for g in ("G1", "G2", "G3") for s in (10, 11)]
ref = dict(o4_e1=float(np.median([n_["ritz"][0] for n_ in o4])), o4_E2=float(np.median([n_["e2_energy"]["median"] for n_ in o4])),
           o1_e1=float(np.median([n_["ritz"][0] for n_ in o1])), o1_E2=float(np.median([n_["e2_energy"]["median"] for n_ in o1])))
L += ["", "For reference, the formal 20,000-iteration runs of the same geometries and seeds: O4 median e_1 %s (E2 %.3f), O1 %s (E2 %.3f)." % (
      fmt(ref["o4_e1"]), ref["o4_E2"], fmt(ref["o1_e1"]), ref["o1_E2"]), ""]
out["x2"] = x2
out["x2_ref"] = ref
with open(os.path.join(HERE, "TABLES_EXPLORE.md"), "w") as f:
    f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- figure 6: online cost
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 86 * MM), gridspec_kw=dict(wspace=0.55))
    nn = np.array([med[h]["n_free"] for h in HT])
    ax = axs[0]
    ser = [("Mesh and assembly", XF.MUTED, "s", [med[h]["t_build"] for h in HT], "-"),
           ("8 eigenpairs (shift-invert)", XF.C_EIG, "D", [med[h]["t_eig8"] for h in HT], "-"),
           ("Mapped reduced basis", XF.C_RB, "^", [med[h]["t_rbm"] for h in HT], "-"),
           ("Network O4: inputs, compiled network, reduced operators", XF.C_O4, "o", [route[h] for h in HT], "-"),
           ("Network O4: first call, compilation included", XF.C_O4, "v", [med[h]["t_net_first"] for h in HT], "--")]
    for nm, col, mk, y, ls in ser:
        ax.plot(nn, y, ls, color=col, lw=1.5, marker=mk, ms=4.5, mec=col if ls == "--" else "white", mew=0.9 if ls == "--" else 0.7,
                mfc="white" if ls == "--" else col, label=nm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Degrees of freedom"); ax.set_ylabel("Time on one core (s)")
    ax.set_title("A reduced model for a new geometry", loc="left")
    ax.set_xlim(nn[0] / 1.5, nn[-1] * 1.5)
    ax = axs[1]
    for nm, col, y, lw in (("Full model: factorisation\nand 640 Newmark steps", XF.MUTED, [med[h]["t_full_sim_640"] for h in HT], 2.6),
                           ("Reduced model, 8 coordinates:\n640 steps", XF.INK2, [med[h]["t_reduced_sim_640"] for h in HT], 1.5)):
        ax.plot(nn, y, "-", color=col, lw=lw, marker="o", ms=4.5, mec="white", mew=0.7)
        ax.annotate(nm, (nn[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5, color=XF.INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Degrees of freedom")
    ax.set_title("One simulation", loc="left")
    ax.set_xlim(nn[0] / 1.5, nn[-1] * 1.5)
    fig.subplots_adjust(bottom=0.32)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03), handlelength=2.4, columnspacing=1.6, title="Left panel",
               title_fontsize=7)
    fig.text(0.5, -0.07, "Laptop, one core, single-threaded libraries; median over the four FINE geometries of the median of three repeats; "
             "h_max 0.2, 0.1, 0.05 and 0.025.", ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig6_online_cost")


# ----------------------------------------------------------------------------- figure 7: accuracy against mesh size
def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), sharey=True, gridspec_kw=dict(wspace=0.08))
    series = [("Supervised network", XF.C_SUP, "s", "-", ["sup_s0", "sup_s1", "sup_s2"]),
              ("Shared network, O4", XF.C_O4, "o", "-", ["o4_s0", "o4_s1", "o4_s2"]),
              ("One network per mode, O4", XF.C_O4, "o", "--", ["sep_o4_s0", "sep_o4_s1", "sep_o4_s2"]),
              ("Mapped reduced basis (sources at h_max 0.2)", XF.C_RB, "^", "-", ["rbm_all"]),
              ("Frequency regression", XF.C_FREQ, "D", "-", ["freq_rbf"])]
    data = {}
    for nm, col, mk, ls, ks in series:
        y = [np.median([mx(ACC[g + "_h" + h][k]) for g in GEOMS for k in ks]) for h in H3]
        x = [nfree[h] for h in H3]
        kw = dict(color=col, marker=mk, ms=4.5, mec=col if ls == "--" else "white", mew=0.9 if ls == "--" else 0.7,
                  mfc="white" if ls == "--" else col)
        axs[0].plot(x, y, ls, label=nm, **kw)
        H4 = H3 + ("0.025",)
        yf = [np.median([mx((FINE if h == "0.025" else ACC)[g + "_h" + h][k]) for g in FINE_G for k in ks]) for h in H4]
        xf = [np.median([(FINE if h == "0.025" else ACC)[g + "_h" + h]["n_free"] for g in FINE_G]) for h in H4]
        axs[1].plot(xf, yf, ls, **kw)
        data[nm + " xf"] = [float(v) for v in xf]
        data[nm] = dict(held_out=[float(v) for v in y], fine=[float(v) for v in yf])
    # item 22: the mapped reduced basis with sources on the evaluation mesh (rbfine, sandbox), h_max 0.2 (same basis), 0.1, 0.05
    RBF = os.path.join(HERE, "rbfine")
    if os.path.exists(os.path.join(RBF, "rbfine_h0.05_part1.json")):
        FG = {}
        for hh in ("0.1", "0.05"):
            for p in (0, 1):
                FG.update(jload(os.path.join(RBF, "rbfine_h%s_part%d.json" % (hh, p)))["geoms"])
        kw = dict(color=XF.C_RB, marker="^", ms=4.5, mec=XF.C_RB, mew=0.9, mfc="white")
        val = lambda g, h: mx(ACC[g + "_h0.2"]["rbm_all"]) if h == "0.2" else mx(FG[g + "_h" + h]["rbm_fine"])
        y0 = [np.median([val(g, h) for g in GEOMS]) for h in H3]
        axs[0].plot([nfree[h] for h in H3], y0, "--", label="Mapped reduced basis (sources on the evaluation mesh)", **kw)
        y1 = [np.median([val(g, h) for g in FINE_G]) for h in H3]
        x1 = [np.median([ACC[g + "_h" + h]["n_free"] for g in FINE_G]) for h in H3]
        axs[1].plot(x1, y1, "--", **kw)
        data["Mapped reduced basis (sources on the evaluation mesh)"] = dict(held_out=[float(v) for v in y0], fine=[float(v) for v in y1])
    xt0 = [nfree[h] for h in H3]
    xt1 = data["Shared network, O4 xf"]
    for ax, t, xt, hs in zip(axs, ("16 held-out geometries", "4 FINE geometries"), (xt0, xt1), (H3, H3 + ("0.025",))):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("h_max (degrees of freedom, median)")
        ax.set_title(t, loc="left")
        ax.set_xticks(xt); ax.set_xticklabels(["%s\n(%s)" % (h, "{:,}".format(int(round(v)))) for h, v in zip(hs, xt)])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(xt[0] / 1.4, xt[-1] * 1.4)
    axs[0].set_ylabel("Largest frequency error,\nmedian over geometries")
    fig.subplots_adjust(bottom=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), handlelength=2.4, columnspacing=1.8)
    fig.text(0.5, -0.12, "All networks trained at h_max 0.2; every method evaluated against the finite element solution on each mesh.\n"
             "Networks: median over geometries and three seeds. Solid green: sources of the mapped reduced basis computed at h_max 0.2.\n"
             "Dashed green: sources computed on the evaluation mesh (sandbox control, not run at 0.025); at 0.2 the two are the same basis.\n"
             "On the right the shared O4 and the supervised network nearly coincide.",
             ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig7_accuracy_mesh_refinement")
    return data


# ----------------------------------------------------------------------------- figure 8: longer training
def fig8():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM), sharey=True, gridspec_kw=dict(wspace=0.16))
    for ax, o, col, nm in ((axs[0], "o2", XF.C_O2, "O2 trace"), (axs[1], "o3", XF.C_O3, "O3 sequential Rayleigh quotients")):
        runs = [v for k, v in x2.items() if k.startswith(o)]
        t = np.array([a for a, _ in runs[0]["e1"]])
        E = np.array([[b for _, b in r["e1"]] for r in runs])
        ax.fill_between(t, E.min(0), E.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(t, np.median(E, 0), "-", color=col, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.5, label="100,000 iterations (X2)")
        tf = np.array([a for a, _ in runs[0]["formal_logs"]])
        Ef = np.array([[b for _, b in r["formal_logs"]] for r in runs])
        ax.plot(tf, np.median(Ef, 0), "--", color=col, lw=1.2, label="20,000 iterations (formal)")
        ax.axhline(ref["o4_e1"], color=XF.C_O4, lw=1.0, ls=":", label="O4 after 20,000 iterations (formal)")
        ax.set_yscale("log"); ax.set_xlim(0, 100000)
        ax.set_xticks([0, 25000, 50000, 75000, 100000]); ax.set_xticklabels(["0", "25,000", "50,000", "75,000", "100,000"])
        ax.set_xlabel("Adam iteration"); ax.set_title("Shared network, " + nm, loc="left")
    axs[0].set_ylabel("Error of the first frequency, $e_1$")
    fig.subplots_adjust(bottom=0.28)
    h, l = axs[0].get_legend_handles_labels()
    hd = [plt.Line2D([], [], color=XF.MUTED, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.5),
          plt.Line2D([], [], color=XF.MUTED, lw=1.2, ls="--"), h[2]]
    fig.legend(hd, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    fig.text(0.5, -0.07, "Median over the six geometry-seed runs (G1 to G3, seeds 10 and 11); band: smallest to largest of the 100,000-iteration runs. "
             "The learning rate falls by a cosine\nfrom 1e-3 to 1e-5 over each run's length, so the 100,000-iteration run is not comparable "
             "with the formal run at 20,000 iterations.", ha="center", va="top", color=XF.INK2, fontsize=7)
    XF.save(fig, "fig8_longer_training_O2_O3")


fig6()
out["fig7"] = fig7()
fig8()
with open(os.path.join(XF.FIGS, "figure_data_x12.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("tables and figures written")
