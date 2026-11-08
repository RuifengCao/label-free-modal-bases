"""X3 figures (EXPLORE_PLAN.md, X3; DEVIATIONS_EXPLORE.md, items 10 and 11). Reads the formal outputs and the outputs of
x3_analysis.py; writes x3/figs/fig*.pdf and .png.

  python -I -B x3_figures.py [FORMAL_RESULTS_DIR]

Colours: the categorical slots of the reference palette (validated on white, light mode, adjacent and all pairs for
the first three slots); objectives and methods keep the same colour in every figure.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
X3 = os.path.join(HERE, "x3")
FIGS = os.path.join(HERE, "figs")                     # DEVIATIONS_EXPLORE.md, item 17 (first versions: x3/figs)
os.makedirs(FIGS, exist_ok=True)
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "formal", "results")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
# One colour per objective or method in every figure (item 17)
C_O1, C_O2, C_O3, C_O4 = YELLOW, AQUA, ORANGE, BLUE
C_SUP, C_RB, C_FREQ, C_EIG = RED, GREEN, MAGENTA, VIOLET
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fde3d6", "#f9c1a6", "#f39a75", "#eb6834", "#c4501f", "#9a3d17", "#6e2b10"]
OBJ_COL = dict(o1=C_O1, o2=C_O2, o3=C_O3, o4=C_O4)
OBJ_MK = dict(o1="o", o2="s", o3="^", o4="D")
OBJ_NAME = dict(o1="O1 incremental potential", o2="O2 trace", o3="O3 sequential Rayleigh quotients", o4="O4 log-determinant")
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "text.color": INK, "axes.titlecolor": INK, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def jload(fn):
    with open(fn) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- figure 1: Part I learning curves

def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), sharey=True)
    table = {}
    for ax, (arch, objs, title) in zip(axs, (("p1", ("o1", "o2", "o3", "o4"), "Shared network"),
                                             ("p1sep", ("o2", "o3", "o4"), "One network per mode"))):
        for o in objs:
            runs = [jload(os.path.join(RES, "%s_%s_%s_s%d.json" % (arch, g, o, s))) for g in ("G1", "G2", "G3") for s in (10, 11)]
            t = np.array([l["t"] for l in runs[0]["logs"]])
            E = np.array([[l["ritz"][0] for l in r["logs"]] for r in runs])          # (6 runs, logs)
            med = np.median(E, axis=0)
            ax.fill_between(t, E.min(axis=0), E.max(axis=0), color=OBJ_COL[o], alpha=0.12, linewidth=0)
            ax.plot(t, med, color=OBJ_COL[o], label=OBJ_NAME[o])
            ax.plot(t[-1], med[-1], OBJ_MK[o], color=OBJ_COL[o], ms=5, mec="white", mew=0.8, zorder=5, clip_on=False)
            table["%s_%s" % (arch, o)] = dict(t=t.tolist(), median=med.tolist(), min=E.min(axis=0).tolist(), max=E.max(axis=0).tolist())
        ax.set_yscale("log")
        ax.set_xlim(0, 20000)
        ax.set_xticks([0, 5000, 10000, 15000, 20000]); ax.set_xticklabels(["0", "5,000", "10,000", "15,000", "20,000"])
        ax.set_xlabel("Adam iteration")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("Error of the first frequency, $e_1$")
    fig.subplots_adjust(bottom=0.3, wspace=0.12)
    handles = [plt.Line2D([], [], color=OBJ_COL[o], marker=OBJ_MK[o], ms=5, mec="white", mew=0.8) for o in OBJ_COL]
    fig.legend(handles, [OBJ_NAME[o] for o in OBJ_COL], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.06),
               handlelength=2.2, columnspacing=1.6)
    fig.text(0.5, 0.0, "Lines: median over the six geometry-seed runs (G1 to G3, seeds 10 and 11); bands: smallest to largest.",
             ha="center", color=INK2, fontsize=7)
    save(fig, "fig1_partI_learning_curves")
    return table


# ----------------------------------------------------------------------------- figure 2: Part II per held-out geometry

def fig2():
    d = {}
    for k in range(4):
        d.update(jload(os.path.join(RES, "e2_shard%d.json" % k))["geoms"])
    inside = sorted([g for g in d if d[g]["inside_hull"]], key=lambda g: d[g]["dist_nearest"])
    outside = sorted([g for g in d if not d[g]["inside_hull"]], key=lambda g: d[g]["dist_nearest"])
    order = inside + outside
    xs = np.array([i + (1.0 if i >= len(inside) else 0.0) for i in range(len(order))])        # a gap between groups
    mx = lambda r: max(abs(x) for x in r["ritz"])
    series = [("Shared network, O4 (three seeds)", C_O4, "o", lambda r: [mx(r["o4_s%d" % s]) for s in range(3)]),
              ("Supervised network (three seeds)", C_SUP, "s", lambda r: [mx(r["sup_s%d" % s]) for s in range(3)]),
              ("Mapped reduced basis", C_RB, "^", lambda r: [mx(r["rbm_all"])]),
              ("Frequency regression", C_FREQ, "D", lambda r: [mx(r["freq_rbf"])])]
    fig, ax = plt.subplots(figsize=(180 * MM, 70 * MM))
    off = np.array([-0.27, -0.09, 0.09, 0.27])
    table = {g: {} for g in order}
    for j, (name, col, mk, f) in enumerate(series):
        for x, g in zip(xs, order):
            v = np.array(f(d[g]))
            table[g][name] = v.tolist()
            if len(v) > 1:
                ax.plot([x + off[j]] * 2, [v.min(), v.max()], color=col, lw=1.2, solid_capstyle="round")
            ax.plot(x + off[j], np.median(v), mk, color=col, ms=5.6 if mk == "^" else 4.8, mec="white", mew=0.7, zorder=5,
                    label=name if g == order[0] else None)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.2f, %.2f" % tuple(map(float, g.split("_"))) for g in order], rotation=50, ha="right")
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Largest frequency error,\n" + r"$|e_i|$ over modes 1 to 8")
    ax.set_xlabel(r"Held-out geometry (notch position $x_c$, depth $d$)")
    y_top = ax.get_ylim()[1]
    ax.text(np.mean(xs[:len(inside)]), y_top, "Inside the hull of the training geometries", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.text(np.mean(xs[len(inside):]), y_top, "Outside", ha="center", va="bottom", color=INK2, fontsize=7)
    ax.axvline(xs[len(inside)] - 1.0, color=AXIS, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4, handlelength=1.2, columnspacing=1.4)
    fig.text(0.5, -0.36, "Networks: marker at the median of the three seeds, line from the smallest to the largest.\n"
             "Geometries in each group sorted by the distance to the nearest training geometry; $h_{max}$ = 0.2.",
             ha="center", va="top", color=INK2, fontsize=7, transform=fig.transFigure)
    save(fig, "fig2_partII_heldout_frequency_errors")
    return table


# ----------------------------------------------------------------------------- figure 3: tip displacement histories

def fig3():
    f3 = jload(os.path.join(X3, "x3_fig3.json"))
    w1 = jload(os.path.join(X3, "x3_shape.json"))["geoms"][f3["geom"]]["w"][0]
    T1 = 2 * np.pi / w1
    fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 95 * MM), sharex=True, gridspec_kw=dict(height_ratios=(3, 2)))
    titles = dict(tip_pulse="Tip pulse", random_1="First random case")
    table = {}
    for c, name in enumerate(("tip_pulse", "random_1")):
        cr = f3["cases"][name]
        full = np.array(cr["full"]); t = np.arange(len(full)) * f3["dt"] / T1
        sc = np.max(np.abs(full))
        ax = axs[0, c]
        ax.plot(t, full / sc, color=MUTED, lw=3.0, label="Full model", zorder=2)
        ax.plot(t, np.array(cr["o4_s0"]) / sc, color=C_O4, lw=0.9, label="Shared network, O4, seed 0", zorder=4)
        ax.plot(t, np.array(cr["rbm_all"]) / sc, color=C_RB, lw=1.4, label="Mapped reduced basis", zorder=3)
        ax.set_title(titles[name], loc="left")
        ax2 = axs[1, c]
        e_net = (np.array(cr["o4_s0"]) - full) / sc; e_rb = (np.array(cr["rbm_all"]) - full) / sc
        ax2.plot(t, e_net, color=C_O4, lw=1.1)
        ax2.plot(t, e_rb, color=C_RB, lw=1.1)
        ax2.axhline(0, color=AXIS, lw=0.6)
        ax2.set_xlabel(r"Time / first period, $t / T_1$")
        table[name] = dict(scale=float(sc), max_abs_err_net=float(np.max(np.abs(e_net))), max_abs_err_rb=float(np.max(np.abs(e_rb))),
                           e1_net=float(np.linalg.norm(e_net) / np.linalg.norm(full / sc)),
                           e1_rb=float(np.linalg.norm(e_rb) / np.linalg.norm(full / sc)))
    axs[0, 0].set_ylabel("Tip displacement\n(scaled by its maximum)")
    axs[1, 0].set_ylabel("Reduced minus full\n(same scale)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), handlelength=2.2)
    xc, d = map(float, f3["geom"].split("_"))
    fig.text(0.5, -0.07, "Held-out geometry $x_c$ = %.4f, $d$ = %.4f (inside the training hull); vertical displacement at "
             "the middle of the free end; $h_{max}$ = 0.2.\nAt the scale of the upper panels the three curves coincide; "
             "the lower panels show the differences." % (xc, d), ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig3_tip_displacement_history")
    return table


# ----------------------------------------------------------------------------- figure 4: first four modes of G1

def fig4():
    z = np.load(os.path.join(X3, "x3_fig4.npz"))
    p, t, fe, nn = z["p"], z["t"], z["fe"], z["net"]
    tri = Triangulation(p[:, 0], p[:, 1], t)
    cen = p[t].mean(axis=1)
    root = np.array([float(z["xc"]), 1.0 - float(z["d"])])
    near = np.linalg.norm(cen - root, axis=1) < 0.3
    cm_b = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cm_o = LinearSegmentedColormap.from_list("seq_orange", SEQ_ORANGE)
    fig, axs = plt.subplots(4, 3, figsize=(180 * MM, 82 * MM),
                            gridspec_kw=dict(wspace=0.08, hspace=0.38, width_ratios=(10.6, 10.6, 3.0)))
    table = []
    for i in range(4):
        amp = np.max(np.linalg.norm(fe[i], axis=1))
        s = 0.5 / amp
        for c, U in enumerate((fe[i], nn[i])):
            ax = axs[i, c]
            tr = Triangulation(p[:, 0] + s * U[:, 0], p[:, 1] + s * U[:, 1], t)
            ax.tripcolor(tr, np.linalg.norm(U, axis=1) / amp, cmap=cm_b, vmin=0, vmax=1, shading="gouraud", rasterized=True)
        rel = z["dens_diff"][i] / z["dens_fe"][i].max()
        share = float(np.sum(z["dens_diff"][i][near] * 1.0) / np.sum(z["dens_diff"][i]))
        im = axs[i, 2].tripcolor(tri, facecolors=np.clip(rel, 1e-6, None), cmap=cm_o,
                                 norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1e-1), rasterized=True)
        e = float(np.sqrt(z["theta"][i]) / z["w"][i] - 1)
        axs[i, 0].set_title("Mode %d" % (i + 1), loc="left", fontsize=7.5, pad=2)
        axs[i, 1].set_title(r"$e_{%d}$ = %.1e, $\varepsilon_{%d}$ = %.3f" % (i + 1, e, i + 1, z["eps"][i]), loc="left",
                            fontsize=7.5, color=INK2, pad=2)
        axs[i, 2].set_title("%.0f%% inside" % (100 * share), loc="left", fontsize=7.5, color=INK2, pad=2)
        axs[i, 2].add_patch(matplotlib.patches.Circle(root, 0.3, fill=False, ec=INK2, lw=0.6, zorder=5))
        table.append(dict(mode=i + 1, e=e, eps=float(z["eps"][i]), delta=float(z["delta"][i]), share_near_notch=share,
                          max_rel_density=float(rel.max())))
        for c, ax in enumerate(axs[i]):
            ax.set_aspect("equal")
            if c < 2:
                ax.set_xlim(-0.2, 10.4); ax.set_ylim(-0.55, 1.55)
            else:                                              # zoom on the notch
                ax.set_xlim(root[0] - 0.75, root[0] + 0.75); ax.set_ylim(0.0, 1.05)
            ax.axis("off")
    for c, name in enumerate(("Finite element", "Network (Ritz vector)", "Difference, near the notch")):
        axs[0, c].text(0.5, 1.45, name, transform=axs[0, c].transAxes, ha="center", va="bottom", fontsize=8)
    cax = fig.add_axes([0.945, 0.18, 0.01, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("strain energy density of the difference,\nrelative to the largest value of the mode", color=INK2, fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    fig.text(0.5, 0.02, "G1 ($x_c$ = %.4f, $d$ = %.4f); Part I shared network, O4, seed 10; $h_{max}$ = 0.2. Columns 1 and 2: "
             "deformed shape (largest displacement drawn as 0.5),\nshaded by displacement magnitude (0 to the largest). "
             "Column 3: element mean of the strain energy density of\n(Ritz vector minus mode), log scale, "
             "near the notch; circle: radius 0.3 around the notch root, with the share of the difference's energy inside it."
             % (float(z["xc"]), float(z["d"])), ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig4_G1_modes")
    return table


# ----------------------------------------------------------------------------- figure 5: shape errors and the E2 prediction

def fig5():
    d = jload(os.path.join(X3, "x3_shape.json"))
    G = d["geoms"]
    groups = (("Shared network, O4 (planned)", C_O4, "o", d["bases_planned"]),
              ("Supervised network (added)", C_SUP, "s", ["sup_s0", "sup_s1", "sup_s2"]),
              ("Mapped reduced basis (added)", C_RB, "^", ["rbm_all"]))
    fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 118 * MM), gridspec_kw=dict(hspace=0.45, wspace=0.28))
    table = {}
    for c, (name, col, mk, ks) in enumerate(groups):
        sd = np.array([np.sqrt(max(x, 1e-300)) for g in G.values() for k in ks for x in g[k]["delta"]])
        ep = np.array([x for g in G.values() for k in ks for x in g[k]["eps"]])
        ax = axs[0, c]
        lo, hi = 1e-3, 3e-1
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
        ax.plot(sd, ep, mk, color=col, ms=3.2, mec="white", mew=0.4, alpha=0.9, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(r"$\sqrt{\delta_i}$, $\delta_i = \theta_i / \lambda_i - 1$")
        if c == 0:
            ax.set_ylabel(r"Shape error $\varepsilon_i$ (energy norm)")
        ax.set_title(name, loc="left")
        r = ep / sd
        ax.text(0.04, 0.96, "median ratio %.3f\n90%% below %.2f" % (np.median(r), np.quantile(r, 0.9)), transform=ax.transAxes,
                va="top", fontsize=6.5, color=INK2)
        E2 = np.array([x for g in G.values() for k in ks for x in g[k]["E2_formal"]])
        Ep = np.array([x for g in G.values() for k in ks for x in g[k]["E2_pred"]])
        Ep2 = np.array([x for g in G.values() for k in ks for x in g[k]["E2_pred2"]])
        ax = axs[1, c]
        lo2, hi2 = 0.0, 0.16
        ax.plot([lo2, hi2], [lo2, hi2], color=MUTED, lw=0.8, zorder=1)
        ax.plot(E2, Ep, mk, color=col, mfc="white", ms=3.4, mew=0.7, alpha=0.9, zorder=3)
        ax.plot(E2, Ep2, mk, color=col, ms=3.2, mec="white", mew=0.4, alpha=0.9, zorder=4)
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2); ax.set_aspect("equal")
        ax.set_xlabel("E2, formal evaluation")
        if c == 0:
            ax.set_ylabel("E2 predicted mode by mode")
        rr, rr2 = Ep / E2, Ep2 / E2
        ax.set_xticks([0, 0.05, 0.10, 0.15]); ax.set_yticks([0, 0.05, 0.10, 0.15])
        ax.text(0.04, 0.96, "median ratio\nopen %.3f\nfilled %.3f" % (np.median(rr), np.median(rr2)),
                transform=ax.transAxes, va="top", fontsize=6.2, color=INK2, linespacing=1.3)
        table[name] = dict(n_modes=int(len(ep)), eps_over_sqrt_delta=dict(median=float(np.median(r)), q10=float(np.quantile(r, 0.1)),
                           q90=float(np.quantile(r, 0.9)), min=float(r.min()), max=float(r.max())),
                           n_cases=int(len(E2)), E2pred_over_E2=dict(median=float(np.median(rr)), q10=float(np.quantile(rr, 0.1)),
                           q90=float(np.quantile(rr, 0.9)), min=float(rr.min()), max=float(rr.max())),
                           E2pred2_over_E2=dict(median=float(np.median(rr2)), q10=float(np.quantile(rr2, 0.1)),
                           q90=float(np.quantile(rr2, 0.9)), min=float(rr2.min()), max=float(rr2.max())))
    hd = [plt.Line2D([], [], ls="", marker="o", color=MUTED, mfc="white", ms=4, mew=0.8),
          plt.Line2D([], [], ls="", marker="o", color=MUTED, ms=4, mec="white", mew=0.5)]
    fig.subplots_adjust(bottom=0.16)
    fig.legend(hd, ["Open: shape errors only", "Filled: shape, amplitude and phase errors"], loc="upper center",
               bbox_to_anchor=(0.5, 0.075), ncol=2, columnspacing=2.0)
    fig.text(0.5, -0.04, "Top: each point one mode (i = 1 to 8) of one basis on one of the 16 held-out geometries; grey line: equality. "
             "Bottom: each point one evaluation case.\nOpen: E2 from the shape errors alone, "
             r"$\sqrt{\sum_{i \leq 8} p_i \varepsilon_i^2 + \sum_{i > 8} p_i}$, $p_i$ the share of the reference response's "
             r"energy in exact mode $i$." "\nFilled: also the errors of the modal amplitudes over time (amplitude and phase), "
             "from the reduced model's amplitudes on its Ritz vectors. Both ignore coupling between different modes.",
             ha="center", va="top", color=INK2, fontsize=7)
    save(fig, "fig5_shape_errors_and_E2")
    return table


if __name__ == "__main__":
    out = dict(fig1=fig1(), fig2=fig2(), fig3=fig3(), fig4=fig4(), fig5=fig5())
    with open(os.path.join(FIGS, "figure_data.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(dict(fig3=out["fig3"], fig4=out["fig4"], fig5=out["fig5"]), indent=1))
