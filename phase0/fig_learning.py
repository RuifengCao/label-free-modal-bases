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
