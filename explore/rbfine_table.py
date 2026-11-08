"""Table for section 1.3 of RESULTS_EXPLORE.md: the control of DEVIATIONS_EXPLORE.md items 18 to 21.

  python -I -B rbfine_table.py RBFINE_DIR EXPLORE_RESULTS_DIR      # writes RBFINE_DIR/table.md and prints it

Laptop X1a values (x1_acc_shard*.json) for the networks and for the mapped reduced basis with the h_max 0.2 sources;
sandbox values (rbfine_h*_part*.json, third attempt) for the basis with its sources on the evaluation mesh.
At h_max 0.2 the two bases are the same, so that column is taken from the laptop for both.
"""
import os, sys, json
import numpy as np

RB, RES = sys.argv[1], sys.argv[2]
mx = lambda r: max(abs(x) for x in r["ritz"])
e2 = lambda r: r["e2_energy"]["median"]
e1 = lambda r: r["e1_tip_disp"]["median"]
ACC = {}
for s in range(4):
    with open(os.path.join(RES, "x1_acc_shard%d.json" % s)) as f:
        ACC.update(json.load(f)["geoms"])
FG = {}
for h in ("0.1", "0.05"):
    for p in (0, 1):
        with open(os.path.join(RB, "rbfine_h%s_part%d.json" % (h, p))) as f:
            FG.update(json.load(f)["geoms"])
H = ("0.2", "0.1", "0.05")
G = sorted(set(k.rsplit("_h", 1)[0] for k in FG))
assert len(G) == 16 and all(g + "_h" + h in ACC for g in G for h in H)


def fine(g, h):
    return ACC[g + "_h0.2"]["rbm_all"] if h == "0.2" else FG[g + "_h" + h]["rbm_fine"]


def coarse(g, h):
    return ACC[g + "_h" + h]["rbm_all"]


def seeds(g, h, f):          # O4 over its three seeds
    return [f(ACC[g + "_h" + h]["o4_s%d" % s]) for s in range(3)]


rows, data = [], {}
def row(name, vals, fmt):
    rows.append("| %s | %s |" % (name, " | ".join(fmt % v for v in vals)))
    data[name] = [float(v) for v in vals]

for lab, f, fmt in (("Largest frequency error", mx, "%.2e"), ("E2", e2, "%.4f"), ("E1", e1, "%.4f")):
    row("%s, mapped reduced basis, sources on the evaluation mesh" % lab, [np.median([f(fine(g, h)) for g in G]) for h in H], fmt)
    row("%s, mapped reduced basis, sources at h_max 0.2" % lab, [np.median([f(coarse(g, h)) for g in G]) for h in H], fmt)
    row("%s, shared O4 (geometries and seeds)" % lab, [np.median([v for g in G for v in seeds(g, h, f)]) for h in H], fmt)
for lab, f in (("largest frequency error", mx), ("E2", e2)):
    for nm, b in (("sources on the evaluation mesh", fine), ("sources at h_max 0.2", coarse)):
        row("Geometries (of 16) where the %s of the basis with %s is below O4 (seed median)" % (lab, nm),
            [sum(f(b(g, h)) < np.median(seeds(g, h, f)) for g in G) for h in H], "%d")
r = [[mx(fine(g, h)) / mx(fine(g, "0.2")) for g in G] for h in H[1:]]
row("Largest frequency error, sources on the evaluation mesh, relative to h_max 0.2: median", [np.median(x) for x in r], "%.2f")
row("same: smallest", [min(x) for x in r], "%.2f")
row("same: largest", [max(x) for x in r], "%.2f")
q = [[mx(coarse(g, h)) / mx(fine(g, h)) for g in G] for h in H[1:]]
row("Largest frequency error, sources at h_max 0.2 over sources on the evaluation mesh: median", [np.median(x) for x in q], "%.2f")
row("same: smallest", [min(x) for x in q], "%.2f")
row("same: largest", [max(x) for x in q], "%.2f")
X = "4.0451_0.1017"
row("%s (outside the hull), largest frequency error, sources on the evaluation mesh" % X, [mx(fine(X, h)) for h in H], "%.2e")
row("%s, shared O4 (seed median)" % X, [np.median(seeds(X, h, mx)) for h in H], "%.2e")
row("Median degrees of freedom of the 16 held-out geometries", [np.median([ACC[g + "_h" + h]["n_free"] for g in G]) for h in H], "%.0f")

hdr = ["| Quantity (median over the 16 held-out geometries unless stated) | h_max 0.2 | 0.1 | 0.05 |", "|---|---|---|---|"]
# rows with only two values (ratios) get an empty h_max 0.2 cell
out = []
for line in rows:
    n = line.count("|") - 1
    out.append(line if n == 4 else line.replace(" | ", " |  | ", 1))
txt = "\n".join(hdr + out) + "\n"
with open(os.path.join(RB, "table.md"), "w") as f:
    f.write(txt)
with open(os.path.join(RB, "table.json"), "w") as f:
    json.dump(data, f, indent=1)
print(txt)
