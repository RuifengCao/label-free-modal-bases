"""Independent recomputation of every pre-registered verdict from the raw outputs of the formal run.

Written from the text of PREREG.md (sections 3 and 4), without importing or copying phase1.verdicts_from,
then compared with results/verdicts.json. Also checks the reproduced notch drop against its definition.

Usage: python3 recheck_verdicts.py <results_dir>
"""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"
GEOMS, SEEDS1, SEEDS2 = ("G1", "G2", "G3"), (10, 11), (0, 1, 2)


def p1(arch, g, o, s):
    name = "p1%s_%s_%s_s%d.json" % ("sep" if arch == "sep" else "", g, o, s)
    with open(os.path.join(R, name)) as f:
        d = json.load(f)
    assert d["arch"] == arch and d["geom"] == g and d["obj"] == o and d["seed"] == s, name
    return np.array(d["net"]["ritz"], dtype=float)


def label(n_pass, n):
    full, part = (6, (4, 5)) if n == 6 else (3, (2,))
    return "supported" if n_pass == full else ("partly supported" if n_pass in part else "not supported")


out = {}
# ---- Part I: each prediction checked in each of the six geometry-seed runs
runs = [(g, s) for g in GEOMS for s in SEEDS1]
checks = {
    "H1": ("shared", "o4", lambda e: e.max() <= 0.005),
    "H2": ("shared", "o1", lambda e: e[2:].max() > 0.05),
    "H3": ("shared", "o2", lambda e: e[0] > 0.01 and e[0] > e[2:].max()),
    "H3b": ("shared", "o3", lambda e: e[0] > 0.005 and e[0] > e[2:].max()),
    "H9": ("sep", "o2", lambda e: e[0] > 0.01 and e[0] > e[2:].max()),
    "H10": ("sep", "o4", lambda e: e.max() <= 0.005),
}
for h, (arch, o, f) in checks.items():
    passes = [bool(f(p1(arch, g, o, s))) for g, s in runs]
    out[h] = (sum(passes), 6, label(sum(passes), 6))
h8 = []
for g, s in runs:
    e_sep, e_sh = p1("sep", g, "o3", s)[0], p1("shared", g, "o3", s)[0]
    h8.append((g, s, e_sep, e_sh, bool(e_sep < e_sh / 3.0)))
out["H8"] = (sum(x[-1] for x in h8), 6, label(sum(x[-1] for x in h8), 6))

# ---- Part II: the 16 held-out geometries
G = {}
for fn in sorted(glob.glob(os.path.join(R, "e2_shard*.json"))):
    with open(fn) as f:
        sh = json.load(f)
    w_plain = np.atleast_1d(np.array(sh["w_plain"], dtype=float))
    for k, d in sh["geoms"].items():
        assert k not in G
        d["_w_plain1"] = float(w_plain[0])
        G[k] = d
assert len(G) == 16, len(G)


def drop_def(d, e1):
    """Reproduced drop from its definition: (omega_plain - omega_1,net) / (omega_plain - omega_1,FE)."""
    w1 = float(d["w"][0])
    return (d["_w_plain1"] - w1 * (1.0 + e1)) / (d["_w_plain1"] - w1)


max_drop_dev = 0.0
for k, d in G.items():
    for m, v in d.items():
        if isinstance(v, dict) and "ritz" in v and "drop_reproduced" in v:
            max_drop_dev = max(max_drop_dev, abs(drop_def(d, v["ritz"][0]) - v["drop_reproduced"]))

vals = {}
for s in SEEDS2:
    m = "o4_s%d" % s
    h4 = float(np.median([np.median(np.array(d[m]["ritz"][:5])) for d in G.values()]))
    h5 = float(np.median([drop_def(d, d[m]["ritz"][0]) for d in G.values()]))
    h6 = float(np.median([np.median(np.array(d[m]["e2_energy"]["per_case"])) for d in G.values()]))
    h7 = (float(np.median([d["o3s_s%d" % s]["ritz"][0] for d in G.values()])),
          float(np.median([d["o4_s%d" % s]["ritz"][0] for d in G.values()])))
    h7sep = (float(np.median([d["sep_o3s_s%d" % s]["ritz"][0] for d in G.values()])),
             float(np.median([d["sep_o4_s%d" % s]["ritz"][0] for d in G.values()])))
    vals[s] = dict(H4=h4, H5=h5, H6=h6, H7=h7, H7sep=h7sep)
tests = {"H4": lambda v: v <= 0.01, "H5": lambda v: v >= 0.90, "H6": lambda v: v <= 0.10,
         "H7": lambda v: v[0] > v[1], "H7sep": lambda v: v[0] > v[1]}
for h, f in tests.items():
    n_pass = sum(bool(f(vals[s][h])) for s in SEEDS2)
    out[h] = (n_pass, 3, label(n_pass, 3))

# ---- compare with the stamped code's verdicts
with open(os.path.join(R, "verdicts.json")) as f:
    V = json.load(f)
mism = []
for h, (n_pass, n, lab) in out.items():
    part = "part1" if h in ("H1", "H2", "H3", "H3b", "H8", "H9", "H10") else "part2"
    v = V[part][h]
    if (v["n_pass"], v["n"], v["verdict"]) != (n_pass, n, lab):
        mism.append((h, (v["n_pass"], v["n"], v["verdict"]), (n_pass, n, lab)))
for g, s, e_sep, e_sh, _ in h8:
    v = V["part1"]["H8"]["per_run"]["p1sep_%s_o3_s%d" % (g, s)]
    if abs(v["e1_sep"] - e_sep) > 1e-15 or abs(v["e1_shared"] - e_sh) > 1e-15:
        mism.append(("H8 value", g, s))
maxdev = 0.0
for s in SEEDS2:
    for h in ("H4", "H5", "H6"):
        maxdev = max(maxdev, abs(V["part2"][h]["per_seed"]["o4_s%d" % s]["value"] - vals[s][h]))
    for h, a, b in (("H7", "o3s_median_e1", "o4_median_e1"), ("H7sep", "o3s_median_e1", "o4_median_e1")):
        x = V["part2"][h]["per_seed"][str(s)]
        maxdev = max(maxdev, abs(x[a] - vals[s][h][0]), abs(x[b] - vals[s][h][1]))

print("recomputed verdicts:")
for h, (n_pass, n, lab) in out.items():
    print("  %-6s %s (%d of %d)" % (h, lab, n_pass, n))
print("mismatches with verdicts.json:", mism if mism else "none")
print("largest difference in the Part II values against verdicts.json: %.1e" % maxdev)
print("largest difference between stored drop_reproduced and its definition (all methods, 16 geometries): %.1e" % max_drop_dev)
