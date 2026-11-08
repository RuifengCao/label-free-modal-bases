"""Audit test: phase1.verdicts_from on made-up inputs whose verdicts are known in advance."""
import os, sys
os.environ["P1_SMOKE"] = "1"
os.environ.setdefault("P1_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_test"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1 as F


def run(obj, ritz, g, s, arch="shared"):
    return dict(obj=obj, geom=g, seed=s, arch=arch, net=dict(ritz=list(ritz)))


PAIRS = [(g, s) for g in ("G1", "G2", "G3") for s in (10, 11)]


def p1_set(o4, o1, o2, o3):
    runs = {}
    for i, (g, s) in enumerate(PAIRS):
        runs["p1_%s_o4_s%d" % (g, s)] = run("o4", o4[i], g, s)
        runs["p1_%s_o1_s%d" % (g, s)] = run("o1", o1[i], g, s)
        runs["p1_%s_o2_s%d" % (g, s)] = run("o2", o2[i], g, s)
        runs["p1_%s_o3_s%d" % (g, s)] = run("o3", o3[i], g, s)
    return runs


def p1_sep(o2, o3, o4):
    runs = {}
    for i, (g, s) in enumerate(PAIRS):
        runs["p1sep_%s_o2_s%d" % (g, s)] = run("o2", o2[i], g, s, "sep")
        runs["p1sep_%s_o3_s%d" % (g, s)] = run("o3", o3[i], g, s, "sep")
        runs["p1sep_%s_o4_s%d" % (g, s)] = run("o4", o4[i], g, s, "sep")
    return runs


good_o4 = [[0.001] * 8] * 6
bad_o4 = [[0.001] * 8] * 4 + [[0.006] + [0.001] * 7] * 2          # 4 of 6 pass -> partly
o1 = [[0.0005, 1.0] + [3.0] * 6] * 3 + [[0.0005] + [0.04] * 7] * 3   # 3 of 6 pass -> not supported
o2 = [[0.15, 0.003] + [0.001] * 6] * 6                              # all pass
o3 = [[0.04, 0.001] + [0.0005] * 6] * 5 + [[0.004] + [0.0005] * 7]   # e1 = 0.4% fails (> 0.5% required): 5 of 6 -> partly

v = F.verdicts_from(p1_set(good_o4, o1, o2, o3), {}, (0.01, 0.9, 0.1))
assert v["part1"]["H1"]["verdict"] == "supported", v["part1"]["H1"]
assert v["part1"]["H2"]["verdict"] == "not supported" and v["part1"]["H2"]["n_pass"] == 3, v["part1"]["H2"]
assert v["part1"]["H3"]["verdict"] == "supported", v["part1"]["H3"]
assert v["part1"]["H3b"]["verdict"] == "partly supported" and v["part1"]["H3b"]["n_pass"] == 5, v["part1"]["H3b"]
assert v["part1"]["H8"]["verdict"] == "missing" and v["part1"]["H9"]["n"] == 0, v["part1"]["H8"]
v = F.verdicts_from(p1_set(bad_o4, o1, o2, o3), {}, (0.01, 0.9, 0.1))
assert v["part1"]["H1"]["verdict"] == "partly supported" and v["part1"]["H1"]["n_pass"] == 4, v["part1"]["H1"]

# One network per mode. Shared O3 has e1 = 0.04 in runs 1 to 5 and 0.004 in run 6 (o3 above).
sep_o3 = [[e] + [0.0005] * 7 for e in (0.01, 0.01, 0.01, 0.02, 0.001, 0.001)]  # H8 (e1 < shared / 3): 0.02 fails -> 5 of 6
sep_o2 = [[0.2, 0.003] + [0.001] * 6] * 5 + [[0.005] + [0.001] * 7]            # H9: e1 = 0.5% fails (> 1% required) -> 5 of 6
sep_o4 = [[0.004] * 8] * 6                                                    # H10: max 0.4% <= 0.5% -> all
runs = p1_set(good_o4, o1, o2, o3)
runs.update(p1_sep(sep_o2, sep_o3, sep_o4))
v = F.verdicts_from(runs, {}, (0.01, 0.9, 0.1))
p = v["part1"]
assert list(p) == ["H1", "H2", "H3", "H3b", "H8", "H9", "H10"], list(p)
assert p["H1"]["verdict"] == "supported" and p["H1"]["n"] == 6, p["H1"]          # shared runs only
assert p["H3b"]["n"] == 6 and p["H3b"]["n_pass"] == 5, p["H3b"]                 # shared O3 only, per-mode O3 not mixed in
assert p["H8"]["verdict"] == "partly supported" and p["H8"]["n_pass"] == 5 and p["H8"]["n"] == 6, p["H8"]
assert not p["H8"]["per_run"]["p1sep_G2_o3_s11"]["pass_"] and p["H8"]["per_run"]["p1sep_G3_o3_s11"]["pass_"], p["H8"]
assert p["H9"]["verdict"] == "partly supported" and p["H9"]["n_pass"] == 5, p["H9"]
assert p["H10"]["verdict"] == "supported" and p["H10"]["n"] == 6, p["H10"]


def geom(e1_o4, e15_o4, drop_o4, e2_o4, e1_o3s, sep=None):
    g = {}
    for s in (0, 1, 2):
        g["o4_s%d" % s] = dict(ritz=[e1_o4[s]] + [e15_o4[s]] * 7, drop_reproduced=drop_o4[s], e2_energy=dict(median=e2_o4[s]))
        g["o3s_s%d" % s] = dict(ritz=[e1_o3s[s]] + [0.001] * 7, drop_reproduced=0.5, e2_energy=dict(median=0.3))
        if sep is not None:                                       # one network per mode: (e1 of O4, e1 of O3s) per seed
            g["sep_o4_s%d" % s] = dict(ritz=[sep[s][0]] + [0.001] * 7, drop_reproduced=0.9, e2_energy=dict(median=0.05))
            g["sep_o3s_s%d" % s] = dict(ritz=[sep[s][1]] + [0.001] * 7, drop_reproduced=0.9, e2_energy=dict(median=0.05))
    return g


# 16 held-out geometries; seed 2 fails H4 (median of modes 1 to 5 = 0.02 > 0.01), all pass H5 and H6 except seed 1 on H6
held = {}
for i in range(16):
    held["g%d" % i] = geom(e1_o4=[0.002, 0.002, 0.002], e15_o4=[0.003, 0.004, 0.02], drop_o4=[0.95, 0.93, 0.91],
                           e2_o4=[0.05, 0.12, 0.08], e1_o3s=[0.03, 0.001, 0.03])
v = F.verdicts_from({}, held, (0.01, 0.9, 0.1))
p = v["part2"]
assert p["H4"]["verdict"] == "partly supported" and p["H4"]["n_pass"] == 2, p["H4"]
assert p["H5"]["verdict"] == "supported", p["H5"]
assert p["H6"]["verdict"] == "partly supported" and not p["H6"]["per_seed"]["o4_s1"]["pass_"], p["H6"]
assert p["H7"]["verdict"] == "partly supported" and p["H7"]["n_pass"] == 2, p["H7"]
# median of modes 1 to 5 uses ritz[:5]: with e1 = 0.002 and four entries 0.003 the median is 0.003
assert abs(p["H4"]["per_seed"]["o4_s0"]["value"] - 0.003) < 1e-15
assert p["H7sep"]["verdict"] == "missing" and p["H7sep"]["n"] == 0, p["H7sep"]

# One network per mode: O3s worse than O4 for seeds 0 and 2, better for seed 1 -> partly supported; shared H7 unaffected
held = {}
for i in range(16):
    held["g%d" % i] = geom(e1_o4=[0.002, 0.002, 0.002], e15_o4=[0.003, 0.004, 0.02], drop_o4=[0.95, 0.93, 0.91],
                           e2_o4=[0.05, 0.12, 0.08], e1_o3s=[0.03, 0.001, 0.03], sep=[(0.003, 0.01), (0.003, 0.002), (0.004, 0.006)])
p = F.verdicts_from({}, held, (0.01, 0.9, 0.1))["part2"]
assert p["H7sep"]["verdict"] == "partly supported" and p["H7sep"]["n_pass"] == 2 and not p["H7sep"]["per_seed"]["1"]["pass_"], p["H7sep"]
assert p["H7"]["n_pass"] == 2 and p["H4"]["n"] == 3, p["H7"]                     # per-mode entries do not enter H4 to H7
print("all verdict tests passed")
