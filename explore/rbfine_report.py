"""Summary of the X1a control with fine-mesh sources (DEVIATIONS_EXPLORE.md, items 18 to 21).

  python -I -B rbfine_report.py RBFINE_DIR EXPLORE_RESULTS_DIR      # prints the tables and writes rbfine/summary.json
"""
import os, sys, json
import numpy as np

RB, RES = sys.argv[1], sys.argv[2]
mx = lambda r: max(abs(x) for x in r["ritz"])
ACC = {}
for s in range(4):
    with open(os.path.join(RES, "x1_acc_shard%d.json" % s)) as f:
        ACC.update(json.load(f)["geoms"])
out = dict(h={})
for h in ("0.1", "0.05"):
    parts = [json.load(open(os.path.join(RB, "rbfine_h%s_part%d.json" % (h, p)))) for p in (0, 1)]
    G = {}
    for p in parts:
        G.update(p["geoms"])
    geoms = sorted(k.rsplit("_h", 1)[0] for k in G)
    assert len(geoms) == 16, len(geoms)
    lap = {g: ACC[g + "_h" + h] for g in geoms}
    lap02 = {g: ACC[g + "_h0.2"] for g in geoms}
    # cross-check: rbm_all recomputed in the sandbox against the laptop
    chk = []
    for g in geoms:
        r = G[g + "_h" + h]
        if "rbm_all" in r:
            a, b = r["rbm_all"], lap[g]["rbm_all"]
            chk.append(dict(geom=g, ritz_abs=max(abs(x - y) for x, y in zip(a["ritz"], b["ritz"])),
                            e2_rel=max(abs(x / y - 1) for x, y in zip(a["e2_energy"]["per_case"], b["e2_energy"]["per_case"])),
                            w_rel=max(abs(x / y - 1) for x, y in zip(r["w"], lap[g]["w"]))))
    o4 = np.array([np.median([mx(lap[g]["o4_s%d" % s]) for s in range(3)]) for g in geoms])
    o4e2 = np.array([np.median([lap[g]["o4_s%d" % s]["e2_energy"]["median"] for s in range(3)]) for g in geoms])
    fine = np.array([mx(G[g + "_h" + h]["rbm_fine"]) for g in geoms])
    fine_e2 = np.array([G[g + "_h" + h]["rbm_fine"]["e2_energy"]["median"] for g in geoms])
    fine_e1 = np.array([G[g + "_h" + h]["rbm_fine"]["e1_tip_disp"]["median"] for g in geoms])
    coarse = np.array([mx(lap[g]["rbm_all"]) for g in geoms])
    coarse_e2 = np.array([lap[g]["rbm_all"]["e2_energy"]["median"] for g in geoms])
    rb02 = np.array([mx(lap02[g]["rbm_all"]) for g in geoms])
    rb02_e2 = np.array([lap02[g]["rbm_all"]["e2_energy"]["median"] for g in geoms])
    rec = dict(
        n=len(geoms), check=chk,
        max_e=dict(rbm_fine=float(np.median(fine)), rbm_coarse=float(np.median(coarse)), rbm_at_02=float(np.median(rb02)), o4=float(np.median(o4))),
        e2=dict(rbm_fine=float(np.median(fine_e2)), rbm_coarse=float(np.median(coarse_e2)), rbm_at_02=float(np.median(rb02_e2)), o4=float(np.median(o4e2))),
        e1_fine=float(np.median(fine_e1)),
        ratio_fine_over_02=[float(np.median(fine / rb02)), float((fine / rb02).min()), float((fine / rb02).max())],
        ratio_coarse_over_fine=[float(np.median(coarse / fine)), float((coarse / fine).min()), float((coarse / fine).max())],
        fine_below_o4=int(np.sum(fine < o4)), fine_e2_below_o4=int(np.sum(fine_e2 < o4e2)),
        coarse_below_o4=int(np.sum(coarse < o4)), coarse_e2_below_o4=int(np.sum(coarse_e2 < o4e2)),
        sources_time=dict(at_02=[p["timing"]["sources_0.2"] for p in parts], at_h=[p["timing"]["sources_h"] for p in parts]),
        map_time=dict(fine=float(np.median([G[g + "_h" + h]["t_map_rbm_fine"] for g in geoms])),
                      coarse=float(np.median([G[g + "_h" + h]["t_map_rbm_all"] for g in geoms if "t_map_rbm_all" in G[g + "_h" + h]]))),
        rank=sorted(set(G[g + "_h" + h]["rbm_fine"]["rb_info"]["rank"] for g in geoms)),
        nudged=dict(fine=[int(G[g + "_h" + h]["rbm_fine"]["rb_info"]["nonfinite_points"]) for g in geoms],
                    nearest=[int(G[g + "_h" + h]["rbm_fine"]["rb_info"]["nearest_node_points"]) for g in geoms],
                    check=[int(G[g + "_h" + h]["rbm_all"]["rb_info"]["nonfinite_points"]) for g in geoms if "rbm_all" in G[g + "_h" + h]]),
        per_geom={g: dict(fine=float(f), coarse=float(c), at02=float(r0), o4=float(o), fine_e2=float(fe), o4_e2=float(oe))
                  for g, f, c, r0, o, fe, oe in zip(geoms, fine, coarse, rb02, o4, fine_e2, o4e2)})
    out["h"][h] = rec
    print("h %s: largest frequency error, median: fine sources %.2e | sources at 0.2 %.2e | basis at h 0.2 %.2e | O4 %.2e" % (
        h, rec["max_e"]["rbm_fine"], rec["max_e"]["rbm_coarse"], rec["max_e"]["rbm_at_02"], rec["max_e"]["o4"]))
    print("        E2 median: fine %.4f | coarse %.4f | at 0.2 %.4f | O4 %.4f ; E1 fine %.4f" % (
        rec["e2"]["rbm_fine"], rec["e2"]["rbm_coarse"], rec["e2"]["rbm_at_02"], rec["e2"]["o4"], rec["e1_fine"]))
    print("        fine / (basis at 0.2): median %.2f (%.2f to %.2f); coarse / fine: median %.2f (%.2f to %.2f)" % (
        *rec["ratio_fine_over_02"], *rec["ratio_coarse_over_fine"]))
    print("        below O4: fine %d/16 (E2 %d/16), coarse %d/16 (E2 %d/16)" % (
        rec["fine_below_o4"], rec["fine_e2_below_o4"], rec["coarse_below_o4"], rec["coarse_e2_below_o4"]))
    print("        sources (s, sandbox): at 0.2 %s, at %s %s; mapping per geometry: fine %.2f s, coarse %.2f s; rank %s" % (
        ["%.1f" % t for t in rec["sources_time"]["at_02"]], h, ["%.1f" % t for t in rec["sources_time"]["at_h"]],
        rec["map_time"]["fine"], rec["map_time"]["coarse"], rec["rank"]))
    print("        cross-check:", [(c["geom"], "%.1e" % c["ritz_abs"], "%.1e" % c["e2_rel"], "%.1e" % c["w_rel"]) for c in chk])
    print("        nudged points per geometry (fine): %s; nearest-node points: %d; in the cross-check bases: %s" % (
        rec["nudged"]["fine"], sum(rec["nudged"]["nearest"]), rec["nudged"]["check"]))
with open(os.path.join(RB, "summary.json"), "w") as f:
    json.dump(out, f, indent=1)
