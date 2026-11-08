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
