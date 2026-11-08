"""Post hoc diagnostics D1 to D3 (see POSTHOC2.md)."""
import sys, os, json, time
import numpy as np
import triangle as tr
from skfem import MeshTri, Basis, ElementTriP2, ElementVector, asm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fem_cantilever as fc
import run_check as rc
import lb_model as lb
import posthoc as ph
import jax
import jax.numpy as jnp
from skfem.models.elasticity import linear_elasticity


def plain_cantilever_w1(h_max=0.2):
    v = np.array([(0, 0), (fc.L, 0), (fc.L, fc.H), (0, fc.H)], float)
    s = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    t = tr.triangulate(dict(vertices=v, segments=s), "pq30a%g" % (np.sqrt(3) / 4 * h_max ** 2))
    m = MeshTri(t["vertices"].T.copy(), t["triangles"].T.copy())
    e = ElementVector(ElementTriP2())
    ib = Basis(m, e, intorder=4)
    K = asm(linear_elasticity(fc.LAM_PS, fc.MU), ib).tocsr()
    M = asm(fc._mass, ib).tocsr()
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten()
    free = np.setdiff1d(np.arange(K.shape[0]), D)
    w, _ = fc.eig(K[free][:, free].tocsc(), M[free][:, free].tocsc(), 3)
    return w


def main():
    out = {}
    inst, w, V, a_r, b_r, dt, C, train_cases, eval_cases = rc.setup()
    w_plain = plain_cantilever_w1()
    out["D1_w1_notched"] = float(w[0]); out["D1_w1_plain"] = float(w_plain[0])
    out["D1_plain_over_notched"] = float(w_plain[0] / w[0])
    print("D1", out, flush=True)
    pr = lb.Problem(inst, a_r, b_r, dt, 1)

    def rq_loss(p):
        Phi = lb.raw_basis(p, pr)
        return (Phi[:, 0] @ (pr.K @ Phi)[:, 0]) / (Phi[:, 0] @ (pr.M @ Phi)[:, 0])

    v1 = jnp.asarray(V[:, 0])
    Kv1 = pr.K @ v1[:, None]
    v1Kv1 = float(v1 @ Kv1[:, 0])

    def sup_loss(p):
        phi = lb.raw_basis(p, pr)[:, 0]
        Kphi = (pr.K @ phi[:, None])[:, 0]
        return 1.0 - (phi @ Kv1[:, 0]) ** 2 / ((phi @ Kphi) * v1Kv1)

    for name, loss in (("D2_rayleigh_label_free", rq_loss), ("D3_supervised_eigvec", sup_loss)):
        t0 = time.time()
        params, hist = ph.train_generic(loss, pr, 1, seed=0, iters=3000, log_every=500)
        Phi = np.asarray(lb.raw_basis(params, pr))[:, 0]
        rq = float(Phi @ (inst["K"] @ Phi) / (Phi @ (inst["M"] @ Phi)))
        out[name] = dict(ritz_freq_rel_err=float(np.sqrt(rq) / w[0] - 1), final_loss=float(hist[-1][1]),
                         t_s=time.time() - t0, hist=hist)
        print(name, out[name]["ritz_freq_rel_err"], flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "posthoc2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
