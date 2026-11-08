"""Notched cantilever: geometry, graded P2 mesh, K, M (consistent), Rayleigh C, load patterns, Newmark reference.

Plane stress, E = 1, nu = 0.3, density 1. Beam [0, 10] x [0, 1], left edge clamped.
U-shaped notch on the top edge: centre x_c, depth d, root radius rho (see PLAN.md).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import triangle as tr
from skfem import MeshTri, Basis, FacetBasis, ElementTriP2, ElementVector, BilinearForm, LinearForm, asm
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity

E, NU, RHO_MAT = 1.0, 0.3, 1.0
MU = E / (2 * (1 + NU))
LAM_PS = E * NU / (1 - NU ** 2)
L, H = 10.0, 1.0
BETA, GAMMA = 0.25, 0.5


def pslg(xc, d, rho, n_arc=24):
    pts = [(0.0, 0.0), (L, 0.0), (L, H), (xc + rho, H), (xc + rho, H - d + rho)]
    for k in range(1, n_arc):
        a = -np.pi * k / n_arc                        # from 0 to -pi through -pi/2 (the root)
        pts.append((xc + rho * np.cos(a), H - d + rho + rho * np.sin(a)))
    pts += [(xc - rho, H - d + rho), (xc - rho, H), (0.0, H)]
    n = len(pts)
    return np.array(pts), np.array([(i, (i + 1) % n) for i in range(n)])


def make_mesh(xc, d, rho, h_max=0.2):
    v, s = pslg(xc, d, rho)
    t = tr.triangulate(dict(vertices=v, segments=s), "pq30a%g" % (np.sqrt(3) / 4 * h_max ** 2))
    root = np.array([xc, H - d])
    h_min = rho / 4.0
    for _ in range(4):
        cen = t["vertices"][t["triangles"]].mean(axis=1)
        r = np.linalg.norm(cen - root, axis=1)
        h = np.minimum(h_max, h_min + (h_max - h_min) * np.clip(r / 1.0, 0.0, 1.0))
        t["triangle_max_area"] = np.sqrt(3) / 4 * h ** 2
        t = tr.triangulate(t, "rpq30a")
    return MeshTri(t["vertices"].T.copy(), t["triangles"].T.copy())


@BilinearForm
def _mass(u, v, w):
    return RHO_MAT * dot(u, v)


def build(xc=2.0, d=0.3, rho=0.05, h_max=0.2):
    m = make_mesh(xc, d, rho, h_max)
    e = ElementVector(ElementTriP2())
    ib = Basis(m, e, intorder=4)
    K = asm(linear_elasticity(LAM_PS, MU), ib).tocsr()
    M = asm(_mass, ib).tocsr()
    n_all = K.shape[0]
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).flatten()
    free = np.setdiff1d(np.arange(n_all), D)
    fmap = -np.ones(n_all, dtype=int); fmap[free] = np.arange(len(free))
    # load patterns, each with unit total force
    right = m.facets_satisfying(lambda x: np.isclose(x[0], L))
    top = m.facets_satisfying(lambda x: np.isclose(x[1], H))
    fr = FacetBasis(m, e, facets=right, intorder=4)
    ft = FacetBasis(m, e, facets=top, intorder=4)
    P_ty = asm(LinearForm(lambda v, w: v[1]), fr)
    P_tx = asm(LinearForm(lambda v, w: v[0]), fr)
    P_top = asm(LinearForm(lambda v, w: v[1]), ft)
    P_ty /= P_ty.sum(); P_tx /= P_tx.sum(); P_top /= P_top.sum()
    P = np.stack([P_ty, P_tx, P_top], axis=1)[free]          # (n_free, 3)
    # node locations (vertices, then edge midpoints) and their x / y dofs in the free numbering
    pts = np.hstack([m.p, m.p[:, m.facets].mean(axis=1)]).T
    ix = np.concatenate([ib.nodal_dofs[0], ib.facet_dofs[0]])
    iy = np.concatenate([ib.nodal_dofs[1], ib.facet_dofs[1]])
    assert np.allclose(ib.doflocs[:, ix].T, pts) and np.allclose(ib.doflocs[:, iy].T, pts)
    ixf, iyf = fmap[ix], fmap[iy]
    keep = (ixf >= 0) & (iyf >= 0)
    assert np.all((ixf >= 0) == (iyf >= 0))
    tip = np.argmin(np.hypot(pts[:, 0] - L, pts[:, 1] - 0.5 * H))
    Kf = K[free][:, free].tocsc(); Mf = M[free][:, free].tocsc()
    return dict(mesh=m, K=Kf, M=Mf, P=P, pts=pts[keep], ixf=ixf[keep], iyf=iyf[keep],
                tip_dof=int(iyf[tip]), tip_xy=pts[tip], n_free=len(free), n_elem=m.t.shape[1],
                params=(xc, d, rho))


def eig(K, M, k):
    w2, V = spla.eigsh(K, k=k, M=M, sigma=0.0, which="LM")
    idx = np.argsort(w2)
    w2, V = w2[idx], V[:, idx]
    V = V / np.sqrt(np.einsum("ij,ij->j", V, M @ V))              # M-orthonormal
    return np.sqrt(w2), V


def time_setup(w1, w3, zeta=0.02):
    """Rayleigh coefficients (zeta at w1 and w3), dt = T3 / 20."""
    a_r = 2 * zeta * w1 * w3 / (w1 + w3)
    b_r = 2 * zeta / (w1 + w3)
    dt = (2 * np.pi / w3) / 20.0
    return a_r, b_r, dt


def newmark_consts(dt, beta=BETA, gamma=GAMMA):
    a0 = 1 / (beta * dt ** 2); a1 = gamma / (beta * dt); a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1; a4 = gamma / beta - 1; a5 = dt / 2 * (gamma / beta - 2)
    return a0, a1, a2, a3, a4, a5


def g_half_sine(t, tau):
    return np.where(t <= tau, np.sin(np.pi * np.minimum(t, tau) / tau), 0.0)


def load_history(case, dt, N):
    """case = dict(w=(w_ty, w_tx, w_top), kind='pulse'|'step', tau=...). Returns g(t_n), n = 0..N."""
    t = dt * np.arange(N + 1)
    if case["kind"] == "step":
        return np.ones(N + 1)
    return g_half_sine(t, case["tau"])


def newmark_full(K, M, C, P, case, dt, N):
    """Full-order Newmark (average acceleration) from rest. Returns U (N+1, n), Acc (N+1, n), A factor, F-builder."""
    a0, a1, a2, a3, a4, a5 = newmark_consts(dt)
    A = (K + a0 * M + a1 * C).tocsc()
    lu = spla.splu(A)
    w = np.asarray(case["w"], float)
    g = load_history(case, dt, N)
    f_pat = P @ w
    n = K.shape[0]
    u = np.zeros(n); v = np.zeros(n)
    acc = spla.splu(M.tocsc()).solve(f_pat * g[0])
    U = np.zeros((N + 1, n)); Acc = np.zeros((N + 1, n)); V = np.zeros((N + 1, n))
    U[0], V[0], Acc[0] = u, v, acc
    for k in range(N):
        rhs = f_pat * g[k + 1] + M @ (a0 * u + a2 * v + a3 * acc) + C @ (a1 * u + a4 * v + a5 * acc)
        un = lu.solve(rhs)
        an = a0 * (un - u) - a2 * v - a3 * acc
        vn = v + dt * ((1 - GAMMA) * acc + GAMMA * an)
        u, v, acc = un, vn, an
        U[k + 1], V[k + 1], Acc[k + 1] = u, v, acc
    return U, V, Acc


def effective_load(M, C, P, case, dt, N, u, v, acc, k):
    """Newmark effective load F_k built from the state (u, v, acc) at step k."""
    a0, a1, a2, a3, a4, a5 = newmark_consts(dt)
    g = load_history(case, dt, N)
    return (P @ np.asarray(case["w"], float)) * g[k + 1] + M @ (a0 * u + a2 * v + a3 * acc) + C @ (a1 * u + a4 * v + a5 * acc)
