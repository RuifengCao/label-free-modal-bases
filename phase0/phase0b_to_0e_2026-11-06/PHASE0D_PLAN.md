# Phase 0d (exploratory): classical controls for the geometry family

Written 6 November 2026, 18:30 UTC, before any of the runs below. Exploratory; not part of the formal study; no formal geometry is used.

## Why

Phase 0b (l) transferred the eigenvectors of the k nearest training geometries to a held-out geometry by linear interpolation in physical coordinates, followed by Rayleigh-Ritz on the held-out geometry. Before writing this plan I saw its results on half of the pilot-2 held-out geometries (evalfam2 shard 1) and on the Phase 0 pilot (rb1): with all 24 training geometries (8 held-out geometries of pilot 2) the largest Ritz frequency error per geometry was 4.6% to 15.7%, the energy-norm response error E2 (median over load cases) was 34% to 89%, and the drop of the first frequency caused by the notch was hardly reproduced (drop reproduction 0.01 to 0.50); on the Phase 0 pilot (8 training geometries) the same figures were 1.8% to 9.4%, 16% to 63% and 0.05 to 0.71. The network trained with O4 on pilot 2 gave 0.2% to 1.8%, 4.9% to 16.4% and 0.82 to 0.98 on the same 8 geometries. The reason is plain: an interpolated snapshot carries the kink of its own notch at its own notch position, so no snapshot can represent the compliance at the new notch position.

This is not a fair stand-in for classical reduced-basis practice, where snapshots of a geometrically parametrised problem are compared on a reference domain through a map of the geometry. A reviewer would read the unmapped version as a straw man. Phase 0d therefore tests two classical controls that a reviewer would accept, before the choice of controls for the pre-registration.

## Controls

(m) Mapped reduced basis. Eigenvectors of the training geometries are transported to the held-out geometry through a map that aligns the notches, then Rayleigh-Ritz exactly as in (l) (same function `rb_basis`, same Gram truncation 1e-10). Variants: k = 4 nearest training geometries (normalised parameter distance, as in (l)) and all 24.

Map. Reference notch centre X = 3.0, reference depth D = 0.25, zone half-width a = 0.5, blend width b = 0.5 (fixed now, not tuned). For a geometry g with notch centre xc and depth d, physical point (x, y) to reference point (x', y'):

- x' is piecewise linear in x through (0, 0), (xc - a, X - a), (xc + a, X + a), (L, L); inside the zone it is a translation, so the notch keeps its width.
- w(x) = clip(1 - (|x - xc| - a) / b, 0, 1): 1 inside the zone, 0 beyond a + b.
- y' = (1 - w) y + w phi(y), with phi piecewise linear through (0, 0), (H - d, H - D), (H, H): inside the zone the notch flank [H - d, H] goes to [H - D, H] and the ligament [0, H - d] to [0, H - D].

The inverse is explicit (x from x' first, then w(x), then y from y' on the two linear pieces). A training eigenvector u_A is carried to the held-out geometry B by evaluating its interpolant (the same Delaunay interpolant as in (l), clamped edge included) at T_A(T_B^-1(p)) for each node p of B; displacement components are carried as scalar fields. Mapped coordinates are clipped to [0, L] x [0, H] against round-off. The root of the notch (radius 0.05) is circular in both geometries but the depth scaling turns it into an ellipse, so a few nodes of B near the root land in the void of A's notch; check B3 reports how many.

(n) Frequency regression. Each of the first eight finite element frequencies of the 24 training geometries is interpolated over the normalised parameters (`norm_xy`) with scipy `RBFInterpolator` (kernel thin_plate_spline, degree 1, smoothing 0). Also the trivial version: the frequencies of the nearest training geometry. Neither gives mode shapes, so only the frequency measures are available (no E1, E2, angles).

## Where

Pilot 2 (24 training, 16 held-out, as in Phase 0b) and the Phase 0 pilot (8 training, 4 held-out). Same finite element references and the same evaluation function `eval_basis` as Phase 0b, so (m) is directly comparable with net_o4, net_o3s and (l).

## Checks (before the evaluation)

- B1: round trip T_g(T_g^-1(p)) = p on all nodes of 4 pilot-2 training geometries, max error below 1e-12.
- B2: mapped reduced basis with the target itself as the only source (map is then the identity) reproduces its first eight frequencies, max relative Ritz error below 1e-8, on the same 4 geometries.
- B3 (reported, no threshold): fraction of the nodes of B that land outside the material of A (inside A's notch void), median and maximum over all pairs used.
- B4 (reported, no threshold): mapped reduced basis with all 24 sources on 4 training geometries (target included among the sources): max relative Ritz error.

## Measures

For (m): as Phase 0b (H4 measure, H5 measure, H6 measure, median e1, median and maximum of the largest Ritz error, drop reproduction for drops of at least 1%, slope and R squared of predicted against finite element first frequency, time). For (n): H4 measure, median e1, median and maximum of the largest frequency error, H5 measure, slope and R squared.

## What the results are for

They are descriptive and decide nothing automatically. Proposed to Ruifeng before the results are seen: the formal Part II includes the mapped reduced basis with all training geometries and the frequency regression as labelled classical controls whatever this pilot shows, and reports them next to the networks; the unmapped version (l) is reported only as an illustration of why a map is needed, or dropped. If a check fails, the failure is reported here and the control is fixed or dropped with a recorded deviation.

## Compute

One core, after the Phase 0b merge. Expected under 15 minutes.
