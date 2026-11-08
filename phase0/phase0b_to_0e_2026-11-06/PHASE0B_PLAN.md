# Phase 0b (exploratory): family thresholds and controls

Written and hashed before any Phase 0b run, 6 November 2026. Exploratory like Phase 0: it tests no hypothesis. It answers the author's request of 6 November 2026, 16:51 ("评估一下几何族门槛，评估一下最近几何插值的对照，其余按照推荐，继续"): evaluate the thresholds of H4 to H6, evaluate the nearest-geometry interpolation control, and, as recommended, add the O3-with-sorting control to Part II. Code: phase0/phase0b.py (imports phase0.py, which is unchanged). Outputs: phase0/out0b/.

## Why a second pilot

The Phase 0 family pilot had 8 training geometries; three of its four held-out geometries lay outside the training hull, so its medians (0.61%, 86.8%, 14.8%) mix the method with poor coverage. The formal Part II has 24 training and 16 held-out geometries (12 of 16 inside the hull). Pilot 2 repeats the formal Part II design exactly on different geometries, so its held-out medians estimate what the formal runs will show.

## Geometries

Same box and Sobol construction as Phase 0 (d): pilot-2 training, seed 31, first 24 of 32 points; pilot-2 held out, seed 32, 16 points. Not part of the formal study; none of the formal geometries is used.

## (j) Pilot 2, O4

Network and inputs as the Phase 0 pilot ('hard_angle' plus the normalised notch position and depth). O4 on each geometry; each iteration uses 4 of the 24 training geometries, a fresh random permutation of the 24 every six iterations (each geometry once per six iterations), loss = mean over the 4. Seed 0, 60,000 iterations, Adam 1e-3 cosine to 1e-5. Monitoring every 5,000 iterations: Ritz errors of modes 1 to 8 on all 40 geometries.

## (k) Pilot 2, O3 with per-geometry sorting

Same as (j) with the loss O3s: on each geometry the raw columns are put in ascending order of their own Rayleigh quotients (computed with the gradient stopped), then O3 (column-by-column Gram-Schmidt in the M inner product, earlier columns stopped, sum of the Rayleigh quotients) is applied in that order. This is the objective of Chang et al. (2025) and Zhang et al. (2026), written for this network.

## (l) Reduced-basis controls built from training eigenvectors (no training)

For each held-out geometry: the 8 finite element eigenvectors of k training geometries (nearest first, distance in ((x_c - 3)/2, (d - 0.25)/0.15)) are interpolated onto the held-out P2 nodes as in Phase 0 (i) (piecewise linear over the source node cloud with the clamped edge added); Rayleigh-Ritz in their span (M-orthonormalisation by eigen-decomposition of the Gram matrix, directions with eigenvalue below 1e-10 of the largest dropped) gives 8 Ritz vectors as the basis. Variants: k = 1 (the Phase 0 (i) control), k = 4 (32 vectors), all training geometries (64 in the Phase 0 pilot, 192 in pilot 2). This is the classical reduced-basis approach to parametrised eigenproblems with snapshots from other meshes. Like the supervised control, it uses eigen-solves on the training geometries; it uses none on the held-out geometry. Applied to the held-out geometries of both pilots, with the Phase 0 pilot network evaluated alongside for reference.

## Evaluation

Held-out geometries of pilot 2 (and of the Phase 0 pilot for (l)): Ritz errors of modes 1 to 8; sin^2 (M); reproduction of the notch-induced drop of omega_1; E1 and E2 on each geometry's 10 evaluation cases; modal truncation as reference; wall time of each control. Training geometries of pilot 2: Ritz errors and drop reproduction only. Also reported, exploratory only: slope and R^2 of omega_1 (network) against omega_1 (finite element) over the held-out geometries; the H5 measure restricted to held-out geometries whose drop is at least 1%.

## Threshold rule (fixed now, before pilot 2 runs)

Let a4, a5, a6 be pilot 2's (j) held-out values of the H4, H5 and H6 measures (medians over the 16 geometries).

- Keep the plan's thresholds (H4 <= 1%, H5 >= 90%, H6 <= 10%) for each measure that pilot 2 meets with margin: a4 <= 0.67%; a5 >= 93.3%; a6 <= 6.7% (the pilot uses at most two thirds of the allowance).
- Otherwise set that threshold from the pilot with a 1.5 times allowance: H4 <= 1.5 a4 rounded up to 0.1%; H6 <= 1.5 a6 rounded up to 1%; H5 >= 100% - 1.5 (100% - a5) rounded down to 1%.
- H7 (O3s against O4 on mode 1): keep the directional prediction "held-out median e_1 of O3s > that of O4" if pilot 2 shows it; otherwise report O3s without a direction.

The thresholds so obtained go into the pre-registration draft for the author's approval; he can still change them before the stamp.

## Checks before training

- A5: O3s against a loop implementation with an explicit sort: value and gradient within a relative 1e-10, at the seed-0 initialisation and at the F1 parameters; O3s value equal to the trace (relative 1e-10).
- A6: the k = 1 control on the four Phase 0 held-out geometries reproduces the Phase 0 (i) Ritz errors (relative 1e-8).
- A smoke test of every command with a few iterations in a separate folder; only exit codes and output fields are inspected.

## Compute

Two sandbox cores, single-threaded processes pinned with taskset: (j) on one core, (k) on the other (about an hour each); then the evaluation in two shards; (l) on the Phase 0 pilot alongside.
