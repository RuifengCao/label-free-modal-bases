# Pre-registration (draft for approval): which label-free objective learns the vibration modes?

> **Superseded on 6 November 2026, 21:58 UTC**, by `claude/scripts/unified_formal_2026-11-06/PREREG.md` (version 1, for the author's approval): the thresholds of H4 to H6 are kept by the pilot-2 rule (PHASE0B_PLAN.md); the reduced-basis controls are replaced by a mapped reduced basis and a frequency regression (Phase 0d); O3s is added with the shared network and with one network per mode, and Part I gains runs with one network per mode (Phases 0c and 0e). Kept below as it was.

**Status**: draft, 6 November 2026. Written after Phase 0 (exploratory; PHASE0_PLAN.md, RESULTS_PHASE0.md) and before any formal run. On the author's approval it is hashed into PLAN_HASH.txt together with phase1.py and geometries.json; no Part I or Part II run starts before that hash. Points marked [DECISION] need the author's choice; the recommended option is written in.

## 1. Question

For network-generated finite element reduced bases in linear structural dynamics, which label-free training objective learns the eight lowest vibration modes at a fixed training budget, and does the outcome follow the spectral weighting of the objectives (per-step incremental potential: weight by the mode's share of the trajectory; trace: by its eigenvalue; log-det: equal in relative terms)?

## 2. Fixed settings

Identical to Phase 0 (PHASE0_PLAN.md, "Common settings") unless stated here.

- Model problem: plane-stress cantilever 10 x 1 with a U-shaped top notch (root radius 0.05), P2 triangles, h_max = 0.2 with refinement at the notch; Newmark average acceleration, Rayleigh damping 2% at modes 1 and 3, dt = T_3/20, 640 steps; 16 training and 10 evaluation load cases per geometry (seeds 2026 and 2027, pulse widths scaled by the geometry's own periods). m = 8.
- Network: MLP 4 x 64 (tanh); inputs: 20 planned features and 3 notch-angle features (Part II adds the normalised notch position and depth); outputs multiplied by x/L; Cholesky M-orthonormalisation.
- Optimiser: Adam (0.9, 0.999, 1e-8), learning rate cosine from 1e-3 to 1e-5; initialisation PRNGKey(seed), identical across objectives for a given seed.
- Objectives: O1 per-step incremental potential (history stopped); O2 trace; O3 sequential Rayleigh quotients with the earlier columns stopped (the objective of Chang et al. 2025 and Zhang et al. 2026, without their per-geometry sorting, which a single geometry does not need); O4 log-det. Definitions and code as Phase 0 (phase0.py, hash in PLAN_HASH.txt).
- Hardware: the author's laptop (Windows 11, i9-13900HX), CPU, float64, single-threaded processes, about four at a time. Last-digit differences from the sandbox are expected and are not deviations.

**Geometries** (Sobol points, scipy.stats.qmc.Sobol(d=2, scramble=True, seed=s), x_c = 1 + 4u, d = 0.1 + 0.3v, rounded to four decimals; lists in geometries.json):

- Part I: G1 to G3, seed 11 (first 3 of 4 points).
- Part II: 24 training geometries (seed 12, first 24 of 32) and 16 held-out geometries (seed 13).

None of these was trained on in Phase 0; Phase 0 computed only their eigen-gaps (all of them) and the sensitivity tables (G1 to G3).

## 3. Part I: confirmatory comparison on G1 to G3

Runs: O1, O2, O3, O4 x G1, G2, G3 x seeds 10, 11; 20,000 iterations each (24 runs).

Primary measure: the relative Ritz frequency errors e_i = omega~_i / omega_i - 1, i = 1..8 (sorted Ritz values of the final basis against the finite element frequencies).

| | Prediction | Phase 0 on G0 (seeds 0 to 2) |
|---|---|---|
| H1 (O4 learns all eight) | max_i e_i <= 0.5% | 0.08%, 0.10%, 0.10% |
| H2 (O1 misses higher modes) | max_{i=3..8} e_i > 5% | 462%, 649%, 513% |
| H3 (O2 leaves mode 1 behind) | e_1 > 1% and e_1 > max_{i=3..8} e_i | e_1 = 12.0%, 17.2%, 19.9%; max_{3..8} = 0.11%, 0.14%, 0.16% |
| H3b (O3 leaves mode 1 behind) | e_1 > 0.5% and e_1 > max_{i=3..8} e_i | e_1 = 3.66%, 3.01%, 5.11%; max_{3..8} = 0.07%, 0.06%, 0.11% |

Each prediction is checked in each of the six geometry-seed runs. Verdict per hypothesis: supported if it holds in all six; partly supported in four or five; not supported in three or fewer. H3b is new after Phase 0 (the plan document set no direction for O3).

Secondary, reported without thresholds: E1 (tip displacement and acceleration), E2 (trajectory energy-norm error), S/T on the training cases, sin^2 of the angle between each exact eigenvector and the span, learning curves every 1,000 iterations, wall time.

Per-geometry predictions from the sensitivity tables of Phase 0 (e) (how many times more the per-step potential responds to mode 1 than to each other mode) are reported next to the O1 results; they are descriptive, not a test.

## 4. Part II: the notched-cantilever family (demonstration)

Runs: O4 on the family, seeds 0, 1, 2, 60,000 iterations, 4 of the 24 training geometries per iteration (a fresh random permutation of the 24 every six iterations, so each geometry is used once per six iterations, as in the pilot), loss = mean over the 4.

Controls:
- modal truncation (8 finite element eigenvectors) on every geometry;
- supervised: the same network fitted to the eight lowest eigenvectors of the training geometries (K-weighted capture, the B1 loss of set 6), seeds 0, 1, 2, same budget;
- [DECISION, recommended] O3 with per-geometry sorting (columns ordered by their detached Rayleigh quotients before the stopped Gram-Schmidt, as Chang et al. 2025 and Zhang et al. 2026), seeds 0, 1, 2, same budget; about one more hour of laptop time;
- not included: nearest-geometry eigenvector interpolation (tried in Phase 0 (i): largest Ritz error 8.9% to 82%, E2 38% to 96%; a strawman).

Measures on the 16 held-out geometries:

| | Prediction (each seed) | Phase 0 pilot (8 training, 4 held out, seed 0) |
|---|---|---|
| H4 | median over geometries of the median of e_1..e_5 <= 1% | 0.61% |
| H5 | median reproduction of the notch-induced drop of omega_1 >= 90% [DECISION: or 85%] | 86.8% |
| H6 | median E2 (median over each geometry's 10 evaluation cases) <= 10% [DECISION: or 15%] | 14.8% |
| H7 (if the O3 control is run) | held-out median e_1 of O3 > that of O4 | (not piloted) |

Drop reproduction = (omega_plain - omega_1,net) / (omega_plain - omega_1,FE), with omega_plain the un-notched cantilever.

Thresholds of H4 to H6. The plan document proposed 1%, 90% and 10% before the pilot. The pilot meets the first and misses the other two. With only eight training geometries, three of its four held-out geometries lie outside the training hull; two of them, one at the edge of the parameter range and one with a deep notch far from every training point (distances 0.22 and 0.60 in the normalised coordinates), have largest Ritz errors of 1.6% and 1.75%, while the other two give 0.15% to 0.29%, 96% to 98% and 5.6% to 7.6%. The formal family has three times as many training geometries; its held-out geometries lie 0.04 to 0.23 from their nearest training geometry (median 0.19 against 0.31 in the pilot), 12 of 16 inside the training hull. Recommended: keep 1%, 90% and 10%. Alternative: 1%, 85% and 15%, the levels the pilot reached with a third of the training geometries.

Also reported for Part II, without thresholds: every per-geometry number, the training geometries, the distance of each held-out geometry to its nearest training geometry, and the two held-out geometries whose notch drop is below 1% (where the reproduction ratio mostly measures the frequency error).

## 5. Rules

- Thresholds, geometries, seeds, budgets and code are fixed by the hash. Any change after the hash is recorded in DEVIATIONS before it is made, with its reason, and reported.
- No run is excluded or repeated. A run that crashes is rerun once from the same seed and reported as such.
- All hypotheses are reported, supported or not, with the per-run numbers.
- Exploratory analyses beyond this list are labelled as such.
