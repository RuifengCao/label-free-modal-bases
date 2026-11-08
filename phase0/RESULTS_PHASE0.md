# Phase 0 results (exploratory): Unified 2027 dynamics paper

Plan: PHASE0_PLAN.md, hashed 2026-11-06 02:58:50 UTC before any run (PLAN_HASH.txt). Deviations and additions: DEVIATIONS_PHASE0.md (three entries, each hashed before the action). Runs: 6 November 2026, 03:05 to 04:35 UTC, two sandbox cores. Full tables: TABLES_PHASE0.md. Exploratory throughout: no hypothesis is tested here.

## Summary

1. **The four objectives fail on different modes, as their weighting predicts** (G0, 20,000 iterations, seeds 0 to 2, same network, initialisation and schedule):
   - O1, per-step incremental potential: mode 1 within 0.06%; modes 2 to 8 wrong by 166% to 649%.
   - O2, trace: modes 2 to 8 within 0.33%; mode 1 wrong by 12.0%, 17.2%, 19.9%.
   - O3, sequential Rayleigh quotients (the objective of Chang et al. 2025 and Zhang et al. 2026): modes 2 to 8 within 0.25%; mode 1 wrong by 3.7%, 3.0%, 5.1%.
   - O4, log-det: all eight within 0.10% in every seed.

   The response errors follow: tip acceleration error (median over 10 evaluation cases) 86% (O1), 55% to 85% (O2), 14% to 24% (O3), 1.7% to 2.1% (O4), against 1.25% for modal truncation.
2. **O4 trained longer reaches modal truncation**: 60,000 Adam iterations then 5,000 L-BFGS iterations give all eight modes within 0.006%, tip acceleration error 1.26% (modal truncation 1.25%) and E2 1.63% (1.26%).
3. **Notch-angle inputs are not needed**: without them O4 reaches 0.11% (with them 0.08%) after 20,000 iterations.
4. **Eight-mode gap**: lambda_9/lambda_8 >= 1.39 on all 56 geometries; no crossing at the truncation. Modes 3 and 4 come close on some geometries (eigenvalue ratio down to 1.03; irrelevant to O4, which sees only the subspace).
5. **Sensitivity tables on G1 to G3** repeat the G0 pattern: the per-step potential responds to a 1% error in mode 1 42 to 49 times more than to mode 2 and 1,000 to 19,000 times more than to modes 3 to 8 (one negative response, mode 6 on G1).
6. **Family pilot** (8 training, 4 held-out geometries, O4, 60,000 iterations): training geometries within 0.23% (all eight modes); two held-out geometries within 0.29%, the other two (one at the edge of the parameter range, one with a deep notch far from every training point) at 1.6% and 1.75%. Held-out medians: first-five-mode error 0.61%, notch drop of omega_1 reproduced 86.8%, E2 14.8%.
7. **Nearest-geometry interpolation of eigenvectors** is far worse (max Ritz error 8.9% to 82%, E2 38% to 96%); not informative as a control.
8. **Cost** at 7,322 degrees of freedom: network evaluation 4 ms against 113 ms for 16 eigenpairs (after a one-off 0.3 s compilation per mesh); reduced step 19 microseconds against 1.3 ms for a full step.
9. **Literature**: Shape Space Spectra (Chang et al., ACM TOG 2025) had been missed. It already learns vibration modes label-free over continuous shape families and uses them for reduced elastodynamics, with the O3 objective plus per-geometry sorting. The family part of the paper is therefore not new; the objective comparison is.

## A. Code checks

All passed (out/checks.json), A4 after its tolerance was corrected (DEVIATIONS_PHASE0.md, 1: fc.eig starts ARPACK from a random vector, so rc.setup() itself differs between calls by about 1e-13).

| Check | Result |
|---|---|
| A1 O3 vectorised against loop (value; gradient) | init: 5.8e-16; 5.6e-15. F1 parameters: 3.5e-13; 6.6e-13 |
| A2 JAX against NumPy at the F1 parameters (O2; O3; O4) | 1.7e-12; 3.6e-13; 7.4e-13. At V_8: 8.3e-15; 1.4e-14; 2.3e-12 |
| A3 family loss and gradient against the mean | 5.5e-14; 2.7e-14 |
| A4 geom_setup(2.0, 0.3) against rc.setup() | frequencies 4.4e-15, dt 3.3e-15, pulse widths 3.2e-15; weights and names identical |

## (a) Four objectives on G0

Final Ritz frequency errors (sorted Ritz frequencies against the finite element frequencies), 20,000 iterations:

| Objective, seed | mode 1 | mode 2 | modes 3 to 8 (largest) | E1 tip acceleration (median) | E2 (median) | S/T | training time |
|---|---|---|---|---|---|---|---|
| modal truncation | 0 | 0 | 0 | 1.25% | 1.26% | 3.4e-9 | |
| O1, 0 / 1 / 2 | 0.06% / 0.05% / 0.06% | 173% / 183% / 166% | 462% / 649% / 513% | 85.8% (all three) | 19.1% (all three) | 8.2e-8 to 9.8e-8 | 523 to 531 s |
| O2, 0 / 1 / 2 | 12.0% / 17.2% / 19.9% | 0.33% / 0.30% / 0.33% | 0.11% / 0.14% / 0.16% | 54.9% / 76.8% / 84.8% | 77.4% / 100.3% / 109.7% | 1.9e-5 to 3.2e-5 | 286 to 289 s |
| O3, 0 / 1 / 2 | 3.66% / 3.01% / 5.11% | 0.14% / 0.14% / 0.25% | 0.07% / 0.06% / 0.11% | 17.1% / 14.1% / 24.1% | 32.7% / 28.8% / 41.1% | 4.6e-6 to 7.8e-6 | 308 to 314 s |
| O4, 0 / 1 / 2 | 0.07% / 0.07% / 0.09% | 0.04% / 0.04% / 0.05% | 0.08% / 0.10% / 0.10% | 1.72% / 1.86% / 2.09% | 4.03% / 3.93% / 4.42% | 1.1e-7 to 1.4e-7 | 287 to 289 s |
| O1, D1 of set 6 (50,000; 1e-3 to 1e-6) | 0.01% | 0.22% | 452% (mode 3: 7.6%) | 12.3% | 4.75% | 2.9e-8 | 2,504 s |
| O1, D2 of set 6 (50,000; 3e-4 to 1e-6) | 0.05% | 169% | 515% | 85.8% | 19.1% | 8.9e-8 | 2,489 s |
| O4, F1 of set 6 (repeat of O4 seed 0) | 0.07% | 0.04% | 0.08% | 1.72% | 4.03% | 1.2e-7 | 397 s |

Learning curves (figure out/fig_learning_seed0.png; first logged iteration with all eight errors at or below a level): O4 reaches 5% / 1% / 0.5% at 4,000 to 7,000 / 6,000 to 8,000 / 8,000 to 9,000 iterations; O3 reaches 5% at 15,000 to 16,000 in two seeds and never 1%; O1 and O2 never reach 5%. The order in which modes are learnt is reversed between O4 and the trace-type objectives: under O4 mode 1 is at about 1% after 1,000 iterations (1.09%, seed 0); under O2 and O3 mode 1 is the last, still decreasing slowly at 20,000 (O2 seed 0: 294% at 1,000, 28% at 10,000, 12% at 20,000).

Values of the four objectives at each final basis (TABLES_PHASE0.md) show each objective preferring its own optimiser's basis. Two consequences: O3's value equals O2's (Gram-Schmidt in column order is the Cholesky orthonormalisation, so the sum of the column Rayleigh quotients is the trace; only the gradients differ). And O1 rates O1's own bases (value -1.12105e6 to -1.12088e6, S/T about 9e-8) above O4's (-1.11985e6 to -1.11939e6, S/T about 1.3e-7), although O4's trajectory error E2 is 4% against 19%: the per-step quantities measure each step from the basis's own history and do not see the accumulated error.

M-angles mislead here. sin^2 of the M-angle between v_1 and span(Phi) is 1.7e-6 to 2.8e-6 for O2 and 1.2e-7 to 4.1e-7 for O3, yet their first frequency is 3% to 20% high. Post hoc (DEVIATIONS_PHASE0.md, 3), the same angle in the K (energy) inner product, seed 0: v_1 0.20 (O2), 0.069 (O3), 1.5e-3 (O4), 1.2e-3 (O1); v_2 to v_8 at most 6.5e-3 (O2), 2.9e-3 (O3), 1.5e-3 (O4) and 0.43 to 0.99 (O1). The trace-type bases carry high-frequency content in their first mode that the mass norm hides and the frequency does not: a contamination of mode i by mode j raises theta_i by about (lambda_j - lambda_i) times its squared size, so relative to lambda_i it weighs lambda_j / lambda_i.

## (b) O4 for longer, then L-BFGS

| | modes 1 to 8 (largest error) | E1 tip acceleration | E2 | S/T | time |
|---|---|---|---|---|---|
| 60,000 Adam | 0.017% | 1.27% | 2.21% | 3.0e-8 | 853 s |
| then 5,000 L-BFGS-B (5,295 evaluations; stopped at the iteration limit) | 0.0056% | 1.26% | 1.63% | 1.2e-8 | 85 s |
| modal truncation | 0 | 1.25% | 1.26% | 3.4e-9 | |

## (c) O4 without the notch-angle inputs

Variant 'hard', seed 0, 20,000 iterations: largest error 0.11% (with angle inputs 0.08%); 0.5% reached at 8,000 iterations (9,000 with them); E1 1.99%, E2 4.66% (1.72%, 4.03%). By the reading rule of the plan, the angle inputs are not needed and speed learning up only slightly. The formal runs keep them.

## (d) Gap and mode order (no training)

| Group | n | lambda_9/lambda_8 (min / median / max) | lambda_4/lambda_3 (min) | drop of omega_1 by the notch (min to max) |
|---|---|---|---|---|
| G0 | 1 | 1.625 | 1.160 | 9.46% |
| G1 to G3 | 3 | 1.586 / 1.636 / 1.715 | 1.031 | 0.48% to 5.70% |
| family, training | 24 | 1.395 / 1.646 / 1.732 | 1.030 | 0.55% to 15.78% |
| family, held out | 16 | 1.392 / 1.643 / 1.716 | 1.029 | 0.50% to 13.44% |
| pilot, training | 8 | 1.577 / 1.649 / 1.728 | 1.035 | 0.67% to 17.94% |
| pilot, held out | 4 | 1.465 / 1.646 / 1.692 | 1.035 | 0.49% to 13.19% |

Un-notched cantilever omega_1 = 0.010083. Mostly axial (x share of kinetic energy above 0.5): modes 3 and 7 on most geometries; sometimes 4 as well (where modes 3 and 4 nearly coincide and mix) and 9. No geometry has lambda_9/lambda_8 below 1.1 (smallest 1.392 at x_c = 3.9503, d = 0.3504). Two held-out family geometries have a notch drop below 1% (0.50%, 0.89%); there, the drop-reproduction ratio is dominated by the frequency error itself.

## (e) Sensitivity tables on G1 to G3 (no training)

G1 = (2.4139, 0.2475), G2 = (3.0944, 0.2760), G3 = (4.3651, 0.1099); lambda_8/lambda_1 = 4,006, 3,735, 3,648 (G0: 4,404). For a 1% frequency error in mode i, mode 1's increase of S/T divided by mode i's:

| Geometry | mode 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| G0 (set 7) | 52 | 5,864 | 2,522 | 4,000 | negative | 10,191 | 12,479 |
| G1 | 49 | 1,222 | 1,740 | 5,038 | negative | 8,537 | 19,167 |
| G2 | 48 | 1,565 | 5,692 | 4,012 | 9,077 | 11,052 | 10,924 |
| G3 | 42 | 1,577 | 1,017 | 4,017 | 7,594 | 10,171 | 12,525 |

The trace excess grows with the eigenvalue (1.8e-6 to 2.0e-6 for mode 1, 6.8e-3 to 7.4e-3 for mode 8); the log-det excess is 0.0199 for every mode on every geometry.

## (f) Family pilot and (i) nearest-geometry control

O4, seed 0, 60,000 iterations, 3,569 s. Training geometries (8): largest Ritz error 0.12% to 0.23%; notch drop reproduced 90.6% to 99.0%; E1 tip acceleration 1.66% to 4.99% (modal 1.12% to 1.51%); E2 4.0% to 6.7% (modal 1.2% to 1.4%).

| Held-out geometry (x_c, d) | distance to nearest training geometry* | largest Ritz error | median of modes 1 to 5 | notch drop (FE) | reproduced | E1 tip acc. (net / modal) | E2 (net / modal) | nearest-geometry control: largest Ritz error, E2 |
|---|---|---|---|---|---|---|---|---|
| (1.2756, 0.3196) | 0.39, outside the training hull | 0.29% | 0.10% | 13.19% | 98.1% | 3.4% / 1.2% | 7.6% / 1.3% | 10.7%, 70.7% |
| (4.9366, 0.1361) | 0.22, outside | 1.60% | 1.11% | 0.49% | -224% | 34.7% / 1.2% | 21.9% / 1.4% | 8.9%, 37.7% |
| (3.7122, 0.3651) | 0.60, outside | 1.75% | 1.18% | 7.24% | 77.6% | 33.5% / 1.3% | 22.9% / 1.3% | 82.5%, 83.9% |
| (2.1215, 0.1838) | 0.13, inside | 0.15% | 0.09% | 3.65% | 96.0% | 1.9% / 1.4% | 5.6% / 1.3% | 15.9%, 95.5% |

*In the normalised coordinates ((x_c - 3)/2, (d - 0.25)/0.15). For the formal family (24 training geometries) the 16 held-out geometries lie at 0.04 to 0.23 from their nearest training geometry (median 0.19, pilot 0.31), and 12 of 16 inside the training hull.

Held-out medians (the measures of H4 to H6): median of the per-geometry median error of modes 1 to 5, 0.61%; drop reproduced, 86.8%; E2, 14.8%. The held-out error stopped improving after about 35,000 iterations (largest 1.79% at 35,000, 1.75% at 60,000) while the training error kept falling (0.35% to 0.23%).

The nearest-geometry control interpolates eigenvectors whose notch is elsewhere; it is a strawman and is not proposed for Part II.

## (g) Literature

Details in the project document on novelty (section 6, added 03:25). In short:

- Missed earlier and now read: Chang, Benchekroun, Chiaramonte, Chen, Grinspun, Shape Space Spectra, ACM TOG 44(4) 121 (2025). Label-free neural eigenfunctions over continuously parameterised shape families, extended to linear elasticity and used for reduced elastodynamics on unseen shapes. Objective: sum of the energies with column-by-column Gram-Schmidt, earlier columns stopped ("causal filtering"), plus per-geometry reordering by eigenvalue ("causal sorting"). This is O3 with sorting.
- Rowan et al. (arXiv 2506.04375, full text): modes learnt one at a time, each in its own minimisation with the earlier converged modes fixed; higher modes need more iterations; no time integration; no trace or log-det.
- Zhang, Xiong, Wang (arXiv 2610.33925, full text, sections 2.4 and 2.5): O3 with per-geometry sorting, following Chang et al. Their spectral-variance relation compares the Rayleigh quotients of one approximate mode in two norms (velocity against potential); ours compares ways of combining the modes into one objective. Related, not overlapping.
- log-det as a training objective for neural eigensolvers or reduced bases: still not found (OpenAlex, three queries of 50 results; two extended web searches). Google Scholar refused automated access in both routes tried (robots.txt; an "unusual traffic" page in the author's browser, not bypassed); the author should run the Scholar search himself.

## (h) Cost (one sandbox core; median of repeats)

| h_max | degrees of freedom | mesh and assembly | 16 eigenpairs | network evaluation (compiled) | compilation | reduced step | full factorisation | full step |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 7,322 | 0.074 s | 0.113 s | 4.0 ms | 0.29 s | 19 us | 30 ms | 1.3 ms |
| 0.14 | 13,404 | 0.143 s | 0.240 s | 7.2 ms | 0.30 s | 20 us | 69 ms | 2.5 ms |
| 0.10 | 23,526 | 0.258 s | 0.488 s | 13.3 ms | 0.36 s | 20 us | 174 ms | 5.1 ms |

Training costs (single core): O4 on one geometry, 20,000 iterations, about 290 s; O1, about 525 s; family pilot, 60,000 iterations with 4 geometries each, 3,569 s.

## Files

phase0/: PHASE0_PLAN.md, DEVIATIONS_PHASE0.md, phase0.py, queue.sh, smoke.sh, analyse.py, fig_learning.py, posthoc_k.py, make_compact_p0.py, TABLES_PHASE0.md, PREREG_DRAFT.md; out/: checks.json, evalg0.json, evalfam.json, gap.json, sens.json, cost.json, geometries.json, posthoc_k.json, phase0_compact.json, fig_learning_seed0.png, parameter files (*.pkl); logs/.
