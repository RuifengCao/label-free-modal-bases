# Phase 0: deviations and check failures

Each entry is written before the action it describes.

## 1. Check A4 failed as written (2026-11-06, about 03:05 UTC, before any training)

A4 asked that geom_setup(2.0, 0.3) reproduce rc.setup() exactly. It did not: the frequencies differed by 1.2e-14 (relative), so dt and the pulse widths differed at the same level and the load-case lists were not equal.

Cause: fc.eig uses ARPACK in shift-invert mode with a random starting vector, so two calls of rc.setup() itself differ by about 1e-13 in the frequencies (checked: rc.setup() against rc.setup(), largest relative difference 1.1e-13; rc.setup() against geom_setup(2.0, 0.3), 1.1e-13; load weights identical; pulse widths within 1.1e-13). The check was too strict, not the code wrong.

Fix: A4 now asks for the same number of degrees of freedom, identical load weights and case names, and frequencies, dt and pulse widths within a relative 1e-10. The JSON writer also failed on NumPy booleans; fixed by converting them. A1 to A3 passed as written and are not changed.

## 2. Smoke test covers the evaluation code too (2026-11-06, about 03:06 UTC, before any training)

The plan says the smoke test "prints losses only". To catch errors in the evaluation code before the long runs, the smoke test also runs every command (long, fam, gap, sens, cost, evalfam, evalg0) with a few iterations, shortened evaluations (P0_SMOKE=1: two evaluation cases, two training cases for S/T) and its own output folder (phase0/smoke). Only exit codes and the presence of the output fields are inspected; the numbers it prints concern networks trained for at most a few hundred iterations and are not read or used.

## 3. Post hoc diagnostic added after seeing the G0 evaluation (2026-11-06, about 04:35 UTC)

Not in the plan; labelled post hoc wherever reported. The O2 and O3 bases have sin^2 (M-angle) of about 1e-6 to 1e-7 for every exact eigenvector, including v_1, yet their first Ritz frequency is 3% to 20% high. The likely reason: the M-norm hides high-frequency contamination, and the frequency error weights a contamination of mode i by mode j with lambda_j / lambda_i. To check this, posthoc_k.py computes, for the final G0 bases (modal truncation; O1 to O4 seed 0; the O4 long run before and after L-BFGS), sin^2 of the angle between each v_i and span(Phi) in the K (energy) inner product. It reads only saved parameters and eigenvectors; nothing is retrained.

## Note on Phase 0b check A5 (2026-11-06, 16:58 UTC; not a deviation)

The first run of `phase0b.py checks` crashed before producing any number: the loop form used for A5 sorted the columns with `float()` inside the JAX trace. The check script now sorts the columns in NumPy outside the trace and passes the order in; the loss under test (o3s_loss) was not changed. The crash and the fix happened before any Phase 0b training; recorded here for completeness, after the fix.

## 4. Phase 0d check B4 shows a floor from the Gram truncation (2026-11-06, about 18:32 UTC, before the held-out evaluation of (m))

B4 (mapped reduced basis with all 24 pilot-2 training geometries as sources, target included among them) gave largest relative Ritz errors of 0.037% to 0.105% on 4 training geometries, although the exact eigenvectors of the target are in the span: with the target as its only source (B2) the errors are below 6e-12. The likely cause is the Gram truncation (directions whose mass-norm Gram eigenvalue is below 1e-10 of the largest are dropped), which can remove directions that are small in the mass norm but not in the energy norm.

Correction (2026-11-06, 19:16 UTC): "below 6e-12" above is wrong; I read only the last of the four B2 values. The four are 1.0e-11, 1.0e-12, 2.0e-11 and 5.9e-12, so the largest is 2.0e-11 (still far below the 1e-8 of check B2). Nothing else in this entry changes.

Action, decided before any held-out result of (m) is seen: a diagnostic of B4 on the same 4 training geometries with tolerances 1e-12, 1e-13 and 1e-14 and with the truncation done on the Gram matrix of K instead of M (tolerance 1e-10). The held-out evaluation then runs the planned variants (tolerance 1e-10 on the mass Gram matrix) unchanged and adds, for k = 4 and all, the variant with the smallest B4 error, labelled as added after B4. The choice uses the training-geometry check only.

## 5. Phase 0e: per-mode networks too slow as implemented (2026-11-06, about 18:55 UTC, before any Phase 0e result)

The first Phase 0e run (O3s, seed 0) started at 18:46 UTC. Phase 0c on G0 took 1024 to 1213 s per 20,000 iterations with one network per mode, against 286 to 312 s for the shared network in Phase 0, so the loop over eight MLPs costs about 3.5 times as much; at that ratio one 60,000-iteration family run would take about 3.7 hours, beyond the 2-hour limit of a sandbox background job. The run was stopped after about 8 minutes, before its first log line after iteration 1; nothing from it is used.

Action: a batched implementation of the same eight MLPs (parameters stacked layer by layer, one batched product per layer, one scatter per displacement component), with the same initial parameters. Before use, check E2: on a pilot-2 training geometry with family inputs, at the seed-0 initialisation with random biases added, the batched raw fields equal the loop fields (relative 1e-13), the values of O3s and O4 agree (relative 1e-12) and their gradients agree (relative 1e-10). Then a timing of both. If one batched 60,000-iteration run still cannot finish within 2 hours, the number of iterations is reduced for both Phase 0e runs, with the shared-network O4 rerun at the same reduced number for a like-for-like comparison, and that change is recorded here before the runs start.

## 6. Phase 0e: fewer iterations, and the shared network rerun at the same number (2026-11-06, about 18:56 UTC, before any Phase 0e result)

Check E2 passed (batched and loop forms identical to round-off: raw fields equal, values equal, gradients within 7e-16), but the batched form is no faster: one value-and-gradient on a pilot-2 training geometry takes 36 ms (O4) and 41 ms (O3s) in both forms. The extra cost of one network per mode is in the activations (8 x 24 = 192 hidden units per layer, against 64), not in the loop. A family iteration (4 geometries) therefore costs about 0.17 s, and 60,000 iterations about 3 hours.

Action, as foreseen in entry 5: all Phase 0e family runs use 30,000 iterations (same cosine schedule from 1e-3 to 1e-5 over the 30,000, same batching and seed 0), about 90 minutes per per-mode run. To compare like with like, the shared network is rerun at 30,000 iterations for O4 and for O3s. The four runs (per-mode O3s and O4, shared O4 and O3s) are evaluated as planned; the 60,000-iteration shared runs of Phase 0b stay as they are and are reported beside them. The batched form is used for the per-mode runs (same parameters, same initialisation). Order: per-mode O3s on core 1 now; per-mode O4 on core 0 after the Phase 0c seed-0 runs; then shared O4 and shared O3s.

## 7. Run order (2026-11-06, about 19:00 UTC; not a deviation in content)

To shorten the Phase 0e critical path, the Phase 0c run O4 seed 0 (G0, one network per mode), started at 18:58:53, is stopped about a minute in and rerun from scratch later with the same seed, after the Phase 0e runs; per-mode O4 for Phase 0e starts on core 0 now instead of after it. Nothing from the stopped run is used.

## Note on entry 7 (2026-11-06, 21:56 UTC; not a deviation)

The rerun of Phase 0c O4 seed 0 ran from 20:03 to 20:21 UTC, after the per-mode O4 run of Phase 0e and before the two shared 30,000-iteration runs, not after all Phase 0e runs as entry 7 says. The Phase 0c seed-1 runs followed from 20:56 to 21:47. No result depends on the order.
