# Phase 0 (exploratory): Unified 2027 dynamics paper

Written and hashed before any Phase 0 run. Phase 0 is exploratory. It tests no hypothesis and gives no verdict on the method. Its results set the design and the thresholds of the pre-registration (Parts I and II), which is written and stamped separately before any formal run. Every result is reported, whatever it shows.

Decisions recorded on 6 November 2026 (02:50, the author): the paper's focus is how each label-free objective weights the error of each vibration mode; the notched-cantilever family is a demonstration. Decisions 2 and 3 of the plan document were taken as recommended: log-det is the only main method (no per-step fine-tune), and the confirmatory comparison uses three new geometries and two seeds. The author can override both.

Code: phase0/phase0.py. Outputs: phase0/out/. Earlier sets (PLAN.md, POSTHOC*.md) are not changed.

## Common settings

- **G0**: notch centre x_c = 2.0, depth d = 0.3, root radius 0.05, h_max = 0.2 (rc.setup()); 7,322 free degrees of freedom. 16 training load cases (seed 2026), 10 evaluation cases (8 random, seed 2027, plus the vertical tip pulse and tip step). N = 640 Newmark steps (average acceleration), dt = T_3/20, Rayleigh damping 2% at modes 1 and 3. m = 8.
- **Other geometries** use the same recipe through geom_setup(x_c, d): same mesh rules, own eigenvalues for dt and damping, load cases from the same seeds with pulse widths scaled by the geometry's own periods.
- **Network**: as set 6 variant 'hard_angle': the 20 planned input features plus the 3 notch-angle features, MLP with 4 hidden layers of 64 (tanh) and 2m outputs, outputs multiplied by x/L (exact clamp), M-orthonormalisation by Cholesky. Initialisation PRNGKey(seed); for a given seed, every objective starts from the same parameters.
- **Optimiser**: Adam (0.9, 0.999, 1e-8), learning rate cosine from 1e-3 to 1e-5 over the run (posthoc6.train_logged). Adam is invariant to a constant rescaling of the loss (up to its epsilon), so the learning rate is not tuned per objective.
- **Objectives** (same network, only the loss changes):
  - O1, per-step incremental potential: lb.loss_fn on the 16 training cases, gradient stopped on the history (as 5 November).
  - O2, trace: tr(Phi_o^T K Phi_o), Phi_o the M-orthonormalised basis.
  - O3, sequential Rayleigh quotients: Q = stop_gradient(Phi_o); R = Phi - Q (U .* (Q^T M Phi)) with U the strictly upper triangular matrix of ones; loss = sum_k r_k^T K r_k / r_k^T M r_k. This is column-by-column Gram-Schmidt in the M inner product with the earlier columns stopped, written without a loop. Our own implementation of the idea behind Rowan et al. and NeuralEF, not their code.
  - O4, log-det: log det(Phi_o^T K Phi_o) through Cholesky, equal to log det(Phi^T K Phi) - log det(Phi^T M Phi) (as F1 of set 6).
- **Monitoring** every 1,000 iterations and at iteration 1: the loss; Ritz frequency errors of modes 1 to 8 (sorted Ritz values against the finite element frequencies); sin^2 of the M-angle between each exact eigenvector v_1 to v_8 and span(Phi). Eigenvectors are used only to monitor and evaluate, never in a loss.
- **Evaluation** of each final basis: Ritz errors (8); eigenvector sin^2 (8); rc.metrics on the 10 evaluation cases (E1 tip displacement and tip acceleration, E2 trajectory energy-norm error, E3 per-step distance); S/T on the 16 training cases (posthoc6.S_split); the values of all four objectives at the basis; conditioning of the M-Gram matrix. The same for modal truncation (the exact V_8).

## A. Code checks (run first; training starts only if all pass)

- A1: O3 as written above against the loop implementation of bench.py (seqrq_loss): value and gradient agree to a relative 1e-10, at the seed-0 initialisation and at the F1 parameters.
- A2: O2, O3 and O4 in JAX against independent NumPy computations from the basis (sum of Ritz values, Gram-Schmidt Rayleigh quotients, sum of log Ritz values): relative 1e-9 at the F1 parameters. At V_8 the NumPy values equal sum lambda_i and sum log lambda_i (relative 1e-10).
- A3: the family loss equals the mean of the single-geometry losses on four geometries, and so does its gradient (relative 1e-12).
- A4: geom_setup(2.0, 0.3) reproduces rc.setup() (degrees of freedom, frequencies, load cases).

If a check fails: stop, record it in phase0/DEVIATIONS_PHASE0.md, fix, record the fix, then train.

## (a) Four objectives on G0

O1, O2, O3 and O4, seeds 0, 1 and 2, 20,000 iterations each (12 runs).

The longer O1 budget is not rerun. D1 of set 6 (O1, the same network and inputs, seed 0, 50,000 iterations, learning rate 1e-3 cosine to 1e-6, Ritz errors of modes 1 to 5 every 5,000 iterations) serves as the 50,000-iteration O1 run, and D2 (3e-4 to 1e-6) as a second learning rate. This differs from the plan document ("O1 另跑到 50,000 步") in the end point of the schedule and in the logging; it is stated here before running. The final D1 and D2 parameters are evaluated in the same way as the new runs. F1 of set 6 (O4, seed 0, 20,000 iterations) is evaluated too, as a reproducibility check of (a) O4 seed 0.

What is looked at: per-mode learning curves; which modes each objective has learnt after 20,000 iterations; spread over seeds.

Expectations from the weighting analysis, stated before running (not tested here): O1 learns modes 1 and 2, the higher modes late or not at all; O2 leaves larger relative errors in the low modes at a fixed budget; O4 is balanced across modes; for O3 no direction is expected.

## (b) O4 for longer, then L-BFGS

O4, seed 0, 60,000 iterations (1e-3 cosine to 1e-5), monitored every 2,000 iterations; then L-BFGS-B from SciPy on the ravelled parameters with the jitted value and gradient (at most 5,000 iterations, 20 correction pairs, ftol 0, gtol 1e-14). Report Ritz errors, E1 and E2 after Adam and after the polish, against modal truncation (tip acceleration 1.3% and E2 1.3% on 5 November).

## (c) O4 without the notch-angle inputs

Variant 'hard' (the 20 planned features, exact clamp), O4, seed 0, 20,000 iterations. Compared with (a) O4 seed 0. Reading rule: if the largest 8-mode Ritz error after 20,000 iterations is at most 0.5%, the angle inputs speed learning up but are not needed; otherwise they are needed. The formal runs keep the angle inputs either way; the result is reported.

## (d) Gap and mode-order check (no training)

Geometries: G0; G1 to G3 (Sobol seed 11, 3 points); the formal family (Sobol seed 12, 24 training; seed 13, 16 held out); the pilot family (seed 21, 8 training; seed 22, 4 held out). 56 geometries.

Sobol points: scipy.stats.qmc.Sobol(d=2, scramble=True, seed=s).random_base2(k) with 2^k the smallest power of two not below n, first n points, mapped to x_c = 1 + 4u and d = 0.1 + 0.3v, rounded to four decimals. The lists are written to phase0/out/geometries.json. Phase 0 trains on none of the G1 to G3 or formal-family geometries.

For each geometry: degrees of freedom; omega_1 to omega_16; lambda_9/lambda_8; the share of kinetic energy in x for modes 1 to 9 (axial modes) and the index of the first axial mode; omega_1 against the un-notched cantilever (posthoc2.plain_cantilever_w1). Flags: lambda_9/lambda_8 < 1.1 (the eight-mode subspace is poorly separated); a mode crossing at the boundary between modes 8 and 9. The crossing of modes 3 and 4 is reported (it does not affect O4, which sees only the subspace).

## (e) Sensitivity tables on G1 to G3 (no training)

As W2 of set 7 (posthoc7.cmd_w2), on each geometry with its own training cases: the excess of the trace and log-det objectives and the increase of S/T for a 1% frequency error in mode i, i = 1 to 8; and mode 1's increase of S/T divided by mode i's. These give the per-geometry predictions for the pre-registration, in particular which modes O1 is expected to miss.

## (f) Family pilot

Training geometries: Sobol seed 21 (8). Held out: seed 22 (4). Not part of the formal study.

Network inputs: the 'hard_angle' inputs plus ((x_c - 3)/2, (d - 0.25)/0.15), 25 in all. Loss: O4 on each geometry; each iteration uses 4 of the 8 training geometries (a fresh random permutation of the 8 every two iterations, seeded by the run seed), loss = mean over the 4. Seed 0, 60,000 iterations, 1e-3 cosine to 1e-5. Training problems are built without any eigen solve (O4 reads only K, M and the inputs; the damping and time-step entries of the problem are unused placeholders). Monitoring every 5,000 iterations: Ritz errors of modes 1 to 8 on all 12 geometries.

Evaluation on all 12 geometries: Ritz errors (8); eigenvector sin^2 (8); reproduction of the notch-induced drop of omega_1, (omega_plain - omega_1,net) / (omega_plain - omega_1,FE); E1 and E2 on each geometry's 10 evaluation cases; modal truncation on the same cases as reference.

Use: the held-out values set the thresholds of H4 to H6 in the pre-registration, which is written after this pilot and before any formal run.

## (i) Nearest-geometry eigenvector baseline (optional control, tried here)

For each held-out pilot geometry: the 8 finite element eigenvectors of the nearest pilot training geometry (distance in the normalised (x_c, d)), interpolated onto the held-out P2 nodes (piecewise linear over the training node cloud, scipy LinearNDInterpolator; the clamp is exact because only free degrees of freedom are kept), M-orthonormalised; evaluated as in (f). Decides whether this control enters Part II.

## (g) Literature

A Google Scholar search (or Semantic Scholar and Crossref if Scholar refuses automated access) for log-det used as a neural eigensolver loss or a reduced-basis training loss; full texts of Rowan et al. (arXiv 2506.04375) and Zhang, Xiong and Wang (arXiv 2610.33925): which objectives they use, what they say about weighting, and the overlap with this work.

## (h) Cost table

For G0 at h_max = 0.2, 0.14 and 0.1: mesh and assembly; 16 eigenpairs; network evaluation (inputs, MLP, M-orthonormalisation; compiled); one reduced step; factorisation and one step of the full model. Wall time on one sandbox core, median of five repeats.

## Compute

Two sandbox cores; single-threaded processes pinned with taskset. Queue A: (f), (b), then the family evaluation. Queue B: the checks, (a), (c), (d), (e), (h). Then the G0 evaluation. A short smoke test with a few iterations, written to a separate folder, precedes the real runs; it prints losses only.
