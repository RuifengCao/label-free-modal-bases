# Pipeline check of the label-free reduced basis on one notched cantilever: results

Status: the exploratory pipeline check fixed in PLAN.md (hashed 2026-11-05 20:27:13 UTC). As fixed there, its purpose is to run every code path, verify the identities and measure magnitudes for designing the study; no verdict on the method is drawn from it and none of its results is evidence for the method. The post hoc diagnostics in section 4 were each written down and hashed before they were run (POSTHOC.md to POSTHOC5.md; times in PLAN_HASH.txt). They are exploratory and are reported with every result, including those that went against the reading stated before running.

**Correction (2026-11-05, after set 6).** Two conclusions of sections 5 and 6, written after set 5, were wrong; section 7 supersedes them. (a) The planned inputs can represent the notch; they are slow to learn it (one supervised field: +9.1% after 3,000 iterations, +0.045% after 30,000). (b) Training length is a main part of the limit for modes 2 to 5, and the objective values used in F3 and F5a do not measure the training signal: they mostly track the amplitude of each basis's own trajectory. No coding error was found. Sections 5 and 6 are kept as written, for the record.

## 1. What was run

- One geometry: notch at x_c = 2.0, depth 0.3, root radius 0.05. Mesh of 1,745 quadratic triangles, 7,322 free degrees of freedom.
- FE natural frequencies (E = 1, density 1, beam 10 by 1): 0.009129, 0.06044, 0.1475, 0.1589, 0.2776, 0.4301, 0.4676, 0.6058. The third mode couples bending and axial motion.
- dt = T3 / 20 = 2.130, N = 640 steps; Rayleigh coefficients a = 3.439e-4, b = 0.2554.
- Planned runs: m = 8 and 16, network seeds 0 and 1, 3,000 iterations each. Baselines: modal truncation and POD at the same m.
- Deviations (DEVIATIONS.md): POD uses every fourth step of the training trajectories (2,560 snapshots); I3 was evaluated on the first random evaluation case and the tip pulse for each trained basis, and on the tip pulse for 16 eigenmodes.

## 2. Identity checks: all pass

- I1 (per-step identity, random vectors): relative error 2.3e-16 to 3.5e-16 for the value, 2.9e-16 for the gradient (tolerance 1e-9).
- I2 (Galerkin step is the best approximation in the step-operator norm): 1.3e-14 for a random 16-dimensional basis; 1.7e-13 to 1.5e-12 for the four trained bases.
- I3 (computable bound beta dt^2 g^T M^-1 g on the squared step error along a reduced rollout): no violation in nine rollouts; the largest ratio of actual to bound was 0.022. The bound is loose: it over-estimates the error norm by a factor of 66 to 77 (median over steps) for the trained bases and 34 for 16 eigenmodes.

## 3. Planned measurements

E1: relative L2-in-time error of the tip vertical displacement and acceleration. E2: trajectory error in the energy norm. E3: relative step-operator-norm distance from the subspace to the exact step from the reference state, median over steps. E4: relative errors of the five lowest Ritz frequencies. Medians over the ten evaluation cases, with the maximum after the slash.

| Basis | E1 displacement | E1 acceleration | E2 | E3 | E4, modes 1 to 5 |
|---|---|---|---|---|---|
| Modal, m = 8 | 1.4e-4 / 2.6e-4 | 0.013 / 1.00 | 0.013 / 0.025 | 2e-11 | 0, 0, 0, 0, 0 |
| POD, m = 8 | 1.8e-5 / 7.9e-5 | 0.025 / 1.00 | 8e-4 / 0.004 | 7e-7 | 0, 0, 0, 0, +1.0% |
| Modal, m = 16 | 4.7e-5 / 8.6e-5 | 0.011 / 1.00 | 0.008 / 0.017 | 2e-11 | 0, 0, 0, 0, 0 |
| POD, m = 16 | 1.2e-7 / 2.1e-6 | 0.010 / 0.99 | 2.5e-5 / 3e-4 | 3e-8 | 0, 0, 0, 0, 0 |
| Trained, m = 8, seed 0 | 0.60 / 0.64 | 0.92 / 1.00 | 0.74 / 0.84 | 0.025 | +10.1%, +966%, +393%, +521%, +422% |
| Trained, m = 8, seed 1 | 0.54 / 0.58 | 0.91 / 1.00 | 0.68 / 0.80 | 0.022 | +9.0%, +529%, +381%, +530%, +379% |
| Trained, m = 16, seed 0 | 0.50 / 0.55 | 0.90 / 1.00 | 0.65 / 0.78 | 0.019 | +8.4%, +399%, +237%, +327%, +210% |
| Trained, m = 16, seed 1 | 0.53 / 0.57 | 0.91 / 1.00 | 0.67 / 0.80 | 0.017 | +8.8%, +261%, +190%, +316%, +224% |

The acceleration maximum near 1 for every basis comes from the tip step case, whose initial acceleration is dominated by modes beyond those kept.

E5, costs (two CPU cores, float64): training 70 s (m = 8) and 144 s to 149 s (m = 16) for 3,000 iterations; network evaluation and projections for this geometry 4.0 ms (m = 8) and 6.0 ms (m = 16), against 31 ms to factorise the step operator and 146 ms for 16 eigenpairs; one full step 1.20 ms, one reduced step 32 us (m = 8) and 35 us (m = 16) including set-up spread over the 640 steps; reference trajectories 15.5 s for the 16 training cases and 9.9 s for the 10 evaluation cases.

## 4. Post hoc diagnostics (exploratory)

### Set 1 (POSTHOC.md): is the step operator's mass dominance the cause?

| Run | First Ritz frequency | E1 displacement | E2 | Modes 2 to 5 |
|---|---|---|---|---|
| V1 static potential, m = 8 | +10.1% | 0.60 | 0.74 | +416% to +723% |
| V1 static potential, m = 16 | +9.6% | 0.58 | 0.71 | +196% to +340% |
| V2 Rayleigh-Ritz trace, m = 8 | +187% | 1.04 | 1.25 | +3% to +12% |
| V2 Rayleigh-Ritz trace, m = 16 | +704% | 1.00 | 1.00 | +10% to +77% |
| V3 training step T1 / 20, m = 8 | +9.9% | 0.58 | 0.73 | +399% to +1163% |
| V3 training step T1 / 20, m = 16 | +8.4% | 0.50 | 0.65 | +201% to +419% |

Not supported: the pure stiffness objective (V1) and the larger training step (V3) leave the first frequency 8% to 10% high, like the planned runs. The trace (V2) learns modes 2 to 5 and loses mode 1.

### Set 2 (POSTHOC2.md): representation or objective?

- D1: without the notch, the first frequency is 1.1045 times that of the notched beam; the notch lowers it by 9.5%, about the error of every basis above.
- D2: one field, label-free Rayleigh quotient: +87.6%.
- D3: one field, supervised to the first eigenvector with a stiffness-angle loss: +3,350%.

D2 and D3 fell outside both readings stated beforehand. POSTHOC3.md explains why D3 could not separate the two explanations: its loss equals 1 - cos^2_M(phi, v1) w1^2 / R(phi), so it measures excess strain energy, as D2 does.

### Set 3 (POSTHOC3.md): where is the excess strain energy?

- F1: in the four planned bases, the first Ritz vector matches the first mode in the mass norm (cosine 0.99995) and is as small near the clamp as the mode (largest value within 0.15 of the clamp 0.0021 of the tip value, against 0.0018). Of its error energy, 69% to 78% lies within 0.5 of the notch root and 3% to 4% in the clamp zone (x < 0.5). The learnt field carries 12% to 13% of its strain energy near the notch, the mode 19%, with a fifth of that in shear.
- F2a, F2b: with the clamp imposed exactly (outputs multiplied by x / L), one field reaches +20.7% (label-free Rayleigh quotient) and +9.1% (supervised).
- F2c: the planned objective with the exact clamp, m = 8: +8.7%, E1 0.52, E2 0.67; no change from the planned runs.
- F3: the planned objective at fixed bases: -1.1214e6 for modal and POD (m = 8 and 16), -0.875e6 to -0.918e6 for the four trained bases. Caveat in section 5.

Reading as stated beforehand: the notch, not the clamp, is the main source of the excess energy. The exact clamp matters for a single field only.

### Set 4 (POSTHOC4.md): inputs that separate the two faces of the notch

Three inputs added: theta / pi, sin(theta / 2), cos(theta / 2), with theta the angle around the centre of the notch root arc measured from the downward direction, branch cut inside the gap. Exact clamp as in set 3.

- F4a: one field, supervised: +0.37% (against +9.1% in F2b).
- F4b: the planned label-free objective, m = 8, 3,000 iterations: first Ritz frequency +0.34%; E1 displacement 3.6% (maximum 10.4%); E2 21% (maximum 58%); E1 acceleration 86%; modes 2 to 5 +313% to +442%. The first Ritz vector carries 18.5% of its strain energy near the notch, against 18.6% for the mode. Its remaining error energy is 30 times smaller than in the planned bases (0.0069 w1^2 against 0.18 to 0.21 w1^2) and now lies mostly in the clamp zone (52%).

### Set 5 (POSTHOC5.md): what stops modes 2 to 5?

- F5a: the planned objective for the lowest m eigenmodes: m = 1, -1,120,848; m = 2, -1,121,381; m = 3, -1,121,384; m = 4, -1,121,386; m = 6 and 8, -1,121,386. Mode 2 is worth 5e-4 of the objective; modes 3 to 8 together 4e-6.
- F5b: F4b with 12,000 iterations: first Ritz frequency +0.11%; E1 displacement 3.0% (maximum 10.2%); E2 19% (maximum 57%); E1 acceleration 86%; modes 2 to 5 +232% to +491%. Against the reading stated beforehand, training length is not the limit.
- F5c: E2 of the F4b basis by case: 46% to 58% for the three shortest pulses (duration 0.05 T1 to 0.09 T1, one with a large axial load), 9% to 35% for the others; modal truncation with 8 modes 0.7% to 2.5% on the same cases.

## 5. What the check shows (as written after set 5; items 2 to 4 superseded by section 7)

1. The code paths and the identities work. The computable bound holds and is 34 to 77 times too large in norm.
2. The planned network cannot represent the notch. Its inputs are smooth in position, so the two faces of the 0.1-wide notch receive nearly the same inputs while the structure lets them move apart; every subspace it produced was too stiff at the notch, by about the notch's own effect on the first frequency. Inputs built from the angle around the notch root remove this: the first mode is then learnt without labels to 0.1% to 0.4%. Imposing the clamp exactly matters for one field, not for 8 or 16.
3. The per-step objective weights each mode roughly by its share of the response energy in the training cases. Mode 1 dominates, so label-free training learns mode 1 and not modes 2 to 5, also with four times the iterations. Tip displacement then comes out within about 3% (median), but accelerations and the responses to short or axial pulses do not; modal truncation with the same number of fields is near exact.
4. Caveat on objective values (F3, F5a): along its own reduced trajectory each basis sees its own effective loads, so the value of the objective compares how much compliance a subspace captures, as a static potential does, rather than per-step error. A lower value is still better in that sense, but these values are not error measures.

## 6. Consequences for the study design (as written after set 5; superseded by section 7)

- Inputs: give the network the topology near notches (angle inputs at each notch root, computable from the geometry parameters), or use a representation that follows the mesh.
- Clamp: impose it exactly.
- Objective: the higher modes must be made worth learning, for example by greedy enrichment (one field at a time against what the fields already found leave unexplained, so that each new field sees a signal of order one), by training cases chosen to excite them, or by a different norm.
- Costs: on this model modal truncation is near exact and costs one 0.15 s eigensolve; a cost advantage of the network can only come from families of geometries and much larger models, which this check did not measure.
- Each of these changes needs its own pre-registered study on new seeds and geometries.

## 7. Re-examination (set 6, POSTHOC6.md with two addenda)

Ruifeng asked whether the failure was a coding error or too little training. Set 6 was written and hashed before running, with two addenda written and hashed before the runs they cover (PLAN_HASH.txt). S below is the sum over training cases and steps of ||u*_n - Phi c_n+1||_A^2 along the basis's own rollout: under the gradient stop it is the quantity whose gradient the training follows. T is the sum of ||u*_n||_A^2.

### 7.1 Code checks

- A1, the training objective against an independent full-space computation (numpy rollout, effective loads and step operator in full space): relative difference 7.0e-11 (planned m = 8, seed 0) and 1.4e-12 (F4b). Pass.
- A2, the training gradient. At the initialisation: 1.9e-14 against the gradient of an explicitly frozen-history functional; central finite differences converge as h^2 down to 1.4e-10 to 5.8e-10. Pass. At F4b: 3.8e-8 and, in the worst direction, 2.1e-5, which miss the stated tolerances (1e-10 and 1e-6). There the finite-difference error falls as h^2 down to h = 1e-4 and then rises: the loss is 1.1e6 and its gradient 1.5, so about six digits are lost to cancellation. I read this as rounding, not a coding error.
- For information: the derivative of the loss with the history allowed to move is 9 times the training gradient at the initialisation and 3,000 to 24,000 times at F4b, with the same sign. The loss value mostly tracks the amplitude of the model's own trajectory; it is not a measure of the per-step error the training reduces.
- A3, condition numbers of Phi_raw^T M Phi_raw: 66 to 896 for the planned bases, 557 (F4b), 1,014 (F5b), up to 15,560 (B1). No ill-conditioning.

### 7.2 What the training gradient rewards

- E3, S / T along each basis's own rollout: modal m = 8, 3.4e-9; F5b, 1.7e-7; F4b, 5.2e-7; planned m = 8 seed 0, 1.2e-5. The training's own measure prefers the modal subspace, by a factor of 50 over F5b.
- E1: for F5b, adding modes 2 to 5 to the span removes 3.4% of S, the three load responses 4.4%, both 5.8%. For modal m = 8, the load responses remove all of S: its per-step error is entirely load injection.
- E4: adding the exact first mode removes 22% (F5b), 26% (F4b) and 43% (planned) of S; modes 1 to 8, 98%, 99% and 97%; modes 1 to 16, 99% or more.
- E2, relative residual of the load responses (tip vertical, tip axial, top) outside the span: modal m = 8, 0.29, 0.59, 0.20; F5b, 0.78, 0.24, 0.35.

The sharper mechanism of POSTHOC6.md (training spends fields on the load responses) is not supported: the trained bases' per-step error lies in the span of the eight lowest modes.

### 7.3 Capacity, warm start, training length, and a balanced label-free objective

| Run | Ritz frequency errors, modes 1 to 5 | E1 displacement | E1 acceleration | E2 (max) | S / T |
|---|---|---|---|---|---|
| Modal, m = 8 | 0 | 0.014% | 1.3% | 1.3% (2.5%) | 3.4e-9 |
| B1, supervised to modes 1 to 8, 20,000 iterations | modes 1 to 8 all within 0.08% | 0.42% | 1.7% | 3.9% (4.3%) | 1.1e-7 |
| C1, B1 then per-step objective, 1e-4 | 0.06%, 0.05%, 0.63%, 0.46%, 2.7% | 0.37% | 2.9% | 3.6% (4.3%) | 9.4e-8 |
| C2, B1 then per-step objective, 1e-3 | 0.05%, 0.11%, 5.3%, 65%, 71% | 0.35% | 12% | 4.2% (12.8%) | 8.9e-8 |
| D1, per-step objective, 50,000 iterations, 1e-3 | 0.014%, 0.22%, 7.6%, 350%, 414% | 0.35% | 12% | 4.8% (13%) | 2.9e-8 |
| D2, per-step objective, 50,000 iterations, 3e-4 | 0.05%, 169%, 372%, 515%, 272% | 2.9% | 86% | 19% (57%) | 8.9e-8 |
| F1, log-determinant objective, 20,000 iterations | modes 1 to 8 all within 0.08% | 0.46% | 1.7% | 4.0% (4.4%) | 1.2e-7 |
| F2, F1 then per-step objective, 1e-4 | 0.06%, 0.05%, 0.69%, 0.48%, 2.1% | 0.38% | 2.9% | 3.6% (4.4%) | 9.6e-8 |

All runs: exact clamp and notch-angle inputs, m = 8, seed 0. E1 and E2 are medians over the ten evaluation cases.

- B2, one field with the planned inputs and the exact clamp, supervised as in F2b: 0.91% at 5,000 iterations, 0.31% at 10,000, 0.045% at 30,000 (F2b: +9.1% at 3,000).
- D1, modes over training: mode 2 went from +175% at 30,000 iterations to +0.39% at 35,000; mode 3 from +109% to 12%, then 7.6% at 50,000; modes 4 and 5 were not learnt. D2, with a three times smaller learning rate, learnt only mode 1.
- F1 needed no labels: within about 1% for modes 1 to 7 at 2,000 iterations, and its loss was still falling at 20,000.
- Training times on the two sandbox cores, with other jobs running: B1 377 s, F1 397 s, C and F2 108 s to 123 s, D1 and D2 about 2,500 s each (one thread each).

Readings against those stated beforehand:
- A1 pass. A2 pass at the initialisation; at F4b the stated tolerances were missed, for the rounding reason above. A3: no ill-conditioning.
- B1: modes 1 to 5 within 1%, so the network can represent the subspace. B2: well under 1%, so the notch failure of the planned inputs was largely training length.
- C1: modes 2 to 5 stay within a few per cent (largest 2.7%), so the objective holds them at a small learning rate. C2: the first steps at the larger learning rate knocked modes 4 and 5 out, and the objective did not bring them back within 3,000 iterations.
- D: the stated criterion (modes 2 to 5 within 5% and E2 below 5%) is met by neither run. D1 shows the higher modes arriving one at a time after long plateaus.
- E: (i) removes 3.4% and (ii) 4.4%, so neither explanation holds; every trained basis has a larger S / T than modal m = 8, by factors of 8.6 (D1) to 3,500 (planned).
- F1: modes 1 to 8 within 0.08% without labels, so a balanced label-free objective solves the optimisation problem. F2: the per-step objective keeps modes 1 and 2 and lets modes 3 to 5 drift by up to 2.1%; it improves tip displacement and E2 slightly and worsens tip acceleration.

### 7.4 What the re-examination shows

1. No coding error was found.
2. The planned failure was an optimisation problem: far too little effective training for the objective used.
   - The planned inputs can represent the notched first mode (B2); they learn it slowly. With the notch-angle inputs one supervised field reached 0.37% in 3,000 iterations (F4a); with the planned inputs it took about 10,000 to reach 0.31%.
   - The per-step objective is smallest near the modal subspace (E3), the network can represent that subspace (B1), and the objective holds it once found (C1, F2). From a random start, however, the objective is dominated by the accuracy of the first mode: D1, which lacks modes 4 and 5, has a smaller per-step error (S / T 2.9e-8) than B1 and F1, which have all eight modes within 0.1% (1.1e-7 and 1.2e-7). The higher modes therefore arrive one at a time after long plateaus (D1: mode 2 between 30,000 and 35,000 iterations; modes 4 and 5 not within 50,000).
3. A balanced label-free objective solves this. Ritz values bound the eigenvalues from above one by one (classical min-max), so the sum of the logarithms of the Ritz values, log det(Phi^T K Phi) - log det(Phi^T M Phi), is minimised by the span of the m lowest eigenvectors, and its gradient weights each Ritz value by its relative change. It uses only K and M. It learnt all eight modes to within 0.08% (F1). After per-step fine-tuning (F2): tip displacement 0.38%, tip acceleration 2.9%, E2 3.6% (maximum 4.4%), against 0.014%, 1.3% and 1.3% for modal truncation and 50% to 60%, about 90% and 65% to 74% for the planned runs.
4. The remaining gap to modal truncation is the accuracy of the mode shapes: Ritz frequency errors of 0.04% to 0.08% (F1) correspond to energy-norm shape errors of about 3% to 4% (the eigenvalue error is about the square of the energy-norm error), the size of E2. F1's loss was still falling when it stopped.
5. In this setting most of the accuracy comes from the Rayleigh-Ritz objective; the per-step potential adds a small refinement. The log-determinant objective must be checked against prior work on neural eigensolvers before it is claimed as new.
6. Caveats: one geometry, one seed per setting, all post hoc. A redesigned study needs its own pre-registration on new seeds and geometries.

### 7.5 Compute

- One iteration of the per-step objective (m = 8, 16 cases, 640 steps) takes about 25 ms on the two sandbox cores; the log-determinant objective is cheaper. The time goes into 640 sequential steps of 8 by 8 algebra, so a GPU would not shorten a single run. More cores help by running settings in parallel (seeds, learning rates, geometries).

### 7.6 Spectral weighting of the three objectives (set 7, POSTHOC7.md)

Exploratory: it checks a derivation and measures a sensitivity on the exact eigenvectors of this geometry; it involves no training. Written and hashed before running (2026-11-06 00:35:42 UTC).

- W1, the second-order formulas. For Phi = V_8 + W E, with W the eigenvectors 9 to 16 and E random of size eps, the excess of the trace objective over its minimum matches sum E_ji^2 (lambda_j - lambda_i): ratio 0.9978 to 0.9987 at eps = 1e-2, within 2.3e-5 of 1 at 1e-3 and within 1.9e-7 at 1e-4. The log-determinant excess matches sum E_ji^2 (lambda_j - lambda_i) / lambda_i once the relative Ritz errors are small: ratio 0.21 to 0.39 at eps = 1e-2, where the expected relative error of the first Ritz value is about 12 (the trace formula still holds there, and the trace barely registers that error); 0.90 to 0.96 at 1e-3; 0.9991 to 0.9998 at 1e-4. Both pass. To first order in the Ritz errors, the excess of sum f(theta_i) weights the relative error of mode i by lambda_i f'(lambda_i): by lambda_i for the trace, by 1 for the log-determinant.
- W2, the same 1% frequency error in each mode. Mode i is turned towards a fixed combination of modes 9 to 16 until its Ritz frequency is 1% high; only that frequency moves (checked: the others stay within 4e-12).

| Mode i | Trace excess | Log-det excess | Increase of S / T (16 training loads) | Ratio of mode 1's increase to mode i's |
|---|---|---|---|---|
| 1 | 1.68e-6 | 0.0199 | 1.10e-6 | 1 |
| 2 | 7.34e-5 | 0.0199 | 2.13e-8 | 52 |
| 3 | 4.37e-4 | 0.0199 | 1.88e-10 | 5,864 |
| 4 | 5.07e-4 | 0.0199 | 4.37e-10 | 2,522 |
| 5 | 1.55e-3 | 0.0199 | 2.76e-10 | 4,000 |
| 6 | 3.72e-3 | 0.0199 | -3.0e-11 | negative |
| 7 | 4.39e-3 | 0.0199 | 1.08e-10 | 10,191 |
| 8 | 7.38e-3 | 0.0199 | 8.84e-11 | 12,479 |

S / T of the exact eight-mode basis itself is 3.38e-9.

Readings against those stated beforehand: as expected for the trace (mode 8 counts 4,404 times as much as mode 1, which is lambda_8 / lambda_1) and the log-determinant (flat). The per-step objective falls as expected and more steeply than f = -1/theta would (that weights mode 1 4,404 times as much as mode 8; the per-step objective, 12,479 times). Not expected: for modes 3 to 8 the increase is below the per-step error of the exact eight-mode basis, and for mode 6 it is negative. On these loads a 1% error in mode 6 that brings in some content of modes 9 to 16 lowers the per-step error, so the per-step objective cannot be relied on to hold modes 3 to 8. This is consistent with the plateaus of D1 and with F2's drift of modes 3 to 5. For the trace, the table supports the reading offered for V2's lost first mode, which still has to be confirmed by a fair re-run.

## 8. Files

- Plans and records: PLAN.md, DEVIATIONS.md, POSTHOC.md to POSTHOC7.md, PLAN_HASH.txt.
- Code: fem_cantilever.py (geometry, mesh, matrices, Newmark), lb_model.py (network, loss, training), run_check.py (planned runs and evaluation), posthoc.py to posthoc7.py, make_compact.py. posthoc6.py was extended (commands e4, f1, f2) while D1 and D2 were running; the code of command d did not change.
- Results: results_compact.jsonl (sets up to 5) and results6_compact.jsonl (set 6), built by make_compact.py from the result files in out/. Commands: `python -I run_check.py prep`, `train M SEED`, `eval`; `python -I posthoc.py train V M` and `eval`; `python -I posthoc2.py`; `python -I posthoc3.py f1|f3|f2`; `python -I posthoc4.py`; `python -I posthoc5.py a|c|b`; `python -I posthoc6.py d|a|e|e4|b1|b2|c|f1|f2|eval` (see its docstring); `python -I posthoc7.py w1|w2` (results in out/posthoc7_w1.json and out/posthoc7_w2.json).
- Smoke test recorded: before D1 and D2, `posthoc6.py d smoke 1e-3 20` (20 iterations, loss and Ritz errors printed; output deleted).
