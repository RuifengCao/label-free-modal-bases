# Pre-registration: which label-free objective learns the vibration modes? Parts I and II

**Version 1, 6 November 2026, approved by the author on 6 November 2026 at 23:02 UTC; apart from this sentence, the file is the approved draft, whose hash is in PLAN_HASH.txt.** On approval, this file, the code it names and geometries.json are hashed into STAMP.txt and PLAN_HASH.txt; run_formal.py refuses to start the formal runs unless every hash matches. Exploratory work behind it: Phase 0 (PHASE0_PLAN.md, RESULTS_PHASE0.md) and Phases 0b to 0e (PHASE0B_PLAN.md to PHASE0E_PLAN.md, RESULTS_PHASE0B.md, TABLES_PHASE0B.md, DEVIATIONS_PHASE0.md). No formal geometry was trained on or evaluated in any of them, apart from smoke tests whose numbers were not read.

## 1. Question

For network-generated finite element reduced bases in linear structural dynamics, which label-free training objective learns the eight lowest vibration modes at a fixed training budget, and does the outcome follow the spectral weighting of the objectives: the per-step incremental potential weights a mode by its share of the trajectory, the trace by its eigenvalue, and the log-det equally in relative terms? Because Adam rescales the gradient of each parameter, the eigenvalue weighting should hurt mode 1 when the modes share their parameters and much less when each mode has its own network, unless the objective couples the modes itself (the trace, through the orthonormalisation). Part I tests this on three geometries, Part II on a family of geometries.

## 2. Fixed settings

- **Model problem**: plane-stress cantilever 10 x 1, E = 1, nu = 0.3, density 1, clamped at x = 0, with a U-shaped notch in the top edge (root radius 0.05); quadratic triangles, h_max = 0.2, refined at the notch. Newmark average acceleration; Rayleigh damping 2% at modes 1 and 3; dt = T_3/20; 640 steps. Per geometry: 16 training and 10 evaluation load cases (seeds 2026 and 2027; tip vertical, tip axial and distributed top load with random weights times a half-sine pulse of log-uniform width between T_3/2 and T_1/2; plus a tip pulse of width T_2/2 and a tip step). m = 8.
- **Networks**. Shared: one MLP with 4 hidden layers of 64 (tanh) and 2m outputs. One network per mode: 8 MLPs, each with 4 hidden layers of 24 (tanh) and 2 outputs (the x and y displacement of one mode), evaluated together in batched form (identical to evaluating them one by one, to round-off: Phase 0e check E2); 19,408 parameters against 15,056 for the shared network in Part I, 19,792 against 15,184 in Part II. Both: inputs are 20 planned features and 3 notch-angle features, plus in Part II the normalised notch position and depth ((x_c - 3)/2, (d - 0.25)/0.15); outputs multiplied by x/L (exact clamp); Cholesky M-orthonormalisation of the assembled basis.
- **Optimiser**: Adam (0.9, 0.999, 1e-8); learning rate cosine from 1e-3 to 1e-5 over the run. Initialisation: shared, PRNGKey(seed); one network per mode, PRNGKey(seed) split into 8 keys. The same initialisation for every objective at a given seed and architecture.
- **Objectives** (functions of the assembled basis, the same code for both architectures):
  - O1, per-step incremental potential, gradient stopped on the history (shared network only);
  - O2, trace of the projected stiffness of the M-orthonormalised basis;
  - O3, sequential Rayleigh quotients: column-by-column Gram-Schmidt in the M inner product with the earlier columns stopped, in the network's fixed column order (as NeuralEF);
  - O3s, the same after ordering the columns on each geometry by their own Rayleigh quotients with the gradient stopped (as Shape Space Spectra, Chang et al. 2025, which uses one network per eigenfunction, and Neural Eigenfluid Bases, Zhang et al. 2026, which uses one network);
  - O4, log det of the projected stiffness of the M-orthonormalised basis.
- **Geometries** (geometries.json; Sobol points from scipy.stats.qmc.Sobol(d=2, scramble=True, seed=s), x_c = 1 + 4u, d = 0.1 + 0.3v, rounded to four decimals): G1 to G3, seed 11 (first 3 of 4); family training, seed 12 (first 24 of 32); family held out, seed 13 (16).
- **Primary measure**: e_i = omega~_i / omega_i - 1, i = 1..8, the sorted Ritz frequencies of the final basis against the finite element frequencies of the same mesh.
- **Code**: phase1.py, with phase0.py, phase0b.py, phase0c.py, phase0d.py and phase0e.py (objectives, architectures, set-up, controls and evaluation, unchanged from the exploratory phases) and the finite element and model modules named in run_formal.py; started by run_formal.py on the author's laptop (Windows 11, i9-13900HX, CPU, float64), one single-threaded process per logical CPU 2, 4, 6 and 8. Last-digit differences from the sandbox are expected and are not deviations.

## 3. Part I: G1 to G3

Runs, 20,000 iterations each: shared network, O1, O2, O3 and O4 on G1, G2 and G3 with seeds 10 and 11 (24 runs); one network per mode, O2, O3 and O4 on the same geometries and seeds (18 runs).

| | Prediction, checked in each of the six geometry-seed runs | Phase 0 and 0c on G0 |
|---|---|---|
| H1 (shared O4 learns all eight) | max_i e_i <= 0.5% | 0.08%, 0.10%, 0.10% (seeds 0 to 2) |
| H2 (shared O1 misses higher modes) | max over i = 3..8 of e_i > 5% | 462%, 649%, 513% |
| H3 (shared O2 leaves mode 1 behind) | e_1 > 1% and e_1 > max over i = 3..8 of e_i | e_1 = 12.0%, 17.2%, 19.9%; others at most 0.16% |
| H3b (shared O3 leaves mode 1 behind) | e_1 > 0.5% and e_1 > max over i = 3..8 of e_i | e_1 = 3.66%, 3.01%, 5.11%; others at most 0.11% |
| H8 (per-mode O3 no longer does) | e_1 of per-mode O3 < e_1 of shared O3 on the same geometry and seed, divided by 3 | seeds 0 and 1: 0.37% against 3.66%, 0.25% against 3.01% (ratios 10 and 12) |
| H9 (per-mode O2 still leaves mode 1 behind) | e_1 > 1% and e_1 > max over i = 3..8 of e_i | seeds 0 and 1: e_1 = 26.4%, 22.4%; others at most 0.38% |
| H10 (per-mode O4 learns all eight) | max_i e_i <= 0.5% | seeds 0 and 1: 0.12%, 0.11% |

Verdict for each: supported if the prediction holds in all six runs (for H8, all six geometry-seed pairs); partly supported in four or five; not supported in three or fewer. H3b was added after Phase 0; H8 to H10 after Phase 0c.

## 4. Part II: the notched-cantilever family

Runs, with 4 of the 24 training geometries per iteration (a fresh random permutation of the 24 every six iterations, seeded by the run seed), loss = mean over the 4:

- shared network, 60,000 iterations: O4, seeds 0, 1, 2 (the method); O3s, seeds 0, 1, 2 (the objective of the geometry-family methods); supervised control, seeds 0, 1, 2: the same network fitted to the eight lowest eigenvectors of the training geometries (mean over the modes of the energy share outside the span; the B1 loss of set 6);
- one network per mode, 30,000 iterations (about the training time of 60,000 shared iterations): O4 and O3s, seeds 0, 1, 2.

Controls without training, on every held-out geometry: modal truncation (its own 8 eigenvectors); a mapped reduced basis (the 8 eigenvectors of all 24 training geometries, carried to the held-out geometry by the notch-aligning map of Phase 0d, then Rayleigh-Ritz on the held-out geometry with Gram truncation 1e-13; phase0d.rbm_basis); a frequency regression (each of the 8 finite element frequencies of the training geometries interpolated over the normalised notch position and depth with a thin-plate-spline radial basis function, degree 1, no smoothing; frequencies only). The two classical controls and the supervised control use eigen-solutions of the training geometries; the networks trained with O4 and O3s do not. The controls are reported in full whatever they show; no hypothesis is attached to them.

Measures on the 16 held-out geometries, for each seed of shared O4 (H4 to H7) and of per-mode O4 and O3s (H7sep):

| | Prediction (each seed) | Phase 0b and 0e, pilot 2 (24 training, 16 held out, seed 0) |
|---|---|---|
| H4 | median over the geometries of the median of e_1..e_5 <= 1% | 0.13% |
| H5 | median over the geometries of the reproduced notch drop of omega_1 >= 90% | 96.4% |
| H6 | median over the geometries of E2 (median over each geometry's 10 evaluation cases) <= 10% | 6.2% |
| H7 | median over the geometries of e_1 for shared O3s > the same for shared O4 (same seed) | 46.1% against 0.18% |
| H7sep | median over the geometries of e_1 for per-mode O3s > the same for per-mode O4 (same seed) | 0.71% against 0.29% (Phase 0e, 30,000 iterations) |

Reproduced drop = (omega_plain - omega_1,net) / (omega_plain - omega_1,FE), omega_plain the un-notched cantilever on the same mesh rules. E2 = the square root of the sum over all steps of the squared stiffness-norm error of the reduced displacement against the full model, divided by the same sum for the full model's displacement. E1 = the relative L2 error over time of the tip displacement and of the tip acceleration histories.

Verdict for each: supported if it holds for all three seeds; partly supported for two; not supported for one or none. The thresholds of H4 to H6 come from pilot 2 by the rule fixed in PHASE0B_PLAN.md before pilot 2 ran: the pilot values (0.13%, 96.4%, 6.2%) lie inside the bounds under which the planned thresholds are kept (0.67%, 93.3%, 6.7%), so all three are kept. H7sep was added after Phase 0e.

## 5. Reported without thresholds

Part I: e_1..e_8 of every run; E1 (tip displacement and acceleration); E2; S/T on the training cases; sin^2 of the M-angle between each exact eigenvector and the span; the four objectives' values at each basis; the conditioning of the raw basis; learning curves every 1,000 iterations; wall time; the same quantities compared between architectures. Part II: every per-geometry number for every run and control (including the controls' reproduced drop, E1 and E2 where defined, and the mapped reduced basis' rank and share of nodes mapped into a notch void); the training geometries (e_i and the reproduced drop); each held-out geometry's distance to its nearest training geometry and whether it lies inside the training hull; the held-out geometries whose notch drop is below 1%; learning curves every 5,000 iterations; wall time per run and per architecture.

## 6. Rules

- Thresholds, geometries, seeds, budgets and code are fixed by the stamp. Any change after it is written into DEVIATIONS with its reason before it is made, and reported.
- No run is excluded. A run that fails or is cut off is rerun once from the same seed (run_formal.py does this and records it); a second failure is reported as such.
- Every hypothesis is reported, supported or not, with the per-run numbers. Analyses beyond this list are labelled exploratory.

## 7. Where each item is computed

| Item | Code (phase1.py) | Output |
|---|---|---|
| Part I runs and their evaluation | cmd_p1 (shared: P.make_loss; per mode: PE.lossv on PE.stack_sep(PC.init_sep)); p6.train_logged; P.eval_basis; p6.S_split; P.objectives_np | results/p1_{G}_{O}_s{seed}.json, results/p1sep_{G}_{O}_s{seed}.json |
| e_i | p6.ritz_errors (sorted Ritz frequencies against fc.eig) | field net.ritz |
| H1 to H3b, H8 to H10 | verdicts_from (tested on made-up inputs in test_verdicts.py) | results/verdicts.json, part1 |
| Part II training | cmd_p2 (shared O4: P.o4_loss; shared O3s: PB.o3s_loss; supervised: _sup_loss; per mode: PE.lossv) | results/p2_{O}_s{seed}.pkl, results/p2sep_{O}_s{seed}.pkl |
| Part II measures and controls | cmd_e2: P.eval_basis, _drop; PD.sources, PD.rbm_basis (mapped reduced basis); RBFInterpolator (frequency regression) | results/e2_shard{k}.json: o4_s{seed}, sep_o4_s{seed}, rbm_all, freq_rbf, ... |
| H4 to H7, H7sep | verdicts_from | results/verdicts.json, part2 |
| Training geometries | cmd_e2train | results/e2_train.json |
