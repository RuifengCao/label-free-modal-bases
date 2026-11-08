# Exploratory runs after the formal runs: results

**Status.** Exploratory (EXPLORE_PLAN.md, hashed 7 November 2026 at 17:42:57 UTC; changes and additions in DEVIATIONS_EXPLORE.md, items 1 to 17). No hypotheses, thresholds or verdicts. X1 and X2 ran on the laptop on 7 November 2026 from 19:37:32 to 22:33:23 UTC (21 jobs, each finished at the first attempt); X3 ran in the sandbox from the saved formal networks and outputs. Every number below comes from the output files named with it; the tables are in TABLES_EXPLORE.md (written by x12_report.py), the figures in explore/figs. Added on 8 November 2026 at 01:57 UTC: section 1.3, a control run in the sandbox after the laptop runs (DEVIATIONS_EXPLORE.md items 18 to 22), the pointer to it at the end of the third point of section 1.1, and a second green line in figure 7 (item 22); nothing else in this file changed.

## 0. Laptop smoke test (7 November 2026, 19:31:45 to 19:32:41 UTC; outputs not used)

All 6 jobs finished at the first attempt (explore/results_smoke/progress.txt); the file hashes in environment.json match PLAN_HASH.txt (explore.py 372becc5..., run_explore.py acc9fa0d..., EXPLORE_PLAN.md 8e829ba3...), and the 37 input hashes (formal code against STAMP.txt, formal networks and records against SHA256SUMS_formal_results.txt) matched. On this machine, the one that ran the formal evaluation, the checks at h_max = 0.2 on two held-out geometries gave: streaming against phase0.eval_basis 1.7e-11 and 1.4e-11; Ritz errors against the formal record, absolute 8.3e-12 and 3.8e-12, relative 9.7e-9 (frequency regression) and 3.9e-9 (mapped reduced basis); finite element frequencies 7.7e-14 and 5.1e-14; E1 and E2 against the formal record 5.9e-9 and 1.2e-8 (largest for the mapped reduced basis). Timing paths: the compiled basis with mesh arguments and the NumPy basis equal lb.orthonormal_basis to 7.6e-14 and 4.6e-14. Nothing was written into formal/ or the parent folder.

## 1. X1, mesh refinement and online cost

Outputs: explore/results (x1_acc_shard0 to 3, x1_fine0 to 3, x1_timing), checksums in SHA256SUMS_explore_results.txt (verified on the copies used here). Run on the laptop with the same environment as the formal runs (environment.json).

### 1.0 The check at h_max = 0.2, read first

The plan asks that any relative difference above 1e-8 in a Ritz error, between this run at h_max = 0.2 and the formal record, be reported and explained before anything else is read. It was read first (23:51 UTC).

- Absolute differences of the Ritz errors, which are the relative differences of the Ritz frequencies: at most 6.7e-11 over the 16 geometries and 18 bases (networks at most 4.3e-13; modal truncation 3.7e-11; mapped reduced basis 6.7e-11; frequency regression 4.4e-13).
- Relative differences of the errors themselves above 1e-8: 12 cases, 11 for the mapped reduced basis (largest 2.0e-7, on 3.1654_0.2769, for an error of 3.4e-4 and an absolute difference of 6.7e-11) and 1 for the frequency regression (3.2e-8 on 3.3505_0.1723; absolute 4.4e-13).
- **Explanation.** Both bases are built from the eigen-solutions of the 24 training geometries, and fem_cantilever.eig starts ARPACK from a random vector, so the source eigenvectors differ at round-off from one run to the next; the mapped reduced basis passes them through the Gram matrix of 192 vectors, which amplifies the difference. Repeating these random-start eigen-solves twice within one process in the sandbox (check_h02/repeat_sources.py, three of these geometries) gives differences of the same size: mapped reduced basis up to 1.3e-11 absolute and 5.3e-8 relative; frequency regression up to 2.1e-13 and 5.6e-8; modal truncation 2.4e-11 absolute. The networks, which do not use these eigen-solves, agree with the formal record to 4.3e-13. So the code path at h_max = 0.2 is that of the formal evaluation, as intended.
- Other checks: streaming against phase0.eval_basis at most 4.6e-11; finite element frequencies 4.0e-13; E1 and E2 against the formal record at most 4.6e-7 (modal truncation; networks 2.9e-8), the same round-off carried through the time integration.

### 1.1 X1a, accuracy on refined meshes

All networks were trained, and all sources of the mapped reduced basis computed, at h_max = 0.2; each basis is evaluated against the finite element solution of the mesh it is used on. Median degrees of freedom: 7,366 (h 0.2), 23,352 (0.1), 76,893 (0.05); about 265,000 at 0.025 on the four FINE geometries.

- **The networks keep their accuracy.** The largest frequency error (max over modes 1 to 8) grows by a median 3% from h 0.2 to 0.05 for every network (shared O4 2.54e-3 to 2.61e-3, ratio range 1.01 to 1.10; supervised 1.01 to 1.07; one network per mode, O4, 1.00 to 1.06). On the FINE geometries the growth from 0.2 to 0.025 is 2 to 6% for O4 seed 0. E1 and E2 change by similar amounts (shared O4: E2 0.0619, 0.0629, 0.0635; E1 0.0115 to 0.0121), the largest sin^2 not at all, and the reproduced drop of omega_1 stays at 0.97 (Tables X1a.1 to X1a.3; figure 7).
- **The finite element frequencies hardly change** between h 0.2 and 0.05: median -1.3e-4 for omega_1, at most 3.6e-4 over modes 1 to 8. The formal evaluation at h 0.2 was therefore not limited by the mesh: every basis's error is larger than the discretisation change.
- **The mapped reduced basis with sources at h 0.2 loses accuracy as the mesh is refined:** the largest frequency error grows by a median 4.5-fold from h 0.2 to 0.05 (3.0 to 6.5), and 3.6- to 7.5-fold to h 0.025 on the FINE geometries; E2 grows from 0.038 to 0.079 and E1 from 0.0044 to 0.0193. It is below the O4 networks (seed median) in largest frequency error on 14 of the 16 geometries at h 0.2, 10 at 0.1 and 8 at 0.05, and in E2 on 14, 9 and 5. Likely cause, not tested: the sources are interpolated linearly from the coarse mesh, and on a finer mesh the kinks of that interpolant add spurious strain energy. Sources computed on each evaluation mesh would remove this, at the cost of fine-mesh eigen-solves for the 24 training geometries. [Added 8 November 2026: tested afterwards in section 1.3; with sources on the evaluation mesh most of the loss goes.]
- The frequency regression's error grows by 3% (it predicts the coarse-mesh frequencies), and the shared O3s networks stay failed (largest error 1.9).

### 1.2 X1b, online cost

One core, single-threaded libraries; median over the four FINE geometries of the median of three repeats (Table X1b; figure 6). Growth exponents are slopes of log time against log DOFs over the four meshes.

| Step | 7,356 DOFs | 23,352 | 76,980 | 265,196 | exponent |
|---|---|---|---|---|---|
| Mesh and assembly (fem_cantilever.build) | 0.044 s | 0.38 s | 1.26 s | 5.16 s | 1.29 |
| 8 eigenpairs, shift-invert | 0.041 s | 0.20 s | 1.01 s | 5.22 s | 1.35 |
| Mapped reduced basis | 0.099 s | 0.27 s | 0.84 s | 2.92 s | 0.94 |
| Network O4: inputs + compiled network + reduced operators | 0.009 s | 0.033 s | 0.107 s | 0.383 s | 1.04 |
| Network O4, first call (compilation included) | 0.19 s | 0.21 s | 0.23 s | 0.37 s | 0.17 |
| Full simulation: factorisation and 640 Newmark steps | 0.47 s | 2.4 s | 11.3 s | 55.4 s | 1.33 |
| Reduced simulation, 640 steps | 0.013 s | 0.013 s | 0.013 s | 0.013 s | 0.00 |

- The network route is 4.5 times faster than the eigen-solve at 7,356 DOFs and 13.6 times at 265,196, and 8 to 11 times faster than the mapped reduced basis. In NumPy without compilation it takes about twice the compiled time.
- But every route needs the mesh and the assembled matrices, and assembly costs about as much as the eigen-solve. From mesh to reduced model, the network route is therefore 1.4 to 1.9 times faster than the eigen-solve route, not 5 to 14 times.
- Offline: one shared O4 network took about 5,100 s of training on one core (formal Part II log); the 24 sources of the mapped reduced basis take 2.4 s. Derived, not measured: the training is repaid against the eigen-solve after about 160,000 new geometries at h_max 0.2 and about 1,000 at 265,000 DOFs (the saving per geometry is 32 ms and 4.8 s).
- Against the expectations written before running: the network route grows linearly (exponent 1.04), as expected, but it is 5 to 13 times cheaper than mesh and assembly, not "about as much"; the eigen-solve grows faster (1.35), as expected, yet at 265,000 DOFs it still costs no more than assembly; at about 7,000 DOFs every step is below a second (largest: the full simulation, 0.47 s).
- The timed bases equal lb.orthonormal_basis to 2.3e-13 (compiled with mesh arguments) and 2.7e-13 (NumPy).

### 1.3 Control: the mapped reduced basis with its sources on the evaluation mesh (added 8 November 2026)

Approved by the author on 8 November 2026 at 00:08 UTC and written down before any computation as item 18 of DEVIATIONS_EXPLORE.md; items 19 to 21 record two failed attempts and what was changed before each rerun, item 22 the change to figure 7. Run in the sandbox (third attempt; two processes, each doing h_max 0.1 and then 0.05 for 8 of the 16 held-out geometries), started after 00:59:00 UTC, when its code was hashed, and finished at 01:45:14 UTC. Outputs: explore/rbfine (rbfine_h0.1_part0 and part1, rbfine_h0.05_part0 and part1; summary.json from rbfine_report.py; table.md and table.json from rbfine_table.py), all hashed in PLAN_HASH.txt; the outputs of the second attempt are kept in rbfine/attempt2 and not used. The values of the networks and of the basis with the h_max 0.2 sources are the laptop X1a values at the same mesh sizes.

- **Checks.** The basis with the h_max 0.2 sources, recomputed in the sandbox on two held-out geometries per mesh size, agrees with the laptop: Ritz errors to 9.5e-11 at h 0.1 and 2.9e-10 at 0.05 (absolute), E2 to 1.2e-9 and 1.0e-8 (relative); the finite element frequencies of the two machines differ by at most 3.8e-11 and 1.1e-10 (relative). Mapped points with a non-finite interpolated value (items 19 to 21): 1 to 5 per geometry at h 0.1 and 6 to 22 at 0.05, over the 24 sources; every one was resolved by interpolating again 1e-9 inside the domain, and the nearest-node value was never needed. One such point lies in the cross-check basis on 2.6263_0.3995 at both mesh sizes; that basis still agrees with the laptop, where no point needed it, to 9.5e-11 and 2.9e-10 (with the nearest-node value of the second attempt the difference at h 0.1 was 2.4e-5).
- **Result** (table below; figure 7, dashed green line). With its sources on the evaluation mesh, the mapped reduced basis keeps its h_max 0.2 accuracy at 0.1 (largest frequency error, median 7.51e-4 against 6.75e-4; per geometry a median 1.03 times its h 0.2 value, range 0.39 to 1.67) and loses some at 0.05 (1.06e-3; a median 1.38 times, range 0.22 to 2.63). With the sources at h_max 0.2 it was 2.4 times (1.4 to 9.7) and 2.7 times (1.6 to 24.9) worse than this on the same meshes. E2 and E1 follow: E2 0.0384, 0.0401, 0.0469 at h 0.2, 0.1, 0.05, against 0.0644 and 0.0794 with the 0.2 sources; E1 0.0044, 0.0049, 0.0066, against 0.0126 and 0.0193.
- **Against the O4 networks.** The basis with sources on the evaluation mesh is below O4 (seed median) in largest frequency error on 14 of the 16 geometries at every mesh size, with the same two exceptions as at h 0.2: 3.3505_0.1723 (2.05e-3 and 2.13e-3 at h 0.1 and 0.05, against 1.27e-3 and 1.31e-3) and 4.0451_0.1017, outside the hull of the training geometries (1.4e-2 and 1.7e-2, against 2.2e-3: O4 is 6.4 and 7.6 times more accurate there, 3.9 times at h 0.2). In E2 it is below O4 on 13 and 12 geometries (the further exceptions are 4.6308_0.3736, outside the hull, and at 0.05 4.4704_0.2994, 0.0539 against 0.0535). In the medians it is 3.4 and 2.5 times below O4 in largest frequency error and 1.6 and 1.4 times in E2, at h 0.1 and 0.05.
- **Against the expectation written before running** (item 18: back to a largest frequency error near 7e-4 and E2 near 0.04, and an offline cost that grows like the eigen-solves): met at h 0.1; at 0.05 only in part, with a median loss of 1.4 times and a wide spread. The cause of the remaining loss was not tested. The rank of the basis does not account for it in any obvious way: the Gram matrix keeps all 192 vectors at h 0.1 and 187 to 191 at 0.05 (tolerance 1e-13, as in the formal runs), and the geometry with rank 187 has a ratio of 0.55.
- **Offline cost** (sandbox; the two processes ran at the same time, so only ratios are used). The 24 sources took 3.7 s at h 0.2 (four values, 3.67 to 3.79 s), 14.6 and 15.0 s at 0.1 and 75.2 and 75.9 s at 0.05 (one value per process): 4.0 and 20.3 times the h 0.2 cost, a growth exponent of 1.28 in the median degrees of freedom of the held-out geometries from h 0.2 to 0.05 (on the laptop, Table X1b: eigen-solve 1.35, mesh and assembly 1.29). As expected, the cost grows like the eigen-solves. Scaled to the laptop's 2.4 s at h 0.2 (derived, not measured): about 10 s at 0.1 and 50 s at 0.05, at most about 1% of the training of one shared O4 network (about 5,100 s). Mapping the basis onto one held-out geometry took 0.49 s with the fine sources against 0.57 s with the 0.2 sources at h 0.1, and 1.82 s against 1.29 s at 0.05 (sandbox).
- **What this changes.** Section 1.1 attributed the loss of the mapped reduced basis on refined meshes, untested, to its coarse sources; the control confirms this for most of the loss, and figure 7 now shows both versions. With sources on the evaluation mesh, the mapped reduced basis stays more accurate than every network in the medians of the largest frequency error, E2 and E1 on all three meshes, for an offline cost far below training. In this problem the eigen-solves of the 24 training geometries that it needs are cheap, so the label-free networks gain nothing that matters offline from avoiding them. What remains on the networks' side in these runs is one of the four geometries outside the hull, 4.0451_0.1017 (on the other three the basis is still below O4 in largest frequency error), and the online cost: the network route is 8 to 11 times faster than the mapping with the 0.2 sources (section 1.2), and with sources on the evaluation mesh the mapping took 0.9 and 1.4 times as long at h 0.1 and 0.05 (sandbox).

Table for section 1.3 (written by rbfine_table.py into explore/rbfine/table.md). Networks and the basis with the h_max 0.2 sources: laptop X1a; the basis with sources on the evaluation mesh: sandbox, third attempt. At h_max 0.2 the two bases are the same.

| Quantity (median over the 16 held-out geometries unless stated) | h_max 0.2 | 0.1 | 0.05 |
|---|---|---|---|
| Largest frequency error, mapped reduced basis, sources on the evaluation mesh | 6.75e-04 | 7.51e-04 | 1.06e-03 |
| Largest frequency error, mapped reduced basis, sources at h_max 0.2 | 6.75e-04 | 1.99e-03 | 3.00e-03 |
| Largest frequency error, shared O4 (geometries and seeds) | 2.54e-03 | 2.58e-03 | 2.61e-03 |
| E2, mapped reduced basis, sources on the evaluation mesh | 0.0384 | 0.0401 | 0.0469 |
| E2, mapped reduced basis, sources at h_max 0.2 | 0.0384 | 0.0644 | 0.0794 |
| E2, shared O4 (geometries and seeds) | 0.0619 | 0.0629 | 0.0635 |
| E1, mapped reduced basis, sources on the evaluation mesh | 0.0044 | 0.0049 | 0.0066 |
| E1, mapped reduced basis, sources at h_max 0.2 | 0.0044 | 0.0126 | 0.0193 |
| E1, shared O4 (geometries and seeds) | 0.0115 | 0.0119 | 0.0121 |
| Geometries (of 16) where the largest frequency error of the basis with sources on the evaluation mesh is below O4 (seed median) | 14 | 14 | 14 |
| Geometries (of 16) where the largest frequency error of the basis with sources at h_max 0.2 is below O4 (seed median) | 14 | 10 | 8 |
| Geometries (of 16) where the E2 of the basis with sources on the evaluation mesh is below O4 (seed median) | 14 | 13 | 12 |
| Geometries (of 16) where the E2 of the basis with sources at h_max 0.2 is below O4 (seed median) | 14 | 9 | 5 |
| Largest frequency error, sources on the evaluation mesh, relative to h_max 0.2: median |  | 1.03 | 1.38 |
| same: smallest |  | 0.39 | 0.22 |
| same: largest |  | 1.67 | 2.63 |
| Largest frequency error, sources at h_max 0.2 over sources on the evaluation mesh: median |  | 2.40 | 2.66 |
| same: smallest |  | 1.42 | 1.62 |
| same: largest |  | 9.73 | 24.91 |
| 4.0451_0.1017 (outside the hull), largest frequency error, sources on the evaluation mesh | 8.47e-03 | 1.41e-02 | 1.67e-02 |
| 4.0451_0.1017, shared O4 (seed median) | 2.16e-03 | 2.19e-03 | 2.21e-03 |
| Median degrees of freedom of the 16 held-out geometries | 7366 | 23352 | 76893 |

## 2. X2, longer training of shared O2 and O3

Twelve runs of 100,000 iterations (G1 to G3, seeds 10 and 11), same code path as the formal Part I runs; outputs x2_*.json and .pkl. Table X2 lists every run; figure 8.

| | Formal, 20,000 iterations: median e_1 | 100,000 iterations: median e_1 | below 0.5% | below 1% | median E2 (formal) | training time per run |
|---|---|---|---|---|---|---|
| O2 trace | 1.66e-1 | 5.42e-3 | 2 of 6 | 6 of 6 | 0.107 (0.99) | 1,640 to 2,060 s (formal 370 to 410 s) |
| O3 sequential Rayleigh quotients | 7.03e-2 | 2.54e-3 | 5 of 6 | 6 of 6 | 0.072 (0.55) | 2,220 to 2,410 s (formal 430 to 470 s) |

- e_1 keeps falling, as expected. For O2 it falls at every log in all six runs. For O3 it falls overall but not steadily: there are temporary rises in every run, up to 2.24 at 10,000 iterations (G1, seed 11) and 0.26 at 35,000 (G2, seed 10).
- The higher modes are fine: the largest of e_3 to e_8 at 100,000 iterations is 2.0e-5 to 1.5e-4.
- Even with five times the budget, mode 1 stays 4 to 9 times less accurate than O4 after 20,000 iterations (median e_1 5.8e-4, formal Part I), and E2 stays above O4's 0.036. This supports the formal reading of O2 and O3 as slow for mode 1 rather than unable to reach it.
- Because each run's cosine schedule spans its own length, the 100,000-iteration run at 20,000 iterations is not comparable with the end of the formal run; it is shown in figure 8 only for orientation.

## 3. X3, figures and the shape-error relation

Scripts: x3_analysis.py (data), x3_figures.py (figures 1 to 5). Outputs: x3/x3_shape.json, x3/x3_fig3.json, x3/x3_fig4.npz, x3/x3_notch.json; figures in explore/figs (first versions, with other colours, in x3/figs; item 17). Mesh h_max = 0.2 throughout, as in the formal runs.

### 3.1 Checks

- The Ritz errors recomputed in the sandbox equal the formal values (laptop) to 7.3e-11 at most (absolute difference of the relative frequency errors), over the 16 held-out geometries and the seven bases analysed; for the Part I network on G1, to 3e-11.
- Modal energy shares (item 11): for modal truncation the prediction is an identity, sqrt(1 - sum over i <= 8 of p_i) = E2 of modal_m8; it holds to a relative 2.4e-6 over the 160 cases.
- Element strain energy densities (items 13, 14 and figure 4) sum to D^T K D to a relative 1.2e-13.

### 3.2 Figures

1. **Part I learning curves** (fig1). Median e_1 over the six geometry-seed runs at 20,000 iterations: shared O1 4.3e-4, O4 5.8e-4, O3 7.0e-2 (runs from 4.0e-2 to 7.2e-1), O2 1.7e-1; one network per mode O4 6.9e-4, O3 3.8e-3, O2 3.1e-1. O1 and O4 are below 1.5e-2 after 1,000 iterations; O2 and O3 are still above 1 at that point with the shared network.
2. **Part II, largest frequency error on each held-out geometry** (fig2). Median over the 16 geometries: mapped reduced basis 6.8e-4, supervised network 2.2e-3, shared O4 2.4e-3, frequency regression 5.3e-3. The mapped reduced basis is below the O4 median on 14 of the 16 geometries (not on 4.0451_0.1017, outside the training hull, where it reaches 8.5e-3, nor on 3.3505_0.1723). The frequency regression is below the O4 median on 3 of 16. The supervised network and O4 are within a factor 0.72 to 1.22 of each other.
3. **Tip displacement histories** on 3.1654_0.2769 (fig3). At the scale of the response the full model, O4 seed 0 and the mapped reduced basis coincide. The difference grows with time and reaches 1.3% (O4) and 0.36% (mapped reduced basis) of the peak tip displacement in the tip pulse case by about two periods of mode 1; E1 of the tip displacement is 0.88% and 0.25% (tip pulse) and 0.84% and 0.24% (first random case).
4. **First four modes of G1, finite element against the Part I O4 network** (fig4, seed 10). The shapes are indistinguishable by eye; e_1 to e_4 = 5.8e-4, 4.2e-4, 6.4e-4, 4.7e-4 and eps_1 to eps_4 = 0.034, 0.029, 0.036, 0.031. Of the energy of the difference (Ritz vector minus mode), 98, 92, 99 and 88% lie within 0.3 of the notch root.
5. **Shape errors and E2** (fig5), below. Figures 6 to 8 belong to X1 and X2 (sections 1 and 2).

### 3.3 Shape errors against frequency errors

For each Ritz pair of a basis, matched to the exact modes by order, eps_i = ||vt_i - v_i||_K / ||v_i||_K against sqrt(delta_i), delta_i = theta_i / lambda_i - 1 (16 held-out geometries, modes 1 to 8):

| Basis | Ratio eps_i / sqrt(delta_i): median | 10% | 90% | largest |
|---|---|---|---|---|
| Shared network O4, three seeds (planned; 384 pairs) | 1.012 | 1.000 | 1.149 | 1.52 |
| Supervised network, three seeds (added; 384) | 1.012 | 1.000 | 1.125 | 1.43 |
| Mapped reduced basis (added; 128) | 1.001 | 1.000 | 1.008 | 1.32 |

The relation eps_i ≈ sqrt(delta_i) holds closely for all three; theory gives eps_i >= sqrt(delta_i) to leading order, with equality when the error lies in modes of much higher frequency, which is the case here. Since the frequency error is e_i ≈ delta_i / 2, a frequency error of 0.2% goes with a shape error of about 6% in the energy norm. Median eps_1: 0.059 (O4), 0.037 (mapped reduced basis).

### 3.4 E2 explained mode by mode

The reference response's energy lies almost entirely in mode 1 (median share 0.967; mode 2: 0.031; above mode 8: 1.7e-4). Ratio of the predicted to the formal E2, over all evaluation cases (16 geometries x 10 cases per basis):

| Basis | Shape errors only (item 11): median (10% to 90%) | With amplitude and phase errors (item 12): median (10% to 90%); smallest, largest |
|---|---|---|
| O4, three seeds | 0.977 (0.888 to 0.991) | 1.001 (0.994 to 1.004); 0.925, 1.047 |
| Supervised, three seeds | 0.977 (0.886 to 0.991) | 1.001 (0.995 to 1.005); 0.935, 1.052 |
| Mapped reduced basis | 0.993 (0.968 to 1.007) | 1.001 (1.000 to 1.011); 0.998, 1.056 |

Median shares of the predicted E2^2 for O4: shape errors 0.90, amplitude and phase 0.05, above mode 8 0.04, cross term below 0.001 (mapped reduced basis: 0.86, 0.02, 0.11). The shape-only prediction misses by more than 15% in 28 of the 480 O4 cases; those carry a large share of mode 2 (median 0.26), are on the five geometries with the notch at x_c above 3.9, and there the amplitude and phase part is a median 41% of E2^2 (at most 59%), consistent with a phase error accumulating over the about 12 periods of mode 2 in the window. The largest remaining misfits of the second prediction (0.925 and 1.047) are in the tip step case, where coupling between modes, ignored by both predictions, matters most.

So the E2 of about 6% of the O4 networks, against frequency errors of about 0.1 to 0.3%, is mainly the energy-norm shape error of mode 1, eps_1 ≈ sqrt(2 e_1); phase errors add a smaller part, large only in cases dominated by mode 2. This replaces the qualitative explanation in RESULTS.md section 7.4, item 4, with a quantitative one, and it applies to the mapped reduced basis in the same way.

### 3.5 Where the shape error lies

- Part I network on G1 (fig4): 88 to 99% of the error energy of modes 1 to 4 within 0.3 of the notch root.
- Part II networks on the 16 held-out geometries (items 13 and 14), modes 1 to 8: within 0.3 of the notch root (about 2.5% of the area), median share 0.25 for O4 (0.29 for mode 1; 10% quantile 0.15), 0.25 for the supervised network, 0.04 for the mapped reduced basis. For mode 1 of O4, a median 0.51 of the error energy lies in the band of width 1.0 along the beam that contains the notch, 0.14 in 0 <= x < 1 (0.07 within x < 0.3 of the clamp), and little beyond x = 5. The family networks therefore spread their error over the clamped half of the beam, with a concentration at the notch about ten times its area share, while the single-geometry network concentrates it at the notch root.
