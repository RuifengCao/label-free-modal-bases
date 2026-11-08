# Phase 0e (exploratory): one network per mode on the geometry family

Written 6 November 2026, about 18:35 UTC, before any Phase 0e run. Exploratory; no formal geometry is used.

## Why

What was seen before writing this plan. Phase 0b (k): O3s (sequential Rayleigh quotients after per-geometry sorting) with the shared network on pilot 2 left the first mode far off on the held-out geometries (median first-frequency error 46%, median drop reproduction -11). Phase 0c on G0 (seed 0, still running): with one network per mode, O3 ended at 20,000 iterations with errors of 0.37% (mode 1), 0.33% (mode 2) and below 0.1% for modes 3 to 8, against 3.66% for mode 1 with the shared network; O3s with one network per mode was at 0.63% for mode 1 after 14,000 iterations.

So the architecture matters for O3 and O3s. Shape Space Spectra (Chang et al. 2025), whose sorting O3s copies, uses one network per eigenfunction. A Part II comparison of O4 against O3s on the shared network only would invite the objection that O3s was run on an architecture that disadvantages it. Phase 0e repeats the pilot-2 family training with one network per mode, for O3s and for O4, so that the pre-registration can compare like with like.

## Runs

Pilot 2 exactly as Phase 0b (j) and (k) (24 training geometries, Sobol seed 31; 16 held-out geometries, seed 32; 60,000 iterations; 4 geometries per step, a new permutation every 6 steps; Adam 1e-3 cosine to 1e-5; seed 0), with the architecture of Phase 0c: 8 MLPs with 4 hidden layers of 24 (tanh) and 2 outputs each, inputs 'hard_angle' plus the normalised notch position and depth (as `fam_problem`), exact clamp, PRNGKey(seed) split into 8 keys. Objectives: O3s and O4 (`phase0c.loss_from_raw`). One run each, in parallel on the two cores, after Phase 0c seed 0 and Phase 0d; the Phase 0c seed-1 runs move after these.

## Check before training

E1: on the first pilot-2 training geometry with family inputs, at the seed-0 initialisation, the JAX values of O3 and O4 on the assembled basis agree with `phase0.objectives_np` (relative 1e-9), O3s agrees with the trace, and the gradient of O4 and of O3s is nonzero for every one of the 8 MLPs. A smoke run (20 iterations, 2 evaluation geometries, own folder) of `fam` and `evalfam` before the long runs.

## Evaluation

As Phase 0b evalfam2 and merge, on the 16 held-out geometries (modal truncation and the two networks; Ritz errors, sin^2, E1, E2, drop reproduction, coverage) and on the 24 training geometries (Ritz errors). Compared with the shared-network results of Phase 0b (net_o4: H4 measure 0.13%, H5 measure 96.4%, H6 measure 6.2%; net_o3s: 2.15%, -1106%, 138%) and with Phase 0d.

## What the results are for

Descriptive. Proposed to Ruifeng before the results are seen: Part II gives O3s the architecture under which it does better in this pilot (the stronger baseline); if that is one network per mode, Part II also runs O4 with one network per mode, so that H7 compares like with like, and the paper states which architecture each result belongs to. Expectation, stated now and not tested: with one network per mode O3s learns mode 1 on the training geometries far better than with the shared network; whether the sorting still mismatches modes across geometries is open.
