# Exploratory runs after the formal runs: cost and mesh refinement (X1), longer training (X2), figures and the shape-error relation (X3)

**Status.** Written on 7 November 2026 and hashed into PLAN_HASH.txt before any of these runs. Exploratory: no hypotheses, no thresholds, no verdicts; every result is reported, including any that goes against the expectations written here. Requested by the author on 7 November 2026 at 17:38 UTC ("everything else as recommended; start"), after the evaluation note of the same day. The rotor-dynamics question is left open by the author and is not part of this plan.

The formal runs (PREREG.md, STAMP.txt sha256 120ae745b2a7...) and their outputs are not changed. These runs use the saved formal networks and import the stamped code without modifying it. New code: explore.py (X1, X2), run_explore.py (laptop launcher), and the X3 scripts.

## X1. Online cost and mesh refinement

**Purpose.** (i) Compare the cost of obtaining a reduced model for a new geometry with the trained network, with an eigen-solve and with the mapped reduced basis, and see how each grows with the number of degrees of freedom. (ii) See whether the Part II networks, trained on meshes with h_max = 0.2, keep their accuracy when evaluated on finer meshes of the same geometries.

**Meshes.** h_max = 0.2, 0.1, 0.05 and 0.025, with the mesh rules of fem_cantilever.make_mesh (the grading towards the notch root is unchanged, so the number of degrees of freedom grows less than four-fold per halving).

**X1a, accuracy.** For each of the 16 held-out geometries at h_max = 0.2, 0.1 and 0.05, and for the four FINE geometries also at 0.025: the evaluation of phase1.cmd_e2 with the mesh size as a parameter. On each mesh: finite element frequencies and modes; the plain cantilever's first frequency on the same h_max; the 10 evaluation cases with reference Newmark trajectories on that mesh. Bases: every Part II network (15), modal truncation (8 modes), the mapped reduced basis (sources: the 24 training geometries at h_max = 0.2, as in the formal runs; Gram truncation 1e-13) and the frequency regression (as in the formal runs; frequencies only). Measures: e_1 to e_8, the reproduced drop of omega_1, E1, E2 and sin^2, as phase0.eval_basis.

*Check.* At h_max = 0.2 the code path is that of the formal evaluation apart from the mesh-size parameter, so the Ritz errors must equal the formal values (e2_shard*.json) to round-off. A relative difference above 1e-8 in any Ritz error is reported and explained before anything else is read.

**FINE geometries** (also the timing set), chosen before running to span the family: 1.1698_0.2501 (left, inside the training hull), 2.6263_0.3995 (deepest, outside), 3.5026_0.2488 (middle, inside), 4.0451_0.1017 (shallowest, outside).

**X1b, timing.** Run alone on one CPU after every other job, with single-threaded numerical libraries. For each FINE geometry and h_max, the median of 3 repeats of:
- mesh generation (fem_cantilever.make_mesh);
- mesh and assembly of K, M and the loads (fem_cantilever.build);
- 8 eigenpairs (fem_cantilever.eig, shift-invert);
- network inputs and problem set-up (phase0.fam_problem);
- network basis, first call on the mesh, JAX compilation included (one measurement);
- network basis, compiled;
- network basis in NumPy without compilation (checked against the JAX basis to a relative 1e-10);
- reduced operators (Phi^T K Phi, Phi^T M Phi, Phi^T P);
- mapped reduced basis (mapping, Gram truncation and Rayleigh-Ritz); the preparation of the 24 sources is offline and is timed once per run;
- factorisation of the step operator; one full Newmark step (median over 20); one reduced step (median over 1,000).

Derived: a full simulation of 640 steps = factorisation + 640 full steps; a reduced simulation = 640 reduced steps. The timed network is shared O4, seed 0.

**Expectations, written before running.** The network route costs about as much as meshing and assembly and grows roughly linearly; the eigen-solve grows faster; at about 7,000 degrees of freedom everything takes less than a second. On finer meshes the networks' accuracy stays close to that at h_max = 0.2, with a possible loss near the notch. Reported whatever is found.

## X2. Longer training of shared O2 and O3

**Purpose.** Does mode 1 catch up with a five-fold training budget?

**Runs.** Shared network; O2 and O3; G1, G2, G3; seeds 10 and 11 (12 runs); 100,000 iterations; Adam as in the formal runs, learning rate cosine from 1e-3 to 1e-5 over the 100,000 iterations; Ritz errors and sin^2 logged every 5,000 iterations; then the evaluation of phase1.cmd_p1 (the network and modal truncation). Same code path as phase1.cmd_p1 for the shared network apart from the number of iterations and the logging interval.

**Reported.** e_1 and the largest of e_3 to e_8 at every log, the final evaluation and the wall time, next to the formal 20,000-iteration run of the same geometry and seed. Because the cosine schedule is stretched over 100,000 iterations, the log at 20,000 iterations is not comparable to the end of the formal run.

**Expectation.** e_1 keeps falling; whether it gets below 0.5% (O3) or 1% (O2) by 100,000 iterations is open.

## X3. Figures and the shape-error relation (sandbox; no training)

**Figures**, from the saved results and networks:
1. Part I learning curves: median over the six geometry-seed runs of e_1 against iterations, O1 to O4, both architectures.
2. Part II: largest frequency error on each held-out geometry for shared O4 (three seeds), the supervised network (three seeds), the mapped reduced basis and the frequency regression.
3. Tip vertical displacement against time on the held-out geometry 3.1654_0.2769 (inside the hull), for the tip pulse case and the first random evaluation case: full model, shared O4 seed 0 and the mapped reduced basis.
4. The first four modes of G1: finite element against the Part I shared O4 network, seed 10.

**Analysis.** For each held-out geometry and each shared O4 network: the Ritz vectors of the basis, matched to the exact eigenvectors by order; eps_i = ||vt_i - v_i||_K / ||v_i||_K with the sign of vt_i chosen to minimise it, against sqrt(delta_i), delta_i = theta_i / lambda_i - 1; and E2 against the eps_i. Expected from the single-geometry check of 5 and 6 November (RESULTS.md, section 7.4, item 4): eps_i of the order of sqrt(delta_i), and E2 of the order of the eps_i of the low modes. Reported as found.

## Rules

- This plan is hashed before any of these runs. Any change after the hash is written into DEVIATIONS_EXPLORE.md, with its reason, before it is made.
- New code runs in a sandbox smoke test, with every path executed, before the laptop run; smoke numbers are not used.
- Laptop: four processes on logical CPUs 2, 4, 6 and 8; X1b alone afterwards on CPU 2. A job that fails is run once more and the failure is recorded.
- No result is excluded. In the paper all of this is labelled exploratory.
