# Pipeline check: label-free reduced basis with per-step Galerkin, one notched cantilever

Status: exploratory pipeline check for the Unified 2027 paper (dynamics). It is not the pre-registered study. Its purpose is to execute every code path the study will use, verify the identities the method rests on, and measure the magnitudes needed to set the study's design and thresholds. The plan is fixed, and its SHA-256 recorded, before any network is trained. Deviations, if any, are recorded with reasons before the run they affect.

## Question

On one notched cantilever, can a network-generated basis of m fields, trained only on the incremental potentials of implicit Newmark steps along its own reduced trajectory (no solved step is ever read by the training code), give a reduced model whose trajectories, per-step approximation errors and natural frequencies are close to those of the full FE model, and how does it compare with the m lowest eigenmodes (modal truncation) and with POD of solved trajectories at the same m?

## FE model (one geometry)

- Plane stress, E = 1, nu = 0.3, density 1. Beam x in [0, 10], y in [0, 1]; left edge x = 0 clamped (all dofs removed).
- U-shaped notch on the top edge: centre x_c = 2.0, depth d = 0.3, root radius rho = 0.05 (notch width 2 rho).
- Mesh: Triangle (minimum angle 30 degrees), element size graded from rho / 4 at the notch root to h_max = 0.2 at distance 1.0 and beyond; quadratic six-node triangles (P2); consistent mass matrix.
- Rayleigh damping C = a M + b K with 2% damping at the first and third natural frequencies of this geometry.
- Newmark average acceleration (beta = 1/4, gamma = 1/2); time step dt = T3 / 20, with T3 the period of the third mode; horizon N = 640 steps (about two periods of the first mode). Initial state at rest; the initial acceleration solves M a0 = F(0).
- Load patterns (each normalised to unit total force): P_ty, vertical traction on the free end x = 10; P_tx, axial traction on the free end; P_top, uniform vertical pressure on the top edge. A load case is F(t) = (w_ty P_ty + w_tx P_tx + w_top P_top) g(t).
- Time functions g: half-sine pulse g(t) = sin(pi t / tau) for t <= tau and 0 after, with tau log-uniform in [0.5 T3, 0.5 T1]; and the step g(t) = 1 (evaluation only).
- Training load cases: 16, data seed 2026: weights (w_ty, w_tx, w_top) uniform in [-1, 1]^3, half-sine pulses. Evaluation load cases: 8 half-sine cases with data seed 2027, plus two fixed cases, a vertical tip pulse (w = (1, 0, 0), tau = T2 / 2) and a vertical tip step (w = (1, 0, 0), step).

## Reduced model

- Network: multilayer perceptron, tanh, 4 hidden layers of width 64, float64. Inputs per node location (20): x / 10, y; sin(k pi x / 10), cos(k pi x / 10) for k = 1..6; and, for s = rho and s = 0.3, with (xi, eta) = ((x - x_c) / s, (y - (1 - d)) / s) and r = sqrt(xi^2 + eta^2), the features xi / (1 + r), eta / (1 + r), exp(-r). Outputs: 2m values per node, i.e. m vector fields, scattered to the free dofs to give Phi (free dofs by m).
- Phi is M-orthonormalised (Cholesky of Phi^T M Phi) before use; this changes no subspace.
- Reduced step: (Phi^T A Phi) c = Phi^T F_n, with A = K + M / (beta dt^2) + gamma C / (beta dt) and F_n the Newmark effective load built from the state at step n. Velocities and accelerations are updated by the Newmark formulas in reduced coordinates.

## Training (no solved examples)

- Loss: mean over training load cases and steps of Pi_n(Phi c_n+1), where Pi_n(u) = 1/2 u^T A u - F_n^T u and c_n+1 is the Galerkin solution above. The history (displacement, velocity, acceleration at step n) enters F_n with its gradient stopped, so each term is, up to a constant fixed by the history, half the squared A-norm distance from the subspace to the exact step taken from the visited state.
- Adam, learning rate 1e-3 with cosine decay to 1e-5, 3,000 full-batch iterations (all 16 training cases, full horizon).
- Runs: m in {8, 16}, network seeds {0, 1}: four runs.
- The training code reads only K, M, C, the load patterns and the time functions. Reference solutions and eigenvectors are computed for evaluation and for the baselines only.

## Baselines at the same m

- Modal truncation: the m lowest eigenvectors of (K, M) for this geometry (needs an eigensolve).
- POD with labels: the m leading M-orthonormal POD modes of the solved displacement snapshots of the 16 training cases (needs 16 solved trajectories).

## Evaluation (the 10 evaluation cases)

- E1 Tip response: relative L2-in-time error of the vertical displacement and of the vertical acceleration at the free-end node at mid-height (x = 10, y = 0.5).
- E2 Trajectory error in the energy norm: sqrt(sum_n ||u_n - U_n||_K^2) / sqrt(sum_n ||U_n||_K^2), with U the full reference.
- E3 Per-step approximation: for each n, the relative A-norm distance from the subspace to the reference step U_n+1 (the exact step from the reference state); reported as median and maximum over n and cases.
- E4 Frequencies: relative errors of the five lowest Ritz frequencies of (Phi^T K Phi, Phi^T M Phi) against the FE frequencies.
- E5 Cost: training wall time; per-step time of the reduced and the full step; set-up time for this geometry (network evaluation and projections, against factorisation and eigensolve).

## Identity checks (machine precision; a failure means a code error, not a finding)

- I1 Per-step identity: for random u, |Pi_n(u) - Pi_n(U*) - 1/2 ||u - U*||_A^2| / (1/2 ||u - U*||_A^2) <= 1e-9, and the gradient identity to the same tolerance.
- I2 Galerkin best approximation: Pi_n(Phi c*) - Pi_n(U*) equals 1/2 of the squared A-norm distance from the subspace to U*, to 1e-9.
- I3 Computable bound: ||u - U*||_A^2 <= beta dt^2 g^T M^-1 g with g = F_n - A u, for the reduced predictions along a rollout; no violation beyond 1e-9 relative, and the median ratio (bound over actual) reported.

## Readings fixed in advance

- If any identity check fails, stop and fix the code; the failure is recorded.
- Everything else is a measurement for designing the study (family of geometries, thresholds, m, horizon, training length). No verdict on the method is drawn from this check, and no result of it will be presented as evidence for the method.
