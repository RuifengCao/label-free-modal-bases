# Post hoc diagnostics, seventh set: spectral weighting of the label-free objectives (exploratory)

Written and hashed before running. Exploratory: it checks a derivation and measures a sensitivity; it tests no hypothesis about the method and gives no verdict.

## Background

Let K v = lambda M v with M-orthonormal eigenvectors v_1, v_2, ... and eigenvalues lambda_1 <= lambda_2 <= ...; let theta_1 <= ... <= theta_m be the Ritz values of a basis Phi with m = 8. Derivation to be checked: for Phi = V_8 + W E, where the columns of W are eigenvectors beyond the eighth and E is small,

- trace objective: sum theta_i - sum lambda_i = sum_{i,j} E_ji^2 (lambda_j - lambda_i) + O(|E|^4);
- log-determinant objective: sum log theta_i - sum log lambda_i = sum_{i,j} E_ji^2 (lambda_j - lambda_i) / lambda_i + O(|E|^4).

Equivalently, to first order in the Ritz value errors, the excess of sum f(theta_i) is sum_i lambda_i f'(lambda_i) (theta_i - lambda_i) / lambda_i: the relative error of mode i is weighted by lambda_i (trace), by 1 (log-determinant) and by 1 / lambda_i (f = -1/theta).

## W1: check of the second-order formulas

Geometry and set-up as in PLAN.md (rc.setup()). W = eigenvectors 9 to 16. E with independent N(0, eps^2) entries, eps = 1e-2, 1e-3, 1e-4, five draws each (seed 0). Report the ratio of the exact excess (Ritz values from the projected pencil) to the second-order prediction, for both objectives. Expected: the ratio tends to 1 as eps decreases, with |ratio - 1| of order eps^2.

## W2: per-mode sensitivity of the three objectives

z = V[:, 8:16] c with c a fixed random unit vector (seed 0), so z is M-normalised and K- and M-orthogonal to v_1..v_8. For each i = 1..8: Phi_i equals V_8 with column i replaced by (v_i + t z) / sqrt(1 + t^2), t chosen so that the i-th Ritz frequency is exactly 1% high. Report for each i:

- the Ritz frequency errors of Phi_i (check: only mode i moves, by +1%);
- the excess of the trace and log-determinant objectives and of sum (-1/theta), from the Ritz values;
- the per-step error ratio S/T of Phi_i along its own rollout on the 16 training cases (S_split, as in set 6), and its increase over S/T of V_8.

Question: how strongly does each objective respond to the same 1% frequency error in mode i? Reading expected before running: trace response grows like lambda_i, log-determinant response is flat, and the S/T increase falls steeply with i (mode 1 largest). Any other pattern is reported as found.
