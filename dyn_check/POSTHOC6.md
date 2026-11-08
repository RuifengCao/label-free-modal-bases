# Post hoc diagnostics, sixth set: re-examination (written before running them)

## Why

Ruifeng asked (2026-11-05 21:42) whether the failure comes from a code error or from too little training, and offered more compute. Two points in the earlier sets need re-examination before that question can be answered:

- Under the gradient stop on the history, the training gradient is the gradient of the per-step error part only: Pi_n(Phi c) = 1/2 ||Phi c - u*_n||_A^2 + Pi_n(u*_n), and the second term has no gradient. The objective values compared in F3 and F5a are dominated by Pi_n(u*_n), which differs between rollouts and reflects trajectory amplitude. So "mode 2 is worth 5e-4 of the objective" (F5a) was not measured on the quantity the training follows, and the weighting explanation of set 5 is untested.
- A sharper candidate mechanism: along the model's own trajectory, the exact step leaves the subspace mainly while a load acts, through the load responses A^-1 P_j, which spread over many modes. If so, the per-step objective rewards fields that capture A^-1 P_j, not modes 2 to 5, whose absence shows only as error accumulated over many steps. Longer training would then converge to a subspace without modes 2 to 5, and more compute would not help.

## What (exploratory; for the redesign only; none of it is evidence for the paper)

A. Code checks (no training)
- A1: the training objective (lb.loss_fn) against an independent full-space computation: numpy reduced rollout, effective loads built in full space with fem_cantilever.effective_load, Pi_n evaluated with the full step operator; mean over the 16 training cases and 640 steps. For the planned basis (m = 8, seed 0) and the F4b basis. Pass: relative difference below 1e-9.
- A2: the training gradient, at the seed-0 initialisation and at the F4b parameters (exact clamp and notch-angle inputs, m = 8): (i) against the gradient of an explicitly frozen-history functional built from the same rollout (pass: relative difference below 1e-10); (ii) against central finite differences of that functional along four unit directions (the normalised gradient and three random directions mixed with it), step sizes 1e-2 to 1e-5 (pass: best agreement below 1e-6 relative). For information only: finite differences of the loss itself (history not frozen) along the same directions.
- A3: conditioning of the raw fields: eigenvalues of Phi_raw^T M Phi_raw for the four planned bases, F4b, F5b and the new runs below.

B. Capacity (supervised, with eigenvectors; reference only)
- B1: exact clamp and notch-angle inputs, m = 8, seed 0, trained to span the eight lowest eigenvectors: loss = mean over i of the relative energy-norm residual of v_i after energy-orthogonal projection onto the span; 20,000 iterations, 1e-3 cosine to 1e-5. Report Ritz frequency errors of modes 1 to 8, E1 to E4, the planned objective and S (part E).
  Reading: modes 1 to 5 within 1% means the network can represent the subspace; otherwise capacity is a limit too.
- B2: planned inputs with the exact clamp, one field, supervised to v1 as in F2b, 30,000 iterations (F2b had 3,000 and reached +9.1%).
  Reading: well below 9% (under 1%) means the notch failure of the planned inputs was partly training length; near 9% means it is the inputs.

C. Warm start
- C1, C2: from B1, the planned label-free objective for 3,000 iterations, learning rate 1e-4 cosine to 1e-6 (C1) and 1e-3 cosine to 1e-5 (C2). Ritz errors every 500 iterations; E1 to E4, objective and S at the end.
  Reading: modes 2 to 5 staying within a few per cent means the objective holds them, and the failure from random initialisation is one of optimisation; degrading toward the F4b and F5b levels (hundreds of per cent) means the objective drives the fields away from them.

D. Training length (from scratch; exact clamp and notch-angle inputs; m = 8; seed 0; one CPU thread each)
- D1: 50,000 iterations, 1e-3 cosine to 1e-6. D2: 50,000 iterations, 3e-4 cosine to 1e-6. Loss and Ritz errors every 5,000 iterations; E1 to E4, objective and S at the end.
  Reading: Ritz frequencies 2 to 5 within 5% and E2 below 5% would make training length the limit; otherwise it is not.

E. What the training gradient rewards (no training)
- S = sum over training cases and steps of ||u*_n - Phi c_n+1||_A^2 along the basis's own rollout (the quantity whose gradient the training follows, targets frozen); also S / T, with T the sum of ||u*_n||_A^2.
- E1: for the F5b basis, with its own rollout and targets fixed, the fraction of S removed by adding to the span (i) the exact modes 2 to 5, (ii) the three load responses A^-1 P_ty, A^-1 P_tx, A^-1 P_top, (iii) both. For the modal m = 8 basis, the fraction removed by (ii).
- E2: relative A-norm residual of each load response after projection onto the spans of the planned m = 8 seed 0, F4b and F5b bases.
- E3: S / T for modal m = 8, the planned m = 8 seed 0, F4b, F5b, and B1, C1, C2, D1, D2.
  Reading: if (i) removes a small fraction of S and (ii) most of it, the training gradient has almost no reason to learn modes 2 to 5, and more training cannot be expected to bring them in; if (i) removes a large fraction, the gradient does reward them and the failure is one of optimisation. If the trained bases have smaller S / T than modal m = 8, the training has already done better than the modal subspace on its own objective.

Smoke tests (a few iterations, loss printed only) may be run before the long runs to check that the code executes; they will be recorded.

## Addendum (written after A and E1 to E3, before running E4)

E1 to E3 went against the sharper mechanism: for F5b, adding modes 2 to 5 removes 3.4% of S and adding the three load responses 4.4%; 94% of S remains. Along their own rollouts, modal m = 8 has S / T = 3.4e-9 and F5b 1.7e-7, so the training's own objective prefers the modal subspace by a factor of 50.

- E4 (no training): what the remaining S consists of. For F5b, F4b and the planned m = 8 seed 0 basis, with their own rollouts and targets fixed, the fraction of S removed by adding (iv) the exact first mode v1, (v) v1 to v8, (vi) v1 to v16.
  Reading: if (iv) removes most of S, the per-step error of the trained bases is dominated by the error in the shape of the first mode (the subspace is not invariant under the full dynamics); if only (v) or (vi) does, by low-mode content more generally.

## Second addendum (written after E4, B1, B2, C1 and C2, before running F)

So far: no code error found (A1, A2); the training's own per-step error prefers the modal subspace by a factor of 50 (E3), and 97% to 99% of the trained bases' per-step error lies in the span of the eight lowest modes (E4); the network can represent those modes to below 0.1% (B1); with the planned inputs one field reaches 0.045% given 30,000 supervised iterations (B2), so the notch failure of the planned inputs was largely training length; from the near-modal network, label-free training at a small learning rate keeps modes 1 and 2 and lets modes 3 to 5 drift by up to 2.7% in 3,000 iterations (C1). The failure is one of optimisation: the per-step objective is very flat in the directions that carry modes 2 to 5.

A label-free objective that weights all modes equally would avoid this. Since Ritz values bound eigenvalues from above one by one, the sum of the logarithms of the Ritz values, log det(Phi^T K Phi) - log det(Phi^T M Phi), is minimised by the span of the m lowest eigenvectors, and its gradient weights each Ritz value by its relative change. It uses only K and M.

- F1: exact clamp and notch-angle inputs, m = 8, seed 0, this log-determinant objective, 20,000 iterations, 1e-3 cosine to 1e-5.
- F2: from F1, the planned per-step objective for 3,000 iterations, 1e-4 cosine to 1e-6.
- Both evaluated like the other runs (E1 to E4, S / T).
  Reading: if F1 brings modes 1 to 5 within 1% without labels, a balanced label-free objective solves the optimisation problem; F2 then shows whether the per-step objective keeps the modes and what it adds on the evaluation cases.
