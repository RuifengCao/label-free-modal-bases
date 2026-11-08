# Formal runs: results

Unified 2027 dynamics paper. Results of pre-registration Version 1 (PREREG.md, approved on 6 November 2026, stamped on 7 November 2026; STAMP.txt sha256 120ae745b2a7...). Full numbers: TABLES_FORMAL.md, generated from the outputs by tables_formal.py. Everything in this note is either a pre-registered measure or labelled as descriptive.

## 1. Run and checks

- The 63 jobs ran on the author's laptop on 7 November 2026 from 01:31:10 to 09:11:23 UTC, four single-threaded processes. All 63 finished at the first attempt; none was rerun or excluded.
- The run recorded the hashes of the 18 stamped files (environment.json); they equal STAMP.txt.
- The 70 output files analysed here are byte-identical to the laptop's (SHA256SUMS_formal_results.txt, computed on the laptop for all 172 output files).
- recheck_verdicts.py recomputes every verdict from the raw outputs, written from the text of PREREG.md rather than from the stamped verdict code: same labels, same counts, same values (largest difference 0). It also checks the stored reproduced drop against its definition for every basis and held-out geometry (largest difference 0).
- spot_check.py rebuilds six saved networks in the sandbox (Linux) and recomputes their Ritz errors with its own eigen-solve: largest difference from the stored errors 1.5e-10, and from the stored finite element frequencies 3.1e-11 (relative).

## 2. Verdicts

All twelve pre-registered predictions are supported, each in every run or seed.

| | Prediction | Formal result (range over the runs or seeds) | Pilot value in PREREG.md | Verdict |
|---|---|---|---|---|
| H1 | shared O4: max_i e_i <= 0.5% | 0.0655% to 0.0869% | 0.08% to 0.10% | supported, 6 of 6 |
| H2 | shared O1: max e_3..e_8 > 5% | 398% to 598% | 462% to 649% | supported, 6 of 6 |
| H3 | shared O2: e_1 > 1% and > max e_3..e_8 | e_1 14.4% to 21.6%; max e_3..e_8 at most 0.236% | e_1 12.0% to 19.9% | supported, 6 of 6 |
| H3b | shared O3: e_1 > 0.5% and > max e_3..e_8 | e_1 3.95% to 71.8%; max e_3..e_8 at most 2.88% | e_1 3.01% to 5.11% | supported, 6 of 6 |
| H8 | per-mode O3: e_1 < shared O3's e_1 / 3 | per-mode e_1 0.331% to 0.419%; ratio 12 to 176 | ratios 10 and 12 | supported, 6 of 6 |
| H9 | per-mode O2: e_1 > 1% and > max e_3..e_8 | e_1 19.5% to 56.9%; max e_3..e_8 at most 0.439% | e_1 22.4% and 26.4% | supported, 6 of 6 |
| H10 | per-mode O4: max_i e_i <= 0.5% | 0.0939% to 0.126% | 0.11% and 0.12% | supported, 6 of 6 |
| H4 | shared O4: median of median e_1..e_5 <= 1% | 0.0967%, 0.108%, 0.108% | 0.13% | supported, 3 of 3 |
| H5 | shared O4: median reproduced drop >= 90% | 97.2%, 96.4%, 96.6% | 96.4% | supported, 3 of 3 |
| H6 | shared O4: median E2 <= 10% | 5.93%, 6.39%, 6.20% | 6.2% | supported, 3 of 3 |
| H7 | median e_1: shared O3s > shared O4 | 8.50% vs 0.161%; 386% vs 0.194%; 190% vs 0.171% | 46.1% vs 0.18% | supported, 3 of 3 |
| H7sep | median e_1: per-mode O3s > per-mode O4 | 0.581% vs 0.332%; 0.717% vs 0.347%; 0.729% vs 0.276% | 0.71% vs 0.29% | supported, 3 of 3 |

## 3. Part I (G1 to G3, 20,000 iterations)

- **O4 learns all eight modes with either architecture.** The largest error over the six runs is 0.0869% with the shared network and 0.126% with one network per mode; median E2 3.62% and 3.96%, against 1.29% to 1.37% for the eight exact modes.
- **O1 learns the low modes and misses the high ones.** e_1 is 0.034% to 0.054%, but the largest error is 398% to 598% in every run.
- **The eigenvalue weighting of O2 and O3 leaves mode 1 behind when the modes share their parameters.** Shared O2: e_1 14.4% to 21.6% while modes 3 to 8 are within 0.24%. Shared O3: e_1 3.95% to 71.8%. Seed 10 gave 43.7% on G1 and 71.8% on G3, against 3.95% to 7.56% in the other four runs; in those two runs mode 2 is also off (4.55% and 10.0%).
- **One network per mode removes the effect for O3 but not for O2.** Per-mode O3: e_1 0.331% to 0.419%, 12 to 176 times smaller than shared O3 on the same geometry and seed. Per-mode O2: e_1 19.5% to 56.9% (median 30.8%, against 16.6% shared), because the orthonormalisation in the trace couples the modes whatever the architecture.
- **Descriptive: at this budget the lag is slow convergence, not a fixed point.** Median e_1 of shared O3 falls from 42.0% at 10,000 iterations to 11.2% at 15,000 and 7.03% at 20,000; shared O2 from 32.3% to 20.7% to 16.6%. Per-mode O3 is at 1.11% after 10,000 iterations. The pre-registered statements are about the fixed budget of 20,000 iterations and hold there; the paper should describe O2 and O3 as converging much more slowly on mode 1, not as unable to learn it.

## 4. Part II (24 training geometries, 16 held out)

- **Shared O4 meets the three pre-registered bounds with margin** (H4 to H6 above). The largest error over the 16 held-out geometries is 0.668% to 0.745% across the three seeds; the median of max_i e_i is 0.214% to 0.257% on the held-out geometries and 0.202% to 0.229% on the training geometries, so the gap from training to held-out geometries is small. Inside the training hull (12 geometries) the median of max_i e_i is 0.200% to 0.231%; outside it (4 geometries), 0.441% to 0.452%.
- **Shared O3s fails, and erratically.** Median e_1 8.50%, 386% and 190% for the three seeds; median E2 63% to 123%. The learning curves are not monotone: for seed 1 the median e_1 is 12.0% at 30,000 iterations, 239% at 40,000 and 386% at 60,000.
- **With one network per mode, O3s works but is worse than O4** (H7sep): median e_1 0.581% to 0.729% against 0.276% to 0.347%; reproduced drop 84.9% to 88.8% and median E2 12.9% to 13.7%, which would fail the H5 and H6 bounds had they applied to it. Per-mode O4 (30,000 iterations, about the same training time as 60,000 shared iterations) is itself worse than shared O4: median E2 7.60% to 8.45%, and a smaller largest error than shared O4 (same seed) on only 0, 1 and 0 of the 16 geometries.
- **The label-free network matches the supervised one.** The same shared network fitted to the eight lowest eigenvectors of the training geometries gives H4 to H6 measures of 0.100% to 0.113%, 96.5% to 96.8% and 6.05% to 6.48%, against 0.0967% to 0.108%, 96.4% to 97.2% and 5.93% to 6.39% for O4. Geometry by geometry, the supervised network has the smaller largest error on 9, 10 and 11 of the 16 geometries.

### Classical controls (pre-registered without hypotheses; reported in full)

- **Mapped reduced basis** (eigenvectors of the 24 training geometries carried by the notch-aligning map, then Rayleigh-Ritz on the held-out geometry): median of max_i e_i 0.0675%, median E2 3.84%, median reproduced drop 98.5%. It has a smaller largest error than shared O4 on 14, 15 and 14 of the 16 geometries (seeds 0, 1, 2) and a smaller E2 on 14 of 16 for each seed. Its one clear failure is the shallowest notch, outside the training hull (x_c = 4.0451, d = 0.1017, finite element drop of omega_1 0.50%): largest error 0.847%, E2 13.3% and reproduced drop -67%, where shared O4 (seed 0) gives 0.216%, 6.4% and 66%.
- **Frequency regression** (thin-plate spline over notch position and depth; frequencies only, no modes): the most accurate for mode 1 (median |e_1| 0.0411%) but less accurate for the higher modes (median |e_6| 0.210% and |e_8| 0.249%, against 0.123% to 0.127% and 0.201% to 0.221% for O4); its largest |e_i| is smaller than shared O4's on only 4, 4 and 3 of the 16 geometries.
- **Modal truncation** (the 8 exact modes of each held-out geometry): median E2 1.33%, the floor for an 8-mode basis.

## 5. What this supports in the paper

- The mechanism claim is supported in full and should be stated for networks whose modes share parameters: with Adam, eigenvalue-weighted objectives (O2, O3) let mode 1 converge far more slowly; giving each mode its own network removes this for O3 but not for O2, whose orthonormalisation couples the modes; the equal relative weighting of O4 learns all eight modes with either architecture.
- Part II supports: a single network trained with O4 and without any eigen-solution gives a basis for unseen notch geometries with the accuracy of the same network trained on eigenvectors.
- Part II does not support any claim that the network is more accurate than classical model reduction: the mapped reduced basis, which uses the training geometries' eigen-solutions and a hand-designed map, is more accurate on 14 or 15 of 16 geometries. The network's advantages are that it needs no eigen-solutions and no map, and that it degrades less on the one geometry where the map fails; that last point rests on a single geometry.
- Scope: one two-parameter family of plane-stress notched cantilevers, eight modes, one network size per architecture, three seeds in Part II and two in Part I.
