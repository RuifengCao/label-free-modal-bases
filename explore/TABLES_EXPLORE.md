# Exploratory runs: tables for X1 and X2

Written by x12_report.py from the laptop outputs (explore/results, SHA256SUMS_explore_results.txt). Medians are over the 16 held-out geometries (and, for the networks, the three seeds) unless stated.

## X1a.1 Largest frequency error, max over modes 1 to 8 of |e_i|, against mesh size

| Basis | h 0.2 | h 0.1 | h 0.05 | ratio 0.05 / 0.2: median (range) |
|---|---|---|---|---|
| Shared O4 | 2.54e-03 | 2.58e-03 | 2.61e-03 | 1.03 (1.01 to 1.10) |
| Shared O3s | 1.90e+00 | 1.90e+00 | 1.90e+00 | 1.00 (1.00 to 1.01) |
| Supervised | 2.25e-03 | 2.30e-03 | 2.33e-03 | 1.03 (1.01 to 1.07) |
| Per mode O4 | 3.84e-03 | 3.92e-03 | 3.96e-03 | 1.02 (1.00 to 1.06) |
| Per mode O3s | 1.20e-02 | 1.21e-02 | 1.22e-02 | 1.01 (1.00 to 1.03) |
| Modal truncation | 2.03e-11 | 2.28e-11 | 6.58e-11 | 2.81 (0.09 to 11.27) |
| Mapped reduced basis | 6.75e-04 | 1.99e-03 | 3.00e-03 | 4.45 (2.97 to 6.45) |
| Frequency regression | 5.33e-03 | 5.45e-03 | 5.50e-03 | 1.03 (0.90 to 1.12) |

Degrees of freedom (median): 7,366 at h 0.2, 23,352 at h 0.1, 76,893 at h 0.05. Finite element first frequency, h 0.05 against 0.2: median change -1.29e-04 (range -1.67e-04 to -9.42e-05); largest change over modes 1 to 8: median 2.62e-04, largest 3.57e-04.

## X1a.2 E1 (tip displacement), E2 and the largest sin^2, against mesh size

Median over the evaluation cases, then over geometries (and seeds).

| Basis | E1 h 0.2 | E1 h 0.1 | E1 h 0.05 | E2 h 0.2 | E2 h 0.1 | E2 h 0.05 | max sin^2 h 0.2 | h 0.05 | drop reproduced h 0.2 | h 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| Shared O4 | 0.0115 | 0.0119 | 0.0121 | 0.0619 | 0.0629 | 0.0635 | 1.95e-03 | 1.95e-03 | 0.967 | 0.966 |
| Shared O3s | 1.0003 | 1.0003 | 1.0003 | 1.0522 | 1.0522 | 1.0522 | 8.79e-03 | 8.82e-03 | -34.539 | -34.537 |
| Supervised | 0.0115 | 0.0120 | 0.0122 | 0.0631 | 0.0639 | 0.0645 | 1.77e-03 | 1.74e-03 | 0.966 | 0.964 |
| Per mode O4 | 0.0200 | 0.0204 | 0.0207 | 0.0822 | 0.0833 | 0.0838 | 2.01e-03 | 2.07e-03 | 0.942 | 0.939 |
| Per mode O3s | 0.0472 | 0.0478 | 0.0480 | 0.1339 | 0.1354 | 0.1359 | 2.79e-03 | 2.79e-03 | 0.872 | 0.869 |
| Modal truncation | 0.0002 | 0.0002 | 0.0002 | 0.0133 | 0.0133 | 0.0133 | 2.00e-15 | 5.44e-15 | 1.000 | 1.000 |
| Mapped reduced basis | 0.0044 | 0.0126 | 0.0193 | 0.0384 | 0.0644 | 0.0794 | 3.95e-06 | 2.27e-05 | 0.985 | 0.931 |
| Frequency regression | | | | | | | | | 1.000 | 0.997 |

Drop reproduced: the share of the finite element drop of the first frequency (notched against plain cantilever on the same mesh) that the basis reproduces, as in the formal evaluation.

## X1a.3 The four FINE geometries down to h_max 0.025

| Geometry | DOFs at 0.025 | Basis | max e_i at 0.2 | 0.1 | 0.05 | 0.025 | E2 at 0.2 | 0.025 |
|---|---|---|---|---|---|---|---|---|
| 1.1698_0.2501 | 265,432 | Shared O4 s0 | 1.88e-03 | 1.94e-03 | 1.98e-03 | 1.99e-03 | 0.0619 | 0.0641 |
| 1.1698_0.2501 | 265,432 | Supervised s0 | 2.02e-03 | 2.09e-03 | 2.14e-03 | 2.15e-03 | 0.0647 | 0.0668 |
| 1.1698_0.2501 | 265,432 | Per mode O4 s0 | 3.88e-03 | 3.97e-03 | 4.02e-03 | 4.03e-03 | 0.0891 | 0.0914 |
| 1.1698_0.2501 | 265,432 | Mapped reduced basis | 3.29e-04 | 1.11e-03 | 1.67e-03 | 1.91e-03 | 0.0275 | 0.0541 |
| 1.1698_0.2501 | 265,432 | Frequency regression | 1.01e-02 | 9.89e-03 | 9.82e-03 | 9.80e-03 |  |  |
| 1.1698_0.2501 | | finite element w_1 change 0.2 to 0.025 | -1.69e-04 | | | | | |
| 2.6263_0.3995 | 267,118 | Shared O4 s0 | 6.88e-03 | 6.96e-03 | 7.05e-03 | 7.06e-03 | 0.0951 | 0.0964 |
| 2.6263_0.3995 | 267,118 | Supervised s0 | 5.64e-03 | 5.69e-03 | 5.73e-03 | 5.74e-03 | 0.1084 | 0.1095 |
| 2.6263_0.3995 | 267,118 | Per mode O4 s0 | 1.11e-02 | 1.12e-02 | 1.13e-02 | 1.13e-02 | 0.1424 | 0.1435 |
| 2.6263_0.3995 | 267,118 | Mapped reduced basis | 9.82e-04 | 3.39e-03 | 5.66e-03 | 7.39e-03 | 0.0455 | 0.1306 |
| 2.6263_0.3995 | 267,118 | Frequency regression | 6.47e-03 | 6.73e-03 | 6.83e-03 | 6.85e-03 |  |  |
| 2.6263_0.3995 | | finite element w_1 change 0.2 to 0.025 | -1.37e-04 | | | | | |
| 3.5026_0.2488 | 264,960 | Shared O4 s0 | 2.11e-03 | 2.11e-03 | 2.14e-03 | 2.15e-03 | 0.0452 | 0.0471 |
| 3.5026_0.2488 | 264,960 | Supervised s0 | 1.84e-03 | 1.83e-03 | 1.87e-03 | 1.88e-03 | 0.0480 | 0.0500 |
| 3.5026_0.2488 | 264,960 | Per mode O4 s0 | 2.71e-03 | 2.72e-03 | 2.76e-03 | 2.77e-03 | 0.0637 | 0.0654 |
| 3.5026_0.2488 | 264,960 | Mapped reduced basis | 2.02e-04 | 7.67e-04 | 1.10e-03 | 1.23e-03 | 0.0233 | 0.0469 |
| 3.5026_0.2488 | 264,960 | Frequency regression | 2.99e-03 | 3.13e-03 | 3.20e-03 | 3.22e-03 |  |  |
| 3.5026_0.2488 | | finite element w_1 change 0.2 to 0.025 | -1.35e-04 | | | | | |
| 4.0451_0.1017 | 260,280 | Shared O4 s0 | 2.16e-03 | 2.19e-03 | 2.21e-03 | 2.22e-03 | 0.0637 | 0.0656 |
| 4.0451_0.1017 | 260,280 | Supervised s0 | 2.05e-03 | 2.08e-03 | 2.13e-03 | 2.15e-03 | 0.0631 | 0.0650 |
| 4.0451_0.1017 | 260,280 | Per mode O4 s0 | 3.81e-03 | 3.86e-03 | 3.91e-03 | 3.93e-03 | 0.0821 | 0.0841 |
| 4.0451_0.1017 | 260,280 | Mapped reduced basis | 8.47e-03 | 2.01e-02 | 2.71e-02 | 3.08e-02 | 0.1334 | 0.2942 |
| 4.0451_0.1017 | 260,280 | Frequency regression | 3.17e-03 | 2.96e-03 | 2.91e-03 | 2.89e-03 |  |  |
| 4.0451_0.1017 | | finite element w_1 change 0.2 to 0.025 | -1.85e-04 | | | | | |

## X1a.4 Checks at h_max 0.2 against the formal evaluation (same laptop)

| Geometry | streaming vs eval_basis | Ritz, absolute | Ritz, relative (basis) | FE frequencies | E1, E2 vs formal |
|---|---|---|---|---|---|
| 1.1698_0.2501 | 3.0e-11 | 1.6e-11 | 1.2e-08 (rbm_all) | 4.4e-14 | 1.3e-07 |
| 1.3461_0.1508 | 2.7e-11 | 3.0e-11 | 1.7e-08 (rbm_all) | 1.2e-14 | 2.7e-07 |
| 1.5060_0.2220 | 4.6e-11 | 3.5e-11 | 9.9e-08 (rbm_all) | 2.7e-13 | 4.6e-07 |
| 1.9470_0.3289 | 4.5e-11 | 2.8e-11 | 1.9e-08 (rbm_all) | 9.7e-15 | 4.0e-07 |
| 2.0416_0.1276 | 1.9e-11 | 2.9e-11 | 4.3e-08 (rbm_all) | 1.3e-14 | 1.2e-07 |
| 2.4737_0.3212 | 3.7e-11 | 2.5e-11 | 3.6e-08 (rbm_all) | 7.7e-14 | 3.6e-08 |
| 2.6263_0.3995 | 3.0e-11 | 2.3e-11 | 2.4e-08 (rbm_all) | 3.9e-14 | 1.4e-07 |
| 2.8270_0.1994 | 1.9e-11 | 2.2e-12 | 4.0e-09 (rbm_all) | 1.2e-14 | 1.1e-07 |
| 3.1654_0.2769 | 3.2e-11 | 6.7e-11 | 2.0e-07 (rbm_all) | 4.4e-14 | 2.1e-07 |
| 3.3505_0.1723 | 1.9e-11 | 3.4e-11 | 3.2e-08 (freq_rbf) | 4.0e-13 | 4.5e-07 |
| 3.5026_0.2488 | 4.0e-11 | 3.7e-11 | 6.6e-08 (rbm_all) | 1.9e-14 | 3.2e-07 |
| 3.9503_0.3504 | 2.7e-11 | 1.8e-11 | 9.3e-09 (rbm_all) | 4.8e-14 | 2.5e-07 |
| 4.0451_0.1017 | 2.4e-11 | 2.6e-11 | 3.1e-09 (rbm_all) | 8.0e-14 | 2.0e-07 |
| 4.4704_0.2994 | 3.2e-11 | 2.0e-11 | 3.1e-08 (rbm_all) | 1.2e-14 | 6.7e-08 |
| 4.6308_0.3736 | 2.4e-11 | 2.9e-11 | 9.9e-09 (rbm_all) | 5.2e-14 | 4.5e-07 |
| 4.8227_0.1775 | 2.0e-11 | 1.0e-11 | 1.6e-08 (rbm_all) | 6.6e-15 | 2.6e-08 |

## X1b Online cost on one core (seconds; median over the four FINE geometries of the median of three repeats)

| Component | h 0.2 (7,356 DOFs) | h 0.1 (23,352 DOFs) | h 0.05 (76,980 DOFs) | h 0.025 (265,196 DOFs) | growth exponent |
|---|---|---|---|---|---|
| mesh | 0.001869 | 0.005237 | 0.0177 | 0.0672 | 1.00 |
| mesh and assembly | 0.04432 | 0.3817 | 1.26 | 5.161 | 1.29 |
| 8 eigenpairs | 0.04119 | 0.2035 | 1.006 | 5.218 | 1.35 |
| network inputs | 0.003885 | 0.01557 | 0.04772 | 0.1658 | 1.04 |
| network, first call | 0.1847 | 0.2087 | 0.2299 | 0.3469 | 0.17 |
| network, compiled | 0.004508 | 0.01389 | 0.04872 | 0.1767 | 1.03 |
| network, NumPy | 0.01327 | 0.04255 | 0.1478 | 0.5271 | 1.03 |
| reduced operators | 0.0007239 | 0.00326 | 0.01102 | 0.04021 | 1.11 |
| mapped reduced basis | 0.09912 | 0.2732 | 0.8396 | 2.924 | 0.94 |
| factorisation | 0.01851 | 0.09008 | 0.4833 | 2.951 | 1.41 |
| full step | 0.000707 | 0.003645 | 0.0169 | 0.082 | 1.32 |
| reduced step | 1.99e-05 | 1.995e-05 | 2.005e-05 | 1.99e-05 | 0.00 |
| full simulation, 640 steps | 0.4712 | 2.423 | 11.3 | 55.43 | 1.33 |
| reduced simulation, 640 steps | 0.01274 | 0.01277 | 0.01283 | 0.01274 | 0.00 |

Network route = network inputs + compiled network + reduced operators: 0.00912 s at h 0.2, 0.0327 s at h 0.1, 0.107 s at h 0.05, 0.383 s at h 0.025. Growth exponent: slope of log time against log DOFs over the four meshes. Offline preparation of the 24 sources of the mapped reduced basis: 2.38 s. Largest differences of the timed bases from lb.orthonormal_basis: compiled with mesh arguments 2.3e-13, NumPy 2.7e-13.

## X2 Shared O2 and O3, 100,000 iterations, next to the formal 20,000-iteration runs

| Objective | Geometry | Seed | formal: e_1 | formal: max e_3..e_8 | 100k run: e_1 at 20k | 50k | 100k | max e_3..e_8 at 100k | E2 formal | E2 100k | train s, formal | 100k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O2 | G1 | 10 | 2.16e-01 | 1.50e-03 | 1.22e-01 | 2.22e-02 | 7.50e-03 | 7.27e-05 | 1.151 | 0.126 | 372 | 1890 |
| O2 | G1 | 11 | 1.44e-01 | 2.04e-03 | 7.93e-02 | 2.02e-02 | 7.25e-03 | 7.69e-05 | 0.911 | 0.124 | 383 | 1922 |
| O2 | G2 | 10 | 1.67e-01 | 1.48e-03 | 9.25e-02 | 1.72e-02 | 5.12e-03 | 6.83e-05 | 1.011 | 0.103 | 409 | 2056 |
| O2 | G2 | 11 | 1.82e-01 | 2.36e-03 | 1.14e-01 | 1.92e-02 | 5.72e-03 | 7.93e-05 | 1.062 | 0.110 | 407 | 1904 |
| O2 | G3 | 10 | 1.47e-01 | 1.44e-03 | 9.48e-02 | 1.32e-02 | 4.24e-03 | 5.79e-05 | 0.900 | 0.093 | 382 | 1729 |
| O2 | G3 | 11 | 1.65e-01 | 1.95e-03 | 9.54e-02 | 1.50e-02 | 4.23e-03 | 1.47e-04 | 0.977 | 0.093 | 371 | 1636 |
| O3 | G1 | 10 | 4.37e-01 | 1.15e-02 | 1.49e-01 | 5.72e-03 | 1.99e-03 | 3.14e-05 | 1.385 | 0.064 | 444 | 2260 |
| O3 | G1 | 11 | 3.95e-02 | 9.36e-04 | 2.94e-01 | 2.27e-02 | 4.84e-03 | 8.49e-05 | 0.348 | 0.100 | 435 | 2259 |
| O3 | G2 | 10 | 6.50e-02 | 1.25e-03 | 2.61e-01 | 1.68e-02 | 5.18e-03 | 7.06e-05 | 0.518 | 0.104 | 469 | 2402 |
| O3 | G2 | 11 | 7.56e-02 | 1.27e-03 | 6.15e-02 | 1.16e-02 | 2.84e-03 | 5.32e-05 | 0.578 | 0.077 | 472 | 2407 |
| O3 | G3 | 10 | 7.18e-01 | 2.88e-02 | 4.67e-02 | 7.16e-03 | 1.95e-03 | 2.28e-05 | 1.328 | 0.063 | 440 | 2260 |
| O3 | G3 | 11 | 5.78e-02 | 6.12e-04 | 3.90e-02 | 1.28e-02 | 2.24e-03 | 2.03e-05 | 0.454 | 0.067 | 430 | 2219 |

O2: median e_1, formal 1.66e-01, 100k run 5.42e-03; below 0.5%: 2 of 6, below 1%: 6 of 6; median E2 0.107 (formal 0.994).

O3: median e_1, formal 7.03e-02, 100k run 2.54e-03; below 0.5%: 5 of 6, below 1%: 6 of 6; median E2 0.072 (formal 0.548).

For reference, the formal 20,000-iteration runs of the same geometries and seeds: O4 median e_1 5.76e-04 (E2 0.036), O1 4.28e-04 (E2 0.084).

