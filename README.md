# label-free-modal-bases

Code and records of the study reported in *Learning the Vibration Modes of Damaged Structures Without Finite Element Eigen-Solutions* (Ruifeng Cao, in preparation).

The study asks which label-free training objective lets a network generate a finite element reduced basis that holds the lowest vibration modes of a structure, across a family of geometries, without any finite element eigen-solution in training. The model problem is a plane-stress cantilever with a U-shaped notch in its top edge, the notch standing for damage. The objectives compared are the per-step incremental potential of Newmark time stepping (O1), the trace of the projected stiffness (O2), sequential Rayleigh quotients in a fixed or a sorted column order (O3, O3s) and the log-determinant of the projected stiffness (O4). The finite element eigen-solutions are used only to evaluate the trained bases.

## Layout

Each stage was planned in writing and the plan hashed before it ran. `PLAN_HASH.txt` is the ledger of those hashes, of the code and of the records, in the order they were written.

**Top level: the model and the first checks** (5 and 6 November 2026, exploratory)
- `fem_cantilever.py`: geometry, graded quadratic triangle mesh, stiffness, consistent mass, Rayleigh damping, load patterns and the Newmark reference.
- `lb_model.py`: the network-generated basis with a per-step Galerkin solve, trained on Newmark incremental potentials (JAX, CPU, float64).
- `run_check.py`: the pipeline check; `posthoc.py`, `posthoc2.py`, `posthoc3.py`, `posthoc4.py`, `posthoc6.py` and `posthoc7.py`: its post hoc diagnostics.
- `bench.py`, `bench_laptop.py` and the `.bat` files: timing and the set-up of the laptop's environment; `bench_results/` holds their output. No result is read from them.

**`dyn_check/`**: the plan (`PLAN.md`), the results (`RESULTS.md`) and the notes of the sixth and seventh sets of post hoc diagnostics (`POSTHOC6.md`, `POSTHOC7.md`, with `posthoc7_w1.json` and `posthoc7_w2.json`) of the pipeline check. The notes of sets 1 to 5 (`POSTHOC.md` to `POSTHOC5.md`) are not kept; section 4 of `RESULTS.md` summarises them. In `PLAN_HASH.txt` the note of set 7 appears as `POSTHOC7.md`.

**`phase0/`: Phase 0** (exploratory): `PHASE0_PLAN.md`, `phase0.py` and its scripts, `RESULTS_PHASE0.md`, `TABLES_PHASE0.md`, `DEVIATIONS_PHASE0.md` (its final version, which also records Phases 0b to 0e), `PREREG_DRAFT.md`, the logs in `logs/` and the outputs in `out/`. `phase0/PLAN_HASH.txt` is the ledger as it stood after Phase 0.

**`phase0/phase0b_to_0e_2026-11-06/`: Phases 0b to 0e** (exploratory): their plans, code, results, tables and outputs, as they were returned in one archive, with the ledger as it stood then.

**`formal/`: the pre-registered formal runs**
- `PREREG.md`: the pre-registration, approved on 6 November 2026 at 23:02 UTC.
- `STAMP.txt`: the hashes of the code and of `geometries.json`. `run_formal.py` refuses to start unless every hash matches.
- `phase1.py`: the runs, with `phase0.py` and `phase0b.py` to `phase0e.py` unchanged from the exploratory phases; `run_formal.py` starts them; `test_stamp.py` and `test_verdicts.py` are its audit tests, and `tests_log.txt` their output.
- `results/`: the records of the 63 jobs, their hashes in `SHA256SUMS_formal_results.txt`, the progress log and `verdicts.md`; `results_smoke/`: the smoke test.
- `analysis/`: written after the runs from the records: `RESULTS_FORMAL.md`, the tables `TABLES_FORMAL_part1.md` and `TABLES_FORMAL_part2.md` with `tables_formal.py`, and the checks `recheck_verdicts.py` and `spot_check.py`.

**`explore/`: exploratory runs X1 to X3**, made after the formal verdicts to a plan hashed before they ran (`EXPLORE_PLAN.md`): `explore.py`, started by `run_explore.py`; the deviations (`DEVIATIONS_EXPLORE.md`), results (`RESULTS_EXPLORE.md`) and tables (`TABLES_EXPLORE.md`); the control with fine-mesh sources (`rbfine.py`, `rbfine_report.py`, `rbfine_table.py`, records in `rbfine/`); the shape errors of X3 (`x3_analysis.py`, `x3_figures.py`, records in `x3/`); the reports of X1 and X2 (`x12_report.py`); a check in `check_h02/`; the records in `results/` and `results_smoke/`, with their hashes in `results/SHA256SUMS_explore_results.txt`; the data of the figures in `figs/`; and `code_explore.md`, the code as it was reviewed.

## Where the runs were made

The pipeline check, Phases 0 to 0e and the analyses written after the runs (`formal/analysis/`, `explore/x3_analysis.py`, `explore/rbfine.py`) ran in a Linux sandbox. The formal runs and the exploratory runs X1 and X2 ran on the author's laptop (Windows 11, Intel Core i9-13900HX, CPU only, float64), in an environment made by `setup_venv_and_bench.bat`: Python 3.13 with jax 0.12.0.1, jaxlib 0.12.0.1, numpy 2.5.4, scipy 1.18.1, scikit-fem 12.0.2, triangle 20250106, psutil 7.2.2, ml-dtypes 0.6.0 and opt-einsum 3.4.0.

To repeat the formal runs, put the repository's files in one folder as here, install those packages and run `python run_formal.py --smoke` from `formal/` (about 10 minutes), then `python run_formal.py` (63 jobs, about eight hours on that laptop).

## Hashes and line endings

- `PLAN_HASH.txt`, `formal/STAMP.txt` and the `SHA256SUMS` files hold SHA-256 hashes of the files as they were written. Many records were written on Windows and end their lines with CR LF. `.gitattributes` keeps every file byte for byte, so the hashes can be checked on a checkout.
- `MANIFEST.sha256` lists the SHA-256 hash of every file of the first commit.
- Two hashed plans quote the author's instructions in Chinese, and are kept unchanged so that their hashes still match. In English, the quotation in `phase0/PHASE0_PLAN.md` reads "run O1 separately to 50,000 steps", and the one in `phase0/phase0b_to_0e_2026-11-06/PHASE0B_PLAN.md` reads "evaluate the thresholds of the geometry family, evaluate the nearest-geometry interpolation control, the rest as recommended, continue".

## Not in this repository

- The trained networks and their training histories: 113 pickle files, about 16 MB. The hashes of the 69 from the formal and exploratory runs are in `formal/results/SHA256SUMS_formal_results.txt` and `explore/results/SHA256SUMS_explore_results.txt`; the others are from Phases 0 to 0e and the smoke tests. The author keeps them; they are to be archived separately with the paper.
- The figures of the exploratory runs as first drawn (`explore/figs/*.pdf`, `*.png`), whose labels are in Chinese. Their data are in `explore/figs/*.json`.
- Three run notes in Chinese, two notes in Chinese and a script that wrote tables in Chinese (`zh_tables.py`) from the archive of Phases 0b to 0e, and the earlier version of `phase0/DEVIATIONS_PHASE0.md`.
- Six files named in `PLAN_HASH.txt` that were not kept: `explore/sbx_tests/test_launcher.py`, a test of the launcher run in the sandbox before the exploratory runs, and five early versions of the figures in `explore/x3/figs/`.
- The Python environment.

## Licence

The code in this repository (the `.py`, `.bat` and `.sh` files) is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): it may be used, changed and shared for any noncommercial purpose, as the licence defines it, including research and teaching. Commercial use needs a separate licence from the author, Ruifeng Cao (ruifeng.cao@manchester.ac.uk).

The plans, pre-registration, deviations, results and tables (the `.md` files), the hash ledgers and the records (the `.json`, `.log`, `.txt` and `.npz` files) are licensed instead under the [Creative Commons Attribution 4.0 International licence](LICENSE-CC-BY-4.0).

These terms apply to every version of these files in this repository, including earlier commits.
