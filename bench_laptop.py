"""Laptop calibration for the Unified 2027 runs (timing only; no result is read from it).

Run from the folder that holds bench.py and the model scripts:
    python bench_laptop.py            # about 5 to 8 minutes
    python bench_laptop.py --quick    # a 1-minute smoke test of this script
Writes bench_results/laptop_<UTC time>.json and prints a summary.

What it measures:
  1. which logical CPUs are fast (performance cores) and which are slow (efficient cores), by a short spin test;
  2. one single-thread process at a time on a fast and on a slow core: ms per iteration of
     o4 (log-det, one geometry), o1 (per-step potential, 16 cases x 640 steps), fam4 (log-det, four geometries);
  3. many single-thread o4 processes at once (one per physical fast core; plus all slow cores; all logical CPUs):
     total iterations per second, expressed in sandbox-core equivalents (one sandbox core: 14.1 ms per o4 iteration).
"""
import os, sys, json, time, platform, subprocess, datetime
from importlib import metadata
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "bench_results")
SANDBOX_MS = dict(o4=14.1, o1=26.0, fam4=60.1)            # one sandbox core, measured 2026-11-06
QUICK = "--quick" in sys.argv
PLAN = dict(total_core_s=49723, phase0_core_s=12644, parts12_core_s=37079,   # sandbox single-core seconds
            fam_iters=60000, o1_iters=50000)


_LOG = []


def log(*a):
    """Print and keep a copy, written to bench_results/log_<time>.txt at the end and after each stage."""
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _LOG.append(msg)


def flush_log(tag):
    with open(os.path.join(OUTDIR, "log_%s.txt" % tag), "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")


def env1(n_time=None):
    e = dict(os.environ)
    e.update(XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
             OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if n_time:
        e["BENCH_N_TIME"] = str(n_time)
    else:
        e.pop("BENCH_N_TIME", None)
    return e


def start(which, cpu, n_time=None):
    p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "bench.py"), which], cwd=HERE, env=env1(n_time),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        psutil.Process(p.pid).cpu_affinity([cpu])
    except Exception as ex:                                   # report, keep going
        log("could not pin process to CPU %d: %s" % (cpu, ex))
    return p


def collect(p, timeout=1800):
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return dict(error="timeout", tail=out[-1500:])
    recs = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    if p.returncode != 0 or not recs:
        return dict(error="exit code %s" % p.returncode, tail=out[-1500:])
    return recs[-1]


def spin(cpu, seconds):
    me = psutil.Process()
    me.cpu_affinity([cpu])
    time.sleep(0.05)
    n, t0 = 0, time.perf_counter()
    while True:
        for _ in range(20000):
            n += 1
        dt = time.perf_counter() - t0
        if dt >= seconds:
            return n / dt


def classify(seconds):
    me = psutil.Process()
    cpus = sorted(me.cpu_affinity())
    speed = {c: spin(c, seconds) for c in cpus}
    me.cpu_affinity(cpus)
    vals = sorted(speed.values())
    gaps = [(vals[i + 1] / vals[i], i) for i in range(len(vals) - 1)]
    ratio, i = max(gaps) if gaps else (1.0, 0)
    if ratio < 1.15:                                          # no clear split
        return speed, cpus, [], ratio
    thr = vals[i + 1]
    return speed, [c for c in cpus if speed[c] >= thr], [c for c in cpus if speed[c] < thr], ratio


def versions():
    v = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor())
    for pkg in ("jax", "jaxlib", "numpy", "scipy", "scikit-fem", "triangle", "psutil"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()
    res = dict(versions=versions(), logical=psutil.cpu_count(logical=True), physical=psutil.cpu_count(logical=False),
               memory_gb=round(psutil.virtual_memory().total / 2 ** 30, 1), quick=QUICK,
               utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    tag = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + ("_quick" if QUICK else "")
    log(json.dumps(res))
    flush_log(tag)

    # 1. fast and slow logical CPUs
    speed, fast, slow, ratio = classify(0.15 if QUICK else 0.4)
    smt_pairs = res["logical"] - res["physical"]
    reps = fast[::2] if (smt_pairs > 0 and len(fast) == 2 * smt_pairs) else fast   # one logical CPU per physical fast core
    res.update(spin_speed={str(k): round(v) for k, v in speed.items()}, fast=fast, slow=slow, split_ratio=round(ratio, 3),
               fast_physical_reps=reps)
    log("fast CPUs:", fast, "slow CPUs:", slow, "speed ratio at the split: %.2f" % ratio)
    flush_log(tag)

    # 2. one process at a time
    n_single = dict(o4=60, o1=10, fam4=20) if QUICK else dict(o4=600, o1=100, fam4=200)
    single = {}
    for label, cpu in (("fast", reps[0]), ("slow", slow[0] if slow else None)):
        if cpu is None:
            continue
        for which in ("o4", "o1", "fam4"):
            r = collect(start(which, cpu, n_single[which]))
            single["%s_%s" % (label, which)] = dict(cpu=cpu, **r)
            log(label, which, json.dumps(r))
            flush_log(tag)
    res["single"] = single

    # 3. many processes at once (o4)
    n_many = 150 if QUICK else 1500
    batches = [("fast_physical", reps), ("fast_physical_plus_slow", reps + slow), ("all_logical", sorted(speed))]
    seen, many = set(), {}
    for name, cpus in batches:
        key = tuple(cpus)
        if not cpus or key in seen:
            continue
        seen.add(key)
        t0 = time.time()
        procs = [(c, start("o4", c, n_many)) for c in cpus]
        rs = [(c, collect(p)) for c, p in procs]
        ok = [r for _, r in rs if "per_iter_ms" in r]
        thr = sum(1000.0 / r["per_iter_ms"] for r in ok)
        many[name] = dict(cpus=cpus, n_ok=len(ok), n=len(rs), wall_s=round(time.time() - t0, 1),
                          per_iter_ms={str(c): (r.get("per_iter_ms") or r.get("error")) for c, r in rs},
                          rss_mb_max=max([r["rss_mb"] for r in ok], default=None),
                          iters_per_s=round(thr, 1), sandbox_core_equivalents=round(thr / (1000.0 / SANDBOX_MS["o4"]), 2))
        log(name, json.dumps({k: v for k, v in many[name].items() if k != "per_iter_ms"}))
        flush_log(tag)
    res["many"] = many

    # 4. summary and projection for the planned runs
    summ = {}
    for label in ("fast", "slow"):
        for which in ("o4", "o1", "fam4"):
            r = single.get("%s_%s" % (label, which))
            if r and "per_iter_ms" in r:
                summ["%s_%s_speed_vs_sandbox" % (label, which)] = round(SANDBOX_MS[which] / r["per_iter_ms"], 2)
    eq_all = max([m["sandbox_core_equivalents"] for m in many.values()], default=None)
    fam_fast = single.get("fast_fam4", {}).get("per_iter_ms")
    fam_slow = single.get("slow_fam4", {}).get("per_iter_ms")
    o1_slow = single.get("slow_o1", {}).get("per_iter_ms") or single.get("fast_o1", {}).get("per_iter_ms")
    if eq_all and fam_fast:
        crit = PLAN["fam_iters"] * fam_fast / 1000 / 60
        summ.update(
            best_throughput_sandbox_core_equivalents=eq_all,
            family_run_minutes_on_fast_core=round(crit, 1),
            family_run_minutes_on_slow_core=round(PLAN["fam_iters"] * fam_slow / 1000 / 60, 1) if fam_slow else None,
            o1_run_minutes_on_slow_core=round(PLAN["o1_iters"] * o1_slow / 1000 / 60, 1) if o1_slow else None,
            phase0_minutes=round(max(crit, PLAN["phase0_core_s"] / eq_all / 60), 1),
            parts12_minutes=round(max(crit, PLAN["parts12_core_s"] / eq_all / 60), 1),
            note="phase0 and parts12 minutes assume the family runs sit on fast cores; otherwise use the slow-core figure")
    res["summary"] = summ
    res["bench_wall_s"] = round(time.time() - t_start, 1)
    fn = os.path.join(OUTDIR, "laptop_%s.json" % tag)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log("\nSUMMARY", json.dumps(summ, indent=1))
    log("written:", fn)
    flush_log(tag)


if __name__ == "__main__":
    main()
