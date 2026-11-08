"""Start the exploratory runs X1 and X2 of the Unified 2027 dynamics paper (EXPLORE_PLAN.md) on this computer.

  python run_explore.py --smoke     every job type on small meshes with a few iterations (6 jobs, a few minutes)
  python run_explore.py             the exploratory runs (21 jobs, about 3 to 4 hours)

Before starting, it checks that the formal code it imports still matches formal/STAMP.txt and that the formal networks
and evaluation records it reads still match formal/results/SHA256SUMS_formal_results.txt; it refuses to start otherwise.
Each job is a separate Python process pinned to one logical CPU (default 2, 4, 6, 8). The four FINE-mesh jobs run one
after another; the timing job runs last, alone, on the first CPU. Progress is written to results/progress.txt
(results_smoke/ for --smoke). Run it again after an interruption: finished jobs are skipped; a job that failed or was
cut off is run once more and the rerun is recorded. While it runs, Windows is asked not to sleep.
"""
import os, sys, json, time, hashlib, subprocess, argparse, platform

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAL = os.path.join(os.path.dirname(HERE), "formal")
OWN = ["EXPLORE_PLAN.md", "explore.py", "run_explore.py"]
NET_FILES = (["p2_%s_s%d.pkl" % (o, s) for o in ("o4", "o3s", "sup") for s in (0, 1, 2)]
             + ["p2sep_%s_s%d.pkl" % (o, s) for o in ("o4", "o3s") for s in (0, 1, 2)]
             + ["e2_shard%d.json" % k for k in range(4)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(fn):
    """'sha256 relpath' lines (STAMP.txt and SHA256SUMS files); other lines are ignored."""
    out = {}
    with open(fn) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                out[parts[1]] = parts[0]
    return out


def check_inputs(nets_dir, sums_fn):
    """The formal code must match formal/STAMP.txt; the formal networks and records read here must match the
    checksum list written on this computer after the formal run."""
    bad, hashes = [], {}
    stamp = os.path.join(FORMAL, "STAMP.txt")
    if not os.path.exists(stamp):
        return ["formal/STAMP.txt not found"], hashes
    for rel, h in read_sums(stamp).items():
        p = os.path.join(FORMAL, rel)
        cur = sha256(p) if os.path.exists(p) else "missing"
        hashes["formal/" + rel] = cur
        if cur != h:
            bad.append("formal/%s changed since the stamp" % rel)
    if not os.path.exists(sums_fn):
        return bad + ["%s not found" % sums_fn], hashes
    sums = read_sums(sums_fn)
    for rel in NET_FILES:
        p = os.path.join(nets_dir, rel)
        cur = sha256(p) if os.path.exists(p) else "missing"
        hashes["nets/" + rel] = cur
        if sums.get(rel) != cur:
            bad.append("%s: %s" % (rel, "missing" if cur == "missing" else "not in the checksum list" if rel not in sums
                                   else "differs from the checksum list"))
    return bad, hashes


class Job:
    def __init__(self, name, args, output, after=(), cpu=None):
        self.name, self.args, self.output, self.after, self.cpu_req = name, args, output, set(after), cpu
        self.attempts, self.status, self.cpu, self.t0, self.proc, self.log = 0, "pending", None, None, None, None


def job_list(smoke, cpus):
    if smoke:
        fine = [Job("fine_0", ["fine", "0"], "x1_fine0.json")]
        acc = [Job("acc_%d" % k, ["acc", str(k), "4"], "x1_acc_shard%d.json" % k) for k in (0, 1)]
        long_ = [Job("x2_%s_%s_s%d" % (g, o, s), ["long", g, o, str(s)], "x2_%s_%s_s%d.json" % (g, o, s))
                 for (g, o, s) in (("G1", "o2", 10), ("G2", "o3", 11))]
    else:
        fine = [Job("fine_%d" % k, ["fine", str(k)], "x1_fine%d.json" % k, ["fine_%d" % (k - 1)] if k else [])
                for k in range(4)]                                # one at a time (memory)
        acc = [Job("acc_%d" % k, ["acc", str(k), "4"], "x1_acc_shard%d.json" % k) for k in range(4)]
        long_ = [Job("x2_%s_%s_s%d" % (g, o, s), ["long", g, o, str(s)], "x2_%s_%s_s%d.json" % (g, o, s))
                 for o in ("o3", "o2") for g in ("G1", "G2", "G3") for s in (10, 11)]
    others = fine + acc + long_
    timing = Job("timing", ["timing"], "x1_timing.json", [j.name for j in others], cpu=cpus[0])   # alone, at the end
    return others + [timing]


def keep_awake(on):
    if platform.system() != "Windows":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpus", default="2,4,6,8")
    ap.add_argument("--nets", default=os.path.join(FORMAL, "results"))
    ap.add_argument("--sums", default=None, help="default: SHA256SUMS_formal_results.txt in the --nets folder")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sums_fn = a.sums or os.path.join(a.nets, "SHA256SUMS_formal_results.txt")
    out = a.out or os.path.join(HERE, "results_smoke" if a.smoke else "results")
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    progress_txt, state_fn = os.path.join(out, "progress.txt"), os.path.join(out, "state.json")

    def note(msg):
        line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(progress_txt, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    bad, in_hashes = check_inputs(a.nets, sums_fn)
    if bad:
        note("NOT STARTED. " + "; ".join(bad))
        sys.exit(2)
    with open(os.path.join(out, "environment.json"), "w") as f:
        import numpy, scipy, jax
        json.dump(dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                       numpy=numpy.__version__, scipy=scipy.__version__, jax=jax.__version__,
                       logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                       memory_gb=round(psutil.virtual_memory().total / 2 ** 30, 1),
                       file_hashes={rel: sha256(os.path.join(HERE, rel)) for rel in OWN}, inputs=in_hashes,
                       nets=os.path.abspath(a.nets), smoke=a.smoke, started=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=1)
    cpus = [int(x) for x in a.cpus.split(",")]
    jobs = job_list(a.smoke, cpus)
    by_name = {j.name: j for j in jobs}
    prev = json.load(open(state_fn)) if os.path.exists(state_fn) else {}
    for j in jobs:
        if os.path.exists(os.path.join(out, j.output)):
            j.status = "done"
        elif j.name in prev and prev[j.name]["attempts"] > 0:     # failed or cut off in an earlier start
            j.attempts = prev[j.name]["attempts"]
            if j.attempts >= 2:
                j.status = "failed"; note("%s: failed twice in an earlier start; not run again" % j.name)
            else:
                note("%s: earlier attempt did not finish (%s); it will be run once more" % (j.name, prev[j.name]["status"]))
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               EXP_OUT=out, EXP_NETS=os.path.abspath(a.nets), EXP_SMOKE="1" if a.smoke else "0")

    def save_state():
        with open(state_fn, "w") as f:
            json.dump({j.name: dict(status=j.status, attempts=j.attempts, cpu=j.cpu) for j in jobs}, f, indent=1)

    note("START %s: %d jobs, %d already done; CPUs %s" % ("smoke test" if a.smoke else "exploratory runs", len(jobs),
                                                          sum(j.status == "done" for j in jobs), cpus))
    keep_awake(True)
    free = list(cpus)
    try:
        while True:
            for j in jobs:                                         # collect finished jobs
                if j.status == "running" and j.proc.poll() is not None:
                    rc_, el = j.proc.returncode, time.time() - j.t0
                    j.log.close(); free.append(j.cpu)
                    if rc_ == 0 and os.path.exists(os.path.join(out, j.output)):
                        j.status = "done"; note("%s: done in %.0f s (CPU %d)" % (j.name, el, j.cpu))
                    elif j.attempts < 2:
                        j.status = "pending"; note("%s: FAILED (exit %s) after %.0f s; it will be run once more" % (j.name, rc_, el))
                    else:
                        j.status = "failed"; note("%s: FAILED again (exit %s); not run again" % (j.name, rc_))
                    save_state()
            for j in jobs:                                         # the order is for memory and timing only, so a
                if j.status == "pending":                          # failed job does not hold back the jobs after it
                    j.after = {x for x in j.after if by_name[x].status != "failed"}
            for j in jobs:                                         # start what can start
                if not free:
                    break
                if j.status == "pending" and all(by_name[x].status == "done" for x in j.after):
                    if j.cpu_req is not None:
                        if j.cpu_req not in free:
                            continue
                        cpu = j.cpu_req; free.remove(cpu)
                    else:
                        cpu = free.pop(0)
                    j.attempts += 1; j.cpu = cpu; j.t0 = time.time(); j.status = "running"
                    j.log = open(os.path.join(logdir, j.name + ".log"), "a")
                    j.log.write("\n=== attempt %d, %s ===\n" % (j.attempts, time.strftime("%Y-%m-%d %H:%M:%S"))); j.log.flush()
                    j.proc = subprocess.Popen([sys.executable, "-I", "-B", os.path.join(HERE, "explore.py")] + j.args, cwd=HERE,
                                              env=env, stdout=j.log, stderr=subprocess.STDOUT)
                    try:
                        psutil.Process(j.proc.pid).cpu_affinity([cpu])
                    except Exception as e:                       # noqa: BLE001
                        note("%s: could not pin to CPU %d (%s)" % (j.name, cpu, e))
                    note("%s: started on CPU %d (attempt %d)" % (j.name, cpu, j.attempts))
                    save_state()
            if all(j.status in ("done", "failed") for j in jobs):
                break
            time.sleep(2 if a.smoke else 10)
    finally:
        keep_awake(False)
        for j in jobs:
            if j.status == "running" and j.proc.poll() is None:
                j.proc.terminate()
        save_state()
    n_done = sum(j.status == "done" for j in jobs)
    note("END: %d of %d jobs done%s" % (n_done, len(jobs), "" if n_done == len(jobs) else "; see FAILED lines above"))


if __name__ == "__main__":
    main()
