"""Start the formal runs of the Unified 2027 dynamics paper on this computer.

  python run_formal.py --smoke     every job type with a few iterations (16 jobs, about 10 minutes); needs no stamp
  python run_formal.py             the formal runs (63 jobs, about 8 hours); refuses to start unless STAMP.txt matches the files

Each job is a separate Python process pinned to one logical CPU (default 2, 4, 6, 8: one thread on each of four
performance cores of an i9-13900HX). Progress is written to results/progress.txt (results_smoke/ for --smoke).
Run it again after an interruption: finished jobs are skipped; a job that failed or was cut off is run once more
and the rerun is recorded. While it runs, Windows is asked not to sleep; the request ends when it stops.
"""
import os, sys, json, time, hashlib, subprocess, argparse, platform

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
STAMPED = ["PREREG.md", "phase1.py", "phase0.py", "phase0b.py", "phase0c.py", "phase0d.py", "phase0e.py", "geometries.json", "run_formal.py",
           "../fem_cantilever.py", "../lb_model.py", "../run_check.py", "../posthoc.py", "../posthoc2.py",
           "../posthoc3.py", "../posthoc4.py", "../posthoc6.py", "../posthoc7.py"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_hashes():
    return {rel: (sha256(os.path.join(HERE, rel)) if os.path.exists(os.path.join(HERE, rel)) else "missing")
            for rel in STAMPED}


def check_stamp():
    fn = os.path.join(HERE, "STAMP.txt")
    if not os.path.exists(fn):
        return False, ["STAMP.txt not found: the pre-registration has not been stamped yet"]
    stamped = {}
    with open(fn) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                stamped[parts[1]] = parts[0]
    bad = []
    for rel, h in current_hashes().items():
        if stamped.get(rel) != h:
            bad.append("%s: %s" % (rel, "not in STAMP.txt" if rel not in stamped else "changed since the stamp"))
    return not bad, bad


class Job:
    def __init__(self, name, args, output, after=()):
        self.name, self.args, self.output, self.after = name, args, output, set(after)
        self.attempts, self.status, self.cpu, self.t0, self.proc, self.log = 0, "pending", None, None, None, None


def job_list(smoke):
    p2 = [Job("p2_%s_s%d" % (o, s), ["p2", o, str(s)], "p2_%s_s%d.pkl" % (o, s))
          for (o, s) in ((("o4", 0), ("o3s", 0), ("sup", 1)) if smoke else
                         [(o, s) for o in ("o4", "o3s", "sup") for s in (0, 1, 2)])]
    p1 = [Job("p1_%s_%s_s%d" % (g, o, s), ["p1", g, o, str(s)], "p1_%s_%s_s%d.json" % (g, o, s))
          for (g, o, s) in ((("G1", "o1", 10), ("G2", "o2", 11), ("G3", "o3", 10), ("G1", "o4", 11)) if smoke else
                            [(g, o, s) for g in ("G1", "G2", "G3") for o in ("o1", "o2", "o3", "o4") for s in (10, 11)])]
    p1sep = [Job("p1sep_%s_%s_s%d" % (g, o, s), ["p1", g, o, str(s), "sep"], "p1sep_%s_%s_s%d.json" % (g, o, s))
             for (g, o, s) in ((("G3", "o3", 10), ("G2", "o2", 11), ("G1", "o4", 11)) if smoke else
                               [(g, o, s) for g in ("G1", "G2", "G3") for o in ("o2", "o3", "o4") for s in (10, 11)])]
    p2sep = [Job("p2sep_%s_s%d" % (o, s), ["p2", o, str(s), "sep"], "p2sep_%s_s%d.pkl" % (o, s))
             for (o, s) in ((("o4", 2), ("o3s", 2)) if smoke else [(o, s) for o in ("o4", "o3s") for s in (0, 1, 2)])]
    nshard = 2 if smoke else 4
    after_p2 = [j.name for j in p2 + p2sep]
    e2 = [Job("e2_shard%d" % k, ["e2", str(k), str(nshard)], "e2_shard%d.json" % k, after_p2) for k in range(nshard)]
    e2 += [Job("e2_train", ["e2train"], "e2_train.json", after_p2)]
    ver = [Job("verdicts", ["verdicts"], "verdicts.json", [j.name for j in p1 + p1sep + e2])]
    return p2 + p2sep + p1sep + p1 + e2 + ver                     # longest jobs first


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
    a = ap.parse_args()
    out = os.path.join(HERE, "results_smoke" if a.smoke else "results")
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)
    progress_txt, state_fn = os.path.join(out, "progress.txt"), os.path.join(out, "state.json")

    def note(msg):
        line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(progress_txt, "a") as f:
            f.write(line + "\n")

    hashes = current_hashes()
    if not a.smoke:
        ok, bad = check_stamp()
        if not ok:
            note("NOT STARTED. " + "; ".join(bad))
            sys.exit(2)
    with open(os.path.join(out, "environment.json"), "w") as f:
        import numpy, scipy, jax
        json.dump(dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                       numpy=numpy.__version__, scipy=scipy.__version__, jax=jax.__version__,
                       logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                       file_hashes=hashes, smoke=a.smoke), f, indent=1)
    cpus = [int(x) for x in a.cpus.split(",")]
    jobs = job_list(a.smoke)
    by_name = {j.name: j for j in jobs}
    prev = json.load(open(state_fn)) if os.path.exists(state_fn) else {}
    for j in jobs:
        if os.path.exists(os.path.join(out, j.output)):
            j.status = "done"
        elif j.name in prev and prev[j.name]["status"] in ("running", "failed_once"):
            j.attempts = 1                                         # cut off or failed in an earlier start
            note("%s: earlier attempt did not finish (%s); it will be run once more" % (j.name, prev[j.name]["status"]))
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               P1_OUT=out, P1_SMOKE="1" if a.smoke else "0")

    def save_state():
        with open(state_fn, "w") as f:
            json.dump({j.name: dict(status=j.status, attempts=j.attempts, cpu=j.cpu) for j in jobs}, f, indent=1)

    note("START %s: %d jobs, %d already done; CPUs %s" % ("smoke test" if a.smoke else "formal runs", len(jobs),
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
            for j in jobs:                                         # dependants of failed jobs cannot run
                if j.status == "pending" and any(by_name[x].status == "failed" for x in j.after):
                    j.status = "failed"; note("%s: skipped, a job it needs failed" % j.name); save_state()
            for j in jobs:                                         # start what can start
                if not free:
                    break
                if j.status == "pending" and all(by_name[x].status == "done" for x in j.after):
                    cpu = free.pop(0)
                    j.attempts += 1; j.cpu = cpu; j.t0 = time.time(); j.status = "running"
                    j.log = open(os.path.join(logdir, j.name + ".log"), "a")
                    j.log.write("\n=== attempt %d, %s ===\n" % (j.attempts, time.strftime("%Y-%m-%d %H:%M:%S"))); j.log.flush()
                    j.proc = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "phase1.py")] + j.args, cwd=HERE,
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
    if os.path.exists(os.path.join(out, "verdicts.md")):
        note("verdicts: " + os.path.join(out, "verdicts.md"))


if __name__ == "__main__":
    main()
