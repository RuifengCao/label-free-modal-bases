"""Audit test: run_formal.check_stamp accepts a matching STAMP.txt and refuses a missing, changed or incomplete one.

Works on a temporary STAMP.txt and restores the folder afterwards; it never starts any job.
"""
import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_formal as R

fn = os.path.join(HERE, "STAMP.txt")
backup = fn + ".audit_backup"
had = os.path.exists(fn)
if had:
    shutil.move(fn, backup)
try:
    ok, bad = R.check_stamp()
    assert not ok and "not found" in bad[0], bad                                   # missing
    hashes = R.current_hashes()
    assert "missing" not in hashes.values(), [k for k, v in hashes.items() if v == "missing"]
    with open(fn, "w") as f:
        f.writelines("%s %s\n" % (h, rel) for rel, h in hashes.items())
    ok, bad = R.check_stamp()
    assert ok and not bad, bad                                                     # matching
    rel0 = R.STAMPED[0]
    with open(fn, "w") as f:
        f.writelines("%s %s\n" % (("0" * 64 if rel == rel0 else h), rel) for rel, h in hashes.items())
    ok, bad = R.check_stamp()
    assert not ok and bad == ["%s: changed since the stamp" % rel0], bad          # one file changed
    with open(fn, "w") as f:
        f.writelines("%s %s\n" % (h, rel) for rel, h in list(hashes.items())[1:])
    ok, bad = R.check_stamp()
    assert not ok and bad == ["%s: not in STAMP.txt" % rel0], bad                 # one file not stamped
    print("stamp check tests passed (%d files)" % len(hashes))
finally:
    if os.path.exists(fn):
        os.remove(fn)
    if had:
        shutil.move(backup, fn)
