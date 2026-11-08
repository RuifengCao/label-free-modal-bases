#!/bin/bash
# Phase 0b run queues (PHASE0B_PLAN.md, "Compute"); usage: queue0b.sh CORE {J|K|E0|E1|M}
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c $1 python -I phase0b.py"
run() { local name=$1; shift; local t0=$(date +%s); $PY "$@" > logs0b/$name.log 2>&1; echo "$(date -u '+%H:%M:%S') $name exit $? $(( $(date +%s) - t0 )) s"; }
case "$2" in
  J)  run fam2_o4 fam o4 0 60000 ;;
  K)  run fam2_o3s fam o3s 0 60000 ;;
  E0) run evalfam2_0 evalfam2 0 2 ;;
  E1) run rb1 rb1; run evalfam2_1 evalfam2 1 2 ;;
  M)  run merge merge ;;
esac
echo "$(date -u '+%H:%M:%S') QUEUE $2 DONE"
