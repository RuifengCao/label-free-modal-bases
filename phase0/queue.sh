#!/bin/bash
# Phase 0 run queues (PHASE0_PLAN.md, "Compute"); usage: queue.sh CORE {A|B1|B2|E}
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c $1 python -I phase0.py"
run() { local name=$1; shift; local t0=$(date +%s); $PY "$@" > logs/$name.log 2>&1; echo "$(date -u '+%H:%M:%S') $name exit $? $(( $(date +%s) - t0 )) s"; }
case "$2" in
  A)  run fam fam 0 60000; run long long; run evalfam evalfam ;;
  B1) for obj in o4 o2 o3; do for s in 0 1 2; do run train_${obj}_s${s} train $obj $s 20000 hard_angle; done; done ;;
  B2) for s in 0 1 2; do run train_o1_s${s} train o1 $s 20000 hard_angle; done
      run train_o4_s0_hard train o4 0 20000 hard; run gap gap; run sens sens; run cost cost ;;
  E)  run evalg0 evalg0 ;;
esac
echo "$(date -u '+%H:%M:%S') QUEUE $2 DONE"
