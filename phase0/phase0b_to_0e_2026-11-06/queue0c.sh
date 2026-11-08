#!/bin/bash
# Phase 0c runs (PHASE0C_PLAN.md); usage: queue0c.sh CORE "OBJ:SEED OBJ:SEED ..."
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for js in $2; do o=${js%%:*}; s=${js##*:}; t0=$(date +%s)
  taskset -c $1 python -I phase0c.py train $o $s > logs0c/sep_${o}_s${s}.log 2>&1
  echo "$(date -u '+%H:%M:%S') sep_${o}_s${s} exit $? $(( $(date +%s) - t0 )) s"; done
echo "$(date -u '+%H:%M:%S') QUEUE0C DONE"
