#!/bin/bash
# Smoke test: few iterations, separate folder; usage: smoke.sh CORE PART
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P0_OUT=/home/claude/udc/phase0/smoke P0_SMOKE=1 P0_LOG=10
PY="taskset -c $1 python -I phase0.py"
run() { local name=$1; shift; $PY "$@" > smoke/logs/$name.log 2>&1; echo "$name exit $?"; }
if [ "$2" = "A" ]; then
  run train_o1 train o1 0 20 hard_angle
  run train_o2 train o2 0 20 hard_angle
  run train_o3 train o3 0 20 hard_angle
  run train_o4 train o4 0 20 hard_angle
  run train_o4_hard train o4 0 20 hard
  run long long
  run evalg0 evalg0
else
  run fam fam 0 20
  run evalfam evalfam
  run gap gap
  run sens sens
  run cost cost
fi
