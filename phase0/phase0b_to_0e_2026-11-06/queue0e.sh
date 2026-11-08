#!/bin/bash
# Phase 0e run queue (PHASE0E_PLAN.md, DEVIATIONS 5 to 7); usage: queue0e.sh CORE "arch:obj:seed:iters ..."
cd /home/claude/udc/phase0
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for js in $2; do
  IFS=: read arch obj seed iters <<< "$js"; t0=$(date +%s)
  taskset -c $1 python -I phase0e.py fam $arch $obj $seed $iters > logs0e/fam2${arch}_${obj}_s${seed}_${iters}.log 2>&1
  echo "$(date -u '+%H:%M:%S') fam2${arch}_${obj}_s${seed}_${iters} exit $? $(( $(date +%s) - t0 )) s"
done
echo "$(date -u '+%H:%M:%S') QUEUE0E DONE"
