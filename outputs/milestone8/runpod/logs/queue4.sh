#!/bin/bash
# M8 RunPod queue 4 (after queue 3): VK finalist run (SGGL, FLOWVEL, M3vel), then FLOWVEL ablations on DEV+REP
source /workspace/m8env.sh
L=outputs/milestone8/runpod/logs
until grep -q QUEUE3_DONE $L/queue3.log 2>/dev/null; do sleep 30; done
for A in SGGL FLOWVEL M3vel; do
  for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run $A VK > $L/runVK_${A}_${i}of4.log 2>&1 & done; wait
  echo "$A@VK $(date -u +%FT%TZ)" >> $L/queue4.log
done
for A in FV_PRESENT FLOWCARRY CAMCARRY; do
  for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run $A DEV,REP > $L/run_${A}_${i}of4.log 2>&1 & done; wait
  echo "$A $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$A/*.csv | wc -l)" >> $L/queue4.log
done
echo QUEUE4_DONE >> $L/queue4.log
