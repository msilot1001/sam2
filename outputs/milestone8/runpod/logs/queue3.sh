#!/bin/bash
# M8 RunPod queue 3 (after queue 2): stage-3 combinations, 4 shards each
source /workspace/m8env.sh
L=outputs/milestone8/runpod/logs
until grep -q QUEUE2_DONE $L/queue2.log 2>/dev/null; do sleep 30; done
for A in K_OCLR_M3 K_OCLR_FLOWVEL; do
  for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run $A DEV,REP > $L/run_${A}_${i}of4.log 2>&1 & done; wait
  echo "$A $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$A/*.csv | wc -l)" >> $L/queue3.log
done
echo QUEUE3_DONE >> $L/queue3.log
