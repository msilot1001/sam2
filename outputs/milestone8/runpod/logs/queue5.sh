#!/bin/bash
source /workspace/m8env.sh
L=outputs/milestone8/runpod/logs
until grep -q QUEUE4_DONE $L/queue4.log 2>/dev/null; do sleep 30; done
for A in K_ADMIT_FLOWVEL; do
  for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run $A DEV,REP > $L/run_${A}_${i}of4.log 2>&1 & done; wait
  echo "$A $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$A/*.csv | wc -l)" >> $L/queue5.log
done
echo QUEUE5_DONE >> $L/queue5.log
