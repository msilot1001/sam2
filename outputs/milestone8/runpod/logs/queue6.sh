#!/bin/bash
source /workspace/m8env.sh
L=outputs/milestone8/runpod/logs
until grep -q QUEUE5_DONE $L/queue5.log 2>/dev/null; do sleep 30; done
for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run FLOWCARRY VK > $L/runVK_FLOWCARRY_${i}of4.log 2>&1 & done; wait
echo "FLOWCARRY@VK $(date -u +%FT%TZ)" >> $L/queue6.log
echo QUEUE6_DONE >> $L/queue6.log
