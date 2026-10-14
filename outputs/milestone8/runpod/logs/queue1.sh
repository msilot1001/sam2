#!/bin/bash
# M8 RunPod queue 1: (a) stage-1 diag for the 11 shard-2 units not recorded on Ubuntu, (b) control arms, 3 shards each
source /workspace/m8env.sh
mkdir -p outputs/milestone8/runpod/logs
( OMP_NUM_THREADS=3 M7_SHARD=2/3 M8_DIAG_DIR=outputs/milestone8/runpod/diag M8_SKIP=davis__davis_drift-straight__obj1,davis__davis_lab-coat__obj2,davis__davis_motocross-jump__obj2,davis__davis_soapbox__obj1,pumavos__pumavos_ice_cream__obj1,pumavos__pumavos_smile__obj1,segtrack__segtrack_parachute__obj1,davis__davis_classic-car__obj2,davis__davis_dog-gooses__obj3,davis__davis_drone__obj1 \
    $PY scripts/m8.py diag DEV,REP > outputs/milestone8/runpod/logs/diag_runpod_2of3.log 2>&1; echo DIAG_DONE >> outputs/milestone8/runpod/logs/diag_runpod_2of3.log ) &
for ARM in SGGL M3vel FLOWVEL; do
  for i in 0 1 2; do OMP_NUM_THREADS=3 M7_SHARD=$i/3 $PY scripts/m8.py run $ARM DEV,REP > outputs/milestone8/runpod/logs/run_${ARM}_${i}of3.log 2>&1 & done
  wait -n; wait -n; wait -n 2>/dev/null
  while [ $(ls outputs/milestone8/runpod/runs/$ARM/*.csv 2>/dev/null | wc -l) -lt 64 ] && pgrep -f "m8.py run $ARM" >/dev/null; do sleep 10; done
  echo "$ARM $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$ARM/*.csv | wc -l)" >> outputs/milestone8/runpod/logs/queue1.log
done
wait
echo QUEUE1_DONE >> outputs/milestone8/runpod/logs/queue1.log
