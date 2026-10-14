#!/bin/bash
# M8 RunPod queue 2 (after queue 1): stage-2 rescue arms on all 64 units (4 shards), then oracle arms on the 11 units whose
# stage-1 diag ran on the pod (the other 53 run on Ubuntu, matched to the Ubuntu diag SGGL)
source /workspace/m8env.sh
L=outputs/milestone8/runpod/logs
until grep -q QUEUE1_DONE $L/queue1.log 2>/dev/null; do sleep 30; done
run4() { for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 $PY scripts/m8.py run $1 DEV,REP > $L/run_$1_${i}of4.log 2>&1 & done; wait; echo "$1 $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$1/*.csv | wc -l)" >> $L/queue2.log; }
for A in RESC_OCLR RESC_OCLR_ADMIT RESC_CPFR; do run4 $A; done
for A in ORACLE_POOL ORACLE_POOL_ADMIT; do
  for i in 0 1 2 3; do OMP_NUM_THREADS=3 M7_SHARD=$i/4 M8_SKIP=davis__davis_dogs-jump__obj1,davis__davis_drift-chicane__obj1,davis__davis_drift-straight__obj1,davis__davis_kite-surf__obj2,davis__davis_lab-coat__obj1,davis__davis_lab-coat__obj2,davis__davis_mbike-trick__obj1,davis__davis_mbike-trick__obj2,davis__davis_motocross-jump__obj2,davis__davis_scooter-black__obj1,davis__davis_scooter-black__obj2,davis__davis_soapbox__obj1,davis__davis_soapbox__obj2,davis__davis_soapbox__obj3,pumavos__pumavos_ice_cream__obj1,pumavos__pumavos_queen_car__obj1,pumavos__pumavos_eye_area__obj1,pumavos__pumavos_smile__obj1,segtrack__segtrack_birdfall__obj1,segtrack__segtrack_cheetah__obj1,segtrack__segtrack_parachute__obj1,davis__davis_car-turn__obj1,davis__davis_classic-car__obj1,davis__davis_classic-car__obj2,davis__davis_disc-jockey__obj3,davis__davis_dog-gooses__obj2,davis__davis_dog-gooses__obj3,davis__davis_dog-gooses__obj4,davis__davis_dog-gooses__obj5,davis__davis_drone__obj1,davis__davis_drone__obj2,davis__davis_drone__obj4,davis__davis_lady-running__obj2,davis__davis_longboard__obj2,davis__davis_longboard__obj5,davis__davis_miami-surf__obj1,davis__davis_miami-surf__obj3,davis__davis_miami-surf__obj4,davis__davis_miami-surf__obj6,davis__davis_motocross-bumps__obj1,davis__davis_motorbike__obj1,davis__davis_motorbike__obj2,davis__davis_night-race__obj1,davis__davis_night-race__obj2,davis__davis_rallye__obj1,davis__davis_schoolgirls__obj6,davis__davis_sheep__obj3,davis__davis_sheep__obj4,davis__davis_skate-park__obj1,davis__davis_skate-park__obj2,davis__davis_surf__obj2,davis__davis_swing__obj3,davis__davis_tractor-sand__obj2 $PY scripts/m8.py run $A DEV,REP > $L/run_${A}_${i}of4.log 2>&1 & done; wait
  echo "$A $(date -u +%FT%TZ) units=$(ls outputs/milestone8/runpod/runs/$A/*.csv | wc -l)" >> $L/queue2.log
done
echo QUEUE2_DONE >> $L/queue2.log
