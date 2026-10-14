#!/bin/bash
cd /home/sojak/sam2
for i in 1 2; do
  [ -f outputs/milestone8/STOP_UBUNTU_DIAG_$i ] && continue
  OMP_NUM_THREADS=4 M7_SHARD=$i/3 uv run python scripts/m8.py diag DEV,REP > outputs/milestone8/logs/diag_ubuntu_${i}of3.log 2>&1
done
OMP_NUM_THREADS=4 M7_SHARD=0/3 uv run python scripts/m8.py diag1b DEV,REP > outputs/milestone8/logs/diag1b_ubuntu_0of3.log 2>&1
echo CHAIN_DIAG_DONE >> outputs/milestone8/logs/diag1b_ubuntu_0of3.log
