#!/bin/bash
# Local-disk working dir for the pod: data copied to the container disk (fast), code/outputs/checkpoints linked to the
# persistent /workspace repo (outputs MUST stay on /workspace until Ubuntu has verified them).
set -e
R=/workspace/sam2-m7-test; W=/root/w
mkdir -p $W
for d in src scripts tests configs pyproject.toml outputs checkpoints CODE_SYNC.json; do ln -sfn $R/$d $W/$d; done
date; cp -a $R/data $W/data.tmp && rm -rf $W/data && mv $W/data.tmp $W/data; date
du -sh $W/data
echo LOCALIZE_DONE
