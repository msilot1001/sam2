# source on the pod before any M8 job
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=/workspace/sam2-m7-test/src:/workspace/sam2-m7-test
export TORCH_HOME=/workspace/.cache/torch
export M8_MACHINE=runpod-rtx4090
export M8_RUNS_DIR=outputs/milestone8/runpod/runs
PY=/workspace/.venvs/m7/bin/python
cd /root/w
