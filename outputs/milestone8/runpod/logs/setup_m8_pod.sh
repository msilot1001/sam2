#!/usr/bin/env bash
# M8 RunPod worker setup (resumes the stopped /tmp/setup_m7_runpod.sh from step 4). Idempotent.
set -Eeuo pipefail
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND" >&2' ERR
BASE=/workspace; REPO=$BASE/sam2-m7-test; SAM2=$BASE/sam2-smoke-test; VENV=$BASE/.venvs/m7; PY=$VENV/bin/python
COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
echo "== step 4: research deps"; date
$PY -m pip install -q numpy pandas pillow matplotlib pyyaml scipy pytest ninja tqdm hydra-core iopath setuptools wheel packaging scikit-image imageio
# a dependency can replace the cu128 torchvision by the PyPI build (torchvision::nms missing) -> pin the CUDA build
$PY -m pip install -q --no-deps --force-reinstall torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo "== step 5: SAM2 source @ $COMMIT"; date
[ -d $SAM2/.git ] || git clone -q https://github.com/facebookresearch/sam2.git $SAM2
cd $SAM2
if [ "$(git rev-parse HEAD)" != "$COMMIT" ]; then git fetch -q origin $COMMIT; git checkout -q --detach $COMMIT; fi
SAM2_BUILD_CUDA=0 $PY -m pip install -q --no-build-isolation -e $SAM2
echo "== step 6: SAM2 CUDA extension"; date
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=$($PY -c 'import torch;a,b=torch.cuda.get_device_capability();print(f"{a}.{b}")')
[ -f $SAM2/sam2/_C.so ] || SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 MAX_JOBS=4 $PY setup.py build_ext --inplace > $BASE/build_ext.log 2>&1
$PY -c 'import torch; from sam2 import _C; print("SAM2 _C:", _C.__file__)'
echo "== step 7: checkpoint"; date
mkdir -p $SAM2/checkpoints
CKPT=$SAM2/checkpoints/sam2.1_hiera_tiny.pt
[ -s $CKPT ] || { curl -fsSL --retry 3 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o $CKPT.tmp; mv $CKPT.tmp $CKPT; }
sha256sum $CKPT
echo "== step 9: DAVIS 2017 trainval 480p"; date
mkdir -p $REPO/data/davis2017 $REPO/checkpoints
[ -e $REPO/checkpoints/sam2.1_hiera_tiny.pt ] || ln -s $CKPT $REPO/checkpoints/sam2.1_hiera_tiny.pt
D=$REPO/data/davis2017
if [ ! -f $D/DAVIS/ImageSets/2017/val.txt ]; then
  Z=/root/DAVIS-2017-trainval-480p.zip
  [ -s $Z ] || curl -fsSL --retry 3 https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -o $Z
  unzip -tq $Z && unzip -oq $Z -d $D
fi
ls $D/DAVIS
echo "SETUP_POD_DONE"; date
