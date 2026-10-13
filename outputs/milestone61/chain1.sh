set -u
cd /home/sojak/sam2
L=outputs/milestone61/logs; mkdir -p $L
run() { uv run python scripts/m61.py "$@" > $L/$1.log 2>&1; echo "$1 exit=$?" >> $L/chain1.status; }
run tplr
run dtrain
run head_pc
if uv run python -c "import json,sys;sys.exit(0 if json.load(open('outputs/milestone61/trackB/positive_control.json'))['pass'] else 1)"; then
  run head_train
  run head_image
  run head_video DAVIS_val,PUMaVOS_B
else
  echo "head_pc FAILED -> skip main" >> $L/chain1.status
fi
echo DONE >> $L/chain1.status
