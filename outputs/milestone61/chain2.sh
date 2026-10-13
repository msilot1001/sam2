cd /home/sojak/sam2
L=outputs/milestone61/logs
run() { uv run python scripts/m61.py "$@" > $L/$1_$2.log 2>&1; echo "$1 $2 exit=$?" >> $L/chain2.status; }
uv run python scripts/m61.py head_train > $L/head_train.log 2>&1; echo "head_train exit=$?" >> $L/chain2.status
uv run python scripts/m61.py head_image > $L/head_image.log 2>&1; echo "head_image exit=$?" >> $L/chain2.status
run head_video DAVIS_val,PUMaVOS_B
run head_video PUMaVOS_A,SegTrack
echo DONE >> $L/chain2.status
