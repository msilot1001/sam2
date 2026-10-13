cd /home/sojak/sam2
L=outputs/milestone61/logs
until grep -q DONE $L/chain3.status 2>/dev/null; do sleep 20; done
uv run python scripts/m61.py flip2 DAVIS_val,PUMaVOS_B > $L/flip2.log 2>&1; echo "flip2 exit=$?" >> $L/chain4.status
echo DONE >> $L/chain4.status
