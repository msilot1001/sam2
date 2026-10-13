cd /home/sojak/sam2
L=outputs/milestone61/logs
until grep -q DONE $L/chain2.status 2>/dev/null; do sleep 20; done
uv run python scripts/m61.py dual A2d,A3,A3s > $L/dual.log 2>&1; echo "dual exit=$?" >> $L/chain3.status
echo DONE >> $L/chain3.status
