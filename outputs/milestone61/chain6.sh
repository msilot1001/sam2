cd /home/sojak/sam2
until [ -e outputs/milestone61/logs/chain5.status ]; do sleep 20; done
uv run python scripts/m61.py veto dev,train > outputs/milestone61/logs/veto.log 2>&1; echo "veto exit=$?" > outputs/milestone61/logs/chain6.status
