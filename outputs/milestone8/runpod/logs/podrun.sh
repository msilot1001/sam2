#!/bin/bash
# usage: podrun.sh <log> <command...>   -- runs a command in the M8 pod environment, logging to /workspace/logs/<log>
source /workspace/m8env.sh
mkdir -p /workspace/logs
LOG=/workspace/logs/$1; shift
echo "[$(date -u +%FT%TZ)] START $*" > $LOG
"$@" >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] EXIT $?" >> $LOG
