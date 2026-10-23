#!/bin/sh
# switch.sh bratan|integrated|off : choose which runtime the boot hook starts (off: none), then restart it.
set -e
case "$1" in bratan|integrated|off) ;; *) echo "usage: switch.sh bratan|integrated|off"; exit 2;; esac
echo "$1" > /opt/robotbrain/active-runtime
# Stop whichever runtime is running; the boot hook loop starts the selected one within ~5 s.
for p in $(ps -eo pid,args | awk '/-m runtime.service|-m bratan.main|run_edge_worker|-m runtime.vision|-m runtime.speech|-m bratan.(body|eyes|voice).main/ && !/awk/ {print $1}'); do kill -TERM $p 2>/dev/null || true; done
sleep 3
for p in $(ps -eo pid,args | awk '/-m runtime.service|-m bratan.main|-m runtime.vision|-m runtime.speech|-m bratan.(body|eyes|voice).main/ && !/awk/ {print $1}'); do kill -KILL $p 2>/dev/null || true; done
echo "active: $1"
