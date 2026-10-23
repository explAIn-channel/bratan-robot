#!/bin/sh
# Unpack /tmp/bratan.tgz as a new release and point app -> it. Does not switch runtimes.
set -e
B=/opt/robotbrain/bratan
R=$B/releases/$(date +%Y%m%d-%H%M%S)
mkdir -p $R $B/state $B/logs
tar xzf /tmp/bratan.tgz -C $R
ln -sfn $R $B/app.new && mv -Tf $B/app.new $B/app
[ -f $B/config.json ] || echo '{"autonomy": false}' > $B/config.json
ls -1dt $B/releases/* | tail -n +6 | xargs -r rm -rf
echo installed $R
