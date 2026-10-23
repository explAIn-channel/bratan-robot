#!/bin/sh
# From the PC: package, upload and install (does not switch). Usage: sh deploy/push.sh
set -e
cd "$(dirname "$0")/.."
S="ssh -F ../../access/ssh_config phone"
tar czf /tmp/bratan.tgz --exclude=__pycache__ --exclude=.git --exclude=tests bratan assets deploy ARCHITECTURE.md
$S "cat > /tmp/bratan.tgz" < /tmp/bratan.tgz
$S "sh -s" < deploy/install.sh
