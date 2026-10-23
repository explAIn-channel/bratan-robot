#!/bin/sh
# From the PC: deploy only when the body is idle (no skill running, not talking). Usage: sh deploy/safe_deploy.sh
set -e
cd "$(dirname "$0")/.."
S="ssh -F ../../access/ssh_config phone"
state=$($S 'cd /opt/robotbrain/bratan/app && PYTHONPATH=. timeout 5 /opt/robotbrain/venvs/perception/bin/python -m bratan.cli get body.state | tail -1; PYTHONPATH=. timeout 5 /opt/robotbrain/venvs/perception/bin/python -m bratan.cli get voice.state | tail -1')
echo "$state" | grep -q '"skill": null' || { echo "body busy — not deploying"; exit 3; }
echo "$state" | grep -q '"live": false' || { echo "in a conversation — not deploying"; exit 3; }
sh deploy/push.sh | tail -1
$S 'sh /opt/robotbrain/bratan/app/deploy/switch.sh bratan' | tail -1
