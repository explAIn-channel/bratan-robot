#!/bin/sh
# Started by the Android boot hook (through the old run-phone.sh selector) inside the Ubuntu chroot.
mkdir -p /run/bratan /opt/robotbrain/bratan/logs
cd /opt/robotbrain/bratan/app
export PYTHONPATH=/opt/robotbrain/bratan/app PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TZ=Europe/Moscow
exec /opt/robotbrain/venvs/perception/bin/python -m bratan.main >>/opt/robotbrain/bratan/logs/main.log 2>&1
