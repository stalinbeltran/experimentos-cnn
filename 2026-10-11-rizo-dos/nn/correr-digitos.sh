#!/bin/sh
cd "$(dirname "$0")/.." || exit 2
../.venv/bin/python nn/resistencia.py
node "${COORD_HOME:-$HOME/src/telegram-coordinator}/scripts/notify.mjs" "rizo-2: resistencia terminado (log /tmp/rizo2-dig.log)" || true
exit 0
