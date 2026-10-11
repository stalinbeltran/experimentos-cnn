#!/bin/sh
cd "$(dirname "$0")/.." || exit 2
../.venv/bin/python nn/digitos_cerradas.py
node "${COORD_HOME:-$HOME/src/telegram-coordinator}/scripts/notify.mjs" "rizo-2: digitos_cerradas terminado (log /tmp/rizo2-dig.log)" || true
exit 0
