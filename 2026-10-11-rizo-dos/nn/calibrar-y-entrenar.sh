#!/bin/sh
# rizo-2: calibra R y P y entrena los dos brazos de C, en una unidad persistente (sobrevive al fin del turno).
set -u
cd "$(dirname "$0")/.." || exit 2
PY=../.venv/bin/python
echo "== rayos";       $PY nn/rayos.py
echo "== plantillas";  $PY nn/plantillas.py
for b in c-base c-ancha; do
  echo "== entrenar $b"; $PY nn/entrenar_local.py "$b" --epocas 12
done
echo "== FIN $(date -u +%H:%M:%S)"
node "${COORD_HOME:-$HOME/src/telegram-coordinator}/scripts/notify.mjs" "rizo-2: calibración y entrenamiento terminados (log /tmp/rizo2-cal.log)" || true
exit 0
