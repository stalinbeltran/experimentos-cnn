#!/bin/sh
# rizo-2 — sin huecos y con el mapa continuo (2026-10-13): referencia tinta (98) + mapa suave (98).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-suaves env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-suaves.sh
#   (ENTRADA va DENTRO del comando: el desacoplador sólo pasa COORD_*, HOME y PATH)
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
$PY nn/todo_objetos_entrada.py --nombre tinta-sh --sin-huecos
$PY nn/todo_objetos_entrada.py --nombre suave-k3x3 --entrada "$ENTRADA" --continuo --sin-huecos
$PY nn/fallos_entrada.py --nombre tinta-sh --sin-huecos
$PY nn/fallos_entrada.py --nombre suave-k3x3 --entrada "$ENTRADA" --continuo --sin-huecos
node "$COORD_HOME/scripts/notify.mjs" "rizo2-suaves terminó: resultados/40-entrada-{tinta-sh,suave-k3x3}.json" || true
exit 0
