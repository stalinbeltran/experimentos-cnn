#!/bin/sh
# rizo-2 — compositor con las variantes del detector de huecos suaves (pendientes del dueño, 2026-10-13).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-var env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-variantes.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
$PY nn/todo_objetos_entrada.py --nombre suave-k3x3-hs-v3 --entrada "$ENTRADA" --continuo --huecos-suaves --variante CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS
$PY nn/todo_objetos_entrada.py --nombre suave-k3x3-hs-v4 --entrada "$ENTRADA" --continuo --huecos-suaves --variante CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS,PONDERAR
$PY nn/fallos_entrada.py --nombre suave-k3x3-hs-v3 --entrada "$ENTRADA" --continuo --huecos-suaves
$PY nn/fallos_entrada.py --nombre suave-k3x3-hs-v4 --entrada "$ENTRADA" --continuo --huecos-suaves
node "$COORD_HOME/scripts/notify.mjs" "rizo2-var terminó: resultados/40-entrada-suave-k3x3-hs-v{3,4}.json" || true
exit 0
