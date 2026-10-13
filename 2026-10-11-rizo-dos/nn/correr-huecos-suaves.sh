#!/bin/sh
# rizo-2 — todo objetos con el grupo de HUECOS SUAVES (2026-10-13): tinta y mapa suave de delg-cnn, 118 números cada uno.
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-hs env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-huecos-suaves.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
$PY nn/todo_objetos_entrada.py --nombre tinta-hs --huecos-suaves
$PY nn/todo_objetos_entrada.py --nombre suave-k3x3-hs --entrada "$ENTRADA" --continuo --huecos-suaves
$PY nn/fallos_entrada.py --nombre tinta-hs --huecos-suaves
$PY nn/fallos_entrada.py --nombre suave-k3x3-hs --entrada "$ENTRADA" --continuo --huecos-suaves
node "$COORD_HOME/scripts/notify.mjs" "rizo2-hs terminó: resultados/40-entrada-{tinta-hs,suave-k3x3-hs}.json" || true
exit 0
