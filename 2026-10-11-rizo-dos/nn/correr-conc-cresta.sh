#!/bin/sh
# rizo-2 — el compositor sólo con objetos de la cresta, con las CONCAVIDADES de la cresta en vez de las de rayos (2026-10-14): 152 números.
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-conc env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-conc-cresta.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-todo-cc --entrada "$ENTRADA" --continuo --cresta todo --conc-cresta
$PY nn/fallos_entrada.py --nombre suave-cresta-todo-cc --entrada "$ENTRADA" --continuo
node "$COORD_HOME/scripts/notify.mjs" "rizo2-conc terminó: resultados/40-entrada-suave-cresta-todo-cc.json" || true
exit 0
