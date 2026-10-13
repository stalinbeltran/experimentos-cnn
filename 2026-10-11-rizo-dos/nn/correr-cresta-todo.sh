#!/bin/sh
# rizo-2 — el compositor SÓLO con objetos de la cresta (2026-10-13): rectas + arcos + golpes + lazos (108) y + concavidades suaves v4 (118).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-todo env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-cresta-todo.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
V=CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS,PONDERAR
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-todo --entrada "$ENTRADA" --continuo --cresta todo
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-todo-conc --entrada "$ENTRADA" --continuo --cresta todo --conc-suaves --variante $V
for n in suave-cresta-todo suave-cresta-todo-conc; do $PY nn/fallos_entrada.py --nombre $n --entrada "$ENTRADA" --continuo; done
node "$COORD_HOME/scripts/notify.mjs" "rizo2-todo terminó: resultados/40-entrada-suave-cresta-todo*.json" || true
exit 0
