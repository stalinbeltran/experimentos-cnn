#!/bin/sh
# rizo-2 — el compositor con los LAZOS de la cresta en lugar de los huecos de rayos (el dueño, 2026-10-13):
#   A) rectas + gris + lazos (108) · B) + concavidades suaves v4 (118)
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-lazos env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-lazos.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
V=CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS,PONDERAR
$PY nn/todo_objetos_entrada.py --nombre suave-gris-lazos --entrada "$ENTRADA" --continuo --cresta lazos-solo
$PY nn/todo_objetos_entrada.py --nombre suave-gris-lazos-conc --entrada "$ENTRADA" --continuo --cresta lazos+conc --variante $V
for n in suave-gris-lazos suave-gris-lazos-conc; do $PY nn/fallos_entrada.py --nombre $n --entrada "$ENTRADA" --continuo; done
node "$COORD_HOME/scripts/notify.mjs" "rizo2-lazos terminó: resultados/40-entrada-suave-gris-lazos*.json" || true
exit 0
