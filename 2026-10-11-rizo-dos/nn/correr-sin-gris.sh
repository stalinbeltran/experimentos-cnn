#!/bin/sh
# rizo-2 — el compositor SIN gris (el dueño, 2026-10-13: «¿no lo reemplazamos con el detector mejorado de curvas?»):
#   rectas + curvas de la cresta + lazos de la cresta + concavidades suaves v4 (44 + 54 + 10 + 10 = 118)
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-singris env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-sin-gris.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
V=CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS,PONDERAR
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-lazos-conc --entrada "$ENTRADA" --continuo --cresta lazos --conc-suaves --variante $V
$PY nn/fallos_entrada.py --nombre suave-cresta-lazos-conc --entrada "$ENTRADA" --continuo
node "$COORD_HOME/scripts/notify.mjs" "rizo2-singris terminó: resultados/40-entrada-suave-cresta-lazos-conc.json" || true
exit 0
