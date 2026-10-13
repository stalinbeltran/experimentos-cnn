#!/bin/sh
# rizo-2 — compositor con las curvas sobre la CRESTA (2026-10-13): A) cresta en vez del gris + huecos suaves v4 (118) ·
# B) cresta + lazos, sin rayos (108) · C) gris + cresta + lazos + huecos suaves v4 (182).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-cresta env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-cresta.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
V=CENTRAR,UNO_POR_CELDA,SIN_RENDIJAS,PONDERAR
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-hs --entrada "$ENTRADA" --continuo --cresta solo --huecos-suaves --variante $V
$PY nn/todo_objetos_entrada.py --nombre suave-cresta-lazos --entrada "$ENTRADA" --continuo --cresta lazos
$PY nn/todo_objetos_entrada.py --nombre suave-gris-cresta-hs --entrada "$ENTRADA" --continuo --cresta mas-gris --huecos-suaves --variante $V
for n in suave-cresta-hs suave-cresta-lazos suave-gris-cresta-hs; do $PY nn/fallos_entrada.py --nombre $n --entrada "$ENTRADA" --continuo; done
node "$COORD_HOME/scripts/notify.mjs" "rizo2-cresta terminó: resultados/40-entrada-suave-{cresta-hs,cresta-lazos,gris-cresta-hs}.json" || true
exit 0
