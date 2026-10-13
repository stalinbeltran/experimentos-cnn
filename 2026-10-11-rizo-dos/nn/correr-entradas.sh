#!/bin/sh
# rizo-2 — las cuatro corridas de todo_objetos_entrada.py (2026-10-13): referencia + adelgazado tal cual + dos ajustes.
# Dos lotes (uno por vCPU), cada uno como unidad:
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-entradas-a env ENTRADA=/ruta/delgados-k3x3.npz sh nn/correr-entradas.sh a
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-entradas-b sh nn/correr-entradas.sh b
# Estado: ls resultados/40-entrada-*.json · systemctl status rizo2-entradas-a rizo2-entradas-b · /tmp/rizo2-entradas-{a,b}.log
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
# ENTRADA: ruta al .npz del experimento que adelgaza (nn/exportar_delgados.py del experimento de la CNN delgada); se pasa por entorno
ENTRADA="${ENTRADA:?falta ENTRADA=/ruta/delgados-<pesos>.npz}"
case "${1:-}" in
  a)
    $PY nn/todo_objetos_entrada.py --nombre tinta
    $PY nn/todo_objetos_entrada.py --nombre delgado-k3x3-eng2 --entrada "$ENTRADA" --umbral 0.5 --engrosar 2 ;;
  b)
    $PY nn/todo_objetos_entrada.py --nombre delgado-k3x3 --entrada "$ENTRADA" --umbral 0.5
    # misma imagen binaria → mismos candidatos: se reutilizan, sólo cambia el filtro de huecos
    cp resultados/40-candidatos-delgado-k3x3.pkl resultados/40-candidatos-delgado-k3x3-var14.pkl
    $PY nn/todo_objetos_entrada.py --nombre delgado-k3x3-var14 --entrada "$ENTRADA" --umbral 0.5 --var "medio (h1.5, 14px)" ;;
  *) echo "uso: correr-entradas.sh a|b" >&2; exit 2 ;;
esac
node "$COORD_HOME/scripts/notify.mjs" "rizo2-entradas-$1 terminó: resultados/40-entrada-*.json en rizo-2" || true
exit 0
