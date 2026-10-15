#!/bin/sh
# rizo-2 — el LAZO con tamaño (el dueño, 2026-10-15): dos variantes del bloque de lazos sobre la receta aprobada (§ 74:
# rectas∪restos + arcos + golpes + entrantes conos 60° presencia). 'region' = fracción de celda cubierta por el polígono;
# 'area' = gaussiana con altura área/300. Control: cresta-restos-fundido (0,9340 / 0,9158).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-lazos sh nn/correr-lazos.sh
#   ENTRANTES_SECO=1 sh nn/correr-lazos.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA"; exit 1; }
for V in region area; do
    ORDEN="$PY nn/todo_objetos_entrada.py --nombre cresta-lazos-$V --entrada $ENTRADA --continuo --cresta todo --entrantes60 presencia --restos fundido --lazos $V"
    echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
done
[ "${ENTRANTES_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-lazos terminó: resultados/40-entrada-cresta-lazos-*.json" || true
exit 0
