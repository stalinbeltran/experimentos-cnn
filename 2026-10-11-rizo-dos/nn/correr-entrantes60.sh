#!/bin/sh
# rizo-2 — ENTRANTES en 4 CONOS (derecha/izquierda a ±60° de la horizontal; abajo/arriba el resto) × 3×3 (36), el dueño 2026-10-15.
# Dos corridas sobre rectas + arcos + golpes + lazos (108): presencia sin magnitud (144) y fuerza máxima por celda (144).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-entrantes60 sh nn/correr-entrantes60.sh
#   ENTRANTES_SECO=1 sh nn/correr-entrantes60.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA"; exit 1; }
for V in presencia fuerza; do
    ORDEN="$PY nn/todo_objetos_entrada.py --nombre cresta-entrantes60-$V --entrada $ENTRADA --continuo --cresta todo --entrantes60 $V"
    echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
done
ORDEN="$PY nn/fallos_entrada.py --nombre cresta-entrantes60-presencia --entrada $ENTRADA --continuo"
echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
[ "${ENTRANTES_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-entrantes60 terminó: resultados/40-entrada-cresta-entrantes60-*.json" || true
exit 0
