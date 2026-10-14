#!/bin/sh
# rizo-2 — REJILLA 4×4 contra 3×3 (plan del dueño, 2026-10-15, punto 1): mismas features de la cresta, mismo protocolo,
# sólo cambia --celdas. Corre el control 3×3 (152 números) y luego el 4×4 (250), cada uno con su figura de fallos.
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-rejilla sh nn/correr-rejilla.sh
#   REJILLA_SECO=1 sh nn/correr-rejilla.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA: regenerar con ../2026-10-15-delgado-cnn/nn/exportar_delgados.py --pesos k3x3"; exit 1; }
for C in 3 4; do
    for ORDEN in \
        "$PY nn/todo_objetos_entrada.py --nombre cresta-cc-${C}x${C} --entrada $ENTRADA --continuo --cresta todo --conc-cresta --celdas $C" \
        "$PY nn/fallos_entrada.py --nombre cresta-cc-${C}x${C} --entrada $ENTRADA --continuo"; do
        echo "→ $ORDEN"
        [ "${REJILLA_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
    done
done
[ "${REJILLA_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-rejilla terminó: resultados/40-entrada-cresta-cc-{3x3,4x4}.json" || true
exit 0
