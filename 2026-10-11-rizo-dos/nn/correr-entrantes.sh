#!/bin/sh
# rizo-2 — compositor sólo-cresta con los ENTRANTES (concavidades de arco ∪ esquinas, 2026-10-15) en lugar de las
# concavidades de arco solas; el control es cresta-cc-3x3 (§ 55). Dos corridas: entrantes solos (152) y conc + entrantes (196).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-entrantes sh nn/correr-entrantes.sh
#   ENTRANTES_SECO=1 sh nn/correr-entrantes.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA: regenerar con ../2026-10-15-delgado-cnn/nn/exportar_delgados.py --pesos k3x3"; exit 1; }
for ORDEN in \
    "$PY nn/todo_objetos_entrada.py --nombre cresta-entrantes --entrada $ENTRADA --continuo --cresta todo --entrantes" \
    "$PY nn/fallos_entrada.py --nombre cresta-entrantes --entrada $ENTRADA --continuo" \
    "$PY nn/todo_objetos_entrada.py --nombre cresta-cc-entrantes --entrada $ENTRADA --continuo --cresta todo --conc-cresta --entrantes"; do
    echo "→ $ORDEN"
    [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
done
[ "${ENTRANTES_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-entrantes terminó: resultados/40-entrada-cresta-entrantes.json" || true
exit 0
