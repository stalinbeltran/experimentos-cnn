#!/bin/sh
# rizo-2 — los ENTRANTES reducidos a 4 números (uno por dirección de abertura: → ← ↓ ↑), pedido por el dueño el 2026-10-15.
# Tres corridas sobre rectas + arcos + golpes + lazos (108): + fuerza máx. por dirección (112) · + cuenta (112) · + los dos (116).
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-entrantes4 sh nn/correr-entrantes4.sh
#   ENTRANTES_SECO=1 sh nn/correr-entrantes4.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA"; exit 1; }
for V in fuerza cuenta ambos; do
    ORDEN="$PY nn/todo_objetos_entrada.py --nombre cresta-entrantes4-$V --entrada $ENTRADA --continuo --cresta todo --entrantes4 $V"
    echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
done
ORDEN="$PY nn/fallos_entrada.py --nombre cresta-entrantes4-fuerza --entrada $ENTRADA --continuo"
echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
[ "${ENTRANTES_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-entrantes4 terminó: resultados/40-entrada-cresta-entrantes4-*.json" || true
exit 0
