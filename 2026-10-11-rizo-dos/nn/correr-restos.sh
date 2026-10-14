#!/bin/sh
# rizo-2 — RESTOS de la cresta (2026-10-15, el dueño: «corrige para que funcione con todo tramo»): los trozos que no son recta ni
# arco entran al compositor. Dos variantes sobre el mejor sólo-cresta (§ 66: conos 60° presencia, 144):
#   fundido → las rectas pasan a ser rectas ∪ restos (mismos 144 números) · aparte → bloque propio de 44 (188)
#   COORD_HOME=~/src/telegram-coordinator "$COORD_HOME/scripts/desacoplar-persistente.sh" rizo2-restos sh nn/correr-restos.sh
#   ENTRANTES_SECO=1 sh nn/correr-restos.sh      → imprime las órdenes sin lanzar nada
set -u
cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python
ENTRADA="${ENTRADA:-../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz}"
[ -f "$ENTRADA" ] || { echo "falta $ENTRADA"; exit 1; }
for V in fundido aparte; do
    ORDEN="$PY nn/todo_objetos_entrada.py --nombre cresta-restos-$V --entrada $ENTRADA --continuo --cresta todo --entrantes60 presencia --restos $V"
    echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
done
ORDEN="$PY nn/fallos_entrada.py --nombre cresta-restos-fundido --entrada $ENTRADA --continuo"
echo "→ $ORDEN"; [ "${ENTRANTES_SECO:-}" = 1 ] || $ORDEN || echo "AVISO: falló: $ORDEN"
[ "${ENTRANTES_SECO:-}" = 1 ] || node "$COORD_HOME/scripts/notify.mjs" "rizo2-restos terminó: resultados/40-entrada-cresta-restos-*.json" || true
exit 0
