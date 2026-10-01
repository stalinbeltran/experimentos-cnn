#!/usr/bin/env sh
# `bor-ae` en Vast, con el modo `trabajo` del lanzador (plan del 2026-10-01, §4).
#
#   nn/vast.sh fase2 [k09 ...]    los 27 brazos: una maquina por k, sus 3 semillas a la vez
#   nn/vast.sh --estado           lee el libro del DISCO
#   nn/vast.sh apagar             para las unidades y destruye TODAS las maquinas expc-borae-*
#
#   VAST_SECO=1 nn/vast.sh fase2  imprime unidades, etiquetas y ordenes SIN tocar la API
#
# Copiado del `nn/vast.sh` de `banco-k` el 2026-10-01 (Regla 0: se copia, no se comparte)
# y reducido a la unica fase de este experimento.
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si `LAMBDA` no esta congelado: cada maquina moriria al
# empezar con el mismo error, despues de pagar el arranque y la instalacion (R2).
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el
# siguiente sabe que etiquetas eran suyas. Apagar desde cualquier maquina con el token:
# `vast_instance.py trabajo --apagar expc-borae-`, o desde Telegram `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-borae-

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"
LIBRO="$EXP/resultados/vast/fase2"

case "$MODO" in
    fase2) ;;
    --estado) $V trabajo --estado --libro "$LIBRO"; exit 0 ;;
    apagar)   $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *) echo "uso: $0 fase2 [k...] | --estado | apagar"; exit 2 ;;
esac

if ! (cd "$AQUI" && "$REPO/.venv/bin/python" -c "import sys, entrenar_local as e; sys.exit(e.LAMBDA is None)" 2>/dev/null); then
    echo "✗ LAMBDA no esta congelado en nn/entrenar_local.py. No se alquila nada."
    exit 2
fi
EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
ORDEN="$V trabajo --descriptor $AQUI/vast.json --prefijo $PREFIJO --libro $LIBRO --horas-max 3"
[ "$#" -gt 0 ] && ORDEN="$ORDEN --solo $*"

echo "modo:   $MODO"
echo "orden:  $ORDEN"
if [ -n "$VAST_SECO" ]; then
    $ORDEN --seco
    exit 0
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "bor-ae: libro de la fase 2 en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice que maquinas eran de aqui."
echo
echo "Como va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
