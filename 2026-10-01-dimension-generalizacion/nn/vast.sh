#!/usr/bin/env sh
# `dim-gen` en Vast, con el modo `trabajo` del lanzador (ESPECIFICACION.md §5).
#
#   nn/vast.sh todo [s1 c1 ...]  los 30 brazos en DIEZ maquinas: s<N> = w128 + los 4 baratos de
#                                la semilla N; c<N> = su control w128-de16,
#                                mas la unidad de CIERRE (nn/cierre.sh) que espera, informa,
#                                commitea, copia al volumen y avisa
#   nn/vast.sh --estado          lee el libro del DISCO
#   nn/vast.sh apagar            para las unidades y destruye TODAS las maquinas expc-dimgen-*
#
#   VAST_SECO=1 nn/vast.sh todo  imprime unidades, etiquetas y ordenes SIN tocar la API
#
# Copiado del `nn/vast.sh` de `bor-k` el 2026-10-02 (Regla 0: se copia, no se comparte) y
# adaptado: una fase, una maquina por semilla, y el cierre.
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si `LR` no esta congelado: cada maquina moriria al empezar con
# el mismo error, despues de pagar el arranque y la instalacion (R2).
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el siguiente
# sabe que etiquetas eran suyas. Apagar desde cualquier maquina con el token:
# `vast_instance.py trabajo --apagar expc-dimgen-`, o desde Telegram `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-dimgen-
HORAS_MAX=6

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"
LIBRO="$EXP/resultados/vast/todo"

case "$MODO" in
    todo) ;;
    --estado) $V trabajo --estado --libro "$LIBRO"; echo; systemctl is-active "${PREFIJO}cierre" >/dev/null 2>&1 && echo "cierre: unidad ${PREFIJO}cierre ACTIVA (espera a las maquinas)" || echo "cierre: unidad ${PREFIJO}cierre no activa"; exit 0 ;;
    apagar)   $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *) echo "uso: $0 todo [s1 ...] | --estado | apagar"; exit 2 ;;
esac

# El guardia del lr va DESPUES del seco: un ensayo en seco no alquila nada, asi que no puede
# fallar por esto, y poder mirar el plan antes de congelar el lr es justo para lo que sirve.
if ! (cd "$AQUI" && "$REPO/.venv/bin/python" -c "import sys, entrenar_local as e; sys.exit(e.LR is None)" 2>/dev/null); then
    if [ -n "$VAST_SECO" ]; then
        echo "⚠ (seco) LR no esta congelado en nn/entrenar_local.py: el lanzamiento REAL se negaria."
    else
        echo "✗ LR no esta congelado en nn/entrenar_local.py. No se alquila nada."
        exit 2
    fi
fi
EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
ORDEN="$V trabajo --descriptor $AQUI/vast.json --prefijo $PREFIJO --libro $LIBRO --horas-max $HORAS_MAX"
[ "$#" -gt 0 ] && ORDEN="$ORDEN --solo $*"

echo "modo:   $MODO"
echo "orden:  $ORDEN"
echo "cierre: $COORD_HOME/scripts/desacoplar-persistente.sh ${PREFIJO}cierre sh $AQUI/cierre.sh"
if [ -n "$VAST_SECO" ]; then
    $ORDEN --seco
    exit 0
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "dim-gen: libro de Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice que maquinas eran de aqui."
# El cierre: una unidad aparte (padre PID 1) que espera a las cinco, informa, commitea, copia al
# volumen y avisa. Si ya hay una viva, desacoplar-persistente.sh se niega y lo dice.
(cd "$EXP" && "$COORD_HOME/scripts/desacoplar-persistente.sh" "${PREFIJO}cierre" sh "$AQUI/cierre.sh") \
    || echo "⚠ el cierre NO arranco: lanzalo a mano cuando terminen: sh nn/cierre.sh"
echo
echo "Como va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
