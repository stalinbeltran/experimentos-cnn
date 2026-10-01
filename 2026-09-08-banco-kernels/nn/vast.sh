#!/usr/bin/env sh
# `banco-k` en Vast, con el modo `trabajo` del lanzador (plan del 2026-10-01, §4).
#
#   nn/vast.sh fase1              identidad s3 en UNA maquina: valida la cadena y mide deriva
#   nn/vast.sh fase3 [k09 ...]    evalua los kernels nuevos: una maquina por k (o solo esos)
#   nn/vast.sh --estado [fase]    lee el libro del DISCO (defecto: fase1)
#   nn/vast.sh apagar             para las unidades y destruye TODAS las maquinas expc-bancok-*
#
#   VAST_SECO=1 nn/vast.sh <fase>   imprime unidades, etiquetas y ordenes SIN tocar la API
#
# POR QUE NO VA POR `nn/lanzar.sh`: aquel mete UNA orden en UNA unidad. Aqui cada
# maquina tiene su unidad, y las crea el propio lanzador (`vast_instance.py trabajo`),
# que reparte las ofertas ANTES para que no choquen. Este script solo resuelve rutas,
# imprime la orden, y commitea el libro al alquilar.
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR, no al terminar: si este server muere a
# mitad, las maquinas siguen facturando, y el siguiente solo sabe que etiquetas eran
# suyas si el libro esta en el remoto. Las apaga `apagar`, o desde cualquier maquina
# con el token: `vast_instance.py trabajo --apagar expc-bancok-`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO_TODO=expc-bancok-

# ⚠⚠ EL MODO SE GUARDA AL ENTRAR Y NO SE RELEE DE "$1" (el fallo del 2026-09-08 en
# `nn/lanzar.sh`: un `shift` y un despacho que releia "$1"). Lo vigila `probar_vast.sh`.
MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"

case "$MODO" in
    fase1) DESC="$AQUI/vast-fase1.json"; FASE=fase1; HORAS=1 ;;
    fase3) DESC="$AQUI/vast-fase3.json"; FASE=fase3; HORAS=3 ;;
    --estado)
        $V trabajo --estado --libro "$EXP/resultados/vast/${1:-fase1}"
        exit 0 ;;
    apagar)
        $V trabajo --apagar "$PREFIJO_TODO"
        exit 0 ;;
    *)
        echo "uso: $0 fase1 | fase3 [k...] | --estado [fase] | apagar"
        exit 2 ;;
esac

[ -f "$DESC" ] || { echo "✗ no existe $DESC"; exit 2; }
EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
LIBRO="$EXP/resultados/vast/$FASE"
ORDEN="$V trabajo --descriptor $DESC --prefijo ${PREFIJO_TODO}$(echo $FASE | tr -d 'ase')- --libro $LIBRO --horas-max $HORAS"
[ "$#" -gt 0 ] && ORDEN="$ORDEN --solo $*"

# SIEMPRE se imprime QUE se va a correr, seco o no.
echo "modo:   $MODO"
echo "orden:  $ORDEN"
if [ -n "$VAST_SECO" ]; then
    $ORDEN --seco
    exit 0
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "banco-k: libro de $FASE en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice que maquinas eran de aqui."
echo
echo "Como va:   $0 --estado $FASE"
echo "Apagarlo:  $0 apagar"
