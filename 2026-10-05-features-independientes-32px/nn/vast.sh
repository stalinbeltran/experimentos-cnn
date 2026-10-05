#!/usr/bin/env sh
# `feat-ind32` en Vast, con el modo `trabajo` del lanzador: los 13 detectores A LA VEZ en una maquina.
#
#   nn/vast.sh detectores         alquila, entrena los 13, trae los pesos y DESTRUYE la maquina
#   nn/vast.sh --estado           lee el libro del DISCO
#   nn/vast.sh apagar             para la unidad y destruye TODAS las maquinas expc-fi32-*
#
#   VAST_SECO=1 nn/vast.sh detectores   imprime unidad, etiqueta y orden SIN tocar la API
#
# Copiado del `nn/vast.sh` de `bor-ae` el 2026-10-05 (Regla 0: se copia, no se comparte).
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si el dataset no esta publicado Y empujado al almacen, o si
# `entrenar_local.py --comprobar` no pasa aqui: un fallo dentro de la maquina se paga entero (R2).
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el siguiente
# sabe que etiquetas eran suyas. Apagar desde cualquier maquina con el token:
# `vast_instance.py trabajo --apagar expc-fi32-`, o desde Telegram `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-fi32-
DATASET=feat-ind32-sinteticas-32px-r20261005

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"
LIBRO="$EXP/resultados/vast/detectores"

case "$MODO" in
    detectores) ;;
    --estado) $V trabajo --estado --libro "$LIBRO"; exit 0 ;;
    apagar)   $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *) echo "✗ modo '$MODO' desconocido: detectores | --estado | apagar"; exit 2 ;;
esac

EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
ORDEN="$V trabajo --descriptor $AQUI/vast.json --prefijo $PREFIJO --libro $LIBRO --horas-max 3"
echo "modo:   $MODO"
echo "orden:  $ORDEN"
if [ -n "$VAST_SECO" ]; then
    $ORDEN --seco
    exit 0
fi

# El dataset tiene que estar commiteado y EMPUJADO: estar en disco no es estar guardado.
if [ -n "$(git -C "$EXPCNN_DATOS" status --porcelain -- "experimentos-cnn/$DATASET")" ] \
   || ! git -C "$EXPCNN_DATOS" fetch -q origin \
   || [ -n "$(git -C "$EXPCNN_DATOS" log --oneline origin/main..HEAD -- "experimentos-cnn/$DATASET")" ]; then
    echo "✗ $DATASET no esta commiteado y empujado al almacen. No se alquila nada."; exit 2
fi
if ! "$REPO/.venv/bin/python" "$AQUI/entrenar_local.py" --comprobar >/tmp/feat-ind32-comprobar.log 2>&1; then
    echo "✗ entrenar_local.py --comprobar falla aqui (/tmp/feat-ind32-comprobar.log). No se alquila nada."; exit 2
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "feat-ind32: libro de los detectores en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice que maquinas eran de aqui."
echo
echo "Como va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
