#!/usr/bin/env sh
# `feat-fallos` en Vast, con el modo `trabajo` del lanzador:
#
#   nn/vast.sh detectores         los detectores (vast.json) A LA VEZ en UNA máquina: alquila, entrena, trae los
#                                 pesos (sus carpetas de pesos) y DESTRUYE la máquina
#   nn/vast.sh --estado           lee el libro del DISCO
#   nn/vast.sh apagar             para la unidad y destruye TODAS las máquinas expc-ffal-*
#
#   VAST_SECO=1 nn/vast.sh detectores   imprime unidad, etiqueta y orden SIN tocar la API
#   nn/probar_vast.sh                   comprueba, en seco, que el modo construye SU orden y que uno desconocido se niega
#
# COPIA del `nn/vast.sh` de `feat-bor` (2026-10-06), que lo es del de `feat-ind32`, con un modo. Mantiene lo del 2026-09-08
# (telegram-coordinator/CLAUDE.md): el modo se guarda AL ENTRAR, el último caso SE NIEGA, la orden se IMPRIME antes de
# lanzar, y hay modo seco y prueba.
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si el dataset no está publicado Y empujado al almacén, o si `entrenar_local.py --comprobar`
# no pasa aquí: un fallo dentro de la máquina se paga entero (R2).
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el siguiente sabe qué etiquetas eran
# suyas. Apagar desde cualquier máquina con el token: `vast_instance.py trabajo --apagar expc-ffal-`, o `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-ffal-

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"

case "$MODO" in
    detectores)
        DESC="$AQUI/vast.json"; LIBRO="$EXP/resultados/vast/lineas-grueso"; HORAS=3
        DATASET=feat-fallos-sinteticas-grueso-32px-r20261006; COMPROBAR="entrenar_local.py" ;;
    --estado)
        if [ -d "$EXP/resultados/vast/lineas-grueso" ]; then $V trabajo --estado --libro "$EXP/resultados/vast/lineas-grueso"
        else echo "Sin libro: no se ha alquilado nada para feat-fallos."; fi
        exit 0 ;;
    apagar)
        $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *)
        echo "✗ modo '$MODO' desconocido: detectores | --estado | apagar"; exit 2 ;;
esac

EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
ORDEN="$V trabajo --descriptor $DESC --prefijo $PREFIJO --libro $LIBRO --horas-max $HORAS"
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
    echo "✗ $DATASET no está commiteado y empujado al almacén. No se alquila nada."; exit 2
fi
if ! "$REPO/.venv/bin/python" "$AQUI/$COMPROBAR" --comprobar >"/tmp/feat-fallos-comprobar-$MODO.log" 2>&1; then
    echo "✗ $COMPROBAR --comprobar falla aquí (/tmp/feat-fallos-comprobar-$MODO.log). No se alquila nada."; exit 2
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "feat-fallos: libro de '$MODO' en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice qué máquinas eran de aquí."
echo
echo "Cómo va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
