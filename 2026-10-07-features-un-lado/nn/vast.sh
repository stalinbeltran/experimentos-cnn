#!/usr/bin/env sh
# `feat-1lado` en Vast, con el modo `trabajo` del lanzador. UN MODO = UN BRAZO = UNA MÁQUINA:
#
#   nn/vast.sh control       los 13 detectores del brazo control (vast-control.json): 13 procesos × 2 hilos, tope 3 h
#   nn/vast.sh compartido    los 13 del brazo compartido (vast-compartido.json): 13 procesos × 4 hilos, tope 8 h
#                            Cada uno alquila, entrena, trae los pesos (nn/pesos-<brazo>/) y DESTRUYE la máquina.
#   nn/vast.sh --estado      lee los libros del DISCO
#   nn/vast.sh apagar        para las unidades y destruye TODAS las máquinas expc-f1l-*
#
#   VAST_SECO=1 nn/vast.sh <modo>   imprime unidad, etiqueta y orden SIN tocar la API
#   nn/probar_vast.sh               comprueba, en seco, que cada modo construye SU orden y que uno desconocido se niega
#
# COPIA del `nn/vast.sh` de `feat-bor` (2026-10-07) con dos modos. Mantiene lo del 2026-09-08
# (telegram-coordinator/CLAUDE.md): el modo se guarda AL ENTRAR, el último caso SE NIEGA, la orden se IMPRIME antes de
# lanzar, y hay modo seco y prueba.
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si el dataset no está publicado Y empujado al almacén, si el criterio no está commiteado y
# empujado (R13: se escribe antes de mirar), o si `entrenar_local.py --comprobar --brazo <modo>` no pasa aquí.
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el siguiente sabe qué etiquetas eran
# suyas. Apagar desde cualquier máquina con el token: `vast_instance.py trabajo --apagar expc-f1l-`, o `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-f1l-

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"

case "$MODO" in
    control)
        DESC="$AQUI/vast-control.json"; LIBRO="$EXP/resultados/vast/control"; HORAS=3 ;;
    compartido)
        DESC="$AQUI/vast-compartido.json"; LIBRO="$EXP/resultados/vast/compartido"; HORAS=8 ;;
    --estado)
        hay=0
        for b in control compartido; do
            if [ -d "$EXP/resultados/vast/$b" ]; then $V trabajo --estado --libro "$EXP/resultados/vast/$b"; hay=1; fi
        done
        [ "$hay" -eq 1 ] || echo "Sin libro: no se ha alquilado nada para feat-1lado."
        exit 0 ;;
    apagar)
        $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *)
        echo "✗ modo '$MODO' desconocido: control | compartido | --estado | apagar"; exit 2 ;;
esac
DATASET=feat-ind32-sinteticas-32px-r20261005

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
# El criterio, commiteado y EMPUJADO antes de alquilar (R13): si no, se podría reescribir después de mirar.
if [ -n "$(git -C "$REPO" status --porcelain -- "$EXP/instrucciones" "$EXP/REGLAS.md" "$AQUI")" ] \
   || ! git -C "$REPO" fetch -q origin \
   || [ -n "$(git -C "$REPO" log --oneline origin/main..HEAD -- "$EXP")" ]; then
    echo "✗ el criterio o el código de feat-1lado no están commiteados y empujados. No se alquila nada."; exit 2
fi
if ! "$REPO/.venv/bin/python" "$AQUI/entrenar_local.py" --comprobar --brazo "$MODO" >"/tmp/feat-1lado-comprobar-$MODO.log" 2>&1; then
    echo "✗ entrenar_local.py --comprobar --brazo $MODO falla aquí (/tmp/feat-1lado-comprobar-$MODO.log). No se alquila nada."; exit 2
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "feat-1lado: libro de '$MODO' en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice qué máquinas eran de aquí."
echo
echo "Cómo va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
