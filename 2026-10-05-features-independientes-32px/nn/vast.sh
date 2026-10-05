#!/usr/bin/env sh
# `feat-ind32` en Vast, con el modo `trabajo` del lanzador. Dos trabajos, cada uno A LA VEZ en una máquina:
#
#   nn/vast.sh detectores         los 13 detectores (vast.json): alquila, entrena, trae los pesos y DESTRUYE la máquina
#   nn/vast.sh cnn                las 120 curvas de CNN (vast-cnn.json): ídem, trae nn/curvas-cnn/
#   nn/vast.sh --estado           lee los libros del DISCO (los dos)
#   nn/vast.sh apagar             para las unidades y destruye TODAS las máquinas expc-fi32-*
#
#   VAST_SECO=1 nn/vast.sh <modo> imprime unidad, etiqueta y orden SIN tocar la API
#   nn/probar_vast.sh             comprueba, en seco, que cada modo construye SU orden y que uno desconocido se niega
#
# Copiado del `nn/vast.sh` de `bor-ae` el 2026-10-05 (Regla 0: se copia, no se comparte). Con dos modos aplica lo del
# 2026-09-08 (telegram-coordinator/CLAUDE.md): el modo se guarda AL ENTRAR, el último caso SE NIEGA, la orden se
# IMPRIME antes de lanzar, y hay modo seco y prueba.
#
# ⚠ SE NIEGA ANTES DE ALQUILAR si el dataset del modo no está publicado Y empujado al almacén, o si la comprobación del
# modo no pasa aquí: un fallo dentro de la máquina se paga entero (R2).
#
# ⚠ EL LIBRO SE COMMITEA Y SE EMPUJA AL ALQUILAR: si este server muere a mitad, el siguiente sabe qué etiquetas eran
# suyas. Apagar desde cualquier máquina con el token: `vast_instance.py trabajo --apagar expc-fi32-`, o `/use exp-vast`.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-fi32-

MODO="$1"
[ "$#" -gt 0 ] && shift

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`: mira el error de arriba"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"

case "$MODO" in
    detectores)
        DESC="$AQUI/vast.json"; LIBRO="$EXP/resultados/vast/detectores"; HORAS=3
        DATASET=feat-ind32-sinteticas-32px-r20261005; COMPROBAR="entrenar_local.py" ;;
    cnn)
        DESC="$AQUI/vast-cnn.json"; LIBRO="$EXP/resultados/vast/cnn"; HORAS=2
        DATASET=uci-optdigits-orig-32px-r20261005; COMPROBAR="curvas_cnn.py" ;;
    --estado)
        for l in detectores cnn; do
            if [ -d "$EXP/resultados/vast/$l" ]; then echo "== $l"; $V trabajo --estado --libro "$EXP/resultados/vast/$l"; fi
        done
        exit 0 ;;
    apagar)
        $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *)
        echo "✗ modo '$MODO' desconocido: detectores | cnn | --estado | apagar"; exit 2 ;;
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
if ! "$REPO/.venv/bin/python" "$AQUI/$COMPROBAR" --comprobar >"/tmp/feat-ind32-comprobar-$MODO.log" 2>&1; then
    echo "✗ $COMPROBAR --comprobar falla aquí (/tmp/feat-ind32-comprobar-$MODO.log). No se alquila nada."; exit 2
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "feat-ind32: libro de '$MODO' en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice qué máquinas eran de aquí."
echo
echo "Cómo va:   $0 --estado"
echo "Apagarlo:  $0 apagar"
