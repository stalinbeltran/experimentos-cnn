#!/usr/bin/env sh
# `rect-bor` en Vast con el modo `trabajo` del lanzador: la rejilla ENTERA en 12 trozos, UNA máquina, que alquila,
# entrena, trae `resultados/trozos/` al libro y se DESTRUYE sola. El modo `trabajo` ya lo lanza como UNIDAD de systemd.
#
#   nn/vast.sh rejilla      alquila y lanza (tope 1 h)
#   nn/vast.sh --estado     lee el libro del DISCO y lo cruza con lo vivo
#   nn/vast.sh apagar       para la unidad y destruye TODAS las máquinas expc-rbor-
#   VAST_SECO=1 nn/vast.sh rejilla    imprime la orden y el plan SIN tocar la API
#
# Mantiene lo del 2026-09-08: el modo se guarda AL ENTRAR, el último caso SE NIEGA, la orden se IMPRIME antes.
# ⚠ Se niega antes de alquilar si los datasets o el criterio no están commiteados y empujados (R13).
# Freno desde Telegram: /use exp-vast → `apagar expc-rbor-`.
set -e
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
PREFIJO=expc-rbor-
MODO="$1"
LIBRO="$EXP/resultados/vast/rejilla"

LANZADOR=$(cd "$REPO" && python3 -c "from expcnn import exigir_lanzador; print(exigir_lanzador())") || {
    echo "✗ sin lanzador con modo \`trabajo\`"; exit 2; }
V="python3 $LANZADOR/scripts/vast_instance.py"

case "$MODO" in
    rejilla) ;;
    --estado)
        if [ -d "$LIBRO" ]; then $V trabajo --estado --libro "$LIBRO"; else echo "Sin libro: rect-bor no ha alquilado nada."; fi
        exit 0 ;;
    apagar)
        $V trabajo --apagar "$PREFIJO"; exit 0 ;;
    *)
        echo "✗ modo '$MODO' desconocido: rejilla | --estado | apagar"; exit 2 ;;
esac

EXPCNN_DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())")
export EXPCNN_DATOS
ORDEN="$V trabajo --descriptor $AQUI/vast.json --prefijo $PREFIJO --libro $LIBRO --horas-max 1"
echo "modo:   $MODO"
echo "orden:  $ORDEN"
if [ -n "$VAST_SECO" ]; then $ORDEN --seco; exit 0; fi

for DS in rect-lin-entreno-r20261008 rect-lin-banco-r20261008; do
    if [ -n "$(git -C "$EXPCNN_DATOS" status --porcelain -- "experimentos-cnn/$DS")" ] \
       || ! git -C "$EXPCNN_DATOS" fetch -q origin \
       || [ -n "$(git -C "$EXPCNN_DATOS" log --oneline origin/main..HEAD -- "experimentos-cnn/$DS")" ]; then
        echo "✗ $DS no está commiteado y empujado al almacén. No se alquila nada."; exit 2
    fi
done
if [ -n "$(git -C "$REPO" status --porcelain -- "$EXP/instrucciones" "$EXP/REGLAS.md" "$AQUI")" ] \
   || ! git -C "$REPO" fetch -q origin \
   || [ -n "$(git -C "$REPO" log --oneline origin/main..HEAD -- "$EXP")" ]; then
    echo "✗ el criterio o el código de rect-bor no están commiteados y empujados. No se alquila nada."; exit 2
fi

$ORDEN
cd "$REPO"
git add "$LIBRO"
git commit -qm "rect-bor: libro de la rejilla en Vast, al alquilar (lo escribe nn/vast.sh)" \
    && git push -q && echo "libro commiteado y empujado." \
    || echo "⚠ NO pude commitear/empujar el libro: hazlo a mano, es lo que dice qué máquinas eran de aquí."
echo "Cómo va:   $0 --estado"
echo "Apagarlo:  $0 apagar   (o /use exp-vast → apagar $PREFIJO)"
