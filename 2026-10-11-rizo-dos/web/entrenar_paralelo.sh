#!/bin/sh
# Entrena los compositores del revisor de rizo-2 EN PARALELO: un droplet de DigitalOcean por receta (tipo `dev`, sin
# servicios), cada uno entrena la suya por consola (`servidor.py entrenar <id>`), empuja su `.pt` al almacén y se DESTRUYE
# pase lo que pase. Pedido por el dueño el 2026-10-16 («inicia el entrenamiento en paralelo»): en serie en el dev son ~15 min
# por receta; aquí lo que tarde la más lenta (+ ~6 min de nacer y morir cada máquina).
#
#   entrenar_paralelo.sh lanzar [emulados|todas|<id>…]   arma la UNIDAD `rizo2-entrenar-paralelo` (padre PID 1) y vuelve
#   entrenar_paralelo.sh seco   [emulados|todas|<id>…]   qué lanzaría: recetas, nombres de las máquinas y la orden, SIN lanzar
#   entrenar_paralelo.sh estado                            ¿unidad viva? NRestarts, droplets vivos, qué .pt han llegado al almacén
#   entrenar_paralelo.sh off                               para la unidad y destruye los droplets `rizo2-*` que queden
#   entrenar_paralelo.sh --dentro <id>…                    (lo que corre la unidad: no se teclea)
#
# Las cuatro reglas del lanzador (telegram-coordinator/CLAUDE.md, 2026-09-08): el MODO se guarda al entrar; el caso desconocido
# SE NIEGA; se IMPRIME la orden antes de desacoplar; hay modo SECO y va ANTES del guardia de «ya está corriendo».
# Gasto: 1 s-2vcpu-4gb (0,0357 $/h) por receta durante ~25 min ⇒ ≈0,015 $ por receta (estimado; el real lo dice `estado`).
# Freno: `cerrable.mjs` cuenta los droplets DO sueltos y esta unidad por su nombre de script; `apagar-do` del mini también los ve.
MODO="${1:-}"; [ $# -gt 0 ] && shift
AQUI=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$AQUI/../.." && pwd)
LANZ="$HOME/src/digital-ocean-dropplet-auto-launching"; L="python3 $LANZ/scripts/do_droplet.py"
COORD="${COORD_HOME:-$HOME/src/telegram-coordinator}"
UNIDAD=rizo2-entrenar-paralelo; LOG=/tmp/$UNIDAD.log
ALM="$HOME/src/foveal-vision-data/experimentos-cnn-resultados/rizo-2/revisor"
PY="$REPO/.venv/bin/python"; SRV="$AQUI/servidor.py"

ids_de() {   # qué recetas (IDs de 8 hex) corresponden a lo pedido; vacío si el servidor se niega (receta desconocida)
    SAL=$("$PY" "$SRV" entrenar "$@" --seco 2>/dev/null) || return 1
    echo "$SAL" | awk '/^  [0-9a-f]{8} · /{print $1}'
}

case "$MODO" in
lanzar|seco)
    [ $# -gt 0 ] || set -- emulados
    IDS=$(ids_de "$@") || true
    if [ -z "$IDS" ]; then
        echo "✗ nada que entrenar o receta desconocida:"; "$PY" "$SRV" entrenar "$@" --seco 2>/dev/null; exit 2
    fi
    echo "Recetas ($(echo "$IDS" | wc -l)):"; "$PY" "$SRV" entrenar "$@" --seco 2>/dev/null | grep '^  '
    echo "Máquinas: $(for i in $IDS; do printf 'rizo2-%s ' "$i"; done)(tipo dev, sfo2, sin servicios; se destruyen solas)"
    ORDEN="$COORD/scripts/desacoplar-persistente.sh $UNIDAD sh $AQUI/entrenar_paralelo.sh --dentro $(echo $IDS)"
    echo "Orden: $ORDEN"
    [ "$MODO" = seco ] && { echo "🧪 SECO — no he lanzado nada."; exit 0; }
    if systemctl is-active --quiet "$UNIDAD" 2>/dev/null; then
        echo "✗ la unidad $UNIDAD YA está corriendo: no lanzo dos veces (mira \`estado\`, o \`off\`)"; exit 1
    fi
    if [ -n "$($L list --json 2>/dev/null | grep -o '"rizo2-[0-9a-f]*"')" ]; then
        echo "✗ quedan droplets rizo2-* vivos de otra tanda: \`off\` primero"; $L list; exit 1
    fi
    if ! git -C "$REPO" diff --quiet || [ -n "$(git -C "$REPO" log origin/main..HEAD --oneline 2>/dev/null)" ]; then
        echo "✗ experimentos-cnn tiene cambios sin empujar: cada máquina CLONA main, así que entrenaría OTRO código. Empuja primero."; exit 1
    fi
    $ORDEN
    echo "▶️ Armada. Unidad \`$UNIDAD\` · log $LOG · \`estado\` para seguirla. Avisará por Telegram al terminar."
    ;;
estado)
    printf 'unidad %s: ' "$UNIDAD"; systemctl is-active "$UNIDAD" 2>/dev/null || true
    systemctl show "$UNIDAD" -p Result -p NRestarts -p ExecMainStartTimestamp 2>/dev/null | sed 's/^/  /'
    echo "droplets rizo2-*:"; $L list 2>/dev/null | grep -E 'rizo2-|Gastando' | sed 's/^/  /' || echo "  (ninguno, o la API no contesta)"
    echo "pesos en el almacén (local; \`git -C ~/src/foveal-vision-data pull\` para ver los de las máquinas):"
    git -C "$HOME/src/foveal-vision-data" pull -q --rebase 2>/dev/null || true
    for f in "$ALM"/*.pt; do
        "$PY" - "$f" <<'PYE'
import sys, torch; f=sys.argv[1]; z=torch.load(f, weights_only=False, map_location="cpu"); r=z.get("resumen",{})
print(f"  {'EMULADO' if r.get('emulado') else 'real   '} val {r.get('val')} · {z.get('guardado')} · {z.get('nombre')}")
PYE
    done
    [ -f "$LOG" ] && { echo "últimas líneas de $LOG:"; tail -n 12 "$LOG" | sed 's/^/  /'; }
    ;;
off)
    sudo -n systemctl stop "$UNIDAD" 2>/dev/null && echo "unidad parada" || echo "⚠ no pude parar la unidad (¿sudo?): sudo systemctl stop $UNIDAD"
    for n in $($L list --json 2>/dev/null | grep -o '"rizo2-[0-9a-f]*"' | tr -d '"'); do $L destroy "$n" --yes; done
    $L list
    ;;
--dentro)
    # Lo que corre la UNIDAD. Sale SIEMPRE con 0 (Restart=on-failure: un fallo al final sería un bucle). Cada receta en su
    # subshell; el destroy va fuera de toda condición.
    cd "$LANZ" || exit 0
    paso() { echo; echo "==== $(date -u +%T) UTC · $*"; }
    T0=$(date +%s)
    for ID in "$@"; do
        (
            N="rizo2-$ID"; LG="/tmp/$N.log"
            {
            paso "$N launch"
            timeout 1800 $L launch "$N" --type dev --service '' ; echo "[rc launch=$?]"
            paso "$N entrenar $ID"
            timeout 5400 $L ssh "$N" --cmd "cd ~/src/experimentos-cnn && .venv/bin/python 2026-10-11-rizo-dos/web/servidor.py entrenar $ID" ; echo "[rc entrenar=$?]"
            paso "$N destroy"
            $L destroy "$N" --yes ; echo "[rc destroy=$?]"
            } > "$LG" 2>&1
        ) &
    done
    wait
    paso "¿queda algo vivo?"; $L list
    for ID in "$@"; do N="rizo2-$ID"; echo "--- $N"; grep -E '^\[rc |val [0-9.]+ ·|✗|⚠' "/tmp/$N.log"; done
    # Si algún destroy falló, se repite (y si sigue vivo, el freno lo nombra)
    for n in $($L list --json 2>/dev/null | grep -o '"rizo2-[0-9a-f]*"' | tr -d '"'); do $L destroy "$n" --yes; done
    MIN=$(( ($(date +%s) - T0) / 60 ))
    git -C "$HOME/src/foveal-vision-data" pull -q --rebase 2>/dev/null || true
    RES=$(for ID in "$@"; do grep -h -oE 'val [0-9.]+' "/tmp/rizo2-$ID.log" | head -1 | sed "s/^/$ID: /" ; done | tr '\n' ' ')
    # el log, al almacén (temporal/<máquina>/<fecha>/): lo que queda en /tmp no existe
    D="$HOME/src/foveal-vision-data/temporal/dev/$(date -u +%F)"; mkdir -p "$D"; cp /tmp/rizo2-*.log "$LOG" "$D/" 2>/dev/null || true
    git -C "$HOME/src/foveal-vision-data" add temporal >/dev/null 2>&1 && git -C "$HOME/src/foveal-vision-data" commit -qm "rizo-2: logs del entrenamiento en paralelo ($(date -u +%F))" >/dev/null 2>&1 && git -C "$HOME/src/foveal-vision-data" push -q 2>/dev/null || true
    VIVOS=$($L list 2>/dev/null | grep -c 'rizo2-' || true)
    node "$COORD/scripts/notify.mjs" "✅ Entrenamiento en paralelo de rizo-2 terminado en $MIN min: $# receta(s). $RES· droplets rizo2-* vivos: $VIVOS. Pesos en el almacén; la app los toma al reiniciar la unidad rizo2-revisor (\`/use revisor\` → \`instalar\`). Log: $LOG" || true
    exit 0
    ;;
*)
    echo "✗ no sé qué es «$MODO». lanzar [emulados|todas|<id>…] · seco … · estado · off"; exit 2 ;;
esac
