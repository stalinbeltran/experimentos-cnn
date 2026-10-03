#!/usr/bin/env sh
# Lanza el entrenamiento de los detectores de `feat-ind` como UNIDAD de systemd (padre PID 1), que es
# lo único que sobrevive al fin del turno que lo lanza. Imprime la orden ANTES de desacoplar.
#
#   nn/lanzar.sh todas                      los 13 detectores en serie (unidad feat-ind-detectores)
#   nn/lanzar.sh una <feature> [args...]    un detector; args van a entrenar_local.py (--desde, --contra…)
#   nn/lanzar.sh reentrenar                 corrida 3: arcos y esquinas desde nn/pesos/, enfatizando sus contra-casos
#                                           → nn/pesos-c3/; al final, aplicar.py + compositor.py con sufijo -c3
#   nn/lanzar.sh --estado                   ¿qué unidad hay, está activa, NRestarts, qué pesos existen?
#   SECO=1 nn/lanzar.sh todas               imprime unidad y orden SIN lanzar (va ANTES del guardia)
set -eu
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
PY="$REPO/.venv/bin/python"
COORD_HOME="${COORD_HOME:-$HOME/src/telegram-coordinator}"
DESACOPLAR="$COORD_HOME/scripts/desacoplar-persistente.sh"
MODO="${1:-}"

estado() {
    for u in feat-ind-detectores feat-ind-una feat-ind-reentreno; do
        if systemctl list-units --all --no-legend "$u.service" 2>/dev/null | grep -q "$u"; then
            printf '%s: %s · ' "$u" "$(systemctl is-active "$u" 2>/dev/null || true)"
            systemctl show "$u" -p Result -p NRestarts -p ExecMainStatus --value 2>/dev/null | paste -sd' ' -
        fi
    done
    echo "pesos en $AQUI/pesos:"
    for d in "$AQUI"/pesos/*/ "$AQUI"/pesos-c3/*/; do
        [ -d "$d" ] || continue
        n=$(basename "$d")
        if [ -f "$d/summary.json" ]; then
            "$PY" - "$d/summary.json" <<'EOF'
import json,sys; s=json.load(open(sys.argv[1])); b=s["best"]
c3 = "c3" if "pesos-c3" in sys.argv[1] else "c2"
print(f"  {c3} {s['feature']:<12} {s['veredicto']:<12} F1 {b['f1']:.3f} P {b['precision']:.3f} R {b['recall']:.3f} pos<=1 {b['pos_ok']:.3f} (ep {b['epoca']}, {s['segundos']} s)")
EOF
        else
            echo "  $n: a medias (sin summary.json)"
        fi
    done
    echo "log: /tmp/feat-ind-detectores.log · /tmp/feat-ind-una.log · /tmp/feat-ind-reentreno.log"
}

case "$MODO" in
    --estado) estado; exit 0 ;;
    todas)
        UNIDAD=feat-ind-detectores
        ORDEN="cd '$EXP' && '$PY' -u nn/entrenar_local.py --todas; node '$COORD_HOME/scripts/notify.mjs' 'feat-ind: los 13 detectores terminaron (nn/lanzar.sh --estado)' || true" ;;
    reentrenar)
        UNIDAD=feat-ind-reentreno
        ORDEN="cd '$EXP' && '$PY' -u nn/entrenar_local.py --reentrenar-todos && '$PY' nn/aplicar.py --pesos nn/pesos --pesos nn/pesos-c3 --sufijo -c3 && '$PY' nn/compositor.py --sufijo -c3; node '$COORD_HOME/scripts/notify.mjs' 'feat-ind: re-entreno (corrida 3) terminado (nn/lanzar.sh --estado, resultados/compositores-c3.json)' || true" ;;
    una)
        shift
        [ $# -ge 1 ] || { echo "✗ una <feature> [args]" >&2; exit 2; }
        UNIDAD=feat-ind-una
        ORDEN="cd '$EXP' && '$PY' -u nn/entrenar_local.py --feature $*; node '$COORD_HOME/scripts/notify.mjs' 'feat-ind: detector $1 terminado' || true" ;;
    *) echo "✗ modo '$MODO' desconocido: todas | una <feature> | reentrenar | --estado" >&2; exit 2 ;;
esac

echo "unidad: $UNIDAD"
echo "orden:  $ORDEN"
if [ "${SECO:-0}" = 1 ]; then echo "(SECO: no lanzo nada)"; exit 0; fi
if systemctl is-active --quiet "$UNIDAD" 2>/dev/null; then
    echo "✗ $UNIDAD ya está corriendo: no lanzo dos veces (escribirían los mismos pesos)." >&2; exit 1
fi
[ -x "$DESACOPLAR" ] || { echo "✗ no está $DESACOPLAR (COORD_HOME=$COORD_HOME)" >&2; exit 1; }
exec "$DESACOPLAR" "$UNIDAD" sh -c "$ORDEN"
