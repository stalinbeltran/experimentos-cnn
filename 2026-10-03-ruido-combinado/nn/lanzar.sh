#!/usr/bin/env sh
# `ruido-comb` en el dev: las 15 corridas en serie como UNA unidad de systemd (padre PID 1), que al
# terminar corre el informe, commitea, copia todo al almacén y avisa.
#
#   nn/lanzar.sh todo                  los 5 escenarios × 3 semillas; se niega si ya hay unidad viva
#   nn/lanzar.sh corridas <id> [id…]   corridas sueltas
#   nn/lanzar.sh --estado              lee el DISCO
#   SECO=1 nn/lanzar.sh todo           imprime unidad y orden SIN lanzar (va ANTES del guardia)
#
# Copiado de ruido-nist. El modo se guarda al entrar y el último caso SE NIEGA. Una corrida con
# summary.json en disco se salta.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
UNIDAD=expc-ruidocomb
PY="$REPO/.venv/bin/python"
MODO="${1:-}"
[ "$#" -gt 0 ] && shift
SEMILLAS="1 2 3 4 5"
ESCENARIOS="limpio gaussiano@0.2-linea recorte@0.6-linea recorte@0.6+gaussiano@0.2-linea recorte@0.6~gaussiano@0.2-linea"

con_semillas() { for e in $(cat); do for s in $SEMILLAS; do echo "$e-s$s"; done; done; }

lanzar() {
    IDS="$*"; N=$(echo "$IDS" | wc -w)
    ORDEN="sh $AQUI/lanzar.sh --correr $IDS"
    echo "unidad: $UNIDAD ($N corridas)"
    echo "orden:  $COORD_HOME/scripts/desacoplar-persistente.sh $UNIDAD $ORDEN"
    if [ -n "${SECO:-}" ]; then echo "(seco: no se lanza nada)"; exit 0; fi
    if systemctl is-active --quiet "$UNIDAD" 2>/dev/null; then echo "✗ la unidad $UNIDAD ya está corriendo"; exit 2; fi
    (cd "$EXP" && "$PY" nn/modelo.py --inicializar >/dev/null) || { echo "✗ los pesos iniciales no casan"; exit 2; }
    cd "$EXP" && exec "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" $ORDEN
}

case "$MODO" in
    todo)
        [ "$#" -eq 0 ] || { echo "✗ todo no lleva argumentos"; exit 2; }
        lanzar $(echo "$ESCENARIOS" | tr ' ' '\n' | con_semillas) ;;
    corridas)
        [ "$#" -gt 0 ] || { echo "✗ corridas necesita al menos un <escenario>-s<semilla>"; exit 2; }
        for id in "$@"; do
            (cd "$AQUI" && "$PY" -c "import sys, entrenar_local as e; e.parsear_ident(sys.argv[1])" "$id" 2>/dev/null) || { echo "✗ corrida mal formada: $id"; exit 2; }
        done
        lanzar "$@" ;;
    --correr)
        cd "$EXP"; fallos=0; hechas=0; saltadas=0
        for id in "$@"; do
            if [ -f "nn/pesos/$id/summary.json" ]; then saltadas=$((saltadas + 1)); echo "· $id ya tiene summary.json: se salta"; continue; fi
            esc=${id%-s*}; sem=${id##*-s}
            mkdir -p "nn/pesos/$id"
            if "$PY" -u nn/entrenar_local.py --escenario "$esc" --semilla "$sem" > "nn/pesos/$id/log.txt" 2>&1; then hechas=$((hechas + 1))
            else fallos=$((fallos + 1)); echo "✗ $id falló (nn/pesos/$id/log.txt)"; fi
        done
        echo "corridas: $hechas hechas · $saltadas saltadas · $fallos con fallo"
        "$PY" nn/informe.py > /tmp/ruido-comb-informe.txt 2>&1 && echo "informe hecho" || echo "⚠ informe falló"
        "$PY" nn/ruido.py --muestras >/dev/null 2>&1 || echo "⚠ rejilla falló"
        cd "$REPO" && git add "$EXP/nn/pesos" "$EXP/resultados" 2>/dev/null
        git commit -qm "ruido-comb: resultados de las corridas (metricas, resumenes, logs, informe)" && git push -q && echo "repo publico: commit y push" || echo "⚠ commit/push publico"
        DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())" 2>/dev/null)
        if [ -n "$DATOS" ]; then
            DEST="$DATOS/experimentos-cnn-resultados/ruido-comb"; mkdir -p "$DEST/logs"
            cp -r "$EXP/nn/pesos" "$DEST/"; cp -r "$EXP/nn/init" "$DEST/"; cp -r "$EXP/resultados" "$DEST/" 2>/dev/null; cp "/tmp/$UNIDAD.log" "$DEST/logs/" 2>/dev/null
            printf '# ruido-comb — todo lo que produjo el estudio (%s)\n\nPesos, init, métricas, resúmenes, logs e informe de `ruido-comb` (experimentos-cnn). Al almacén por la regla del 2026-10-01.\n' "$(date -u +%Y-%m-%d)" > "$DEST/LEEME.md"
            cd "$DATOS" && git add experimentos-cnn-resultados && git commit -qm "ruido-comb: pesos, metricas, informe y logs (todo al almacen)" && git push -q && echo "almacen: commit y push" || echo "⚠ push al almacen"
        fi
        res=$(grep -E '^\- \*\*' "$EXP/resultados/RESULTADOS.md" 2>/dev/null | head -4 | tr '\n' ' ')
        node "$COORD_HOME/scripts/notify.mjs" "✅ ruido-comb: terminado ($hechas hechas, $fallos fallo(s)). $res" || true
        exit 0 ;;
    --estado)
        echo "unidad $UNIDAD: $(systemctl is-active "$UNIDAD" 2>/dev/null || echo inactiva) · NRestarts=$(systemctl show "$UNIDAD" -p NRestarts --value 2>/dev/null || echo '?') · Result=$(systemctl show "$UNIDAD" -p Result --value 2>/dev/null || echo '?')"
        echo "resúmenes en disco: $(ls "$EXP"/nn/pesos/*/summary.json 2>/dev/null | wc -l) de 25"
        [ -f "$EXP/resultados/RESULTADOS.md" ] && grep -E '^\- \*\*' "$EXP/resultados/RESULTADOS.md" | head -5
        exit 0 ;;
    *) echo "uso: $0 todo | corridas <id...> | --estado   (SECO=1 para ver sin lanzar)"; exit 2 ;;
esac
