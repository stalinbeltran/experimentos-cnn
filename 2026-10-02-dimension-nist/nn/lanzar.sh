#!/usr/bin/env sh
# `dim-nist` en el dev: los 30 brazos en serie como UNA unidad de systemd (padre PID 1), que al
# terminar corre el informe, commitea, copia todo al volumen y avisa.
#
#   nn/lanzar.sh todo [brazo ...]    lanza (p. ej. `todo w6-s1 w6-s2`); se niega si ya hay unidad viva
#   nn/lanzar.sh --estado            lee el DISCO: unidad, NRestarts, resumenes presentes
#   SECO=1 nn/lanzar.sh todo         imprime la unidad y la orden SIN lanzar (va ANTES del guardia)
#
# Copiado del patron de `banco-k`/`bor-k` (Regla 0: se copia). El modo se guarda al entrar y el
# ultimo caso del despacho SE NIEGA (CLAUDE.md del coordinador, 2026-09-08).
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
UNIDAD=expc-dimnist-todo
PY="$REPO/.venv/bin/python"
MODO="${1:-}"
[ "$#" -gt 0 ] && shift

BRAZOS_TODOS=""
for b in w4 w5 w6 w7 w8 w8-de4; do for s in 1 2 3 4 5; do BRAZOS_TODOS="$BRAZOS_TODOS $b-s$s"; done; done

case "$MODO" in
    todo)
        BRAZOS="${*:-$BRAZOS_TODOS}"
        ORDEN="sh $AQUI/lanzar.sh --correr $BRAZOS"
        echo "unidad: $UNIDAD"
        echo "orden:  $COORD_HOME/scripts/desacoplar-persistente.sh $UNIDAD $ORDEN"
        if [ -n "${SECO:-}" ]; then echo "(seco: no se lanza nada)"; exit 0; fi
        if ! (cd "$AQUI" && "$PY" -c "import sys, entrenar_local as e; sys.exit(e.LR is None)" 2>/dev/null); then
            echo "✗ LR no esta congelado en nn/entrenar_local.py. No se lanza."; exit 2
        fi
        cd "$EXP" && exec "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" $ORDEN ;;
    --correr)
        # el cuerpo de la unidad: cada brazo es un proceso `entrenar_local.py` (el freno casa el nombre)
        cd "$EXP"; fallos=0
        for b in "$@"; do
            mkdir -p "nn/pesos/$b"
            "$PY" -u nn/entrenar_local.py --brazo "$b" > "nn/pesos/$b/log.txt" 2>&1 || { fallos=$((fallos + 1)); echo "✗ $b fallo (nn/pesos/$b/log.txt)"; }
        done
        echo "brazos con fallo: $fallos"
        "$PY" nn/informe.py > /tmp/dim-nist-informe.txt 2>&1 && echo "informe hecho" || echo "⚠ informe fallo"
        cd "$REPO" && git add "$EXP/nn/pesos" "$EXP/resultados" 2>/dev/null
        git commit -qm "dim-nist: resultados de los brazos (metricas, resumenes, logs, informe)" && git push -q && echo "repo publico: commit y push" || echo "⚠ commit/push publico"
        DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())" 2>/dev/null)
        if [ -n "$DATOS" ]; then
            DEST="$DATOS/experimentos-cnn-resultados/dim-nist"; mkdir -p "$DEST/logs"
            cp -r "$EXP/nn/pesos" "$DEST/"; cp -r "$EXP/resultados" "$DEST/" 2>/dev/null; cp "/tmp/$UNIDAD.log" "$DEST/logs/" 2>/dev/null
            printf '# dim-nist — todo lo que produjo el estudio (%s)\n\nPesos (`last.pt`), métricas, resúmenes, logs e informe de los brazos de `dim-nist` (experimentos-cnn, rama tema-2). Orden del dueño del 2026-10-02: todo al volumen.\n' "$(date -u +%Y-%m-%d)" > "$DEST/LEEME.md"
            cd "$DATOS" && git add experimentos-cnn-resultados && git commit -qm "dim-nist: pesos, metricas, informe y logs (todo al volumen)" && git push -q && echo "volumen: commit y push" || echo "⚠ push al almacen"
        fi
        res=$(grep -E '^\- \*\*W' "$EXP/resultados/RESULTADOS.md" 2>/dev/null | head -4 | tr '\n' ' ')
        node "$COORD_HOME/scripts/notify.mjs" "✅ dim-nist: terminado ($fallos fallo(s)). $res — resultados/RESULTADOS.md y copia en el almacén" || true
        exit 0 ;;
    --estado)
        echo "unidad $UNIDAD: $(systemctl is-active "$UNIDAD" 2>/dev/null || echo inactiva) · NRestarts=$(systemctl show "$UNIDAD" -p NRestarts --value 2>/dev/null || echo '?')"
        echo "resumenes: $(ls "$EXP"/nn/pesos/*/summary.json 2>/dev/null | wc -l) de 30"
        [ -f "$EXP/resultados/RESULTADOS.md" ] && grep -E '^\- \*\*W' "$EXP/resultados/RESULTADOS.md"
        exit 0 ;;
    *) echo "uso: $0 todo [brazo ...] | --estado   (SECO=1 para ver sin lanzar)"; exit 2 ;;
esac
