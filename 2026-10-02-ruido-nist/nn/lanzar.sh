#!/usr/bin/env sh
# `ruido-nist` en el dev: las corridas de una fase en serie como UNA unidad de systemd (padre PID 1),
# que al terminar corre el informe, commitea, copia todo al almacén y avisa.
#
#   nn/lanzar.sh fase1                     limpio + los 9 tipos al nivel medio + oblicua@0.6-r2, × 3 semillas (33)
#   nn/lanzar.sh fase2 <tipo> [tipo ...]   los 5 niveles de cada tipo × 3 semillas (el medio ya hecho se salta)
#   nn/lanzar.sh corridas <id> [id ...]    corridas sueltas, p. ej. `horizontal@0.6-s2`
#   nn/lanzar.sh --estado                  lee el DISCO: unidad, NRestarts, resúmenes presentes
#   SECO=1 nn/lanzar.sh fase1              imprime la unidad y la orden SIN lanzar (va ANTES del guardia)
#
# El modo se guarda al entrar y el último caso del despacho SE NIEGA (CLAUDE.md del coordinador,
# 2026-09-08). Una corrida con `summary.json` ya en disco se SALTA (así una fase se puede relanzar
# tras un fallo sin repetir lo hecho, y la fase 2 reutiliza el nivel medio de la fase 1).
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export COORD_HOME
UNIDAD=expc-ruidonist
PY="$REPO/.venv/bin/python"
MODO="${1:-}"
[ "$#" -gt 0 ] && shift
SEMILLAS="1 2 3"
TIPOS="borrado externos horizontal vertical oblicua curva recorte gaussiano sal-pimienta"

escenarios_fase1() {
    # limpio + cada tipo al nivel medio (índice 2 de su tabla, lo imprime nn/ruido.py) + la 2ª copia de oblicua
    echo limpio
    "$PY" -c "
import sys; sys.path.insert(0, '$AQUI'); import ruido
for t in ruido.TIPOS: print(ruido.escenario(t, ruido.NIVELES[t][ruido.INDICE_MEDIO]))
print(ruido.escenario('oblicua', ruido.NIVELES['oblicua'][ruido.INDICE_MEDIO], 2))"
}

escenarios_fase2() {
    "$PY" -c "
import sys; sys.path.insert(0, '$AQUI'); import ruido
for t in sys.argv[1:]:
    for n in ruido.NIVELES[t]: print(ruido.escenario(t, n))" "$@"
}

con_semillas() { for e in $(cat); do for s in $SEMILLAS; do echo "$e-s$s"; done; done; }

lanzar() {
    IDS="$*"
    N=$(echo "$IDS" | wc -w)
    ORDEN="sh $AQUI/lanzar.sh --correr $IDS"
    echo "unidad: $UNIDAD ($N corridas)"
    echo "orden:  $COORD_HOME/scripts/desacoplar-persistente.sh $UNIDAD $ORDEN"
    if [ -n "${SECO:-}" ]; then echo "(seco: no se lanza nada)"; exit 0; fi
    if systemctl is-active --quiet "$UNIDAD" 2>/dev/null; then
        echo "✗ la unidad $UNIDAD ya está corriendo: no se lanza dos veces (nn/lanzar.sh --estado)"; exit 2
    fi
    if ! (cd "$AQUI" && "$PY" -c "import sys, entrenar_local as e; sys.exit(e.LR is None)" 2>/dev/null); then
        echo "✗ LR no está congelado en nn/entrenar_local.py. No se lanza."; exit 2
    fi
    (cd "$EXP" && "$PY" nn/modelo.py --inicializar >/dev/null) || { echo "✗ los pesos iniciales no casan (nn/modelo.py --inicializar)"; exit 2; }
    cd "$EXP" && exec "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" $ORDEN
}

case "$MODO" in
    fase1)
        [ "$#" -eq 0 ] || { echo "✗ fase1 no lleva argumentos"; exit 2; }
        lanzar $(escenarios_fase1 | con_semillas) ;;
    fase2)
        [ "$#" -gt 0 ] || { echo "✗ fase2 necesita los tipos que pasaron la fase 1 (resultados/criterio-aplicado.json → fase2)"; exit 2; }
        for t in "$@"; do
            case " $TIPOS " in *" $t "*) ;; *) echo "✗ tipo desconocido: $t (los tipos: $TIPOS)"; exit 2 ;; esac
        done
        lanzar $(escenarios_fase2 "$@" | con_semillas) ;;
    corridas)
        [ "$#" -gt 0 ] || { echo "✗ corridas necesita al menos un <escenario>-s<semilla>"; exit 2; }
        for id in "$@"; do
            (cd "$AQUI" && "$PY" -c "import sys, entrenar_local as e; e.parsear_ident(sys.argv[1])" "$id" 2>/dev/null) || { echo "✗ corrida mal formada: $id"; exit 2; }
        done
        lanzar "$@" ;;
    --correr)
        # el cuerpo de la unidad: cada corrida es un proceso `entrenar_local.py` (el freno casa el nombre)
        cd "$EXP"; fallos=0; hechas=0; saltadas=0
        for id in "$@"; do
            if [ -f "nn/pesos/$id/summary.json" ]; then saltadas=$((saltadas + 1)); echo "· $id ya tiene summary.json: se salta"; continue; fi
            esc=${id%-s*}; sem=${id##*-s}
            mkdir -p "nn/pesos/$id"
            if "$PY" -u nn/entrenar_local.py --escenario "$esc" --semilla "$sem" > "nn/pesos/$id/log.txt" 2>&1; then hechas=$((hechas + 1))
            else fallos=$((fallos + 1)); echo "✗ $id falló (nn/pesos/$id/log.txt)"; fi
        done
        echo "corridas: $hechas hechas · $saltadas saltadas · $fallos con fallo"
        "$PY" nn/informe.py > /tmp/ruido-nist-informe.txt 2>&1 && echo "informe hecho" || echo "⚠ informe falló (/tmp/ruido-nist-informe.txt)"
        "$PY" nn/ruido.py --muestras >/dev/null 2>&1 || echo "⚠ rejilla de muestras falló"
        cd "$REPO" && git add "$EXP/nn/pesos" "$EXP/resultados" 2>/dev/null
        git commit -qm "ruido-nist: resultados de las corridas (metricas, resumenes, logs, informe)" && git push -q && echo "repo publico: commit y push" || echo "⚠ commit/push publico"
        DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())" 2>/dev/null)
        if [ -n "$DATOS" ]; then
            DEST="$DATOS/experimentos-cnn-resultados/ruido-nist"; mkdir -p "$DEST/logs"
            cp -r "$EXP/nn/pesos" "$DEST/"; cp -r "$EXP/nn/init" "$DEST/"; cp -r "$EXP/resultados" "$DEST/" 2>/dev/null; cp "/tmp/$UNIDAD.log" "$DEST/logs/" 2>/dev/null
            printf '# ruido-nist — todo lo que produjo el estudio (%s)\n\nPesos (`last.pt`), pesos iniciales compartidos (`init/`), métricas, resúmenes, logs e informe de `ruido-nist` (experimentos-cnn). Al almacén por la regla del 2026-10-01: lo que no está empujado al almacén no existe.\n' "$(date -u +%Y-%m-%d)" > "$DEST/LEEME.md"
            cd "$DATOS" && git add experimentos-cnn-resultados && git commit -qm "ruido-nist: pesos, metricas, informe y logs (todo al almacen)" && git push -q && echo "almacen: commit y push" || echo "⚠ push al almacen"
        fi
        res=$(grep -E '^\- \*\*' "$EXP/resultados/RESULTADOS.md" 2>/dev/null | head -5 | tr '\n' ' ')
        node "$COORD_HOME/scripts/notify.mjs" "✅ ruido-nist: terminado ($hechas hechas, $fallos fallo(s)). $res — resultados/RESULTADOS.md y copia en el almacén" || true
        exit 0 ;;
    --estado)
        echo "unidad $UNIDAD: $(systemctl is-active "$UNIDAD" 2>/dev/null || echo inactiva) · NRestarts=$(systemctl show "$UNIDAD" -p NRestarts --value 2>/dev/null || echo '?') · Result=$(systemctl show "$UNIDAD" -p Result --value 2>/dev/null || echo '?')"
        echo "resúmenes en disco: $(ls "$EXP"/nn/pesos/*/summary.json 2>/dev/null | wc -l) (fase 1 son 33)"
        for id in $(escenarios_fase1 | con_semillas); do [ -f "$EXP/nn/pesos/$id/summary.json" ] || echo "  falta (fase 1): $id"; done
        [ -f "$EXP/resultados/RESULTADOS.md" ] && grep -E '^\- \*\*' "$EXP/resultados/RESULTADOS.md" | head -6
        exit 0 ;;
    *) echo "uso: $0 fase1 | fase2 <tipo...> | corridas <id...> | --estado   (SECO=1 para ver sin lanzar)"; exit 2 ;;
esac
