#!/usr/bin/env sh
# El CIERRE de `dim-gen`: espera a que las DIEZ maquinas de Vast terminen (unidades
# expc-dimgen-s1..s5 y c1..c5 paradas y libro en estado terminal), y entonces:
#   1. nn/informe.py                     -> resultados/RESULTADOS.md + figuras
#   2. commit + push en experimentos-cnn  (metricas, resumenes, logs, informe, libro; NO .pt)
#   3. copia al VOLUMEN (repo de datos, experimentos-cnn-resultados/dim-gen/): pesos, metricas,
#      resumenes, informe, libro y logs de las unidades; commit + push al almacen
#   4. aviso por Telegram (notify.mjs || true)
# Corre como unidad de systemd (desacoplar-persistente.sh), padre PID 1, y SALE SIEMPRE CON 0:
# la unidad es Restart=on-failure, y ahi un fallo al final es un bucle (62 relanzamientos el
# 2026-09-04). Cada paso dice si fallo; ninguno tumba a los demas.
#
# Orden del dueno (2026-10-02): «Guarda todo en el volumen (si falta espacio hazlo crecer)».
AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
NODE=$(command -v node || ls "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)
PY="$REPO/.venv/bin/python"
LIBRO="$EXP/resultados/vast/todo"
PREFIJO=expc-dimgen-
FECHA=$(date -u +%Y-%m-%d)
HORAS_TOPE=8

aviso() { [ -n "$NODE" ] && "$NODE" "$COORD_HOME/scripts/notify.mjs" "$1" || true; }
log() { echo "$(date -u +%H:%M:%S) $*"; }

log "cierre de dim-gen: espero a s1..s5 y c1..c5 (tope $HORAS_TOPE h)"
limite=$(( $(date +%s) + HORAS_TOPE * 3600 ))
while :; do
    pend=""
    for t in s1 s2 s3 s4 s5 c1 c2 c3 c4 c5; do
        if systemctl is-active --quiet "${PREFIJO}$t" 2>/dev/null; then pend="$pend $t(unidad)"; continue; fi
        f="$LIBRO/$t.json"
        [ -f "$f" ] || { pend="$pend $t(sin-libro)"; continue; }
        est=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('estado',''))" "$f" 2>/dev/null)
        case "$est" in destruida|NO-DESTRUIDA|sin-alquilar) ;; *) pend="$pend $t($est)" ;; esac
    done
    [ -z "$pend" ] && break
    if [ "$(date +%s)" -gt "$limite" ]; then
        log "tope de $HORAS_TOPE h con pendientes:$pend. Sigo con lo que haya."
        aviso "⚠ dim-gen cierre: $HORAS_TOPE h y siguen pendientes:$pend. Cierro con lo que hay; mira nn/vast.sh --estado"
        break
    fi
    sleep 60
done
log "las maquinas terminaron (o se agoto el tope). Resumen del libro:"
python3 "$(cd "$REPO" && python3 -c 'from expcnn import exigir_lanzador; print(exigir_lanzador())')/scripts/vast_instance.py" trabajo --estado --libro "$LIBRO" 2>&1 | tail -12 || true

cd "$EXP"
n_sum=$(ls nn/pesos/*/summary.json 2>/dev/null | wc -l)
log "summary.json presentes: $n_sum de 30"
if [ "$n_sum" -gt 0 ]; then
    "$PY" nn/informe.py > /tmp/dim-gen-informe.txt 2>&1 && log "informe hecho" || log "⚠ informe fallo: $(tail -3 /tmp/dim-gen-informe.txt)"
fi

# 2. el repo publico: todo menos los .pt (los ignora el .gitignore de la carpeta)
cd "$REPO"
git add "$EXP/nn/pesos" "$EXP/resultados" 2>/dev/null || true
if git commit -qm "dim-gen: resultados de los $n_sum brazos en Vast (metricas, resumenes, logs, informe, libro)"; then
    git push -q && log "repo publico: commit y push hechos" || log "⚠ push del repo publico fallo"
else
    log "repo publico: nada nuevo que commitear"
fi

# 3. el volumen: TODO, pesos incluidos
DATOS=$(cd "$REPO" && python3 -c "from expcnn import exigir_datos; print(exigir_datos())" 2>/dev/null)
if [ -n "$DATOS" ] && [ -d "$DATOS" ]; then
    DEST="$DATOS/experimentos-cnn-resultados/dim-gen"
    mkdir -p "$DEST/logs"
    cp -r "$EXP/nn/pesos" "$DEST/" 2>/dev/null || true
    cp -r "$EXP/resultados" "$DEST/" 2>/dev/null || true
    for u in s1 s2 s3 s4 s5 c1 c2 c3 c4 c5 cierre; do cp "/tmp/${PREFIJO}$u.log" "$DEST/logs/" 2>/dev/null || true; done
    cp "$EXP/nn/vast.json" "$DEST/" 2>/dev/null || true
    cat > "$DEST/LEEME.md" <<FIN
# dim-gen — TODO lo que produjo el estudio, en el volumen ($FECHA)

Orden del dueño (2026-10-02): «Guarda todo en el volumen». El experimento (código, reglas,
criterio) vive en \`experimentos-cnn\` (público); aquí está lo que **allí no cabe o no entra**:

- \`pesos/<brazo>-s<semilla>/last.pt\` — los pesos de la época final de cada uno de los 30 brazos
  (\`w128 w064 w032 w016 w008\` y el control \`w128-de16\`, semillas 1–5), con su \`metrics.jsonl\`
  (una línea por época), \`summary.json\` (IoU train/val, brecha, desglose por factor, máquina) y
  \`log.txt\`.
- \`resultados/RESULTADOS.md\` + figuras: el informe generado por \`nn/informe.py\` con el criterio
  escrito antes de mirar.
- \`resultados/vast/todo/\`: el libro de las diez máquinas de Vast (qué, cuándo, cuánto costó).
- \`logs/\`: la salida de las diez unidades y del cierre.
- \`vast.json\`: el descriptor con el que se lanzó.

summary.json presentes al cerrar: $n_sum de 30. Si falta alguno, su log dice por qué.
FIN
    cd "$DATOS"
    git add experimentos-cnn-resultados .gitignore 2>/dev/null || true
    if git commit -qm "dim-gen: pesos, metricas, informe, libro y logs de los $n_sum brazos (orden del dueno 2026-10-02: todo al volumen)"; then
        git push -q && log "volumen: commit y push al almacen hechos ($(du -sh experimentos-cnn-resultados/dim-gen | cut -f1))" \
            || log "⚠ push al almacen fallo (¿mini apagado?). Lo commiteado se queda y se reintenta a mano: git -C $DATOS push"
    else
        log "volumen: nada nuevo que commitear"
    fi
else
    log "⚠ no encuentro el repo de datos: los pesos se quedan solo en $EXP/nn/pesos"
fi

# 4. el aviso
res=$(grep -E '^\- \*\*W' "$EXP/resultados/RESULTADOS.md" 2>/dev/null | head -4 | tr '\n' ' ')
aviso "✅ dim-gen: cierre hecho. $n_sum/30 brazos con resumen. $res — informe en experimentos-cnn/…/resultados/RESULTADOS.md y copia (con pesos) en el almacén: foveal-vision-data/experimentos-cnn-resultados/dim-gen/ (rama tema-2)"
log "cierre terminado."
exit 0
