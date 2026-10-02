#!/bin/sh
# Despierta la conversación del tema cuando el cierre de dim-gen termine (patrón del CLAUDE.md:
# trabajo | claude-session.mjs | notify.mjs). Sale siempre con 0. Tope 9 h.
cd /home/deploy/src/telegram-coordinator
limite=$(( $(date +%s) + 9*3600 ))
while systemctl is-active --quiet expc-dimgen-cierre || pgrep -f 'claude-session.mjs' >/dev/null; do
  [ "$(date +%s)" -gt "$limite" ] && { echo "tope 9 h"; exit 0; }
  sleep 60
done
sleep 20
printf '%s' "El cierre de dim-gen (experimentos-cnn/2026-10-01-dimension-generalizacion, rama tema-2) ha terminado: lee /tmp/expc-dimgen-cierre.log y nn/vast.sh --estado. Comprueba que resultados/RESULTADOS.md y la copia con pesos en foveal-vision-data/experimentos-cnn-resultados/dim-gen/ existen y están empujadas (y si falta algún brazo, di cuál y por qué). Escribe el veredicto aplicando instrucciones/02-criterio.md sin suavizarlo en el README.md de la carpeta, pon estado cerrado en experimento.json, añade la fila y el reporte en estudios-redes-neuronales (tipo estudios: inicio/fin UTC, máquinas alquiladas con sus reintentos y coste real del libro), commitea y empuja todo en tema-2, y termina con la línea del freno." \
  | node scripts/claude-session.mjs --model fable --effort max | node scripts/notify.mjs || true
exit 0
