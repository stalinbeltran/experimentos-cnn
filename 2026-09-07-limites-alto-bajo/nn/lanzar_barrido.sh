#!/usr/bin/env sh
# El barrido de `lim-ab` (limites alto/bajo), como UNIDAD de systemd.
#
#   nn/lanzar_barrido.sh            lanza los 5 brazos x 300 epocas + las figuras
#   nn/lanzar_barrido.sh --estado   ¿esta vivo? ¿por donde va? ¿fallo? ¿se relanzo?
#
# POR QUE ESTO ES UN SCRIPT Y NO UNA LINEA QUE SE TECLEA
# -----------------------------------------------------
# Por orden del dueno (2026-09-07): «Guarda el script q empleaste para el
# vigilante, y anotalo para claude. Asi podemos saber cual se uso, y si funciono
# como se espera. Ya hemos tenido problemas con los vigilantes antes, queremos
# depurarlos».
#
# Tecleada, la cadena no deja rastro de QUE se lanzo: si el resultado sale raro,
# no hay forma de saber si se corrio esto o algo parecido. Commiteada, «cual se
# uso» es una pregunta con respuesta, y `git log` la fecha.
#
# LO QUE SE MIDIO CON ESTE LANZADOR EN `esq-2d` (heredado: aqui NO se ha corrido)
# ---------------------------------------------------------------------------
# La unidad `esq2d-barrido` arranco a las 14:11:32 UTC. A mitad del barrido, el
# proceso de Claude Code que la lanzo termino. Resultado:
#
#   la UNIDAD          -> siguio `active`, NRestarts=0, y `k09` termino sus 300
#                         epocas DESPUES de que la sesion muriera.        ✅
#   los VIGILANTES del -> los dos (`run_in_background` del harness, un
#   harness               `until systemctl is-active...; do sleep; done`)
#                         quedaron marcados `stopped` SIN registro de
#                         finalizacion.                                    ❌
#
# O sea las dos mitades de la regla del proyecto, medidas en el mismo suceso: el
# trabajo sobrevive porque su padre es PID 1; el vigilante no, porque el suyo es
# la sesion. Detalle y consecuencias en `telegram-coordinator/CLAUDE.md`.
#
# ⚠ Y UNA TERCERA COSA QUE APRENDIMOS, que no estaba escrita: un `sleep` dentro
# de una tarea en segundo plano NO hace esperar al modelo. Las llamadas del
# turno vuelven al instante, asi que "esperar" encadenando sleeps no espera
# nada: gasta turnos y el reloj no avanza. La unica espera real es terminar el
# turno; y como el aviso del harness muere con la sesion, lo que de verdad
# contesta "¿como va?" es `--estado` leyendo el DISCO, que es lo que hay aqui.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
UNIDAD=limab-barrido
BRAZOS="k05 k07 k09 k11 k13 k15 k17"
EPOCAS=300
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"

estado() {
    printf 'unidad   %s\n' "$UNIDAD"
    printf 'activa   %s\n' "$(systemctl is-active "$UNIDAD" 2>&1)"
    # ⚠ `Result` y `NRestarts` son la mitad que se olvida: una unidad que fallo y
    # se relanzo sola parece "corriendo" y esta repitiendo trabajo. Ya paso dos
    # veces en este sistema (2026-09-02 y 2026-09-04, 62 relanzamientos).
    systemctl show "$UNIDAD" -p Result -p NRestarts -p ExecMainStatus \
        -p ActiveEnterTimestamp 2>/dev/null | sed 's/^/         /'
    printf 'epocas   (de %s por brazo)\n' "$EPOCAS"
    for b in $BRAZOS; do
        f="$AQUI/pesos/$b/metrics.jsonl"
        if [ -f "$f" ]; then
            printf '         %-5s %4s\n' "$b" "$(wc -l < "$f")"
        else
            printf '         %-5s    - (sin empezar)\n' "$b"
        fi
    done
    printf 'figuras  %s de %s con etiqueta ep%s\n' \
        "$(ls "$EXP/muestras" 2>/dev/null | grep -c -- "-ep$EPOCAS\.png" || true)" \
        "$(echo $BRAZOS | wc -w)" "$EPOCAS"
    # ⚠ El log puede verse VACIO y estar todo bien: python bufferiza su salida
    # cuando no es un tty. Por eso el estado se lee de `metrics.jsonl`, que se
    # escribe y se cierra en cada epoca, y no de aqui.
    printf 'log      /tmp/%s.log (%s bytes; python bufferiza: vacio no es parado)\n' \
        "$UNIDAD" "$(wc -c < "/tmp/$UNIDAD.log" 2>/dev/null || echo 0)"
}

if [ "$1" = "--estado" ]; then
    estado
    exit 0
fi

if [ "$(systemctl is-active "$UNIDAD" 2>&1)" = "active" ]; then
    echo "✗ '$UNIDAD' YA esta corriendo. No se lanza dos veces: dos procesos"
    echo "  escribiendo los mismos pesos y el mismo metrics.jsonl los corrompen."
    echo "  Mira como va:  $0 --estado"
    exit 1
fi

# ⚠ `desacoplar-persistente.sh` y no `desacoplar.sh`: el fin del turno es
# exactamente "muere su padre", que es la columna donde el `--scope` pierde.
# ⚠ El aviso lleva `|| true` SIEMPRE: los secretos no viajan a la unidad, asi que
# `notify.mjs` puede salir con != 0, y con `set -e` eso tumbaria la cadena ENTERA
# despues de haberla completado -- y `Restart=on-failure` la relanzaria. Paso el
# 2026-09-02 y el 2026-09-04 (62 relanzamientos).
cd "$EXP"
COORD_HOME="$COORD_HOME" "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" sh -c "
set -e
for b in $BRAZOS; do
  $REPO/.venv/bin/python nn/entrenar_local.py --brazo \$b --epocas $EPOCAS
done
$REPO/.venv/bin/python nn/muestras.py --etiqueta ep$EPOCAS
node \"\$COORD_HOME/scripts/notify.mjs\" 'lim-ab: barrido terminado (7 brazos x $EPOCAS epocas). Resultados en nn/pesos/*/metrics.jsonl y muestras/*-ep$EPOCAS.png' || true"

echo
echo "Como va, sin acordarse de nada:   $0 --estado"
echo "⚠ El aviso es una comodidad. La fuente de verdad es nn/pesos/*/metrics.jsonl."
