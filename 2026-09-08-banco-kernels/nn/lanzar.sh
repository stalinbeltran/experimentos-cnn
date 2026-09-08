#!/usr/bin/env sh
# Lo que tarda de `banco-k`, lanzado como UNIDAD de systemd. Y como preguntarle.
#
#   nn/lanzar.sh datos       genera las 1000 imagenes del dataset (~30 min)
#   nn/lanzar.sh calibrar    corre la calibracion del §11 (~1 h)
#   nn/lanzar.sh kernel R    evalua el kernel R contra los criterios del §2 (~9 min)
#   nn/lanzar.sh --estado    ¿esta vivo? ¿por donde va? ¿fallo? ¿se relanzo?
#
# POR QUE ESTO ES UN SCRIPT COMMITEADO Y NO UNA LINEA QUE SE TECLEA
# -----------------------------------------------------------------
# Tecleada, la cadena no deja rastro de QUE se lanzo: si el resultado sale raro, no
# hay forma de saber si se corrio esto o algo parecido. Commiteado, «cual se uso» es
# una pregunta con respuesta y `git log` la fecha.
#
# POR QUE `desacoplar-persistente.sh` Y NO `desacoplar.sh`
# -------------------------------------------------------
# El `--scope` sobrevive al restart del coordinador pero NO a que muera su padre, y
# el fin de una sesion de Claude Code es exactamente «muere su padre». La unidad
# tiene por padre a PID 1. Medido en este sistema el 2026-09-07: la unidad siguio
# `active` a traves del final de la sesion (NRestarts=0) mientras los vigilantes del
# harness quedaron `stopped` sin registro.
#
# ⚠ El aviso lleva `|| true` SIEMPRE. Los secretos no viajan a la unidad, asi que
# `notify.mjs` puede salir con != 0, y con `set -e` eso tumbaria la cadena ENTERA
# despues de haberla completado -- y `Restart=on-failure` la relanzaria. Paso el
# 2026-09-02 y el 2026-09-04 (62 relanzamientos).
#
# ⚠ Un log VACIO no significa parado: python bufferiza cuando no es un tty. Por eso
# `--estado` lee el DISCO (manifiesto, metricas), no el log. Se usa `-u` igualmente.
set -e

AQUI=$(cd "$(dirname "$0")" && pwd)
EXP=$(dirname "$AQUI")
REPO=$(dirname "$EXP")
PY="$REPO/.venv/bin/python"
: "${COORD_HOME:=$HOME/src/telegram-coordinator}"

estado_unidad() {
    printf '  unidad   %s\n' "$1"
    printf '  activa   %s\n' "$(systemctl is-active "$1" 2>&1)"
    # ⚠ `Result` y sobre todo `NRestarts` son la mitad que se olvida: una unidad que
    # fallo y se relanzo sola PARECE «corriendo» y esta repitiendo trabajo desde cero.
    systemctl show "$1" -p Result -p NRestarts -p ExecMainStatus \
        -p ActiveEnterTimestamp 2>/dev/null | sed 's/^/           /'
    printf '  log      /tmp/%s.log (%s bytes)\n' "$1" \
        "$(wc -c < "/tmp/$1.log" 2>/dev/null || echo 0)"
}

if [ "$1" = "--estado" ]; then
    echo
    echo "DATOS"
    estado_unidad bancok-datos
    if [ -f "$EXP/datos/manifiesto.json" ]; then
        printf '  dataset  %s muestras · %s\n' \
            "$("$PY" -c "import json;print(json.load(open('$EXP/datos/manifiesto.json'))['imagenes_validas'])" 2>/dev/null || echo '?')" \
            "$("$PY" -c "import json;m=json.load(open('$EXP/datos/manifiesto.json'));print(' · '.join(f\"{k}={v['n']}\" for k,v in m['particiones'].items()))" 2>/dev/null || echo '?')"
    else
        printf '  dataset  (todavia no hay manifiesto)\n'
    fi
    echo
    echo "CALIBRACION"
    estado_unidad bancok-calibrar
    if [ -d "$EXP/resultados" ]; then
        printf '  corridas %s fichero(s) de metricas\n' \
            "$(find "$EXP/resultados" -name 'metricas.csv' 2>/dev/null | wc -l)"
        for f in "$EXP"/resultados/*/metricas.csv; do
            [ -f "$f" ] || continue
            printf '           %-22s %s filas\n' \
                "$(basename "$(dirname "$f")")" "$(($(wc -l < "$f") - 1))"
        done
    fi
    echo
    exit 0
fi

case "$1" in
    datos)     UNIDAD=bancok-datos ;;
    calibrar)  UNIDAD=bancok-calibrar ;;
    kernel)
        [ -n "$2" ] || { echo "uso: $0 kernel <ruta.npy>"; exit 2; }
        # El contrato del §5 se comprueba AQUI, en primer plano, antes de desacoplar
        # nada: un kernel invalido tiene que fallar donde lo estas mirando, no dentro
        # de una unidad cuyo log hay que ir a buscar.
        "$PY" nn/evaluar_kernel.py --contrato "$2" || exit 2
        UNIDAD="bancok-$(basename "$2" .npy)" ;;
    *) echo "uso: $0 datos|calibrar|kernel <ruta.npy>|--estado"; exit 2 ;;
esac

# No se lanza dos veces: dos procesos escribiendo los mismos ficheros los corrompen,
# y el segundo lanzamiento es lo mas facil de hacer por error justo cuando no sabes
# si el primero sigue vivo.
if [ "$(systemctl is-active "$UNIDAD" 2>&1)" = "active" ]; then
    echo "✗ '$UNIDAD' YA esta corriendo. No se lanza dos veces."
    echo "  Mira como va:  $0 --estado"
    exit 1
fi

cd "$EXP"
if [ "$1" = "kernel" ]; then
    ORDEN="$PY -u nn/evaluar_kernel.py --kernel '$2'
node \"\$COORD_HOME/scripts/notify.mjs\" 'banco-k: kernel $(basename "$2") evaluado. Veredicto en resultados/$(basename "$2" .npy)/criterios.json' || true"
elif [ "$1" = "datos" ]; then
    ORDEN="$PY -u nn/datos.py --imagenes 1000
$PY -u nn/datos.py --muestras 16
node \"\$COORD_HOME/scripts/notify.mjs\" 'banco-k: dataset de 1000 parrafos generado. Mira nn/lanzar.sh --estado' || true"
else
    ORDEN="$PY -u nn/calibrar.py --todo
node \"\$COORD_HOME/scripts/notify.mjs\" 'banco-k: calibracion terminada. Resultados en resultados/' || true"
fi

COORD_HOME="$COORD_HOME" "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" \
    sh -c "set -e
$ORDEN"

echo
echo "Como va, sin acordarse de nada:   $0 --estado"
echo "⚠ El aviso es una comodidad. La fuente de verdad es el disco (manifiesto/metricas)."
