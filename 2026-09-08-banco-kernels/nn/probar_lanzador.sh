#!/usr/bin/env sh
# ¿El lanzador despacha CADA modo a lo que dice, y se NIEGA con lo que no conoce?
#
#   nn/probar_lanzador.sh        sale 0 si los cinco casos pasan
#
# POR QUE EXISTE ESTE FICHERO
# ===========================
# El 2026-09-08 se pidio evaluar tres kernels y `lanzar.sh` corrio la CALIBRACION. La
# unidad salio con Result=success, NRestarts=0 y ExecMainStatus=0 -- o sea que el freno
# la daba por buena --, y lo unico que delataba el fallo era que no aparecia ningun
# resultado nuevo en resultados/.
#
# La causa: la rama `kernel` hacia `shift` para recorrer los .npy, y el despacho de mas
# abajo RELEIA "$1" para elegir la orden. Tras el shift, "$1" era la ruta del primer
# .npy, la condicion daba falso, y la cadena caia al `else` -- que era la calibracion.
#
# Un comentario no impide que alguien vuelva a releer "$1". Esta prueba si: usa el modo
# SECO, que imprime la orden sin lanzar nada, y comprueba que cada modo produce la suya.
# Es la regla del preflight que crece con cada fallo, aplicada al lanzador.
set -e
AQUI=$(cd "$(dirname "$0")" && pwd)
L="$AQUI/lanzar.sh"
K="$AQUI/../kernels/gauss.npy"
fallos=0

comprobar() {   # <titulo> <esperado> <no-esperado> <argumentos...>
    titulo="$1"; esperado="$2"; prohibido="$3"; shift 3
    salida=$(BANCOK_SECO=1 "$L" "$@" 2>&1) || true
    if ! printf '%s' "$salida" | grep -q -- "$esperado"; then
        echo "  FALLA  $titulo: no aparece '$esperado'"; fallos=$((fallos + 1)); return
    fi
    if [ -n "$prohibido" ] && printf '%s' "$salida" | grep -q -- "$prohibido"; then
        echo "  FALLA  $titulo: aparece '$prohibido', que es de OTRO modo"
        fallos=$((fallos + 1)); return
    fi
    echo "  ok     $titulo"
}

echo
echo "El despacho de lanzar.sh"
echo
comprobar "datos    -> datos.py, NO calibrar"    "nn/datos.py"     "calibrar.py"  datos
comprobar "calibrar -> calibrar.py, NO datos"    "nn/calibrar.py"  "nn/datos.py"  calibrar
# El caso que fallo: un modo que CONSUME sus argumentos con shift.
comprobar "kernel   -> evaluar_kernel.py, NO calibrar" \
          "nn/evaluar_kernel.py" "calibrar.py" kernel "$K"
comprobar "kernel x2 -> los DOS kernels" "gauss.npy" "" kernel "$K" "$K"

# Un modo desconocido NO puede caer en una accion por defecto: es exactamente asi como
# se corre lo que nadie pidio.
if BANCOK_SECO=1 "$L" modo-que-no-existe >/dev/null 2>&1; then
    echo "  FALLA  modo desconocido: salio con 0 en vez de negarse"; fallos=$((fallos + 1))
else
    echo "  ok     modo desconocido -> se niega"
fi

echo
if [ "$fallos" -gt 0 ]; then
    echo "✗ $fallos fallo(s) en el despacho del lanzador"
    exit 1
fi
echo "El lanzador despacha cada modo a lo suyo."
