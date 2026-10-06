#!/usr/bin/env sh
# Prueba EN SECO de nn/vast.sh (no toca la API ni alquila nada): el modo construye SU orden —su descriptor, su libro, su
# tope de horas, su prefijo— y un modo desconocido se niega. COPIA de la de `feat-ind32`, reducida a su único modo.
#
#   nn/probar_vast.sh        sale con 0 si todo casa
AQUI=$(cd "$(dirname "$0")" && pwd)
fallos=0

caso() {   # caso <modo> <texto que TIENE que salir en la orden>...
    modo=$1; shift
    salida=$(VAST_SECO=1 "$AQUI/vast.sh" "$modo" 2>&1)
    orden=$(printf '%s\n' "$salida" | grep '^orden:')
    for t in "$@"; do
        if printf '%s' "$orden" | grep -q -- "$t"; then echo "  [   ok] $modo: la orden lleva '$t'"
        else echo "  [FALLA] $modo: la orden NO lleva '$t'"; echo "         $orden"; fallos=$((fallos + 1)); fi
    done
}

caso detectores "/vast.json" "resultados/vast/detectores" "--horas-max 3" "--prefijo expc-fcor-"

if "$AQUI/vast.sh" lo-que-sea >/dev/null 2>&1; then echo "  [FALLA] un modo desconocido NO se negó"; fallos=$((fallos + 1))
else echo "  [   ok] un modo desconocido se niega"; fi
if "$AQUI/vast.sh" >/dev/null 2>&1; then echo "  [FALLA] sin modo NO se negó"; fallos=$((fallos + 1))
else echo "  [   ok] sin modo se niega"; fi

[ "$fallos" -eq 0 ] && echo "el lanzador construye la orden de su modo." || echo "✗ $fallos fallo(s)"
exit "$fallos"
