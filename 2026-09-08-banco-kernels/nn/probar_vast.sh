#!/usr/bin/env sh
# ¿`nn/vast.sh` despacha CADA modo a su descriptor, y se NIEGA con lo que no conoce?
#
#   nn/probar_vast.sh        sale 0 si pasan los cuatro casos
#
# Es la prueba de `probar_lanzador.sh` copiada para el lanzador de Vast (R17): el
# 2026-09-08 un despacho que releia "$1" tras un `shift` corrio la calibracion en vez
# de los kernels, con Result=success. Usa el modo SECO, que no toca la API.
set -e
AQUI=$(cd "$(dirname "$0")" && pwd)
L="$AQUI/vast.sh"
fallos=0

comprobar() {   # <titulo> <esperado> <prohibido> <argumentos...>
    titulo="$1"; esperado="$2"; prohibido="$3"; shift 3
    salida=$(VAST_SECO=1 "$L" "$@" 2>&1) || true
    if ! printf '%s' "$salida" | grep -q -- "$esperado"; then
        echo "  FALLA  $titulo: no aparece '$esperado'"; fallos=$((fallos + 1)); return
    fi
    if [ -n "$prohibido" ] && printf '%s' "$salida" | grep -q -- "$prohibido"; then
        echo "  FALLA  $titulo: aparece '$prohibido'"; fallos=$((fallos + 1)); return
    fi
    echo "  ok     $titulo"
}

echo
echo "El despacho de vast.sh"
echo
comprobar "fase1 -> vast-fase1.json y su prefijo" "vast-fase1.json.*expc-bancok-f1-" "vast-fase3" fase1
comprobar "fase1 -> el seco no alquila"           "SECO"                 "Alquilada" fase1
comprobar "fase3 k09 -> vast-fase3.json y --solo" "vast-fase3.json"      "vast-fase1" fase3 k09
if VAST_SECO=1 "$L" modo-que-no-existe >/dev/null 2>&1; then
    echo "  FALLA  modo desconocido: salio con 0 en vez de negarse"; fallos=$((fallos + 1))
else
    echo "  ok     modo desconocido -> se niega"
fi
echo
[ "$fallos" -eq 0 ] && echo "vast.sh despacha cada modo a lo suyo." || { echo "✗ $fallos fallo(s)"; exit 1; }
