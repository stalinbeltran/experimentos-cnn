#!/usr/bin/env sh
# El despacho de nn/lanzar.sh, con el modo seco: cada modo produce su orden y uno desconocido se niega.
AQUI=$(cd "$(dirname "$0")" && pwd); fallos=0
caso() { if [ "$2" = "$3" ]; then echo "  [   ok] $1"; else echo "  [FALLA] $1: esperado $3, salio $2"; fallos=$((fallos+1)); fi; }
s=$(SECO=1 sh "$AQUI/lanzar.sh" todo 2>&1); caso "todo (seco) imprime la orden con los 30 brazos" "$(echo "$s" | grep -c 'w8-de4-s5')" 1
s=$(SECO=1 sh "$AQUI/lanzar.sh" todo w6-s1 2>&1); caso "todo w6-s1 (seco) solo ese brazo" "$(echo "$s" | grep -o 'w[0-9]-s[0-9]\|w8-de4-s[0-9]' | wc -l)" 1
sh "$AQUI/lanzar.sh" inventado >/dev/null 2>&1; caso "modo desconocido se niega (exit 2)" "$?" 2
sh "$AQUI/lanzar.sh" >/dev/null 2>&1; caso "sin modo se niega (exit 2)" "$?" 2
echo "$([ $fallos -eq 0 ] && echo 'el despacho esta en orden' || echo "✗ $fallos fallo(s)")"; exit $fallos
