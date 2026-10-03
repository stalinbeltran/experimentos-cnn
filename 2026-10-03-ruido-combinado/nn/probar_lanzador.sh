#!/usr/bin/env sh
AQUI=$(cd "$(dirname "$0")" && pwd); fallos=0
caso() { if [ "$2" = "$3" ]; then echo "  [   ok] $1"; else echo "  [FALLA] $1: esperado $3, salio $2"; fallos=$((fallos+1)); fi; }
s=$(SECO=1 sh "$AQUI/lanzar.sh" todo 2>&1); caso "todo (seco) son 25 corridas (5 escenarios × 5 semillas)" "$(echo "$s" | grep -o "(25 corridas)")" "(25 corridas)"
caso "todo (seco) lleva la mezcla con semilla 3" "$(echo "$s" | grep -c 'recorte@0.6~gaussiano@0.2-linea-s3')" 1
SECO=1 sh "$AQUI/lanzar.sh" todo extra >/dev/null 2>&1; caso "todo con argumentos se niega (exit 2)" "$?" 2
s=$(SECO=1 sh "$AQUI/lanzar.sh" corridas "recorte@0.6+gaussiano@0.2-linea-s1" 2>&1); caso "corridas con la secuencial (seco) es 1 corrida" "$(echo "$s" | grep -o '(1 corridas)')" "(1 corridas)"
SECO=1 sh "$AQUI/lanzar.sh" corridas "recorte@0.6+recorte@0.6-s1" >/dev/null 2>&1; caso "corridas con una combinación del mismo tipo se niega (exit 2)" "$?" 2
sh "$AQUI/lanzar.sh" inventado >/dev/null 2>&1; caso "modo desconocido se niega (exit 2)" "$?" 2
sh "$AQUI/lanzar.sh" >/dev/null 2>&1; caso "sin modo se niega (exit 2)" "$?" 2
echo "$([ $fallos -eq 0 ] && echo 'el despacho esta en orden' || echo "✗ $fallos fallo(s)")"; exit $fallos
