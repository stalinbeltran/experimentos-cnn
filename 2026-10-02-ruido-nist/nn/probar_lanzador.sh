#!/usr/bin/env sh
# El despacho de nn/lanzar.sh, con el modo seco: cada modo produce su orden y uno desconocido se niega.
AQUI=$(cd "$(dirname "$0")" && pwd); fallos=0
caso() { if [ "$2" = "$3" ]; then echo "  [   ok] $1"; else echo "  [FALLA] $1: esperado $3, salio $2"; fallos=$((fallos+1)); fi; }
s=$(SECO=1 sh "$AQUI/lanzar.sh" fase1 2>&1); caso "fase1 (seco) son 33 corridas" "$(echo "$s" | grep -o '(33 corridas)')" "(33 corridas)"
caso "fase1 (seco) lleva limpio-s3 y oblicua@0.6-r2-s1" "$(echo "$s" | grep -c 'limpio-s3 .*oblicua@0.6-r2-s1')" 1
s=$(SECO=1 sh "$AQUI/lanzar.sh" fase2 horizontal curva 2>&1); caso "fase2 horizontal curva (seco) son 30 corridas" "$(echo "$s" | grep -o '(30 corridas)')" "(30 corridas)"
caso "fase2 (seco) lleva horizontal@1-s2" "$(echo "$s" | grep -c 'horizontal@1-s2')" 1
SECO=1 sh "$AQUI/lanzar.sh" fase2 inventado >/dev/null 2>&1; caso "fase2 con un tipo desconocido se niega (exit 2)" "$?" 2
SECO=1 sh "$AQUI/lanzar.sh" fase2 >/dev/null 2>&1; caso "fase2 sin tipos se niega (exit 2)" "$?" 2
s=$(SECO=1 sh "$AQUI/lanzar.sh" corridas gaussiano@0.3-s2 2>&1); caso "corridas gaussiano@0.3-s2 (seco) es 1 corrida" "$(echo "$s" | grep -o '(1 corridas)')" "(1 corridas)"
SECO=1 sh "$AQUI/lanzar.sh" corridas gaussiano@0.33-s2 >/dev/null 2>&1; caso "corridas con un nivel fuera de tabla se niega (exit 2)" "$?" 2
s=$(SECO=1 sh "$AQUI/lanzar.sh" fase3 2>&1); caso "fase3 (seco) son 15 corridas en línea" "$(echo "$s" | grep -o '(15 corridas)')" "(15 corridas)"
caso "fase3 (seco) lleva recorte@0.6-linea-s3" "$(echo "$s" | grep -c 'recorte@0.6-linea-s3')" 1
SECO=1 sh "$AQUI/lanzar.sh" fase3 extra >/dev/null 2>&1; caso "fase3 con argumentos se niega (exit 2)" "$?" 2
s=$(SECO=1 sh "$AQUI/lanzar.sh" fase4 2>&1); caso "fase4 (seco) son 18 corridas (3 trazos con ayuda en línea × 2 variantes × 3)" "$(echo "$s" | grep -o "(18 corridas)")" "(18 corridas)"
caso "fase4 (seco) lleva oblicua@1-doble-linea-s2" "$(echo "$s" | grep -c 'oblicua@1-doble-linea-s2')" 1
SECO=1 sh "$AQUI/lanzar.sh" fase4 extra >/dev/null 2>&1; caso "fase4 con argumentos se niega (exit 2)" "$?" 2
SECO=1 sh "$AQUI/lanzar.sh" fase1 extra >/dev/null 2>&1; caso "fase1 con argumentos se niega (exit 2)" "$?" 2
sh "$AQUI/lanzar.sh" inventado >/dev/null 2>&1; caso "modo desconocido se niega (exit 2)" "$?" 2
sh "$AQUI/lanzar.sh" >/dev/null 2>&1; caso "sin modo se niega (exit 2)" "$?" 2
echo "$([ $fallos -eq 0 ] && echo 'el despacho esta en orden' || echo "✗ $fallos fallo(s)")"; exit $fallos
