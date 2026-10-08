#!/bin/sh
# Lanza la rejilla de rect-lin como UNIDAD de systemd (sobrevive al fin de la sesión y al restart del coordinador).
#   sh nn/lanzar.sh            lanza (se niega si ya corre)
#   sh nn/lanzar.sh --estado   unidad, Result, NRestarts y cuántas líneas lleva rejilla.jsonl
#   SECO=1 sh nn/lanzar.sh     imprime la orden sin lanzar
MODO="${1:-lanzar}"
UNIDAD=rect-lin-rejilla
AQUI="$(cd "$(dirname "$0")/.." && pwd)"
COORD="${COORD_HOME:-$HOME/src/telegram-coordinator}"
ORDEN="cd '$AQUI' && ../.venv/bin/python -u nn/entrenar_local.py; node '$COORD/scripts/notify.mjs' \"rect-lin: rejilla terminada (\$(wc -l < resultados/rejilla.jsonl)/432). Resultados en experimentos-cnn/2026-10-08-rectas-kernel-lineal/resultados/rejilla.jsonl\" || true"
case "$MODO" in
  --estado)
    systemctl show "$UNIDAD" -p ActiveState -p Result -p NRestarts 2>/dev/null
    echo "rejilla.jsonl: $(wc -l < "$AQUI/resultados/rejilla.jsonl" 2>/dev/null || echo 0) / 432"
    tail -n 2 "/tmp/$UNIDAD.log" 2>/dev/null; exit 0 ;;
  lanzar) ;;
  *) echo "✗ modo desconocido: $MODO (lanzar | --estado)"; exit 2 ;;
esac
echo "unidad: $UNIDAD"; echo "orden:  $ORDEN"
[ -n "$SECO" ] && { echo "(SECO: no lanzo)"; exit 0; }
if systemctl is-active -q "$UNIDAD"; then echo "✗ $UNIDAD ya está corriendo: no lanzo dos veces."; exit 1; fi
exec "$COORD/scripts/desacoplar-persistente.sh" "$UNIDAD" sh -c "$ORDEN"
