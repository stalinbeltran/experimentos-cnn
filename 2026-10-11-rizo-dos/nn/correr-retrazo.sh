#!/bin/sh
# rizo-2 § 125 — AFINAR el adelgazador de 64 px con el feedback del RE-TRAZO y medirlo (el dueño, 2026-10-19: «procede»).
# Tres pasos en orden: afinar (delgado-cnn/nn/afinar_retrazo.py, ~35 min) → exportar M a 64 px de la base y del afinado →
# medir en val (nn/adelgazador_retrazo_vista.py) → aviso.
#   sh nn/correr-retrazo.sh lanzar     → como UNIDAD de systemd `rizo2-retrazo` (padre PID 1: sobrevive al fin del turno)
#   sh nn/correr-retrazo.sh seco       → imprime unidad y órdenes, sin lanzar nada
#   sh nn/correr-retrazo.sh estado     → unidad, Result, NRestarts, último paso del afinado y qué salidas existen
set -u
MODO="${1:-}"                                     # el modo se guarda al entrar y no se vuelve a leer de $1
cd "$(dirname "$0")/.." || exit 1
RIZO="$(pwd)"; DELG="$RIZO/../2026-10-15-delgado-cnn"; PY="$RIZO/../.venv/bin/python"
COORD_HOME="${COORD_HOME:-$HOME/src/telegram-coordinator}"
UNIDAD=rizo2-retrazo; T4=uci-optdigits-orig-32px-trazo4-r20261018; NOMBRE=k3x3-g0.7x-64-rt
O1="$PY nn/afinar_retrazo.py --desde k3x3-g0.7x-64 --nombre $NOMBRE --lambda 1 --inventado 0 --pasos 1500"
O2="$PY nn/exportar_delgados.py --pesos k3x3-g0.7x-64 --dataset $T4 --ampliar 2"
O3="$PY nn/exportar_delgados.py --pesos $NOMBRE --dataset $T4 --ampliar 2"
O4="$PY nn/adelgazador_retrazo_vista.py"

cadena() {                                         # lo que corre DENTRO de la unidad; un paso que falla avisa y corta, sin salir ≠ 0
    falla() { echo "FALLÓ: $1"; node "$COORD_HOME/scripts/notify.mjs" "❌ rizo2-retrazo falló en: $1 (log /tmp/$UNIDAD.log)" || true; }
    cd "$DELG" || { falla "cd $DELG"; return; }
    echo "→ $O1"; $O1 || { falla "$O1"; return; }
    if [ ! -f "resultados/delgados-k3x3-g0.7x-64-$T4-x2.npz" ]; then echo "→ $O2"; $O2 || { falla "$O2"; return; }; fi
    echo "→ $O3"; $O3 || { falla "$O3"; return; }
    cd "$RIZO" || { falla "cd $RIZO"; return; }
    echo "→ $O4"; $O4 || { falla "$O4"; return; }
    node "$COORD_HOME/scripts/notify.mjs" "✅ rizo2-retrazo terminó: resultados/125-adelgazador-retrazo.{png,json} (el criterio, en el json)" || true
}

case "$MODO" in
    seco)
        echo "unidad: $UNIDAD"; echo "  en $DELG:"; echo "    $O1"; echo "    $O2   (sólo si falta)"; echo "    $O3"
        echo "  en $RIZO:"; echo "    $O4"; exit 0 ;;
    estado)
        systemctl show "$UNIDAD" -p ActiveState -p Result -p NRestarts -p ExecMainStatus 2>/dev/null
        tail -n 1 "$DELG/nn/pesos/$NOMBRE/metrics.jsonl" 2>/dev/null || echo "(sin metrics del afinado todavía)"
        ls -la "$DELG/nn/pesos/$NOMBRE/best.pt" "$DELG/resultados/delgados-$NOMBRE-$T4-x2.npz" resultados/125-adelgazador-retrazo.json 2>&1 | sed 's/^/  /'
        exit 0 ;;
    lanzar)
        if systemctl is-active --quiet "$UNIDAD"; then echo "❌ $UNIDAD ya está corriendo: no lanzo otra (pisarían los mismos pesos)"; exit 1; fi
        sh "$0" seco
        exec "$COORD_HOME/scripts/desacoplar-persistente.sh" "$UNIDAD" sh "$RIZO/nn/correr-retrazo.sh" _cadena ;;
    _cadena)
        cadena; exit 0 ;;                          # sale con 0 siempre: es una unidad con Restart=on-failure
    *)
        echo "uso: sh nn/correr-retrazo.sh lanzar|seco|estado   (modo desconocido: «$MODO»; no lanzo nada)"; exit 2 ;;
esac
