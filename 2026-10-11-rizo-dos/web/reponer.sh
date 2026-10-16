#!/bin/sh
# Repone el REVISOR de rizo-2 en un dev recién hecho, de una vez: venv (torch CPU), el delgado, la unidad + ufw.
# Lo llama el `post` de types/dev.json del lanzador (2026-10-16, el dueño: «prepara todo para que no tengamos que esperar»);
# antes eran 4 comandos del README que había que teclear. Idempotente: lo que ya está no se repite.
# Los pesos, las recetas y la caché de features vienen del ALMACÉN (foveal-vision-data/experimentos-cnn-resultados/rizo-2/revisor),
# así que si `almacen conectar` ya corrió, la app arranca con los compositores «listo» sin entrenar.
set -eu
AQUI=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$AQUI/../.." && pwd); cd "$REPO"
echo "rizo2-revisor · reponer en $REPO"
if [ ! -x .venv/bin/python ]; then
    echo "  venv: creando (torch CPU, ~2 min)…"; python3 -m venv .venv
    .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    .venv/bin/pip install -q numpy scipy scikit-image pillow matplotlib -e .
else
    echo "  venv: ya está"
fi
.venv/bin/python -c "import torch, numpy, scipy, skimage, PIL" || { echo "  ✗ el venv no sirve: bórralo (rm -rf .venv) y repite"; exit 1; }
if [ ! -f 2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz ]; then
    echo "  delgado: exportando…"; (cd 2026-10-15-delgado-cnn && ../.venv/bin/python nn/exportar_delgados.py)
else
    echo "  delgado: ya está"
fi
if [ -d ../foveal-vision-data/experimentos-cnn-resultados/rizo-2/revisor ]; then
    echo "  almacén: $(ls ../foveal-vision-data/experimentos-cnn-resultados/rizo-2/revisor/*.pt 2>/dev/null | wc -l) compositor(es) con pesos, $(ls ../foveal-vision-data/experimentos-cnn-resultados/rizo-2/revisor/cache/*.npz 2>/dev/null | wc -l) caché(s)"
else
    echo "  ⚠ almacén: sin revisor/ en foveal-vision-data (¿corrió «almacen conectar»?): se entrenará todo (~2 h)"
fi
.venv/bin/python 2026-10-11-rizo-dos/web/servidor.py instalar
