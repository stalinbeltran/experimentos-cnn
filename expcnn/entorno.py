"""La ÚNICA puerta a `foveal-vision`, y sólo para el experimento que la pida.

Por qué existe (R4 + R2 de las reglas de diseño): en
`foveal-vision/experimentos/` hay **72 `sys.path.insert`** repartidos por 41
ficheros (medido 2026-09-06 con `grep -rn sys.path.insert --include='*.py'`),
cada uno deduciendo del disco dónde está el repo hermano. Eso es frágil dentro
del repo y **falso** fuera de él. Aquí hay una sola indirección, declarable, y
que se NIEGA antes de empezar en vez de fallar a la época 30.

Orden, copiando la forma de `fv/settings.py` que en este proyecto ya funciona:

    EXPCNN_FV (variable de entorno, declarada)  >  hermano ../foveal-vision  >  se niega
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .registro import raiz

AYUDA = (
    "Este experimento reusa código de `foveal-vision` y no lo encuentro.\n"
    "  → clónalo al lado:  git clone https://github.com/stalinbeltran/foveal-vision "
    "~/src/foveal-vision\n"
    "  → o dime dónde está: EXPCNN_FV=/ruta/a/foveal-vision\n"
    "  → o escribe el experimento autónomo, que es la opción por defecto aquí."
)


def ruta_fv() -> Path | None:
    """Dónde está `foveal-vision`, o None. No importa nada ni toca `sys.path`."""
    declarado = os.environ.get("EXPCNN_FV")
    if declarado:
        p = Path(declarado).expanduser().resolve()
        return p if (p / "src" / "fv").is_dir() else None
    hermano = raiz().parent / "foveal-vision"
    return hermano if (hermano / "src" / "fv").is_dir() else None


def hay_fv() -> bool:
    return ruta_fv() is not None


def exigir_fv() -> Path:
    """Deja `fv` importable, o se niega AHORA diciendo qué falta y cómo se arregla.

    Se llama en la primera línea del experimento que reusa `fv`, nunca a mitad:
    una separación que falla a la época 30 no es una separación (R2)."""
    p = ruta_fv()
    if p is None:
        raise RuntimeError(AYUDA)
    src = str(p / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return p
