"""Las ÚNICAS puertas a los repos hermanos, y sólo para el experimento que las pida.

Son dos y hacen falta las dos: `foveal-vision` (el código) y `foveal-vision-data`
(el dato de entrada, que este repo NO puede copiar porque es público y aquél es
privado — ver `.gitignore`) y `image-text-sample-generator` (el generador de
párrafos, que es de donde sale el dato de este repo).

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

AYUDA_DATOS = (
    "Este experimento lee el dato de entrada de `foveal-vision-data` y no lo encuentro.\n"
    "  → clónalo al lado:  git clone https://github.com/stalinbeltran/foveal-vision-data "
    "~/src/foveal-vision-data\n"
    "  → o dime dónde está: EXPCNN_DATOS=/ruta/a/foveal-vision-data\n"
    "⚠ El dato NO se copia a este repo: es público y aquél es privado."
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


def ruta_datos() -> Path | None:
    """Dónde está `foveal-vision-data`, o None.

    ⚠ Respeta `FV_DATA_ROOT` a propósito, que es la variable con la que
    `foveal-vision/src/fv/settings.py:27` resuelve ESE MISMO repo. Inventarse
    aquí una variable propia y sola dejaría dos mandos para un solo hecho, que
    pueden discrepar sin que nada falle (R15: una colisión se anuncia).

        EXPCNN_DATOS  >  FV_DATA_ROOT  >  hermano ../foveal-vision-data  >  None

    ⚠ Y NO cae al repo de código como hace `fv.settings.data_root()`. Allí ese
    respaldo es correcto —el que no ha clonado nada sigue funcionando—; aquí
    sería un directorio sin `windows.npz`, o sea fallar a mitad (R2).
    """
    for var in ("EXPCNN_DATOS", "FV_DATA_ROOT"):
        v = os.environ.get(var)
        if v:
            p = Path(v).expanduser().resolve()
            return p if p.is_dir() else None
    hermano = raiz().parent / "foveal-vision-data"
    return hermano if hermano.is_dir() else None


def exigir_datos() -> Path:
    """La raíz del repo de datos, o se niega AHORA diciendo qué falta."""
    p = ruta_datos()
    if p is None:
        raise RuntimeError(AYUDA_DATOS)
    return p


# --- los datasets PUBLICADOS -------------------------------------------------
#
# Por orden del dueño (2026-09-07): «Guarda los datasets en el repo de data, de
# modo que sean siempre los mismos, por consistencia». Un dataset que cada
# experimento re-deriva es el mismo *si nada cambia*, y "nada cambia" no es una
# propiedad que se pueda comprobar hacia el futuro: basta que el generador
# cambie una fuente. Publicado, es el mismo porque es EL MISMO FICHERO.
#
# Vive aquí y no en cada experimento porque es exactamente la clase de cosa que
# `expcnn` existe para tener en UN sitio: si dos experimentos deducen la ruta
# por su cuenta, «el mismo dataset» deja de estar garantizado por nada.
#
# ⚠ Va en `experimentos-cnn/` del repo de datos y NO en `window-datasets/`, que
# es de `foveal-vision` y lo resuelve su `fv.settings.window_datasets_root()`:
# meter ahí un dataset de otra forma sería una colisión silenciosa (R15).
SUBDIR_DATASETS = "experimentos-cnn"

AYUDA_DATASET = (
    "El dataset '{n}' no está publicado en el repo de datos.\n"
    "  → genéralo y publícalo:  python nn/datos.py --imagenes 300 --publicar\n"
    "  → o dime dónde está el repo de datos: EXPCNN_DATOS=/ruta/a/foveal-vision-data\n"
    "⚠ No se re-deriva al vuelo a propósito: el dato de entrada tiene que ser EL "
    "MISMO fichero entre experimentos, no uno equivalente."
)


def ruta_dataset(nombre: str) -> Path | None:
    """Dónde está un dataset publicado, o None si no está."""
    raiz_datos = ruta_datos()
    if raiz_datos is None:
        return None
    p = raiz_datos / SUBDIR_DATASETS / nombre
    return p if (p / "manifiesto.json").is_file() else None


def exigir_dataset(nombre: str) -> Path:
    """El dataset publicado, o se NIEGA ahora diciendo cómo se publica.

    Se niega antes de empezar y no a mitad (R2): entrenar sobre un dataset
    re-derivado al vuelo daría números que no se pueden comparar con los de
    nadie, y no fallaría por ningún lado."""
    p = ruta_dataset(nombre)
    if p is None:
        raise RuntimeError(AYUDA_DATASET.format(n=nombre))
    return p


AYUDA_GENERADOR = (
    "Este experimento genera su dataset con `image-text-sample-generator` y no lo "
    "encuentro.\n"
    "  → clónalo al lado:  git clone "
    "https://github.com/stalinbeltran/image-text-sample-generator ~/src/image-text-sample-generator\n"
    "  → o dime dónde está: EXPCNN_GENERADOR=/ruta/al/generador\n"
    "⚠ Y necesita su venv con Playwright y un Chromium: "
    "`uv venv && uv pip install -r requirements.txt && python -m playwright install chromium`"
)


def ruta_generador() -> Path | None:
    """Dónde está el generador de párrafos, o None.

        EXPCNN_GENERADOR  >  hermano ../image-text-sample-generator  >  None
    """
    declarado = os.environ.get("EXPCNN_GENERADOR")
    if declarado:
        p = Path(declarado).expanduser().resolve()
        return p if (p / "app" / "core" / "renderer.py").is_file() else None
    hermano = raiz().parent / "image-text-sample-generator"
    return hermano if (hermano / "app" / "core" / "renderer.py").is_file() else None


def exigir_generador() -> Path:
    """La raíz del generador, o se niega AHORA diciendo qué falta."""
    p = ruta_generador()
    if p is None:
        raise RuntimeError(AYUDA_GENERADOR)
    return p
