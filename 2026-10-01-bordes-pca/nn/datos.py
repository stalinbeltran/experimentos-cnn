#!/usr/bin/env python3
"""Los PARCHES de `bor-pca`: k x k centrados en puntos de BORDE, del dataset publicado.

    python nn/datos.py            cuantos parches salen por k y particion

El dato es `parrafos1000-pagina1024-r4-r20261001` (lo produce `bor-p4`): paginas de 256 x
256 con la SUMA de cada bloque 4x4, y la caja de tinta de cada parrafo.

QUE ES UN PARCHE
    Por cada parrafo de `train` y cada uno de sus 4 bordes, PUNTOS_POR_BORDE puntos
    sorteados UNIFORMES a lo largo del borde (semilla fija). El parche es el k x k de la
    pagina centrado en el pixel donde cae ese punto. El borde cae ENTRE dos pixeles, asi
    que el centro se redondea al pixel del lado de la TINTA (izq/sup: el primero de tinta;
    der/inf: el ultimo): los cuatro bordes quedan igual de centrados.

    No se filtra ningun parrafo: un parche de k <= 19 centrado en un borde no sale del
    papel ni ve otro parrafo, porque `bor-p4` garantiza ventanas limpias de 79 x 79
    alrededor de cualquier punto de borde. Se comprueba al construir.

LA ENTRADA: TINTA, x = 1 - suma / 4080 (papel 0, tinta 1), como en `bor-k` y `bor-ae`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))

from expcnn import exigir_dataset  # noqa: E402

DATASET = "parrafos1000-pagina1024-r4-r20261001"
REDUCCION = 4
BLANCO = 255 * REDUCCION ** 2
PUNTOS_POR_BORDE = 8
SEMILLA_PUNTOS = 0
BORDES = ("izq", "der", "sup", "inf")
PARTES = {"train": 0, "val": 1, "eval": 2}


def parches(k: int, parte: str = "train"):
    """(P, k*k) float32 en TINTA, y (P,) el borde de cada uno (0..3)."""
    p = exigir_dataset(DATASET)
    d = np.load(p / "paginas.npz")
    c = np.load(p / "cajas.npz")
    paginas, particion, cajas, derivadas = d["paginas"], d["particion"], c["cajas"], c["derivadas"]
    rng = np.random.default_rng(SEMILLA_PUNTOS + 1000 * PARTES[parte])
    h = (k - 1) // 2
    xs, bs = [], []
    for j in np.flatnonzero(particion[cajas[:, 0]] == PARTES[parte]):
        pag = cajas[j, 0]
        assert h <= (derivadas[j, 2] - 1) / 2, f"parrafo {j}: ventana limpia {derivadas[j, 2]} < {k}"
        izq, der, sup, inf = cajas[j, 1:] / REDUCCION
        for t, borde in enumerate(BORDES):
            for _ in range(PUNTOS_POR_BORDE):
                if borde in ("izq", "der"):
                    cx = int(np.floor(izq)) if borde == "izq" else int(np.ceil(der)) - 1
                    cy = int(np.floor(rng.uniform(sup, inf)))
                else:
                    cx = int(np.floor(rng.uniform(izq, der)))
                    cy = int(np.floor(sup)) if borde == "sup" else int(np.ceil(inf)) - 1
                sub = paginas[pag, cy - h:cy + h + 1, cx - h:cx + h + 1]
                assert sub.shape == (k, k), (j, borde, cx, cy)
                xs.append(1.0 - sub.astype(np.float32).ravel() / BLANCO)
                bs.append(t)
    return np.stack(xs).astype(np.float32), np.array(bs, dtype=np.int8)


if __name__ == "__main__":
    for k in (3, 19):
        for parte in PARTES:
            x, b = parches(k, parte)
            print(f"k={k:>2} {parte:>5}: {len(x)} parches ({np.bincount(b).tolist()} por borde) · "
                  f"tinta media {x.mean():.3f}")
