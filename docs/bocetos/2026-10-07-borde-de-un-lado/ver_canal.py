#!/usr/bin/env python3
"""Por qué el pre-proceso de bordes de UN lado sólo quita el grosor si cada detector mira UN canal.

Un arco de grosor 2, 6 y 12 px, y lo que ve cada canal (→ ↓ ← ↑, los de ver_proceso.py). Dentro de UN canal, la forma
es la misma a cualquier grosor (sólo se desplaza). Entre canales, la distancia entre el lado → y el lado ← es el grosor:
un detector que mira los 4 canales a la vez (como el `signo` de feat-bor) vuelve a ver el grosor.

    /tmp/vizenv/bin/python ver_canal.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ver_proceso import detectar

AQUI = Path(__file__).resolve().parent
yy, xx = np.mgrid[:32, :32]


def arco(grosor):
    r = np.hypot(yy - 16, xx - 18)
    ang = np.degrees(np.arctan2(yy - 16, xx - 18))
    return ((np.abs(r - 9) <= grosor / 2) & (np.abs(ang) > 50)).astype(float)   # una «C» abierta a la derecha


fig, ax = plt.subplots(3, 6, figsize=(13, 6.6))
for f, g in enumerate((2, 6, 12)):
    img = arco(g)
    tinta = 1 - .25 * img
    rgb = np.dstack([tinta] * 3)
    celdas = [("tinta", img, "gray_r")] + [(f"canal {n}", detectar(img, d), "magma")
                                           for n, d in (("→", 0), ("↓", 90), ("←", 180), ("↑", 270))]
    for c, (t, m, cm) in enumerate(celdas):
        ax[f, c].imshow(m, cmap=cm); ax[f, c].axis("off")
        if f == 0: ax[f, c].set_title(t)
    union = np.clip(sum(detectar(img, d) for d in (0, 90, 180, 270)), 0, 1)
    ax[f, 5].imshow(union, cmap="magma"); ax[f, 5].axis("off")
    if f == 0: ax[f, 5].set_title("los 4 juntos")
    ax[f, 0].text(-3, 16, f"{g} px", ha="right", va="center")
fig.suptitle("Dentro de UN canal la forma no depende del grosor; juntando canales, vuelven los dos bordes", fontsize=11)
fig.tight_layout(); fig.savefig(AQUI / "imagenes" / "5-un-canal-contra-todos.png", dpi=100)
