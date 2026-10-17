#!/usr/bin/env python3
"""rizo-2 § 114 — los mismos 1l de § 113, con la tinta suavizada en GRIS (sin rebinarizar) en varios grados (el dueño, 2026-10-17: «no
necesito el binarizado… muéstrame las mismas imágenes que mostraste binarizadas, pero en gris. No las evalúes»). Sólo imágenes: el
adelgazador es el de siempre (k3x3, entrenado con tinta binaria), y los delgados los produce `delgado-cnn`
(`nn/exportar_delgados.py --suavizar σ --gris`), leídos aquí como dato.

    ../.venv/bin/python nn/suavizado_gris_vista.py   → resultados/114-suavizado-gris.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import cresta_curvas as CC                                               # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Polygon                                    # noqa: E402

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"
VARIANTES = [("hoy (sin suavizar)", "delgados-k3x3.npz"), ("σ 0,7 gris", "delgados-k3x3-s0.7g.npz"), ("σ 1 gris", "delgados-k3x3-s1g.npz"),
             ("σ 1,5 gris", "delgados-k3x3-s1.5g.npz")]
COL = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]
MUESTRA = [736, 2849, 4661, 5538, 3517, 4053, 1668, 5266]                 # los mismos de § 113


def main():
    nv, nm = len(VARIANTES), len(MUESTRA)
    fig, axs = plt.subplots(nv, nm, figsize=(2.0 * nm + 0.6, 2.25 * nv + 1.1))
    for f, (nombre, fich) in enumerate(VARIANTES):
        z = np.load(DELG / fich); M = z["M"]; T = z["tinta"]
        for k, i in enumerate(MUESTRA):
            ax = axs[f, k]; m = M[i].astype(np.float32)
            ax.imshow(T[i].astype(np.float32), cmap="Greys", vmin=0, vmax=2.0, extent=(0, 32, 32, 0))
            for e in CC.objetos(m)["entrantes"]:
                c = CC.cono(e); ax.add_patch(Polygon(np.asarray(e["pts"]), closed=True, fc=COL[c], ec=COL[c], alpha=0.55, lw=1))
            for t in CC.tramos(m)[0]:
                p = np.asarray(t["pts"]); ax.plot(p[:, 0], p[:, 1], "-", color="#111", lw=1.3)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"#{i}", fontsize=9)
            if k == 0:
                ax.set_ylabel(nombre, fontsize=10, fontweight="bold")
    fig.suptitle("Los mismos 1l, con la tinta suavizada en GRIS (sin rebinarizar) · adelgazador de siempre\n"
                 "fondo = tinta que ve la red · negro = cresta del delgado · color = entrante: naranja der · azul izq · verde abajo · morado arriba",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); out = AQUI / "resultados" / "114-suavizado-gris.png"; fig.savefig(out, dpi=90); print(out)


if __name__ == "__main__":
    main()
