#!/usr/bin/env python3
"""rizo-2 — figura 69: el detector de CRUCES (dos rectas que se cortan por el interior; el dueño, 2026-10-15) sobre dígitos de
val: para 4, 7, 8, 9 y 1, los primeros con cruz y los primeros sin cruz. Cada brazo en su color (dos rectas = dos colores),
centro con ×, y el ángulo entre las rectas y el brazo más corto (el parámetro a customizar).

    ../.venv/bin/python nn/cruces_dibujo.py   → resultados/69-cruces.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

C1, C2 = "#c2410c", "#2a78d6"


def main() -> int:
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); y, part = d["etiquetas"], d["particion"]
    M = np.load(AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz")["M"].astype(np.float32)
    va = np.flatnonzero(part == "val"); clases = (4, 7, 8, 9, 1); por_clase = 8
    fig, axs = plt.subplots(len(clases), por_clase, figsize=(por_clase * 2.0, len(clases) * 2.3))
    for fila, cls in enumerate(clases):
        con, sin = [], []
        for i in va:
            if y[i] != cls:
                continue
            ob = CC.objetos(M[i])
            (con if ob["cruces"] else sin).append((i, ob))
            if len(con) >= por_clase // 2 and len(sin) >= por_clase // 2:
                break
        n_con = sum(1 for i in va if y[i] == cls and CC.objetos(M[i])["cruces"]); n_cls = int((y[va] == cls).sum())
        for col, (i, ob) in enumerate((con[:por_clase // 2] + sin[:por_clase // 2])[:por_clase]):
            ax = axs[fila, col]; CC.pintar(ax, M[i])
            for z in ob["cruces"]:
                for k, p in enumerate(z["pts"]):
                    ax.plot(p[:, 0] - 0.5, p[:, 1] - 0.5, "-", color=C1 if k < 2 else C2, lw=2.2, alpha=0.9)
                ax.plot(z["cx"] - 0.5, z["cy"] - 0.5, "x", color="k", ms=11, mew=2.5)
                ax.text(z["cx"] + 1.5, z["cy"] - 1, f"{z['ang']:.0f}° · brazo mín {z['brazo_min']:.0f} px" + (" · puente" if z.get("puente") else ""), fontsize=6.5, color="k")
            ax.set_title(f"{cls} · idx {i} · {len(ob['cruces'])} cruz", fontsize=8, color="#1b7f3b" if ob["cruces"] else "#7a7a7a")
        for col in range(len((con[:por_clase // 2] + sin[:por_clase // 2])[:por_clase]), por_clase):
            axs[fila, col].axis("off")
        axs[fila, 0].text(-6, 16, f"{cls}\n{n_con}/{n_cls}\ncon cruz", fontsize=8, ha="right", va="center")
    fig.suptitle("rizo-2 · 69 · detector de CRUCES (el dueño, 2026-10-15): dos rectas que se cortan por el INTERIOR — un nodo de la cresta con dos pares de brazos\n"
                 "casi colineales (≤ 40° de la colinealidad), cada brazo ≥ 2 px (parámetro a customizar). Naranja = una recta, azul = la otra, × = el cruce.\n"
                 "Por clase: los 4 primeros de val CON cruz y los 4 primeros SIN cruz.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / "69-cruces.png", dpi=110); print("→ resultados/69-cruces.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
