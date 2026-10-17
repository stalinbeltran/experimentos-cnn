#!/usr/bin/env python3
"""rizo-2 — VISTA PREVIA de las REJILLAS del vértice libre: una rejilla por DIRECCIÓN, y nada más (el dueño, 2026-10-17: «cada
rejilla debe corresponder a solo una dirección de las 4… No hay ninguna dirección NE, sino que hasta cierto rango (ej. hasta 45
grados) un vértice NE será considerado N o E»).

La figura 101 rotulaba cada CELDA con letras de brújula (NO, N, NE…) para decir su POSICIÓN, y eso se leía como si hubiera
direcciones diagonales. Aquí las celdas no llevan letras: la posición la da dónde está la celda, y la letra es SÓLO de la
rejilla (su dirección). Una dirección = un sector de 90° del rumbo de la punta: N [315°, 45°), E [45°, 135°), S [135°, 225°),
O [225°, 315°).

Mismo detector que § 102 (`vertice_libre_vista.vertices_libres`, sobre la cresta del DELGADO). Nada se entrena.

    ../.venv/bin/python nn/vertice_rejillas_vista.py   → resultados/103-vertice-rejillas.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import vertice_libre_vista as VL                                          # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Wedge                                      # noqa: E402

ORDEN = ["N", "E", "S", "O"]                                             # el orden de la brújula, en el sentido del reloj
CENTRO = {"N": 0, "E": 90, "S": 180, "O": 270}


def leyenda(a):
    """Los 4 sectores de 90°: hacia dónde APUNTA la punta decide la rejilla."""
    for k in ORDEN:
        c = CENTRO[k]                                                     # matplotlib mide desde el E y en contra del reloj
        a.add_patch(Wedge((0, 0), 1, 90 - c - 45, 90 - c + 45, color=VL.COLOR[k], alpha=0.25))
        t = np.radians(90 - c); a.text(0.62 * np.cos(t), 0.62 * np.sin(t), k, ha="center", va="center", fontsize=16, weight="bold", color=VL.COLOR[k])
    for g in (45, 135, 225, 315):
        t = np.radians(90 - g); a.plot([0, np.cos(t)], [0, np.sin(t)], color="#555", lw=1, ls="--")
        a.text(1.13 * np.cos(t), 1.13 * np.sin(t), f"{g}°", ha="center", va="center", fontsize=8, color="#555")
    a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); a.set_aspect("equal"); a.axis("off")
    a.set_title("la punta apunta a… → su rejilla\n(no hay NE: 30° es N, 60° es E)", fontsize=9)


def rejilla(a, g, k, cual):
    a.imshow(g, cmap="Reds" if cual == 0 else "Blues", vmin=0, vmax=1)
    for t in (0.5, 1.5):
        a.axhline(t, color="#bbb", lw=0.6); a.axvline(t, color="#bbb", lw=0.6)
    for rr in range(3):
        for cc in range(3):
            v = g[rr, cc]
            if v > 0:
                a.text(cc, rr, "1" if cual == 0 else f"{v:.2f}\n{v * 180:.0f}°", ha="center", va="center", fontsize=8,
                       color="white" if v > 0.45 else "black")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor(VL.COLOR[k]); sp.set_linewidth(2)


def main() -> int:
    d = dict(np.load(exigir_dataset(VL.DATASET) / "datos.npz"))
    y = d["etiquetas"].astype(int); va = np.flatnonzero(d["particion"] == "val")
    M = np.load(VL.DELGADOS)["M"].astype(np.float32)                    # la entrada por defecto: el DELGADO
    idx = [390, 740, 762, 678, 596]                                      # cinco de los ejemplos de § 102
    # y un vértice DIAGONAL de verdad (rumbo a < 8° de una frontera), para ver que cae en UNA sola rejilla
    for i in va:
        vs = VL.vertices_libres(M[i])
        if any(min(abs((v["rumbo"] - g + 180) % 360 - 180) for g in (45, 135, 225, 315)) < 8 for v in vs) and int(i) not in idx:
            idx.append(int(i)); break
    filas = len(idx)
    fig = plt.figure(figsize=(16, 2.25 * filas + 3.2))
    gs = fig.add_gridspec(filas + 1, 9, width_ratios=[1.6] + [1] * 8, height_ratios=[1.35] + [1] * filas, hspace=0.6, wspace=0.22)
    leyenda(fig.add_subplot(gs[0, 0]))
    for col, (k, cual) in enumerate([(k, 0) for k in ORDEN] + [(k, 1) for k in ORDEN]):
        a = fig.add_subplot(gs[0, 1 + col]); a.axis("off")
        a.text(0.5, 0.15, k, ha="center", va="center", fontsize=26, weight="bold", color=VL.COLOR[k])
        a.text(0.5, -0.05, "¿hay y dónde?" if cual == 0 else "amplitud", ha="center", va="center", fontsize=9)
    for r, i in enumerate(idx, start=1):
        vs = VL.vertices_libres(M[i]); R = VL.rejillas(vs)
        a = fig.add_subplot(gs[r, 0]); a.imshow(M[i], cmap="Greys", vmin=0, vmax=1, alpha=0.45, extent=(0, 32, 32, 0))
        for t in CC.tramos(M[i])[0] + CC.tramos(M[i])[1]:
            a.plot(t["pts"][:, 0], t["pts"][:, 1], "-", color="#333", lw=1.1)
        for b in (32 / 3, 64 / 3):
            a.axhline(b, color="#aaa", lw=0.6, ls="--"); a.axvline(b, color="#aaa", lw=0.6, ls="--")
        for v in vs:
            t = np.radians(v["rumbo"]); dx, dy = 4.5 * np.sin(t), -4.5 * np.cos(t)
            a.plot(v["cx"], v["cy"], "o", ms=7, mfc="none", mec=VL.COLOR[v["dir"]], mew=2)
            a.annotate("", (v["cx"] + dx, v["cy"] + dy), (v["cx"], v["cy"]), arrowprops=dict(arrowstyle="->", color=VL.COLOR[v["dir"]], lw=1.8))
            a.text(v["cx"] + 1.5, v["cy"] + 3.5, f"{v['dir']}  abre {v['ang']:.0f}°\napunta {v['rumbo']:.0f}°", color=VL.COLOR[v["dir"]],
                   fontsize=7, weight="bold")
        a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"val #{i}, un {y[i]} · {len(vs)} vértice(s)", fontsize=9)
        for col, (k, cual) in enumerate([(k, 0) for k in ORDEN] + [(k, 1) for k in ORDEN]):
            a = fig.add_subplot(gs[r, 1 + col]); rejilla(a, R[k][cual].reshape(3, 3), k, cual)
            if cual == 0:
                a.set_xlabel(f"cuenta {R[k][2]}", fontsize=7.5, labelpad=1)
    fig.suptitle("rizo-2 · 103 · VISTA PREVIA: UNA rejilla por DIRECCIÓN (N, E, S, O) · cada celda es sólo un SITIO del dígito (tercio de "
                 "arriba/medio/abajo × izquierda/centro/derecha), sin letras\nvértice libre de § 102 sobre la cresta del DELGADO · "
                 "presencia en UNA celda + amplitud (0 = 0°, 1 = 180°) + cuenta = 4 × 19 = 76 números · nada entrenado", fontsize=10.5, y=0.995)
    out = AQUI / "resultados" / "103-vertice-rejillas.png"
    fig.savefig(out, dpi=100, bbox_inches="tight"); print(out, idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
