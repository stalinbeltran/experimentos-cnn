#!/usr/bin/env python3
"""rizo-2 — figura 65: ENTRANTES en 4 CONOS (±60° de la horizontal = derecha/izquierda; el resto = abajo/arriba) con su
rejilla 3×3 de PRESENCIA por cono, tal como la recibe el compositor (--entrantes60 presencia). 8 treses y 8 cincos de val
(6 aciertos + 2 fallos de la corrida cresta-entrantes, § 58).

    ../.venv/bin/python nn/entrantes_conos_dibujo.py   → resultados/65-entrantes-conos60.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Polygon                              # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

NOM = ["→ der ±60°", "← izq ±60°", "↓ abajo", "↑ arriba"]; COL = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]


def main() -> int:
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); y, part = d["etiquetas"], d["particion"]
    M = np.load(AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz")["M"].astype(np.float32)
    fallan = {x["idx"] for x in json.loads((AQUI / "resultados" / "40-entrada-cresta-entrantes.json").read_text())["fallos"]}
    va = np.flatnonzero(part == "val")

    def muestra(cls, n_ok, n_mal):
        ok = [i for i in va if y[i] == cls and i not in fallan]; mal = [i for i in va if y[i] == cls and i in fallan]
        return [ok[k] for k in np.linspace(0, len(ok) - 1, n_ok).astype(int)] + [mal[k] for k in np.linspace(0, len(mal) - 1, n_mal).astype(int)]
    ids = muestra(3, 6, 2) + muestra(5, 6, 2); b = CC.bloques()["entrantes60"]
    cols = 2; filas = len(ids) // cols
    fig = plt.figure(figsize=(cols * 7.2, filas * 2.6)); outer = fig.add_gridspec(filas, cols, hspace=0.35, wspace=0.1, top=0.94, bottom=0.02, left=0.02, right=0.98)
    for k, i in enumerate(ids):
        fi, co = divmod(k, cols); e = CC.objetos(M[i])["entrantes"]; P = CC.caract(M[i:i + 1])[0][b[0]:b[1]]
        inner = outer[fi, co].subgridspec(1, 5, width_ratios=[1.7, 1, 1, 1, 1], wspace=0.08)
        ax = fig.add_subplot(inner[0]); CC.pintar(ax, M[i])
        for kk in (1, 2):
            ax.axhline(kk * 32 / 3 - 0.5, color="#aaa", lw=0.7, ls="--"); ax.axvline(kk * 32 / 3 - 0.5, color="#aaa", lw=0.7, ls="--")
        for q in e:
            c = CC.cono(q); col = COL[c]
            ax.add_patch(Polygon(q["pts"] - 0.5, closed=True, fc=col, ec=col, alpha=0.35, lw=1.2)); ax.plot(q["cx"] - 0.5, q["cy"] - 0.5, "+", color=col, ms=8, mew=1.8)
            ang = np.degrees(np.arctan2(q["dir"][1], q["dir"][0])); ax.text(q["cx"] + 1, q["cy"], f"{NOM[c][0]} {ang:+.0f}°", fontsize=6.5, color=col)
        est = "✓" if i not in fallan else "✗"; ax.set_title(f"{y[i]} {est} · idx {i}", fontsize=8.5, color="#1b7f3b" if est == "✓" else "#b91c1c", pad=3)
        for o in range(4):
            G = P[o * 9:(o + 1) * 9].reshape(3, 3); ax = fig.add_subplot(inner[1 + o]); ax.imshow(G, cmap="Greys", vmin=0, vmax=1.6)
            for rr in range(3):
                for cc in range(3):
                    ax.text(cc, rr, f"{G[rr, cc]:.1f}" if G[rr, cc] >= 0.05 else "·", ha="center", va="center", fontsize=8, color="k")
            ax.set_xticks([]); ax.set_yticks([]); [s.set_color(COL[o]) for s in ax.spines.values()]; [s.set_linewidth(1.5) for s in ax.spines.values()]
            ax.set_title(NOM[o], fontsize=7.5, color=COL[o], pad=2)
    fig.suptitle("rizo-2 · 65 · ENTRANTES en 4 CONOS (el dueño, 2026-10-15): derecha/izquierda = abertura a ±60° de la horizontal; abajo/arriba = el resto.\n"
                 "PRESENCIA en 3×3 por cono (gaussiana de altura 1, σ = 2 px, máximo por celda) = 36 números: lo que recibe el compositor con --entrantes60 presencia.\n"
                 "Arriba, 3 (6 aciertos + 2 fallos de la corrida § 58); abajo, 5 (ídem).", fontsize=9)
    fig.savefig(AQUI / "resultados" / "65-entrantes-conos60.png", dpi=110); print("→ resultados/65-entrantes-conos60.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
