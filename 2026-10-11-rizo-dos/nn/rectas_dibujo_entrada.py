#!/usr/bin/env python3
"""rizo-2 — las RECTAS sobre los fallos de una corrida (el dueño, 2026-10-13: «muéstrame las rectas»). Por fallo, dos
paneles sobre el mapa suave: (izq.) las rectas-OBJETO que consume el compositor (rectas_objetos: picos de Gabor par
agrupados por colinealidad, 4 orientaciones 0/45/90/135°; segmento centrado en el objeto, largo = su largo, tono y
grosor = fuerza) y (der.) las rectas de la CRESTA (tramos de curvatura < K_MIN, gris) con sus arcos en color.

    python nn/rectas_dibujo_entrada.py --nombre suave-cresta-lazos-conc --entrada /ruta/delgados-k3x3.npz
        → resultados/49-rectas-<nombre>.png
    python nn/rectas_dibujo_entrada.py --nombre suave-cresta-todo-conc --entrada … --solo-cresta
        → resultados/49-rectas-<nombre>-cresta.png   (sólo las rectas de la cresta, por orientación: 0° naranja · 45° verde · 90° azul · 135° morado)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
import rectas_objetos as RO                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def pintar_rectas_objeto(ax, x):
    ax.imshow(np.where(x > 0.05, x, np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
    objs = RO.objetos(x)[0]
    for z in objs:
        d = RO.DIRS[z["o"]]; h = z["largo"] / 2; r = min(z["fuerza"] / RO.NORMA, 1.0)
        ax.plot([z["cx"] - 0.5 - d[0] * h, z["cx"] - 0.5 + d[0] * h], [z["cy"] - 0.5 - d[1] * h, z["cy"] - 0.5 + d[1] * h], color=RO.COLORES[z["o"]], alpha=0.35 + 0.65 * r, lw=0.8 + 2.2 * r, solid_capstyle="round")
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off")
    return objs


def pintar_rectas_cresta(ax, x):
    ax.imshow(np.where(x > 0.05, x, np.nan), cmap="Greys", vmin=0, vmax=3.0, interpolation="nearest")
    rs = CC.objetos(x)["rectas"]
    for r in rs:
        ax.plot(r["pts"][:, 0] - 0.5, r["pts"][:, 1] - 0.5, "-", color=RO.COLORES[r["o"]], lw=1.5 + 2.5 * r["fuerza"], solid_capstyle="round")
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off")
    return rs


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--nombre", required=True); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M")
    ap.add_argument("--solo-cresta", action="store_true"); a = ap.parse_args()
    res = json.loads((AQUI / "resultados" / f"40-entrada-{a.nombre}.json").read_text()); fallos = sorted(res["fallos"], key=lambda f: (f["real"], f["predicho"]))
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); Mz = np.load(a.entrada)[a.clave].astype(np.float32)
    if a.solo_cresta:
        cols = 14; filas = int(np.ceil(len(fallos) / cols)); fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.5, filas * 1.75 + 1.0)); axs = np.atleast_2d(axs)
        for ax in axs.flat:
            ax.axis("off")
        for ax, f in zip(axs.flat, fallos):
            rs = pintar_rectas_cresta(ax, Mz[f["idx"]]); ax.set_title(f"{f['real']}→{f['predicho']} · idx {f['idx']} · {len(rs)} rectas", color=EV.NAR if f["semillas_que_fallan"] == 3 else "#b45309", fontsize=7, pad=2)
        fig.suptitle(f"rizo-2 · 49 · SÓLO las RECTAS de la cresta en los {len(fallos)} fallos de val de «{a.nombre}» (val {res['val']:.4f}) · color = orientación: 0° naranja · 45° verde · 90° azul · 135° morado · grosor = longitud", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); out = AQUI / "resultados" / f"49-rectas-{a.nombre}-cresta.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)}"); return 0
    cols = 8; filas = int(np.ceil(len(fallos) / cols))
    fig, axs = plt.subplots(filas, 2 * cols, figsize=(2 * cols * 1.45, filas * 1.75 + 1.0)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for k, f in enumerate(fallos):
        i = f["idx"]; r, c = divmod(k, cols); a0, a1 = axs[r, 2 * c], axs[r, 2 * c + 1]
        objs = pintar_rectas_objeto(a0, Mz[i]); ob = CC.pintar(a1, Mz[i])
        col = EV.NAR if f["semillas_que_fallan"] == 3 else "#b45309"
        a0.set_title(f"{f['real']}→{f['predicho']} · idx {i} · {len(objs)} rectas-obj", color=col, fontsize=7, pad=2)
        a1.set_title(f"cresta: {len(ob['rectas'])} rectas · {len(ob['arcos'])} arcos", color=col, fontsize=7, pad=2)
    fig.suptitle(f"rizo-2 · 49 · las RECTAS en los {len(fallos)} fallos de val de «{a.nombre}» (val {res['val']:.4f}) · por pareja: rectas-OBJETO del compositor (Gabor par agrupado; "
                 "color = orientación 0° naranja · 45° verde · 90° azul · 135° morado; tono y grosor = fuerza) | rectas de la CRESTA (gris) con sus arcos", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); out = AQUI / "resultados" / f"49-rectas-{a.nombre}.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
