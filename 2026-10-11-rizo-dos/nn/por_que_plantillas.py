#!/usr/bin/env python3
"""rizo-2 — POR QUÉ las plantillas hallan tantos candidatos (pedido por el dueño, 2026-10-11): un dígito, su mapa de
puntuación, y seis candidatos con la plantilla que mejor casa en cada uno, descompuesta en sus tres partes (trazo,
hueco, vano) y en las tres fracciones que forman la puntuación.

    python nn/por_que_plantillas.py [idx]   → resultados/10-por-que-plantillas.png
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
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def partes(m):
    t, h, v = PL._plantilla(m["radio"], m["grosor"], 0.5 + 0.5 * m["cierre"], m["abertura"] or 0)
    return t, h, v


def main() -> int:
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); x = (d["imagenes"][i] > 0)
    best, idx = PL.mapa(x); cands = PL.rizos(x, 0.8)
    e = next(z for z in EV.etiquetas() if z["i"] == i)
    cerca = sorted(cands, key=lambda c: EV._dist(c["centro"], e["centro"]))
    muestra = [cerca[0]] + [c for c in cands if c not in cerca[:1]][:5]
    fig = plt.figure(figsize=(17, 9.4)); gs = fig.add_gridspec(2, 6, height_ratios=(1.15, 1), hspace=0.45)
    ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(x, cmap="Greys", vmin=0, vmax=1.6)
    for c in cands:
        ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color=EV.NAR, ms=7, mew=1.3)
    ax.plot(e["centro"][0] - 0.5, e["centro"][1] - 0.5, "o", mfc="none", mec=EV.VERDE, ms=12, mew=2)
    ax.set_title(f"idx {i}: {len(cands)} candidatos con S ≥ 0,8 (naranja) · verde = el rizo etiquetado", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2:4]); im = ax.imshow(np.where(x, np.nan, best), cmap="Oranges", vmin=0.4, vmax=1); ax.imshow(np.where(x, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#9a9a9a"]))
    ax.set_title("la puntuación de la MEJOR plantilla en cada píxel de fondo\n(≥ 0,8 en todo lo que esté a un lado de un trazo curvo)", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[0, 4:6])
    ax.hist([m["cierre"] for m in PL.META], bins=4, color="#b9b8b4"); ax.set_xlabel("cierre de la plantilla"); ax.set_ylabel("plantillas en el banco")
    cob = [0.5 + 0.5 * m["cierre"] for m in [PL.META[idx[int(c["centro"][1]), int(c["centro"][0])]] for c in cands]]
    ax.set_title(f"el banco: {len(PL.META)} plantillas; {len(cands)} candidatos, {sum(z < 0.6 for z in cob)} casan\ncon una de MEDIA vuelta (cob 0,55), el resto con 0,7", fontsize=8.5)
    for k, c in enumerate(muestra):
        ax = fig.add_subplot(gs[1, k]); cx, cy = int(c["centro"][0]), int(c["centro"][1]); m = PL.META[idx[cy, cx]]
        t, h, v = partes(m); K = PL.K // 2
        canvas = np.full((32, 32), np.nan); tr = np.zeros((32, 32), bool); hu = np.zeros((32, 32), bool); va = np.zeros((32, 32), bool)
        for arr, dst in ((t, tr), (h, hu), (v, va)):
            ys, xs = np.nonzero(arr); yy, xx = ys - K + cy, xs - K + cx; ok = (yy >= 0) & (yy < 32) & (xx >= 0) & (xx < 32); dst[yy[ok], xx[ok]] = True
        ax.imshow(np.where(x, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#c9c8c4"]))
        ax.imshow(np.where(tr, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([EV.AZU]), alpha=0.45)
        ax.imshow(np.where(hu, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([EV.VERDE]), alpha=0.45)
        ax.imshow(np.where(va, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap([EV.NAR]), alpha=0.45)
        ax.plot(cx, cy, "+", color="k", ms=9, mew=1.5)
        f_t, f_h, f_v = x[tr].mean(), (x[hu].mean() if hu.any() else 0), (x[va].mean() if va.any() else 0)
        es = "EL RIZO" if k == 0 else f"candidato {k}"
        ax.set_title(f"{es} · S {c['puntuacion']:.2f}\nplantilla R {m['radio']:g} · g {m['grosor']} · cob {0.5 + 0.5 * m['cierre']:.2f}\n"
                     f"trazo lleno {f_t:.2f} − hueco lleno {f_h:.2f} − ½·vano {f_v:.2f}", fontsize=7.5)
        ax.set_xlim(max(cx - 10, -0.5), min(cx + 10, 31.5)); ax.set_ylim(min(cy + 10, 31.5), max(cy - 10, -0.5)); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("rizo-2 · 10 · POR QUÉ las plantillas hallan tantos candidatos: la plantilla es TRAZO (azul, debe tener tinta) + HUECO (verde, debe estar vacío) + VANO (naranja, mejor vacío).\n"
                 "Con cobertura 0,55 el «trazo» es media vuelta y el «hueco» un píxel: cualquier curva vista desde su lado cóncavo la llena, y el hueco cae en el fondo de al lado. No exige que el trazo RODEE al punto.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / "10-por-que-plantillas.png", dpi=95)
    print("→ resultados/10-por-que-plantillas.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
