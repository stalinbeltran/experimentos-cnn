#!/usr/bin/env python3
"""rizo-2 — imágenes de CONCAVIDADES (el dueño, 2026-10-14) sobre dígitos de val, dos nociones por dígito:
  (izq.) el mapa de concavidad de los RAYOS (huecos_suaves v4, segundo mapa) en color sobre el mapa M, con sus objetos
         (+ verde, tamaño = fuerza) y los huecos cerrados (+ azul);
  (der.) los ARCOS de la cresta (≥ ARCO_MIN° de barrido) con la región que encierran sombreada: el polígono entre el
         arco y su cuerda, del color del sentido. Es la concavidad vista desde el trazo que la forma.

    python nn/concavidades_dibujo.py --entrada /ruta/delgados-k3x3.npz [--n 30] [--semilla 1]   → resultados/53-concavidades.png
"""
from __future__ import annotations

import argparse
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
import huecos_suaves as HS                                          # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

ARCO_MIN = 60.0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--semilla", type=int, default=1); a = ap.parse_args()
    HS.configurar(CENTRAR=True, UNO_POR_CELDA=True, SIN_RENDIJAS=True, PONDERAR=True)
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); va = np.flatnonzero(d["particion"] == "val"); y = d["etiquetas"]
    idx = np.sort(np.random.default_rng(a.semilla).choice(va, a.n, replace=False)); Mz = np.load(a.entrada)[a.clave].astype(np.float32)
    cols = 10; filas = int(np.ceil(a.n / cols))
    fig, axs = plt.subplots(filas, 2 * cols, figsize=(2 * cols * 1.5, filas * 1.8 + 1.0)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for k, i in enumerate(idx):
        m = Mz[i]; r, c = divmod(k, cols); a0, a1 = axs[r, 2 * c], axs[r, 2 * c + 1]
        cer, con = HS.mapas(m[None]); HS.M_ACTUAL = m
        a0.imshow(np.where(m > 0.05, m, np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest", zorder=2)
        a0.imshow(np.where(con[0] > 0.1, con[0], np.nan), cmap="YlGn", vmin=0, vmax=0.8, alpha=0.85, interpolation="nearest", zorder=1)
        oc = HS.objetos(con[0]); oh = HS.objetos(cer[0])
        for o in oc:
            a0.plot(o["cx"] - 0.5, o["cy"] - 0.5, "+", color="#0b6b2e", ms=4 + 10 * o["fuerza"], mew=2, zorder=3)
        for o in oh:
            a0.plot(o["cx"] - 0.5, o["cy"] - 0.5, "+", color="#2a78d6", ms=4 + 8 * o["fuerza"], mew=2, zorder=3)
        a0.set_xlim(-0.5, 31.5); a0.set_ylim(31.5, -0.5)
        a0.set_title(f"{y[i]} · idx {i} · rayos: {len(oc)} conc.", fontsize=7, pad=2)
        a1.imshow(np.where(m > 0.05, m, np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest", zorder=2)
        ob = CC.objetos(m); n_arc = 0
        for z in ob["arcos"]:
            if z["barrido"] < ARCO_MIN:
                continue
            n_arc += 1; pts = z["pts"] - 0.5
            a1.add_patch(Polygon(pts, closed=True, facecolor=CC.COLORES[z["s"]], edgecolor="none", alpha=0.35, zorder=1))
            a1.plot(pts[:, 0], pts[:, 1], "-", color=CC.COLORES[z["s"]], lw=1.8, zorder=3)
            a1.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], "--", color=CC.COLORES[z["s"]], lw=0.8, zorder=3)
        for z in ob["lazos"]:
            a1.plot(z["cx"] - 0.5, z["cy"] - 0.5, "+", color="#111", ms=8, mew=2, zorder=3)
        a1.set_xlim(-0.5, 31.5); a1.set_ylim(31.5, -0.5); a1.set_title(f"cresta: {n_arc} arcos ≥ {ARCO_MIN:.0f}°", fontsize=7, pad=2)
    fig.suptitle(f"rizo-2 · 53 · CONCAVIDADES en {a.n} dígitos de val (semilla {a.semilla}) · por pareja: rayos v4 (verde = mapa de concavidad; + verde objeto, + azul hueco cerrado) | "
                 "cresta (arcos ≥ 60° con su región encerrada sombreada; color = centro de curvatura a la → naranja · ← azul · ↓ verde · ↑ morado; + negro = lazo)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); out = AQUI / "resultados" / "53-concavidades.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
