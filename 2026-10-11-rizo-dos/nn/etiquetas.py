#!/usr/bin/env python3
"""rizo-2 — dibuja las 48 etiquetas a mano sobre la rejilla, para que el dueño las corrija.

    python nn/etiquetas.py   → resultados/0-etiquetas.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parent))
from expcnn import exigir_dataset                                   # noqa: E402

DATASET = "uci-optdigits-orig-32px-r20261005"
VERDE = "#1b7f3b"


def etiquetas() -> list[dict]:
    return json.loads((AQUI / "instrucciones" / "etiquetas-2.json").read_text())["etiquetas"]


def arco_etiqueta(ax, e, color=VERDE, lw=1.6):
    cx, cy, R = e["centro"][0] - 0.5, e["centro"][1] - 0.5, e["radio"]
    span = (0.5 + 0.5 * e["cierre"]) * 360; mitad = (e["abertura"] + 180) if e.get("abertura") is not None else 0
    t = np.deg2rad(np.linspace(mitad - span / 2, mitad + span / 2, 90))
    ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), color=color, lw=lw)
    ax.plot(cx, cy, "x", color=color, ms=8, mew=1.8)


def main() -> int:
    d = dict(np.load(exigir_dataset(DATASET) / "datos.npz")); E = etiquetas()
    fig, axs = plt.subplots(6, 8, figsize=(16, 12.6))
    for ax, e in zip(axs.flat, E):
        x = d["imagenes"][e["i"]] > 0
        ax.imshow(x, cmap="Greys", vmin=0, vmax=1.6, extent=(-0.5, 31.5, 31.5, -0.5))
        ax.set_xticks(np.arange(-0.5, 32, 4)); ax.set_yticks(np.arange(-0.5, 32, 4)); ax.set_xticklabels(range(0, 33, 4), fontsize=6)
        ax.set_yticklabels(range(0, 33, 4), fontsize=6); ax.grid(color="#6090d0", lw=0.5, alpha=0.6)
        arco_etiqueta(ax, e)
        ax.set_title(f"idx {e['i']} · ({e['centro'][0]:g}, {e['centro'][1]:g}) R {e['radio']:g}" + (" · DUDOSO" if e["dudoso"] else ""),
                     fontsize=7.5, color="#c2410c" if e["dudoso"] else "#0b0b0b")
    fig.suptitle("rizo-2 · 0 · las 48 ETIQUETAS a mano (Claude, 2026-10-11): cruz verde = centro del rizo superior, arco verde = radio, "
                 "cierre y abertura · ±1 px · corríjelas si no estás de acuerdo", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(AQUI / "resultados" / "0-etiquetas.png", dpi=85)
    print("→ resultados/0-etiquetas.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
