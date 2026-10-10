#!/usr/bin/env python3
"""Las curvas que definen al detector, ESQUELETIZADAS (pedido del dueño, 2026-10-10): lo que entra al detector en
`esqueleto.py`. Sólo la forma: el original en azul pálido y su esqueleto (skimage.skeletonize) encima, sin detector.

    21-esqueleto-1-banco.png    el banco de rect-lin: arcos (R × grosor) y rectas (largo × grosor), un ejemplar de cada
    21-esqueleto-2-galeria.png  los 32 trazos de la galería

    python curvas_esqueleto.py   (~5 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import galeria                                                      # noqa: E402
from curvas import plt                                              # noqa: E402
from esqueleto import esqueleto                                     # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

PALIDO, NAR = ListedColormap(["#d6e4f5"]), ListedColormap(["#c2410c"])


def pintar(ax, x, titulo):
    s = esqueleto(x)
    ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=PALIDO, vmin=0, vmax=1)
    ax.imshow(np.where(s > 0, 1.0, np.nan), cmap=NAR, vmin=0, vmax=1)
    ax.set_title(f"{titulo}\n{int((x > 0).sum())} → {int(s.sum())} px", fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)


def main() -> int:
    C._estilo()
    b = dict(np.load(exigir_dataset("rect-lin-banco-r20261008") / "datos.npz"))
    t, g, lg, rad, ang = b["tipo"], b["grosor"], b["largo"], b["radio"], b["angulo"]

    def uno(cond):
        ids = np.flatnonzero(cond)
        return ids[np.argmin(np.abs(((ang[ids] - 30) + 90) % 180 - 90))] if len(ids) else None   # el más cercano a 30°

    radios, gr_arco = sorted(set(rad[t == 2].tolist())), sorted(set(g[t == 2].tolist()))
    largos, gr_recta = sorted(set(lg[t == 0].tolist())), [2, 4, 8, 12, 14]
    filas = len(gr_arco) + len(largos)
    cols = max(len(radios), len(gr_recta))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.9, filas * 2.1 + 0.8))
    for i, gg in enumerate(gr_arco):
        for j in range(cols):
            ax = axs[i, j]
            if j < len(radios) and (k := uno((t == 2) & (rad == radios[j]) & (g == gg))) is not None:
                pintar(ax, b["imagenes"][k], f"arco R {radios[j]:g} · grosor {gg:g}")
            else:
                ax.axis("off")
    for i, L in enumerate(largos):
        for j in range(cols):
            ax = axs[len(gr_arco) + i, j]
            if j < len(gr_recta) and (k := uno((t == 0) & (lg == L) & (g == gr_recta[j]))) is not None:
                pintar(ax, b["imagenes"][k], f"recta {L:g} px · grosor {gr_recta[j]}")
            else:
                ax.axis("off")
    fig.suptitle("21 · Las curvas que definen al detector (banco de rect-lin), ESQUELETIZADAS\n"
                 "azul pálido = la figura · naranja = su esqueleto · «px de tinta → px de esqueleto»", fontsize=11, color=C.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(C.IMG / "21-esqueleto-1-banco.png", dpi=100); plt.close(fig)
    print("→ 21-esqueleto-1-banco.png")

    items = [(etq, x) for grupo in galeria.trazos().values() for etq, x, *_ in grupo[2]]
    fig, axs = plt.subplots(4, 8, figsize=(15, 9))
    nombres = [g[1].split(" · ")[0].split(" ")[0] for g in galeria.trazos().values() for _ in g[2]]
    for ax, (etq, x), n in zip(axs.flat, items, nombres):
        pintar(ax, x, f"{n.lower()}: {etq[:14]}")
    fig.suptitle("21 · Los 32 trazos de la galería, ESQUELETIZADOS · azul pálido = la figura · naranja = su esqueleto",
                 fontsize=11, color=C.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(C.IMG / "21-esqueleto-2-galeria.png", dpi=100); plt.close(fig)
    print("→ 21-esqueleto-2-galeria.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
