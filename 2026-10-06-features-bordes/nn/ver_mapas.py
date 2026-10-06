#!/usr/bin/env python3
"""Qué VEN los detectores de borde en dígitos finos y gruesos: el dígito, la entrada de borde (contorno · con signo) y el mapa
8×8 de cinco detectores, cada uno marcado si su máximo pasa el umbral. Los 3 dígitos más finos y los 3 más gruesos (de clases distintas) de una
muestra fija (grosor = tinta ÷ esqueleto).

    python nn/ver_mapas.py      → resultados/mapas-digitos.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset, por_id       # noqa: E402
import bordes as B                              # noqa: E402
import modelo                                   # noqa: E402

FAMS = ("arco-E", "arco-W", "recta-V", "recta-H", "lazo")


def grosor(x):
    sys.path.insert(0, str(por_id("feat-fallos").carpeta / "nn"))
    import normalizar as N                      # noqa: PLC0415
    e = N.esqueleto((x > 0.5).astype(np.uint8))
    return (x > 0.5).reshape(len(x), -1).sum(1) / np.maximum(1, e.reshape(len(e), -1).sum(1))


def main() -> int:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt     # noqa: E401,E702
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    rng = np.random.default_rng(7); idx = rng.choice(len(d["imagenes"]), 400, replace=False)
    x = d["imagenes"][idx].astype(np.float32); g = grosor(x)
    orden = np.argsort(g); y = d["etiquetas"][idx]
    gruesos, vistas = [], set()                 # los 3 más gruesos, de clases DISTINTAS (si no, salen tres 1)
    for k in orden[::-1]:
        if y[k] not in vistas:
            gruesos.append(k); vistas.add(y[k])
        if len(gruesos) == 3:
            break
    sel = list(orden[:3]) + gruesos[::-1]
    xs = x[sel][:, None]
    cols = ["dígito"]
    filas = []
    for rep in ("contorno", "signo"):
        e = B.bordes(xs, rep).astype(np.float32)
        maps, umb = [], []
        for f in FAMS:
            red, est = modelo.cargar(AQUI / f"pesos-{rep}" / f / "best.pt")
            with torch.no_grad():
                maps.append(torch.sigmoid(red(torch.from_numpy(e)))[:, 0].numpy()); umb.append(est["umbral"])
        filas.append((rep, e.max(1) if rep == "contorno" else e, maps, umb))
    cols += ["contorno"] + [f"{f}\n(contorno)" for f in FAMS] + ["con signo"] + [f"{f}\n(signo)" for f in FAMS]
    SUP = "#fcfcfb"
    fig, axs = plt.subplots(6, len(cols), figsize=(1.15 * len(cols), 7.6), dpi=100, facecolor=SUP)
    sig_col = np.array([[0.16, 0.47, 0.84], [0.92, 0.41, 0.2], [0.11, 0.69, 0.48], [0.55, 0.36, 0.84]])
    for i in range(6):
        c = 0
        def pon(img, **kw):
            nonlocal c
            ax = axs[i, c]; ax.imshow(img, interpolation="nearest", **kw); ax.set_xticks([]); ax.set_yticks([]); c += 1
            return ax
        pon(xs[i, 0], cmap="Greys", vmin=0, vmax=1)
        for rep, e, maps, umb in filas:
            if rep == "contorno":
                pon(e[i], cmap="Greys", vmin=0, vmax=1)
            else:
                rgb = 1 - np.clip(np.tensordot(e[i].transpose(1, 2, 0), 1 - sig_col, 1), 0, 1)
                pon(rgb)
            for m, u in zip(maps, umb):
                ax = pon(m[i], cmap="magma", vmin=0, vmax=1)
                on = m[i].max() >= u
                for s in ax.spines.values():
                    s.set_color("#d63c3c" if on else "#e6e5e1"); s.set_linewidth(2.5 if on else 1)
        axs[i, 0].set_ylabel(f"{'fino' if i < 3 else 'grueso'}\n{g[sel[i]]:.1f} px · «{d['etiquetas'][idx][sel[i]]}»",
                             rotation=0, ha="right", va="center", fontsize=8)
    for j, t in enumerate(cols):
        axs[0, j].set_title(t, fontsize=7.5)
    fig.suptitle("Detectores de BORDE sobre dígitos finos (arriba) y gruesos (abajo). Mapa 8×8; marco rojo = el detector se enciende",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.9, bottom=0.01, wspace=0.08, hspace=0.12)
    out = AQUI.parent / "resultados" / "mapas-digitos.png"; fig.savefig(out, facecolor=SUP); print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
