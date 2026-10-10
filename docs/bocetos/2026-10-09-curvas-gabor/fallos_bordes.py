#!/usr/bin/env python3
"""Los fallos del compositor de dígitos con el detector sobre BORDES (`digitos_bordes.py`, semilla 0), en una imagen
(pedido del dueño, 2026-10-10). Cada dígito de val que falla, ordenado por dígito real y luego por lo que leyó, con lo
que vio el detector encima: borde recto (verde), borde curvo (naranja) y una flecha hacia el centro de cada curva.

    python fallos_bordes.py   → imagenes/15-fallos-bordes.png · resultados-fallos-bordes.json  (~30 s)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import ListedColormap

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import bordes as B                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

COLS = 16


def main() -> int:
    torch.set_num_threads(2)
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    B.configurar()
    X = DB.caract(img)["bordes"]
    pred = DB.entrenar(X[tr], y[tr], 0)(X[va])
    acc = float((pred == y[va]).mean())
    f = np.flatnonzero(pred != y[va])
    orden = f[np.lexsort((pred[f], y[va][f]))]                     # por real, luego por leído
    n = len(orden); filas = int(np.ceil(n / COLS))
    print(f"semilla 0: acierto {acc:.4f} · {n} fallos")

    C._estilo()
    fig, axs = plt.subplots(filas, COLS, figsize=(COLS * 0.95, filas * 1.12 + 0.6))
    real_ant = None
    for k, ax in enumerate(axs.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= n:
            ax.axis("off"); continue
        j = orden[k]; i = va[j]
        b = B.bordes(img[i]); c = C.campo(b); kappa, mask, emin = C.giro(c)
        vs = C.veredicto(c, kappa, mask, emin); curvos, _ = B.trozos(kappa, vs, c["theta"])
        med = mask & np.isfinite(kappa)
        ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=ListedColormap(["#e3edf8"]), vmin=0, vmax=1)   # relleno: azul pálido
        ax.imshow(np.where(b & ~med, 1.0, np.nan), cmap=ListedColormap(["#7d7c78"]), vmin=0, vmax=1)    # borde sin giro: gris
        ax.imshow(np.where(med & (np.abs(kappa) < C.KAPPA_MIN), 1.0, np.nan), cmap="Greens", vmin=0, vmax=1.4)
        ax.imshow(np.where(med & (np.abs(kappa) >= C.KAPPA_MIN), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=10)
        for t in curvos:
            (cx, cy), dd = t["centroide"], np.deg2rad(t["direccion"])
            ax.annotate("", xy=(cx - 0.5 + 4 * np.cos(dd), cy - 0.5 + 4 * np.sin(dd)), xytext=(cx - 0.5, cy - 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=DB.AZU, lw=1.1, mutation_scale=7))
        nuevo = y[i] != real_ant; real_ant = y[i]
        ax.set_title(f"{y[i]}→{pred[j]}", fontsize=10, pad=2, color=DB.NAR if not nuevo else C.T1,
                     fontweight="bold" if nuevo else "normal")
        for s in ax.spines.values():
            s.set_color(C.T1 if nuevo else "#d8d7d3"); s.set_linewidth(1.6 if nuevo else 0.6)
    # resumen: fallos por dígito real y las confusiones más frecuentes
    por_real = {int(r): int((y[va][f] == r).sum()) for r in range(10)}
    conf = {}
    for j in f:
        kk = f"{y[va][j]}→{pred[j]}"; conf[kk] = conf.get(kk, 0) + 1
    top = dict(sorted(conf.items(), key=lambda z: -z[1])[:8])
    fig.suptitle(f"15 · Los {n} dígitos de val (1617) que FALLA el compositor con el detector sobre BORDES (semilla 0, "
                 f"acierto {acc:.3f}) · «real→leído», ordenados por real\n"
                 "azul pálido = el dígito · gris = borde sin giro medible · verde = borde recto · naranja = borde curvo · "
                 "flecha = hacia el centro de cada curva · recuadro negro = primer fallo de cada dígito real\n"
                 "fallos por real: " + " · ".join(f"{r}: {v}" for r, v in por_real.items()) +
                 "   |   más frecuentes: " + " · ".join(f"{k} ({v})" for k, v in top.items()),
                 fontsize=10, color=C.T1, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.3)
    fig.savefig(DB.IMG / "15-fallos-bordes.png", dpi=100); plt.close(fig)
    print("→ 15-fallos-bordes.png")
    (AQUI / "resultados-fallos-bordes.json").write_text(json.dumps(
        {"semilla": 0, "acierto_val": round(acc, 4), "fallos": n, "fallos_por_real": por_real, "confusiones": conf,
         "indices_val_fallados": [int(va[j]) for j in orden]}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
