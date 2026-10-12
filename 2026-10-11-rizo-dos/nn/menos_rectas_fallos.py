#!/usr/bin/env python3
"""rizo-2 — los dígitos que la opción 3 (rectas 4×4) ARREGLA y los que ESTROPEA respecto de la referencia (rectas 8×8),
semilla 0, en val. Pedido por el dueño el 2026-10-12. Cada dígito con «real · lee 8×8 → lee 4×4» y el rizo encontrado.

    python nn/menos_rectas_fallos.py   → resultados/23b-menos-rectas-fallos.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import compositores as CO                                           # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import menos_rectas as MR                                           # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def main() -> int:
    torch.set_num_threads(2); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]; Itr = [R.mover(img[tr], *m) for m in aum]
    gris_tr = np.concatenate([R.gris(I, 8) for I in Itr]); rizo_tr = np.concatenate([CO.compacta(cache[("tr", m)], "rizo (h0, 8px)") for m in aum]); Y = np.concatenate([y[tr]] * len(aum))
    gva, rva = R.gris(img[va], 8), CO.compacta([todos[i] for i in va], "rizo (h0, 8px)")
    pred = {}
    for nombre in ("8×8 (referencia)", "4×4"):
        X = np.concatenate([np.concatenate([MR.rectas_variante(I, nombre) for I in Itr]), gris_tr, rizo_tr], 1); Xva = np.concatenate([MR.rectas_variante(img[va], nombre), gva, rva], 1)
        red, mu, sd = RA.entrenar(X, Y, 0); pred[nombre] = RA.logits(red, mu, sd, Xva).argmax(1)
    p8, p4 = pred["8×8 (referencia)"], pred["4×4"]; yv = y[va]
    arreglados = np.flatnonzero((p8 != yv) & (p4 == yv)); nuevos = np.flatnonzero((p8 == yv) & (p4 != yv)); siguen = int(((p8 != yv) & (p4 != yv)).sum())
    print(f"8×8 falla {int((p8 != yv).sum())} · 4×4 falla {int((p4 != yv).sum())} · arreglados {len(arreglados)} · nuevos {len(nuevos)} · siguen mal en los dos {siguen}")
    filas = [("ARREGLADOS por 4×4 (fallaban con 8×8)", arreglados, "#1b7f3b"), ("NUEVOS fallos con 4×4 (acertaban con 8×8)", nuevos, "#c2410c")]
    cols = max(len(arreglados), len(nuevos), 1); fig, axs = plt.subplots(2, cols, figsize=(cols * 1.55, 2 * 2.1 + 0.9)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for fila, (titulo, idxs, color) in enumerate(filas):
        for q, j in enumerate(idxs):
            ax = axs[fila, q]; i = va[j]; RA.pintar_digito(ax, img[i])
            for k, c in enumerate(DV.filtrar(todos[i], *DV.VARIANTES["rizo (h0, 8px)"])):
                ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU if k == 0 else "#9fc2ec", ms=7 if k == 0 else 4, mew=1.4)
            ax.set_title(f"{yv[j]} · {p8[j]}→{p4[j]}\nidx {i}", fontsize=7.5, color=color, pad=2)
        axs[fila, 0].text(-0.1, 1.35, f"{titulo}: {len(idxs)}", transform=axs[fila, 0].transAxes, fontsize=9, color=color, fontweight="bold", va="bottom")
    fig.suptitle("rizo-2 · 23b · opción 3 (rectas 4×4) frente a la referencia (8×8), semilla 0, val · «real · lee 8×8 → lee 4×4» · cruz = rizo", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(AQUI / "resultados" / "23b-menos-rectas-fallos.png", dpi=110); print("→ resultados/23b-menos-rectas-fallos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
