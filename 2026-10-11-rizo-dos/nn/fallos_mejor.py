#!/usr/bin/env python3
"""rizo-2 — los dígitos NO reconocidos por el mejor compositor hasta hoy (MLP 256 · rectas + gris + rizo, entrenado con
AUMENTO ±1–3 px; `resistencia.py` método B), en val y sin mover. Pedido por el dueño el 2026-10-12. Usa el caché de
candidatos de resistencia.py; semilla 0 (y la cuenta de las 3 semillas en el JSON).

    python nn/fallos_mejor.py   → resultados/19-fallos-mejor.png · 19-fallos-mejor.json
"""
from __future__ import annotations

import json
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
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import resistencia as R                                             # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def main() -> int:
    torch.set_num_threads(2); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    cache = pickle.loads(R.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    X = np.concatenate([R.feats(R.mover(img[tr], *m), 8, cache[("tr", m)]) for m in aum]); Y = np.concatenate([y[tr]] * len(aum))
    Cva = cache[("va", "x", 0)]; Xva = R.feats(img[va], 8, Cva)
    preds = [R.mlp_logits(X, Y, sem)(Xva).argmax(1) for sem in D.SEMILLAS]
    p0 = preds[0]; f = np.flatnonzero(p0 != y[va])
    veces = np.sum([p != y[va] for p in preds], 0)
    out = {"acierto_val_s0": round(float((p0 == y[va]).mean()), 4), "acierto_val_3_semillas": round(float(np.mean([(p == y[va]).mean() for p in preds])), 4),
           "fallos_s0": int(len(f)), "fallan_en_las_3_semillas": int((veces == 3).sum()), "fallan_en_alguna": int((veces > 0).sum()),
           "fallos": [{"idx": int(va[j]), "real": int(y[va][j]), "predicho": int(p0[j]), "semillas_que_fallan": int(veces[j])} for j in f]}
    conf = {}
    for j in f:
        k = f"{y[va][j]}→{p0[j]}"; conf[k] = conf.get(k, 0) + 1
    out["confusiones"] = dict(sorted(conf.items(), key=lambda z: -z[1]))
    (AQUI / "resultados" / "19-fallos-mejor.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "fallos"}, indent=1, ensure_ascii=False))
    orden = f[np.lexsort((p0[f], y[va][f]))]; cols = 8; filas = int(np.ceil(len(orden) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.6, filas * 1.85 + 0.8)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, j in zip(axs.flat, orden):
        i = va[j]; cs = DV.filtrar(Cva[j], *DV.VARIANTES["rizo (h0, 8px)"])
        ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
        for q, c in enumerate(cs):
            ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=8 if q == 0 else 5, mew=1.6 if q == 0 else 1)
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
        ax.set_title(f"{y[va][j]}→{p0[j]} · idx {i} · {veces[j]}/3 sem.", color=EV.NAR if veces[j] == 3 else "#b45309", fontsize=7.5, pad=2)
    fig.suptitle(f"rizo-2 · 19 · los {len(f)} dígitos de val (1617) NO reconocidos por el mejor compositor (MLP 256 · rectas + gris + rizo, con aumento; "
                 f"semilla 0, acierto {out['acierto_val_s0']:.4f}) · «real→predicho» · n/3 = en cuántas semillas falla · cruz azul = el rizo", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(AQUI / "resultados" / "19-fallos-mejor.png", dpi=100)
    print("→ resultados/19-fallos-mejor.png · 19-fallos-mejor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
