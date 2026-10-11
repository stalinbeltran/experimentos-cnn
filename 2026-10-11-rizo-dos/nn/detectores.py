#!/usr/bin/env python3
"""rizo-2 — los detectores YA PROBADOS en el repo, solos y combinados, para ver cuál contribuye (pedido del dueño,
2026-10-11). Cuatro familias de características, cada una el detector tal como quedó publicado en el boceto
curvas-gabor, sin tocar su calibración:

  rectas   — 4 Gabor par (0/45/90/135) integrados en línea, 2 escalas, celdas 8×8 (`digitos.caract_rectas`; 0,955 val)
  delgado  — el detector de curvas DELGADO sobre el dígito tal cual: recto · κ→←↓↑ · golpe (`digitos_bordes.caract`
             con `delgadas` configurado; 0,929 val)
  gris     — el mismo detector de curvas sobre el BORDE de Gabor impar gris λ_b 10 · K 9 re-calibrado (el mejor del
             boceto, 0,960 val)
  rizo     — la salida «rizo» del compuesto de este experimento, compacta (mejor candidato en 2×2 + cuenta, 5 números)

Solos, cada uno + rizo, las tres familias viejas, y todo; con tres compositores (lineal · MLP 256 · bosque 500). Mismo
protocolo: 180 train / 1617 val / 3823 a ciegas, 3 semillas.

    python nn/detectores.py   → resultados/17-detectores.json · 17-detectores.png  (~8 min)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
BOCETO = AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(BOCETO))
import compositores as CO                                           # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
import delgadas as DL                                               # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

CAL_GRIS = {"tau": 0.1, "sigma_e": 0.5, "coh_min": 0.25, "lam": 3.0}   # la de fallos_gris.py (resultados-gris-recalibrado.json, λ_b 10 · K 9)
COMPOSITORES = {k: CO.COMPOSITORES[k] for k in ("lineal (referencia)", "MLP 256", "bosque 500")}


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    d = dict(np.load(exigir_dataset(DV.EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    fam = {}
    fam["rectas"] = D.caract_rectas(img)
    DL.configurar(**json.loads((BOCETO / "resultados-delgadas.json").read_text())["elegida"])
    fam["delgado"] = DB.caract(img, borde=lambda z: z.astype(np.float32))["bordes"]
    GR.configurar(**CAL_GRIS)
    fam["gris"] = DB.caract(img, borde=BG.BordeGabor(10.0, k=9, binarizar=False))["bordes"]
    fam["rizo"] = CO.compacta(pickle.loads(DV.CACHE.read_bytes()), "rizo (h0, 8px)")
    print({k: v.shape[1] for k, v in fam.items()}, f"({time.time() - t0:.0f} s)", flush=True)
    conjuntos = {"rectas": ["rectas"], "delgado": ["delgado"], "gris": ["gris"], "rizo": ["rizo"],
                 "rectas + rizo": ["rectas", "rizo"], "delgado + rizo": ["delgado", "rizo"], "gris + rizo": ["gris", "rizo"],
                 "rectas + gris": ["rectas", "gris"], "rectas + delgado": ["rectas", "delgado"], "gris + delgado": ["gris", "delgado"],
                 "rectas + gris + rizo": ["rectas", "gris", "rizo"], "rectas + gris + delgado": ["rectas", "gris", "delgado"],
                 "todo": ["rectas", "gris", "delgado", "rizo"]}
    res = {}
    for cn, ent in COMPOSITORES.items():
        for xn, partes in conjuntos.items():
            X = np.concatenate([fam[p] for p in partes], 1); v, c = [], []
            for sem in D.SEMILLAS:
                pr = ent(X[tr], y[tr], sem); v.append(float((pr(X[va]) == y[va]).mean())); c.append(float((pr(X[ex]) == y[ex]).mean()))
            res[f"{cn} · {xn}"] = {"compositor": cn, "caracteristicas": xn, "n": int(X.shape[1]), "val": round(float(np.mean(v)), 4),
                                   "val_rango": [round(min(v), 4), round(max(v), 4)], "ciega": round(float(np.mean(c)), 4)}
            print(f"{cn:20s} · {xn:26s} ({X.shape[1]:4d}) val {np.mean(v):.4f} [{min(v):.4f}–{max(v):.4f}] · ciega {np.mean(c):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "17-detectores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True); cols = {"lineal (referencia)": "#52514e", "MLP 256": "#2a78d6", "bosque 500": "#c2410c"}
    for ax, medida in zip(axs, ("val", "ciega")):
        for j, (cn, col) in enumerate(cols.items()):
            ax.barh(np.arange(len(conjuntos)) + (j - 1) * 0.27, [res[f"{cn} · {xn}"][medida] for xn in conjuntos], 0.27, color=col, label=cn)
        ax.set_yticks(range(len(conjuntos))); ax.set_yticklabels(list(conjuntos), fontsize=8); ax.invert_yaxis(); ax.set_xlim(0.85, 1.0)
        ax.set_title(f"acierto {'en val (1617)' if medida == 'val' else 'a ciegas (3823)'}", loc="left", fontsize=9.5); ax.grid(axis="x", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("rizo-2 · 17 · los detectores ya probados (rectas · delgado · gris) y el rizo, solos y combinados, con tres compositores", fontsize=10)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "17-detectores.png", dpi=100)
    print(f"→ resultados/17-detectores.json · 17-detectores.png · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
