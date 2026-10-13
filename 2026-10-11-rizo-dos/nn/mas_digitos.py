#!/usr/bin/env python3
"""rizo-2 — ¿aprende la importancia de los AGUJEROS con MÁS dígitos de entrenamiento? (pedido por el dueño, 2026-10-13).
Compositor «todo objetos» (rectas-objeto ×1 + gris-objetos + agujeros, 108) entrenado con los 180 de train (referencia) y
con 180 + los 3823 «extra» (otros escritores; 4003), los dos con aumento ±1–3 px; evaluado en val (1617), que es donde
están los 9 ochos de la figura 33b. ⚠ Con extra en train ya no queda conjunto a ciegas: val es la única medida.
Las features se cachean en resultados/34-features.pkl (regenerable, fuera de git).

    python nn/mas_digitos.py   → resultados/34-mas-digitos.json · 34-mas-digitos.png  (~25 min la primera vez)
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
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import todo_objetos as TO                                           # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

CACHE = AQUI / "resultados" / "34-features.pkl"


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    cache18 = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    feats = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    if "va" not in feats:
        feats["va"] = TO.features(img[va], [todos[i] for i in va]); CACHE.write_bytes(pickle.dumps(feats))
    for m in aum:
        if ("tr", m) not in feats:
            feats[("tr", m)] = TO.features(R.mover(img[tr], *m), cache18[("tr", m)]); CACHE.write_bytes(pickle.dumps(feats))
        if ("ex", m) not in feats:
            Im = R.mover(img[ex], *m); C = R.desplazar_candidatos([todos[i] for i in ex], *m)   # los candidatos cacheados, desplazados (como resistencia.py: exacto si no se pierde tinta)
            feats[("ex", m)] = TO.features(Im, C); CACHE.write_bytes(pickle.dumps(feats)); print(f"extra {m} ({time.time() - t0:.0f} s)", flush=True)
    Xva = feats["va"]; f33 = json.loads((AQUI / "resultados" / "33-todo-objetos.json").read_text())
    ochos = [f["idx"] for f in f33["fallos"] if f["real"] == 8 and f["predicho"] == 9]; J = [int(np.flatnonzero(va == i)[0]) for i in ochos]; f33set = {f["idx"] for f in f33["fallos"]}
    conjuntos = {"180 de train": [("tr", m) for m in aum], "180 + 3823 extra (4003)": [("tr", m) for m in aum] + [("ex", m) for m in aum]}
    out = {"ochos": ochos}
    for nombre, claves in conjuntos.items():
        X = np.concatenate([feats[k] for k in claves]); Y = np.concatenate([y[tr] if k[0] == "tr" else y[ex] for k in claves])
        r = {"val": [], "cuota": [], "L": [], "fallos": []}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); wt = (W ** 2).sum(); L = RA.logits(red, mu, sd, Xva); pv = L.argmax(1)
            r["val"].append(float((pv == y[va]).mean())); r["cuota"].append({g: float((W[:, a:b] ** 2).sum() / wt) for g, (a, b) in TO.GRUPOS.items()}); r["L"].append(L[J]); r["fallos"].append({int(va[k]) for k in np.flatnonzero(pv != y[va])})
        L = np.mean(r["L"], 0)
        out[nombre] = {"n_train": int(len(X)), "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)],
                       "cuota_peso": {g: round(float(np.mean([c[g] for c in r["cuota"]])), 3) for g in TO.GRUPOS}, "fallos_val": [len(f) for f in r["fallos"]],
                       "de_los_59_arreglados": [len(f33set - f) for f in r["fallos"]], "fallos_nuevos": [len(f - f33set) for f in r["fallos"]],
                       "ochos_pred": L.argmax(1).tolist(), "ochos_logit8_menos_9": np.round(L[:, 8] - L[:, 9], 2).tolist(), "ochos_bien": int((L.argmax(1) == 8).sum())}
        print(nombre, json.dumps(out[nombre]), flush=True)
    (AQUI / "resultados" / "34-mas-digitos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(2, len(ochos), figsize=(len(ochos) * 2.2, 5.4)); nombres = list(conjuntos)
    for k, (i, j) in enumerate(zip(ochos, J)):
        ax = axs[0, k]; TO.dibujar_objetos(ax, img[i], todos[i]); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off"); ax.set_title(f"idx {i}", fontsize=8)
        ax = axs[1, k]; vals = [out[n]["ochos_logit8_menos_9"][k] for n in nombres]; b = ax.bar(range(2), vals, color=["#52514e", "#1b7f3b"]); ax.axhline(0, color="k", lw=.6)
        for bb, n in zip(b, nombres):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), f"lee {out[n]['ochos_pred'][k]}", ha="center", va="bottom" if bb.get_height() >= 0 else "top", fontsize=7)
        ax.set_xticks(range(2)); ax.set_xticklabels(["180", "4003"], fontsize=8); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(-9, 9)
        if k == 0:
            ax.set_ylabel("logit 8 − logit 9\n(> 0 = lee 8)", fontsize=8)
    fig.suptitle("rizo-2 · 34 · los 9 ochos con el compositor «todo objetos» entrenado con 180 dígitos y con 4003 (180 + extra), ambos con aumento; media de 3 semillas\n"
                 + " · ".join(f"{n}: val {out[n]['val']:.4f}, cuota agujeros {out[n]['cuota_peso']['agujeros']:.3f}, ochos bien {out[n]['ochos_bien']}/9" for n in nombres), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(AQUI / "resultados" / "34-mas-digitos.png", dpi=110); print(f"→ resultados/34-mas-digitos.json · 34-mas-digitos.png ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
