#!/usr/bin/env python3
"""rizo-2 — TODO OBJETOS (decisión del dueño, 2026-10-12): el compositor sólo con «responsables» identificables:
rectas-objeto ×1 (44) + gris-objetos (54) + agujeros 3×3 + cuenta (10) = 108 números. Mide val / ciega (3 semillas, con
aumento), guarda los fallos de val de la semilla 0 y los pinta con sus objetos encima (rectas como segmentos, curvas como
círculos por sentido, golpes ×, agujeros +), que es lo que permite ir del fallo a los responsables.

    python nn/todo_objetos.py   → resultados/33-todo-objetos.json · 33-todo-objetos-fallos.png  (~6 min)
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
import gris_objetos as GO                                           # noqa: E402
import rastrear as RA                                               # noqa: E402
import rectas_objetos as RO                                         # noqa: E402
import resistencia as R                                             # noqa: E402
import rizo_huecos as RH                                            # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

GRUPOS = {"rectas-objeto ×1": (0, 44), "gris-objetos": (44, 98), "agujeros": (98, 108)}


def features(img, C):
    return np.concatenate([RO.rectas_objetos(img, (0,)), GO.gris_objetos(img), RH.huecos(img, C)], 1)


def dibujar_objetos(ax, x, cs):
    RA.pintar_digito(ax, x)
    for z in RO.objetos(x)[0]:
        d = RO.DIRS[z["o"]]; h = z["largo"] / 2; r = min(z["fuerza"] / RO.NORMA, 1.0)
        ax.plot([z["cx"] - 0.5 - d[0] * h, z["cx"] - 0.5 + d[0] * h], [z["cy"] - 0.5 - d[1] * h, z["cy"] - 0.5 + d[1] * h], color="#52514e", alpha=0.3 + 0.7 * r, lw=0.6 + 1.6 * r, solid_capstyle="round")
    ob = GO.objetos(DB.mapas_bordes(x, R.GRIS))
    for c in ob["curvas"]:
        ax.add_patch(plt.Circle((c["cx"] - 0.5, c["cy"] - 0.5), 0.6 + 0.1 * c["largo"], fill=False, color=GO.COLORES[c["s"]], lw=0.5 + 1.5 * c["fuerza"]))
    for g in ob["golpes"]:
        ax.plot(g["cx"] - 0.5, g["cy"] - 0.5, "x", color="k", ms=4, mew=1)
    ags, _ = RH.agujeros(x, DV.filtrar(cs, *DV.VARIANTES[RH.VAR]))
    for a in ags:
        ax.plot(a["centro_hueco"][0] - 0.5, a["centro_hueco"][1] - 0.5, "+", color=EV.AZU, ms=9, mew=2)


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    X = np.concatenate([features(R.mover(img[tr], *m), cache[("tr", m)]) for m in aum]); Y = np.concatenate([y[tr]] * 13)
    Xva, Xex = features(img[va], [todos[i] for i in va]), features(img[ex], [todos[i] for i in ex]); print(f"features listas: {X.shape[1]} ({time.time() - t0:.0f} s)", flush=True)
    f19 = {f["idx"] for f in json.loads((AQUI / "resultados" / "19-fallos-mejor.json").read_text())["fallos"]}
    r = {"val": [], "ciega": [], "arr": [], "nue": [], "cuota": []}; preds = []
    for sem in D.SEMILLAS:
        red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); wt = (W ** 2).sum()
        pv = RA.logits(red, mu, sd, Xva).argmax(1); pe = RA.logits(red, mu, sd, Xex).argmax(1); preds.append(pv); mal = {int(va[k]) for k in np.flatnonzero(pv != y[va])}
        r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean())); r["arr"].append(len(f19 - mal)); r["nue"].append(len(mal - f19))
        r["cuota"].append({g: float((W[:, a:b] ** 2).sum() / wt) for g, (a, b) in GRUPOS.items()})
    p0 = preds[0]; veces = np.sum([p != y[va] for p in preds], 0); f = np.flatnonzero(p0 != y[va])
    conf = {}
    for j in f:
        k = f"{y[va][j]}→{p0[j]}"; conf[k] = conf.get(k, 0) + 1
    out = {"n": int(X.shape[1]), "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4),
           "cuota_peso": {g: round(float(np.mean([c[g] for c in r["cuota"]])), 3) for g in GRUPOS}, "fallos19_arreglados": r["arr"], "fallos_nuevos": r["nue"], "fallos_s0": int(len(f)),
           "fallan_en_las_3": int((veces == 3).sum()), "confusiones": dict(sorted(conf.items(), key=lambda z: -z[1])),
           "fallos": [{"idx": int(va[j]), "real": int(y[va][j]), "predicho": int(p0[j]), "semillas_que_fallan": int(veces[j])} for j in f]}
    (AQUI / "resultados" / "33-todo-objetos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n"); print(json.dumps({k: v for k, v in out.items() if k != "fallos"}, ensure_ascii=False))
    orden = f[np.lexsort((p0[f], y[va][f]))]; cols = 9; filas = int(np.ceil(len(orden) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.9, filas * 2.15 + 0.9)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, j in zip(axs.flat, orden):
        i = va[j]; dibujar_objetos(ax, img[i], todos[i]); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
        ax.set_title(f"{y[va][j]}→{p0[j]} · idx {i} · {veces[j]}/3", color=EV.NAR if veces[j] == 3 else "#b45309", fontsize=7.5, pad=2)
    fig.suptitle(f"rizo-2 · 33 · TODO OBJETOS (rectas ×1 + gris-objetos + agujeros, {X.shape[1]} números): los {len(f)} fallos de val, semilla 0 (acierto {r['val'][0]:.4f}) · «real→predicho» · n/3 semillas\n"
                 "objetos: segmentos = rectas (tono = fuerza) · círculos = curvas por sentido (→ naranja · ← azul · ↓ verde · ↑ morado; radio ∝ longitud) · × = golpe · + = agujero", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / "33-todo-objetos-fallos.png", dpi=100); print("→ resultados/33-todo-objetos.json · 33-todo-objetos-fallos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
