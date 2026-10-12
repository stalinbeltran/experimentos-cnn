#!/usr/bin/env python3
"""rizo-2 — pasarle al compositor TODOS los agujeros, no sólo el mejor (pedido por el dueño el 2026-10-12, al ver la figura
25: «logra identificar los agujeros del 8, pero sólo pasa 1»), en 3×3 en vez de 2×2, y válido para dígitos con 1 o 0.

La feature nueva («huecos 3×3»): los rizos que pasan el filtro se AGRUPAN POR AGUJERO —dos candidatos cuyo centro cae en
la misma componente conexa del fondo son el mismo agujero; se queda el de mayor encierro— y los que caen en el fondo
exterior (concavidades abiertas, como el gancho de un 5) se agrupan por distancia (≤ 6 px). Un agujero → una campana σ 1,5
de altura = encierro; el mapa de todos se lee en 3×3 (máximo por celda) y se añade la cuenta de agujeros: 10 números.
Sin agujeros: nueve ceros y un 0.

  1. figura: la feature nueva frente a la actual (2×2 del mejor + cuenta) en varios dígitos (0, 1, 2 agujeros)
  2. medida: compositor MLP 256 · rectas + gris + {rizo actual | huecos 3×3 | las dos}, con aumento, 3 semillas,
     val / ciega, fallos de la figura 19 arreglados y nuevos (protocolo de `menos_rectas.py`).

    python nn/rizo_huecos.py   → resultados/26-huecos.png · 26-huecos.json · 26-huecos-medida.png  (~5 min)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import compositores as CO                                           # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

VAR = "rizo (h0, 8px)"; CELDAS = 3; D_EXT = 6.0
EJEMPLOS = (784, 66, 1562, 1765, 27, 1723)                         # 8 (2 agujeros) · 6 (1) · 9 sin rizo visto · 3 (0) · 7 (0) · 1 (0)


def agujeros(x, cs):
    """Los rizos filtrados, agrupados por AGUJERO: uno por componente conexa del fondo (el de mayor encierro); los del
    fondo exterior, por distancia. Devuelve la lista de agujeros (dicts de rizo) y, para cada rizo, su grupo."""
    lab, n = label(np.asarray(x) == 0); borde = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    grupos, out = [], []
    for c in sorted(cs, key=lambda c: -c["r8"]):
        cx, cy = int(c["centro_hueco"][0]), int(c["centro_hueco"][1]); comp = int(lab[cy, cx])
        if comp and comp not in borde:
            clave = ("hueco", comp)
            if clave in [g[0] for g in grupos]:
                grupos[[g[0] for g in grupos].index(clave)][1].append(c); continue
        else:
            cerca = [k for k, (cl, miembros) in enumerate(grupos) if cl[0] == "ext" and EV._dist(c["centro_hueco"], miembros[0]["centro_hueco"]) <= D_EXT]
            if cerca:
                grupos[cerca[0]][1].append(c); continue
            clave = ("ext", len(grupos))
        grupos.append((clave, [c])); out.append(c)
    return out, grupos


def mapa(ags):
    m = np.zeros((32, 32), np.float32)
    for c in ags:
        m = np.maximum(m, (np.exp(-((DV._XX - c["centro_hueco"][0]) ** 2 + (DV._YY - c["centro_hueco"][1]) ** 2) / (2 * 1.5 ** 2)) * c["r8"]).astype(np.float32))
    return m


def huecos(img, C, celdas=CELDAS):
    out = []
    for x, cs in zip(img, C):
        ags, _ = agujeros(x, DV.filtrar(cs, *DV.VARIANTES[VAR]))
        p = F.adaptive_max_pool2d(torch.from_numpy(mapa(ags))[None], celdas)[0].numpy().ravel()
        out.append(np.concatenate([p, [len(ags)]]))
    return np.array(out, np.float32)


def figura(img, y, todos):
    fig, axs = plt.subplots(3, len(EJEMPLOS), figsize=(len(EJEMPLOS) * 2.7, 9.2))
    for k, i in enumerate(EJEMPLOS):
        x = img[i]; cs = DV.filtrar(todos[i], *DV.VARIANTES[VAR]); ags, grupos = agujeros(x, cs)
        ax = axs[0, k]; RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        cols = ["#2a78d6", "#c2410c", "#1b7f3b", "#7c3aed", "#b45309"]
        for g, (clave, miembros) in enumerate(grupos):
            for c in miembros:
                ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=cols[g % 5], ms=9 if c is miembros[0] else 5, mew=1.8 if c is miembros[0] else 1)
        ax.set_title(f"idx {i} · un {y[i]}\n{len(cs)} rizos → {len(ags)} agujero(s) (un color cada uno)", fontsize=8)
        viejo = CO.compacta([todos[i]], VAR)[0]
        ax = axs[1, k]; ax.imshow(viejo[:4].reshape(2, 2), cmap="Oranges", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{viejo[a * 2 + b]:.2f}", ha="center", va="center", fontsize=9, color="k" if viejo[a * 2 + b] < 0.6 else "w")
        ax.set_title(f"ACTUAL: 2×2 del mejor\ncuenta de rizos = {int(viejo[4])}", fontsize=8)
        nuevo = huecos(img[[i]], [todos[i]])[0]
        ax = axs[2, k]; ax.imshow(nuevo[:9].reshape(3, 3), cmap="Oranges", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{nuevo[a * 3 + b]:.2f}", ha="center", va="center", fontsize=9, color="k" if nuevo[a * 3 + b] < 0.6 else "w")
        ax.set_title(f"NUEVO: 3×3 de todos\ncuenta de agujeros = {int(nuevo[9])}", fontsize=8)
    fig.suptitle("rizo-2 · 26 · pasar TODOS los agujeros al compositor: un rizo por agujero (misma componente del fondo = mismo agujero), mapa de todos en 3×3 + cuenta de agujeros (10 números)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(AQUI / "resultados" / "26-huecos.png", dpi=100); print("→ resultados/26-huecos.png")


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    figura(img, y, todos)
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]; Itr = [R.mover(img[tr], *m) for m in aum]
    base_tr = np.concatenate([np.concatenate([R.rectas(I, 8), R.gris(I, 8)], 1) for I in Itr]); Y = np.concatenate([y[tr]] * len(aum))
    base = {"va": np.concatenate([R.rectas(img[va], 8), R.gris(img[va], 8)], 1), "ex": np.concatenate([R.rectas(img[ex], 8), R.gris(img[ex], 8)], 1)}
    rizo_tr = np.concatenate([CO.compacta(cache[("tr", m)], VAR) for m in aum]); rizo = {"va": CO.compacta([todos[i] for i in va], VAR), "ex": CO.compacta([todos[i] for i in ex], VAR)}
    hue_tr = np.concatenate([huecos(I, cache[("tr", m)]) for I, m in zip(Itr, aum)]); hue = {"va": huecos(img[va], [todos[i] for i in va]), "ex": huecos(img[ex], [todos[i] for i in ex])}
    print(f"features listas ({time.time() - t0:.0f} s)", flush=True)
    fallos19 = {f["idx"] for f in json.loads((AQUI / "resultados" / "19-fallos-mejor.json").read_text())["fallos"]}; j784 = int(np.flatnonzero(va == 784)[0])
    variantes = {"rizo actual (2×2 mejor + cuenta, 5)": (rizo_tr, rizo), "huecos 3×3 + cuenta (10)": (hue_tr, hue), "rizo actual + huecos 3×3 (15)": (np.concatenate([rizo_tr, hue_tr], 1), {k: np.concatenate([rizo[k], hue[k]], 1) for k in rizo})}
    out = {}
    for nombre, (ftr, f) in variantes.items():
        X = np.concatenate([base_tr, ftr], 1); Xva = np.concatenate([base["va"], f["va"]], 1); Xex = np.concatenate([base["ex"], f["ex"]], 1); n = ftr.shape[1]; a0 = 896
        r = {"n_rizo": int(n), "val": [], "ciega": [], "cuota_rizo": [], "arreglados": [], "nuevos": [], "idx784": []}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy()
            pv, pe = RA.logits(red, mu, sd, Xva).argmax(1), RA.logits(red, mu, sd, Xex).argmax(1)
            r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean())); r["cuota_rizo"].append(float((W[:, a0:] ** 2).sum() / (W ** 2).sum()))
            mal = {int(va[k]) for k in np.flatnonzero(pv != y[va])}; r["arreglados"].append(len(fallos19 - mal)); r["nuevos"].append(len(mal - fallos19)); r["idx784"].append(int(pv[j784]))
        out[nombre] = {"n_rizo": n, "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4),
                       "cuota_peso_rizo": round(float(np.mean(r["cuota_rizo"])), 3), "fallos19_arreglados": r["arreglados"], "fallos_nuevos": r["nuevos"], "idx784": r["idx784"]}
        print(f"{nombre:40s} val {out[nombre]['val']:.4f} [{out[nombre]['val_rango'][0]:.4f}–{out[nombre]['val_rango'][1]:.4f}] · ciega {out[nombre]['ciega']:.4f} · cuota rizo {out[nombre]['cuota_peso_rizo']:.3f} · arregla {r['arreglados']} · nuevos {r['nuevos']} · 784→{r['idx784']}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "26-huecos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4)); nombres = list(out); xs = np.arange(len(nombres))
    axs[0].bar(xs - 0.2, [out[n]["val"] for n in nombres], 0.4, color="#2a78d6", label="val"); axs[0].bar(xs + 0.2, [out[n]["ciega"] for n in nombres], 0.4, color="#52514e", label="a ciegas"); axs[0].set_ylim(0.96, 0.995); axs[0].legend(fontsize=7, frameon=False)
    for i, n in enumerate(nombres):
        axs[0].text(i - 0.2, out[n]["val"], f"{out[n]['val']:.4f}", ha="center", va="bottom", fontsize=7); axs[0].text(i + 0.2, out[n]["ciega"], f"{out[n]['ciega']:.4f}", ha="center", va="bottom", fontsize=7)
    axs[0].set_title("acierto (MLP 256, con aumento, 3 semillas)", fontsize=9)
    axs[1].bar(xs, [out[n]["cuota_peso_rizo"] for n in nombres], color="#c2410c"); axs[1].set_title("cuota de ‖W‖² del grupo del rizo", fontsize=9)
    axs[2].bar(xs - 0.2, [np.mean(out[n]["fallos19_arreglados"]) for n in nombres], 0.4, color="#1b7f3b", label="de los 23 (fig. 19), arreglados"); axs[2].bar(xs + 0.2, [np.mean(out[n]["fallos_nuevos"]) for n in nombres], 0.4, color="#c2410c", label="fallos nuevos"); axs[2].legend(fontsize=7, frameon=False); axs[2].set_title("fallos en val (media de 3 semillas)", fontsize=9)
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels(nombres, fontsize=7.5); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("rizo-2 · 26 · el compositor con TODOS los agujeros (3×3 + cuenta) en vez del mejor (2×2 + cuenta)", fontsize=10); fig.tight_layout()
    fig.savefig(AQUI / "resultados" / "26-huecos-medida.png", dpi=100); print("→ resultados/26-huecos.json · 26-huecos-medida.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
