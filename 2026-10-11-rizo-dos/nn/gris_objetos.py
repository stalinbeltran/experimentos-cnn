#!/usr/bin/env python3
"""rizo-2 — el GRIS (sin «recto») como OBJETOS, con la receta de los agujeros y de las rectas-objeto (pedido por el dueño
el 2026-10-12). De los mapas del detector gris (`digitos_bordes.mapas_bordes`: k→ · k← · k↓ · k↑ · golpe), cada píxel
curvo se asigna al SENTIDO dominante de su vector de curvatura; las componentes conexas (8-vecindad) dentro de cada
sentido son UNA CURVA: objeto con centro, longitud (px del borde) y fuerza (|κ| máxima, /NORMA_K). Por sentido, campana
σ 1,5 en el centro de cada curva con altura = fuerza, mapa de todas en 3×3 + cuenta + longitud total (/32): 11 × 4 = 44.
El golpe igual: componentes → objetos → 3×3 + cuenta = 10. Total 54 (contra 320 del gris sin recto en celdas 8×8).

  figura 31-gris-objetos-784.png · medida 31-gris-objetos.json / .png: MLP 256 con aumento, 3 semillas, con rectas =
  objetos ×1 (la decisión del dueño) y rizo = actual + huecos 3×3; también sobre la referencia (rectas 8×8).

    python nn/gris_objetos.py   (~8 min)
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
import rectas_objetos as RO                                         # noqa: E402
import resistencia as R                                             # noqa: E402
import rizo_huecos as RH                                            # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

SENTIDOS = ["k→", "k←", "k↓", "k↑"]; COLORES = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]; NORMA_K = 6.0; MIN_PX = 3; CELDAS = 3
OCHO = np.ones((3, 3), int)


def objetos(M):
    """M (7,32,32) de mapas_bordes → {'curvas': [{s, cx, cy, largo, fuerza}], 'golpes': [{cx, cy, largo}]}."""
    K = M[1:5]; curvo = K.max(0) > 0; sent = K.argmax(0); curvas = []
    for s in range(4):
        lab, n = label(curvo & (sent == s), structure=OCHO)
        for z in range(1, n + 1):
            yy, xx = np.nonzero(lab == z)
            if len(yy) < MIN_PX:
                continue
            curvas.append({"s": s, "cx": float(xx.mean() + 0.5), "cy": float(yy.mean() + 0.5), "largo": int(len(yy)), "fuerza": float(min(K[s][yy, xx].max() / NORMA_K, 1.0))})
    lab, n = label(M[5] > 0, structure=OCHO); golpes = []
    for z in range(1, n + 1):
        yy, xx = np.nonzero(lab == z)
        if len(yy) >= MIN_PX:
            golpes.append({"cx": float(xx.mean() + 0.5), "cy": float(yy.mean() + 0.5), "largo": int(len(yy))})
    return {"curvas": curvas, "golpes": golpes}


def _mapa(objs, altura):
    m = np.zeros((32, 32), np.float32)
    for o in objs:
        m = np.maximum(m, (np.exp(-((DV._XX - o["cx"]) ** 2 + (DV._YY - o["cy"]) ** 2) / (2 * 1.5 ** 2)) * altura(o)).astype(np.float32))
    return m


def caract(ob, celdas=CELDAS):
    f = []
    for s in range(4):
        cs = [c for c in ob["curvas"] if c["s"] == s]
        f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(cs, lambda o: o["fuerza"]))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(cs), sum(c["largo"] for c in cs) / 32])
    f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(ob["golpes"], lambda o: 1.0))[None], celdas)[0].numpy().ravel().tolist()); f.append(len(ob["golpes"]))
    return np.array(f, np.float32)


def gris_objetos(img):
    return np.stack([caract(objetos(DB.mapas_bordes(x, R.GRIS))) for x in img])


def figura(x, real):
    M = DB.mapas_bordes(x, R.GRIS); ob = objetos(M); f = caract(ob)
    fig, axs = plt.subplots(2, 6, figsize=(17, 7.4))
    for ax in axs.flat:
        ax.axis("off")
    ax = axs[0, 0]; RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    K = M[1:5]; curvo = K.max(0) > 0; sent = K.argmax(0)
    for s in range(4):
        yy, xx = np.nonzero(curvo & (sent == s)); ax.plot(xx, yy, "s", color=COLORES[s], ms=2.6, mew=0, alpha=0.9)
    yy, xx = np.nonzero(M[5] > 0); ax.plot(xx, yy, ".", color="k", ms=2)
    ax.set_title(f"idx 784: píxeles curvos por SENTIDO dominante de k\n→ naranja · ← azul · ↓ verde · ↑ morado · golpe negro", fontsize=8)
    ax = axs[0, 1]; RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    for c in ob["curvas"]:
        ax.add_patch(plt.Circle((c["cx"] - 0.5, c["cy"] - 0.5), 0.8 + 0.12 * c["largo"], fill=False, color=COLORES[c["s"]], lw=0.8 + 2.4 * c["fuerza"]))
        ax.plot(c["cx"] - 0.5, c["cy"] - 0.5, ".", color=COLORES[c["s"]], ms=4)
    for g in ob["golpes"]:
        ax.plot(g["cx"] - 0.5, g["cy"] - 0.5, "x", color="k", ms=7, mew=1.5)
    ax.set_title(f"OBJETOS: {len(ob['curvas'])} curvas (radio ∝ longitud, grosor ∝ fuerza)\n+ {len(ob['golpes'])} golpes (×); componentes ≥ {MIN_PX} px", fontsize=8)
    for s in range(4):
        ax = axs[0, 2 + s]; m = f[s * 11:s * 11 + 9].reshape(3, 3); ax.imshow(m, cmap="Oranges", vmin=0, vmax=1); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=8, color="k" if m[a, b] < 0.6 else "w")
        ax.set_title(f"{SENTIDOS[s]} · 3×3 de todas\ncuenta {int(f[s * 11 + 9])} · longitud {f[s * 11 + 10] * 32:.0f} px", fontsize=8, color=COLORES[s])
    ax = axs[1, 0]; m = f[44:53].reshape(3, 3); ax.imshow(m, cmap="Greys", vmin=0, vmax=1); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=8, color="k" if m[a, b] < 0.6 else "w")
    ax.set_title(f"golpe · 3×3 de todos · cuenta {int(f[53])}", fontsize=8)
    P = F.adaptive_max_pool2d(torch.from_numpy(M)[None], 8)[0].numpy()
    for s in range(4):
        ax = axs[1, 2 + s]; ax.imshow(P[1 + s], cmap="Oranges", vmin=0, vmax=max(1.0, float(P[1 + s].max()))); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"por comparar: {SENTIDOS[s]} en celdas 8×8 (64)", fontsize=8, color=COLORES[s])
    ax = axs[1, 1]; ax.imshow(P[5], cmap="Greys", vmin=0, vmax=1); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("por comparar: golpe en celdas 8×8 (64)", fontsize=8)
    fig.suptitle(f"rizo-2 · 31 · el GRIS (sin «recto») como OBJETOS, idx 784 (un {real}): píxel curvo → sentido dominante de k → componente conexa = una curva; por sentido 3×3 + cuenta + longitud; golpe 3×3 + cuenta (54 números)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / "31-gris-objetos-784.png", dpi=100); print("→ resultados/31-gris-objetos-784.png")


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    figura(img[784], int(y[784]))
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]; Itr = [R.mover(img[tr], *m) for m in aum]; Y = np.concatenate([y[tr]] * 13)
    G = {"tr": np.concatenate([R.gris(I, 8) for I in Itr]), "va": R.gris(img[va], 8), "ex": R.gris(img[ex], 8)}
    GO = {"tr": np.concatenate([gris_objetos(I) for I in Itr]), "va": gris_objetos(img[va]), "ex": gris_objetos(img[ex])}
    RZ = {"tr": np.concatenate([np.concatenate([CO.compacta(cache[("tr", m)], RH.VAR), RH.huecos(I, cache[("tr", m)])], 1) for I, m in zip(Itr, aum)])}
    for k, idx in (("va", va), ("ex", ex)):
        RZ[k] = np.concatenate([CO.compacta([todos[i] for i in idx], RH.VAR), RH.huecos(img[idx], [todos[i] for i in idx])], 1)
    OB = {"tr": np.concatenate([RO.rectas_objetos(I, (0,)) for I in Itr]), "va": RO.rectas_objetos(img[va], (0,)), "ex": RO.rectas_objetos(img[ex], (0,))}
    R8 = {"tr": np.concatenate([R.rectas(I, 8) for I in Itr]), "va": R.rectas(img[va], 8), "ex": R.rectas(img[ex], 8)}
    sr = {k: v[:, 64:] for k, v in G.items()}
    print(f"features listas ({time.time() - t0:.0f} s)", flush=True)
    conf = {"objetos ×1 + gris + rizo (elegida)": [OB, G, RZ], "objetos ×1 + gris SIN recto + rizo": [OB, sr, RZ], "objetos ×1 + GRIS-OBJETOS + rizo": [OB, GO, RZ],
            "objetos ×1 + gris sin recto + GRIS-OBJETOS + rizo": [OB, sr, GO, RZ], "rectas 8×8 + GRIS-OBJETOS + rizo": [R8, GO, RZ], "rectas 8×8 + gris + GRIS-OBJETOS + rizo": [R8, G, GO, RZ], "GRIS-OBJETOS solo": [GO]}
    f19 = {f["idx"] for f in json.loads((AQUI / "resultados" / "19-fallos-mejor.json").read_text())["fallos"]}; j = int(np.flatnonzero(va == 784)[0]); out = {}
    for nombre, partes in conf.items():
        X, Xva, Xex = [np.concatenate([p[k] for p in partes], 1) for k in ("tr", "va", "ex")]; r = {"val": [], "ciega": [], "arr": [], "nue": [], "784": []}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Y, sem); pv = RA.logits(red, mu, sd, Xva).argmax(1); pe = RA.logits(red, mu, sd, Xex).argmax(1); mal = {int(va[k]) for k in np.flatnonzero(pv != y[va])}
            r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean())); r["arr"].append(len(f19 - mal)); r["nue"].append(len(mal - f19)); r["784"].append(int(pv[j]))
        out[nombre] = {"n": int(X.shape[1]), "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4), "fallos19_arreglados": r["arr"], "fallos_nuevos": r["nue"], "idx784": r["784"]}
        o = out[nombre]; print(f"{nombre:52s} n {o['n']:4d} · val {o['val']:.4f} [{o['val_rango'][0]:.4f}–{o['val_rango'][1]:.4f}] · ciega {o['ciega']:.4f} · arregla {r['arr']} · nuevos {r['nue']} · 784→{r['784']}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "31-gris-objetos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8)); nombres = list(out); xs = np.arange(len(nombres))
    axs[0].bar(xs - 0.2, [out[n]["val"] for n in nombres], 0.4, color="#2a78d6", label="val"); axs[0].bar(xs + 0.2, [out[n]["ciega"] for n in nombres], 0.4, color="#52514e", label="a ciegas"); axs[0].set_ylim(0.93, 0.995); axs[0].legend(fontsize=7, frameon=False)
    for i, n in enumerate(nombres):
        axs[0].text(i - 0.2, out[n]["val"], f"{out[n]['val']:.4f}", ha="center", va="bottom", fontsize=6.5); axs[0].text(i + 0.2, out[n]["ciega"], f"{out[n]['ciega']:.4f}", ha="center", va="bottom", fontsize=6.5)
    axs[0].set_title("acierto (MLP 256, con aumento, 3 semillas)", fontsize=9)
    axs[1].bar(xs - 0.2, [np.mean(out[n]["fallos19_arreglados"]) for n in nombres], 0.4, color="#1b7f3b", label="de los 23 (fig. 19), arreglados"); axs[1].bar(xs + 0.2, [np.mean(out[n]["fallos_nuevos"]) for n in nombres], 0.4, color="#c2410c", label="fallos nuevos"); axs[1].legend(fontsize=7, frameon=False); axs[1].set_title("fallos en val", fontsize=9)
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in nombres], fontsize=6.5); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("rizo-2 · 31 · el gris (sin recto) como OBJETOS (54 números) frente a sus celdas 8×8", fontsize=10); fig.tight_layout()
    fig.savefig(AQUI / "resultados" / "31-gris-objetos-medida.png", dpi=100); print("→ resultados/31-gris-objetos.json · 31-gris-objetos-medida.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
