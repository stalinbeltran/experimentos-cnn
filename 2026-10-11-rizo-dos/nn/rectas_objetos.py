#!/usr/bin/env python3
"""rizo-2 — opción 5: las RECTAS como OBJETOS, con la receta de los agujeros (pedido por el dueño el 2026-10-12).

De cada mapa integrado (4 orientaciones × 2 escalas) salen los picos de «un voto» (`menos_rectas.un_voto`, L = 15). Los
picos de la misma orientación que son COLINEALES (a ≤ 2 px en la perpendicular) y están a ≤ L px a lo largo se agrupan:
son la MISMA recta. Cada recta es un objeto: centro (media de sus picos), orientación, longitud (extensión de los picos
a lo largo + L) y fuerza (el pico mayor). Por orientación: una campana σ 1,5 en el centro de cada recta con altura =
fuerza (normalizada al máximo de train), el mapa de TODAS leído en 3×3, más la cuenta y la longitud total (/32):
11 números × 4 orientaciones = 44 por escala. Se prueban ×1 sola (44), ×1 y ×2 (88), y junto a las rectas 8×8.

  figura 27-objetos-784.png: los objetos del 784 (segmentos con su longitud, tono = fuerza) y sus 3×3 por orientación
  medida 27-objetos.json · 27-objetos-medida.png: compositor MLP 256 con gris 8×8 + rizo (actual + huecos 3×3, 15),
  con aumento, 3 semillas, val / ciega, fallos de la figura 19 y nuevos.

    python nn/rectas_objetos.py   (~6 min)
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
import rizo_huecos as RH                                            # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

L = 15; CELDAS = 3; ANG = (0, 45, 90, 135); COLORES = ["#c2410c", "#1b7f3b", "#2a78d6", "#7c3aed"]
DIRS = {o: (np.array(v, float) / np.hypot(*v)) for o, v in MR.DIRS.items()}
NORMA = 2.5   # fuerza típica máxima de un pico (el 784 da 2,13); se fija aquí, no se aprende: es la escala de la campana


def objetos(x):
    """(32,32) → lista de rectas por escala: [{o, cx, cy, largo, fuerza, picos}] en coordenadas de 32×32."""
    out = []
    for e, m in enumerate(MR.mapas_rectas(x[None])):
        v = MR.un_voto(m, max(3, L // 2 ** e))[0].numpy(); esc = 2 ** e; Le = L // 2 ** e; objs = []
        for o in range(4):
            ys, xs = np.nonzero(v[o] > 0); d = DIRS[o]; n = np.array([-d[1], d[0]])
            picos = [(xs[k] + 0.5, ys[k] + 0.5, float(v[o, ys[k], xs[k]])) for k in np.argsort(-v[o, ys, xs])]
            grupos = []
            for px, py, pv in picos:
                for g in grupos:
                    gx, gy = np.mean([q[0] for q in g]), np.mean([q[1] for q in g]); rel = np.array([px - gx, py - gy])
                    if abs(rel @ n) <= 2.0 / esc and all(abs((px - qx) * d[0] + (py - qy) * d[1]) <= Le for qx, qy, _ in g):
                        g.append((px, py, pv)); break
                else:
                    grupos.append([(px, py, pv)])
            for g in grupos:
                t = [(qx * d[0] + qy * d[1]) for qx, qy, _ in g]
                objs.append({"o": o, "cx": float(np.mean([q[0] for q in g]) * esc), "cy": float(np.mean([q[1] for q in g]) * esc), "largo": float((max(t) - min(t) + Le) * esc),
                             "fuerza": float(max(q[2] for q in g)), "picos": len(g)})
        out.append(objs)
    return out


def caract(objs, celdas=CELDAS):
    f = []
    for o in range(4):
        m = np.zeros((32, 32), np.float32); n = 0; largo = 0.0
        for ob in (z for z in objs if z["o"] == o):
            m = np.maximum(m, (np.exp(-((DV._XX - ob["cx"]) ** 2 + (DV._YY - ob["cy"]) ** 2) / (2 * 1.5 ** 2)) * min(ob["fuerza"] / NORMA, 1.0)).astype(np.float32)); n += 1; largo += ob["largo"]
        f.extend(F.adaptive_max_pool2d(torch.from_numpy(m)[None], celdas)[0].numpy().ravel().tolist()); f.extend([n, largo / 32])
    return np.array(f, np.float32)


def rectas_objetos(img, escalas=(0,)):
    return np.stack([np.concatenate([caract(ob[e]) for e in escalas]) for ob in (objetos(x) for x in img)])


def figura(x, real):
    ob = objetos(x); fig = plt.figure(figsize=(17, 9)); G = fig.add_gridspec(2, 6, width_ratios=[1.6, 1.6, 1, 1, 1, 1])
    for e in range(2):
        ax = fig.add_subplot(G[e, 0]); RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        for z in ob[e]:
            d = DIRS[z["o"]]; h = z["largo"] / 2; r = min(z["fuerza"] / NORMA, 1.0)
            ax.plot([z["cx"] - 0.5 - d[0] * h, z["cx"] - 0.5 + d[0] * h], [z["cy"] - 0.5 - d[1] * h, z["cy"] - 0.5 + d[1] * h], color=COLORES[z["o"]], alpha=0.25 + 0.75 * r, lw=1 + 3 * r, solid_capstyle="round")
            ax.plot(z["cx"] - 0.5, z["cy"] - 0.5, ".", color="k", ms=3)
        ax.set_title(f"escala ×{2 ** e}: {len(ob[e])} rectas (objetos) · longitud real, tono = fuerza\ncuenta por orientación: " + " · ".join(f"{a}° {sum(z['o'] == o for z in ob[e])}" for o, a in enumerate(ANG)), fontsize=8)
        ax = fig.add_subplot(G[e, 1]); RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("por comparar: celdas 8×8 (figura 22), tono = valor", fontsize=8)
        M = MR.mapas_rectas(x[None])[e]; vmax = float(M.max())
        for o in range(4):
            for (oo, i, j, v, px, py) in __import__("rectas_dibujo").celdas(M[0].numpy(), 8):
                if oo != o or v <= 0:
                    continue
                d = DIRS[o]; h = L * v / vmax / 2; ax.plot([px - d[0] * h, px + d[0] * h], [py - d[1] * h, py + d[1] * h], color=COLORES[o], alpha=0.1 + 0.9 * v / vmax, lw=0.6 + 2 * v / vmax)
        f = caract(ob[e])
        for o in range(4):
            ax = fig.add_subplot(G[e, 2 + o]); m = f[o * 11:o * 11 + 9].reshape(3, 3); ax.imshow(m, cmap="Oranges", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=8, color="k" if m[a, b] < 0.6 else "w")
            ax.set_title(f"{ANG[o]}° · 3×3 de todas\ncuenta {int(f[o * 11 + 9])} · longitud {f[o * 11 + 10] * 32:.0f} px", fontsize=8, color=COLORES[o])
    fig.suptitle(f"rizo-2 · 27 · opción 5: las RECTAS como OBJETOS del idx 784 (un {real}) — picos de «un voto» agrupados por colinealidad = una recta; por orientación, 3×3 de todas + cuenta + longitud (44 números por escala)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(AQUI / "resultados" / "27-objetos-784.png", dpi=100); print("→ resultados/27-objetos-784.png")


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    figura(img[784], int(y[784]))
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]; Itr = [R.mover(img[tr], *m) for m in aum]; Y = np.concatenate([y[tr]] * len(aum))
    gris_tr = np.concatenate([R.gris(I, 8) for I in Itr]); gris = {"va": R.gris(img[va], 8), "ex": R.gris(img[ex], 8)}
    rizo_tr = np.concatenate([np.concatenate([CO.compacta(cache[("tr", m)], RH.VAR), RH.huecos(I, cache[("tr", m)])], 1) for I, m in zip(Itr, aum)])
    rizo = {k: np.concatenate([CO.compacta([todos[i] for i in idx], RH.VAR), RH.huecos(img[idx], [todos[i] for i in idx])], 1) for k, idx in (("va", va), ("ex", ex))}
    obj_tr = {e: np.concatenate([rectas_objetos(I, (0, 1)) for I in Itr]) for e in [(0, 1)]}[(0, 1)]; obj = {"va": rectas_objetos(img[va], (0, 1)), "ex": rectas_objetos(img[ex], (0, 1))}
    r88_tr = np.concatenate([R.rectas(I, 8) for I in Itr]); r88 = {"va": R.rectas(img[va], 8), "ex": R.rectas(img[ex], 8)}
    print(f"features listas ({time.time() - t0:.0f} s)", flush=True)
    variantes = {"rectas 8×8 (referencia, 512)": (r88_tr, r88), "objetos ×1 (44)": (obj_tr[:, :44], {k: v[:, :44] for k, v in obj.items()}), "objetos ×1 y ×2 (88)": (obj_tr, obj),
                 "rectas 8×8 + objetos ×1 y ×2 (600)": (np.concatenate([r88_tr, obj_tr], 1), {k: np.concatenate([r88[k], obj[k]], 1) for k in obj})}
    fallos19 = {f["idx"] for f in json.loads((AQUI / "resultados" / "19-fallos-mejor.json").read_text())["fallos"]}; j784 = int(np.flatnonzero(va == 784)[0]); out = {}
    for nombre, (ftr, f) in variantes.items():
        n = ftr.shape[1]; X = np.concatenate([ftr, gris_tr, rizo_tr], 1); Xva = np.concatenate([f["va"], gris["va"], rizo["va"]], 1); Xex = np.concatenate([f["ex"], gris["ex"], rizo["ex"]], 1)
        r = {"val": [], "ciega": [], "cuota": [], "arreglados": [], "nuevos": [], "idx784": []}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); wt = (W ** 2).sum()
            pv, pe = RA.logits(red, mu, sd, Xva).argmax(1), RA.logits(red, mu, sd, Xex).argmax(1)
            r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean()))
            r["cuota"].append({"rectas": float((W[:, :n] ** 2).sum() / wt), "gris": float((W[:, n:n + 384] ** 2).sum() / wt), "rizo": float((W[:, n + 384:] ** 2).sum() / wt)})
            mal = {int(va[k]) for k in np.flatnonzero(pv != y[va])}; r["arreglados"].append(len(fallos19 - mal)); r["nuevos"].append(len(mal - fallos19)); r["idx784"].append(int(pv[j784]))
        out[nombre] = {"n_rectas": int(n), "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4),
                       "cuota_peso": {g: round(float(np.mean([c[g] for c in r["cuota"]])), 3) for g in ("rectas", "gris", "rizo")}, "fallos19_arreglados": r["arreglados"], "fallos_nuevos": r["nuevos"], "idx784": r["idx784"]}
        o = out[nombre]; print(f"{nombre:36s} val {o['val']:.4f} [{o['val_rango'][0]:.4f}–{o['val_rango'][1]:.4f}] · ciega {o['ciega']:.4f} · cuota {o['cuota_peso']} · arregla {r['arreglados']} · nuevos {r['nuevos']} · 784→{r['idx784']}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "27-objetos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6)); nombres = list(out); xs = np.arange(len(nombres))
    axs[0].bar(xs - 0.2, [out[n]["val"] for n in nombres], 0.4, color="#2a78d6", label="val"); axs[0].bar(xs + 0.2, [out[n]["ciega"] for n in nombres], 0.4, color="#52514e", label="a ciegas"); axs[0].set_ylim(0.9, 1.0); axs[0].legend(fontsize=7, frameon=False)
    for i, n in enumerate(nombres):
        axs[0].text(i - 0.2, out[n]["val"], f"{out[n]['val']:.4f}", ha="center", va="bottom", fontsize=7); axs[0].text(i + 0.2, out[n]["ciega"], f"{out[n]['ciega']:.4f}", ha="center", va="bottom", fontsize=7)
    axs[0].set_title("acierto (MLP 256, con aumento, 3 semillas)", fontsize=9)
    for k, (g, c) in enumerate(zip(("rectas", "gris", "rizo"), ("#52514e", "#2a78d6", "#c2410c"))):
        axs[1].bar(xs + (k - 1) * 0.27, [out[n]["cuota_peso"][g] for n in nombres], 0.27, color=c, label=g)
    axs[1].legend(fontsize=7, frameon=False); axs[1].set_title("cuota de ‖W‖² por grupo", fontsize=9)
    axs[2].bar(xs - 0.2, [np.mean(out[n]["fallos19_arreglados"]) for n in nombres], 0.4, color="#1b7f3b", label="de los 23 (fig. 19), arreglados"); axs[2].bar(xs + 0.2, [np.mean(out[n]["fallos_nuevos"]) for n in nombres], 0.4, color="#c2410c", label="fallos nuevos"); axs[2].legend(fontsize=7, frameon=False); axs[2].set_title("fallos en val (media de 3 semillas)", fontsize=9)
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=7.5); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("rizo-2 · 27 · opción 5: las rectas como OBJETOS (3×3 + cuenta + longitud por orientación) frente a las celdas 8×8 · gris 8×8 y rizo (actual + huecos 3×3)", fontsize=9.5); fig.tight_layout()
    fig.savefig(AQUI / "resultados" / "27-objetos-medida.png", dpi=100); print("→ resultados/27-objetos.json · 27-objetos-medida.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
