#!/usr/bin/env python3
"""rizo-2 — RASTREAR un fallo de todo_objetos_entrada.py: por qué el compositor (semilla 0) lee un dígito como lo lee,
número a número. Pedido por el dueño el 2026-10-13 para el idx 985 (un 3 leído como 0) sobre el mapa suave con huecos
suaves. Reentrena el MLP de la semilla 0 con las mismas features (se cachean en resultados/40-features-<nombre>.npz),
comprueba que reproduce la predicción, y:
  (b) ABLACIÓN por grupo: qué predice si un grupo se sustituye por la media de train;
  (c) ATRIBUCIÓN gradiente × entrada estandarizada del margen logit[predicho] − logit[real] (rastrear.atribucion),
      sumada por grupo y por celda, y las features que más empujan, con su nombre, su valor y la media de las dos clases.

    python nn/rastrear_entrada.py --nombre suave-k3x3-hs --idx 985 --entrada /ruta/delgados-k3x3.npz --continuo --huecos-suaves
        → resultados/42-rastrear-<nombre>-<idx>.png · .json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import evaluar as EV                                                # noqa: E402
import gris_objetos as GO                                           # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import huecos_suaves as HS                                          # noqa: E402
import rastrear as RA                                               # noqa: E402
import rectas_objetos as RO                                         # noqa: E402
import resistencia as R                                             # noqa: E402
import todo_objetos_entrada as TE                                   # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

CEL = ["↖", "↑", "↗", "←", "·", "→", "↙", "↓", "↘"]                  # las 9 celdas 3×3, leídas como posición en el dígito


def nombres() -> list[tuple[str, str]]:
    """(grupo, nombre) de los 118 números: rectas 44 · gris 54 · huecos suaves 20."""
    n = []
    for a in RO.ANG:
        n += [("rectas", f"recta {a}° {c}") for c in CEL] + [("rectas", f"recta {a}° cuenta"), ("rectas", f"recta {a}° largo")]
    for s in GO.SENTIDOS:
        n += [("gris", f"curva {s} {c}") for c in CEL] + [("gris", f"curva {s} cuenta"), ("gris", f"curva {s} largo")]
    n += [("gris", f"golpe {c}") for c in CEL] + [("gris", "golpe cuenta")]
    n += [("huecos", f"cerrado {c}") for c in CEL] + [("huecos", "cerrado cuenta")]
    n += [("huecos", f"concavidad {c}") for c in CEL] + [("huecos", "concavidad cuenta")]
    return n


def features(img, huecos_suaves: bool):
    f = [RO.rectas_objetos(img, (0,)), GO.gris_objetos(img)]
    if huecos_suaves:
        f.append(HS.caract(img))
    return np.concatenate(f, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", required=True); ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--entrada", default=None); ap.add_argument("--clave", default="M"); ap.add_argument("--continuo", action="store_true")
    ap.add_argument("--huecos-suaves", action="store_true"); ap.add_argument("--umbral", type=float, default=0.5)
    a = ap.parse_args(); GR.configurar(**R.CAL_GRIS); torch.set_num_threads(2)
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); y, part = d["etiquetas"].astype(int), d["particion"]; tinta = (d["imagenes"] > 0).astype(np.uint8)
    if a.entrada:
        mapa = np.load(a.entrada)[a.clave].astype(np.float32); img = mapa if a.continuo else (mapa > a.umbral).astype(np.uint8)
    else:
        img = tinta
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    cache = AQUI / "resultados" / f"40-features-{a.nombre}.npz"
    if cache.exists():
        z = np.load(cache); X, Y, Xva = z["X"], z["Y"], z["Xva"]
    else:
        X = np.concatenate([features(R.mover(img[tr], *m), a.huecos_suaves) for m in aum]); Y = np.concatenate([y[tr]] * 13)
        Xva = features(img[va], a.huecos_suaves); np.savez_compressed(cache, X=X, Y=Y, Xva=Xva)
    red, mu, sd = RA.entrenar(X, Y, 0)
    j = int(np.flatnonzero(va == a.idx)[0]); x = Xva[j]; L = RA.logits(red, mu, sd, x)[0]; pred, real = int(L.argmax()), int(y[a.idx])
    print(f"idx {a.idx}: real {real}, predicho {pred}; logits {np.round(L, 2).tolist()}")
    nom = nombres(); grupos = sorted(set(g for g, _ in nom), key=lambda g: ["rectas", "gris", "huecos"].index(g))
    # (b) ablación por grupo
    abl = {}
    for g in grupos:
        xa = x.copy(); m = np.array([gg == g for gg, _ in nom]); xa[m] = X[:, m].mean(0)
        La = RA.logits(red, mu, sd, xa)[0]; abl[g] = {"predice": int(La.argmax()), "margen": round(float(La[pred] - La[real]), 2)}
    # (c) atribución
    at = RA.atribucion(red, mu, sd, x, pred, real); z = (x - mu) / sd
    por_grupo = {g: round(float(at[[gg == g for gg, _ in nom]].sum()), 2) for g in grupos}
    orden = np.argsort(-np.abs(at)); top = []
    for k in orden[:14]:
        top.append({"n": int(k), "grupo": nom[k][0], "nombre": nom[k][1], "atribucion": round(float(at[k]), 2), "valor": round(float(x[k]), 3),
                    "z": round(float(z[k]), 2), f"media_{real}": round(float(X[Y == real, k].mean()), 3), f"media_{pred}": round(float(X[Y == pred, k].mean()), 3)})
    out = {"idx": a.idx, "real": real, "predicho": pred, "logits": np.round(L, 2).tolist(), "margen": round(float(L[pred] - L[real]), 2),
           "ablacion_por_grupo": abl, "atribucion_por_grupo": por_grupo, "top": top}
    (AQUI / "resultados" / f"42-rastrear-{a.nombre}-{a.idx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    # figura
    fig = plt.figure(figsize=(18, 9)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0]); RA.pintar_digito(ax, tinta[a.idx]); ax.set_title(f"idx {a.idx} · tinta · real {real}", fontsize=9)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); TE.dibujar(ax, img[a.idx], None, a.continuo, False, a.huecos_suaves); ax.set_title(f"entrada «{a.nombre}» con sus objetos · predicho {pred}", fontsize=9)
    ax = fig.add_subplot(gs[0, 2]); ax.bar(range(10), L, color=["#e8432b" if c == pred else "#2a78d6" if c == real else "#bbb" for c in range(10)]); ax.set_xticks(range(10)); ax.set_title(f"logits (rojo predicho, azul real) · margen {L[pred] - L[real]:.2f}", fontsize=9)
    ax = fig.add_subplot(gs[0, 3:]); vals = [por_grupo[g] for g in grupos]; ax.barh(grupos, vals, color=["#e8432b" if v > 0 else "#2a78d6" for v in vals]); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"atribución por grupo (>0 empuja a {pred}, <0 a {real})\nablación: " + " · ".join(f"sin {g}→{abl[g]['predice']}" for g in grupos), fontsize=8.5)
    ax = fig.add_subplot(gs[1, :2]); ks = orden[:14][::-1]; ax.barh([f"{nom[k][1]}  (={x[k]:.2f}, z {z[k]:+.1f}; μ{real} {X[Y == real, k].mean():.2f} · μ{pred} {X[Y == pred, k].mean():.2f})" for k in ks], at[ks], color=["#e8432b" if at[k] > 0 else "#2a78d6" for k in ks])
    ax.axvline(0, color="k", lw=0.8); ax.tick_params(axis="y", labelsize=7.5); ax.set_title(f"las 14 features que más empujan (rojo → {pred}, azul → {real}) · valor, z estandarizado y medias de train de cada clase", fontsize=8.5)
    # mapa por celda: suma de atribución de todas las features con celda, por grupo
    for gi, g in enumerate(grupos):
        ax = fig.add_subplot(gs[1, 2 + gi])
        m = np.zeros((3, 3))
        for k, (gg, nn_) in enumerate(nom):
            if gg == g and nn_.split()[-1] in CEL:
                c = CEL.index(nn_.split()[-1]); m[c // 3, c % 3] += at[k]
        v = np.abs(m).max() or 1; ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
        for (i, jj), val in np.ndenumerate(m):
            ax.text(jj, i, f"{val:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{g}: atribución por celda 3×3 (rojo → {pred}, azul → {real})", fontsize=8.5)
    fig.suptitle(f"rizo-2 · 42 · por qué «{a.nombre}» lee el idx {a.idx} ({real}) como {pred} · MLP semilla 0", fontsize=10)
    fig.tight_layout(); out_png = AQUI / "resultados" / f"42-rastrear-{a.nombre}-{a.idx}.png"; fig.savefig(out_png, dpi=110); print(f"→ {out_png.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
