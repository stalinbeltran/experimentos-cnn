#!/usr/bin/env python3
"""rizo-2 — RASTREAR por qué el mejor compositor (MLP 256 · rectas + gris + rizo, con aumento ±1–3 px; semilla 0) lee un
dígito de val como lo lee. Pedido por el dueño el 2026-10-12 para el idx 784 (un 8 leído como 6). Gráficamente:

  (a) el dígito con los candidatos del rizo (cruz = centro del hueco, tamaño = encierro) y los logits de las 10 clases;
  (b) ABLACIÓN por grupo: qué predice si un grupo de features (rectas · gris · rizo) se sustituye por la media de train
      (= «no sé nada de ese grupo»), y si se sustituye por la media de los 8 / de los 6 de train;
  (c) ATRIBUCIÓN (gradiente × entrada estandarizada) del margen logit[predicho] − logit[real], sumada por grupo y pintada
      sobre el dígito celda a celda: rojo = esa celda empuja hacia lo predicho, azul = hacia lo real;
  (d) la feature del rizo (2×2 + cuenta) del dígito frente a la media de su clase y la de la clase predicha;
  (e) sus vecinos más cercanos en train (features estandarizadas) con su clase.

    python nn/rastrear.py --idx 784   → resultados/20-rastrear-784.png · 20-rastrear-784.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

# el vector de features, por tramos (feats de resistencia.py): rectas 4 orient × 2 escalas × 64 · gris 6 canales × 64 · rizo 4 + 1
GRUPOS = {"rectas": (0, 512), "gris": (512, 896), "rizo": (896, 901)}
RECTAS_N = [f"recta {a}° ×{e}" for e in (1, 2) for a in (0, 45, 90, 135)]
GRIS_N = ["recto", "curva k→", "curva k←", "curva k↓", "curva k↑", "golpe"]
FONDO = matplotlib.colors.ListedColormap(["#b9b8b4"])


def entrenar(xtr, ytr, sem, oculta=256):
    """Idéntico a resistencia.mlp_logits, pero devuelve la red (hace falta para el gradiente)."""
    torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    red = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], oculta), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(oculta, 10))
    opt = torch.optim.Adam(red.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(400):
        red.train(); opt.zero_grad(); F.cross_entropy(red(xt), yt).backward(); opt.step()
    red.eval(); return red, mu, sd


def logits(red, mu, sd, X):
    with torch.no_grad():
        return red(torch.from_numpy((np.atleast_2d(X) - mu) / sd).float()).numpy()


def atribucion(red, mu, sd, x, a, b):
    """∂(logit a − logit b)/∂z · z, con z la entrada estandarizada: cuánto empuja cada feature hacia a (>0) o hacia b (<0)."""
    z = torch.from_numpy((x - mu) / sd).float()[None].requires_grad_(True)
    L = red(z); (L[0, a] - L[0, b]).backward()
    return (z.grad * z).detach().numpy()[0]


def pintar_digito(ax, img):
    ax.imshow(np.where(img > 0, 1.0, np.nan), cmap=FONDO, vmin=0, vmax=1); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, required=True); ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--vecinos", type=int, default=6); args = ap.parse_args()
    torch.set_num_threads(2); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    j = int(np.flatnonzero(va == args.idx)[0]) if args.idx in set(va.tolist()) else None
    if j is None:
        print(f"idx {args.idx} no está en val"); return 2
    cache = pickle.loads(R.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    X = np.concatenate([R.feats(R.mover(img[tr], *m), 8, cache[("tr", m)]) for m in aum]); Y = np.concatenate([y[tr]] * len(aum))
    X0 = X[:len(tr)]                                         # los 180 sin mover, para medias por clase y vecinos
    Cva = cache[("va", "x", 0)]; Xva = R.feats(img[va], 8, Cva)
    red, mu, sd = entrenar(X, Y, args.semilla)
    x, real = Xva[j], int(y[va][j]); L = logits(red, mu, sd, x)[0]; pred = int(L.argmax()); p = np.exp(L - L.max()); p /= p.sum()
    cs = DV.filtrar(Cva[j], *DV.VARIANTES["rizo (h0, 8px)"]); crudos = Cva[j]
    out = {"idx": args.idx, "real": real, "predicho": pred, "semilla": args.semilla, "logits": L.round(3).tolist(), "prob": p.round(4).tolist(),
           "rizo_candidatos_filtrados": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in cs],
           "rizo_candidatos_crudos": len(crudos)}
    # (b) ablación por grupo
    abl = {}
    medias = {k: X0[y[tr] == k].mean(0) for k in range(10)}
    for g, (a, b) in GRUPOS.items():
        fila = {}
        for nombre, fuente in [("media train", X.mean(0)), (f"media de los {real}", medias[real]), (f"media de los {pred}", medias[pred])]:
            xx = x.copy(); xx[a:b] = fuente[a:b]; Lg = logits(red, mu, sd, xx)[0]
            fila[nombre] = {"predicho": int(Lg.argmax()), f"logit{pred}-logit{real}": round(float(Lg[pred] - Lg[real]), 3)}
        abl[g] = fila
    out["ablacion"] = abl
    # sólo un grupo (los otros dos a la media de train)
    solo = {}
    for g, (a, b) in GRUPOS.items():
        xx = X.mean(0).copy(); xx[a:b] = x[a:b]; Lg = logits(red, mu, sd, xx)[0]; solo[g] = {"predicho": int(Lg.argmax()), f"logit{pred}-logit{real}": round(float(Lg[pred] - Lg[real]), 3)}
    out["solo_un_grupo"] = solo
    # (c) atribución
    at = atribucion(red, mu, sd, x, pred, real)
    out["atribucion_por_grupo"] = {g: round(float(at[a:b].sum()), 3) for g, (a, b) in GRUPOS.items()}
    out["margen"] = round(float(L[pred] - L[real]), 3)
    # (e) vecinos
    Z0, z = (X0 - mu) / sd, (x - mu) / sd; dist = np.linalg.norm(Z0 - z, axis=1); orden = np.argsort(dist)[:args.vecinos]
    out["vecinos"] = [{"idx": int(tr[k]), "clase": int(y[tr][k]), "dist": round(float(dist[k]), 2)} for k in orden]
    dist8 = {k: round(float(np.linalg.norm(Z0[y[tr] == k] - z, axis=1).mean()), 2) for k in range(10)}; out["dist_media_por_clase"] = dist8
    (AQUI / "resultados" / f"20-rastrear-{args.idx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("logits", "prob")}, indent=1, ensure_ascii=False))

    # ─── figura ───
    fig = plt.figure(figsize=(17, 13)); G = fig.add_gridspec(4, 8, height_ratios=[1.25, 1, 1, 1.15], hspace=0.5, wspace=0.25)
    # fila 0: dígito + rizo · logits · ablación · rizo frente a medias
    ax = fig.add_subplot(G[0, 0:2]); pintar_digito(ax, img[args.idx])
    for q, c in enumerate(crudos):
        ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "x", color="#9a9a9a", ms=5, mew=1)
    for q, c in enumerate(cs):
        ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=4 + 10 * c["r8"], mew=2)
        ax.annotate(f"h {c['h']:.1f} · enc {c['r8']:.2f}", (c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5), xytext=(3, 3), textcoords="offset points", fontsize=7, color=EV.AZU)
    ax.set_title(f"idx {args.idx} · real {real} → leído {pred}\n{len(crudos)} candidatos (×) · {len(cs)} rizos (+)", fontsize=9)
    ax = fig.add_subplot(G[0, 2:4]); cols = [EV.NAR if k == pred else (EV.AZU if k == real else "#b9b8b4") for k in range(10)]
    ax.bar(range(10), L, color=cols); ax.set_xticks(range(10)); ax.set_title(f"logits · margen {pred}−{real} = {L[pred]-L[real]:+.2f}\np({pred}) {p[pred]:.2f} · p({real}) {p[real]:.2f}", fontsize=9)
    ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(G[0, 4:6]); nombres = list(next(iter(abl.values())).keys()); W = 0.26
    for i, n in enumerate(nombres):
        v = [abl[g][n][f"logit{pred}-logit{real}"] for g in GRUPOS]; bars = ax.bar(np.arange(3) + (i - 1) * W, v, W, label=f"grupo ← {n}", color=["#52514e", EV.AZU, EV.NAR][i])
        for b_, g in zip(bars, GRUPOS):
            ax.text(b_.get_x() + W / 2, b_.get_height(), str(abl[g][n]["predicho"]), ha="center", va="bottom" if b_.get_height() >= 0 else "top", fontsize=7)
    ax.axhline(L[pred] - L[real], color="#9a9a9a", ls="--", lw=1, label="sin tocar"); ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(GRUPOS)); ax.set_ylabel(f"logit {pred} − logit {real}"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("ABLACIÓN: sustituir UN grupo por una media\n(número = qué lee entonces; < 0 = ya lee el real)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(G[0, 6:8]); a, b = GRUPOS["rizo"]; et = ["rizo ↖", "rizo ↗", "rizo ↙", "rizo ↘", "nº rizos"]
    ax.bar(np.arange(5) - W, x[a:b], W, color="k", label="este dígito"); ax.bar(np.arange(5), medias[real][a:b], W, color=EV.AZU, label=f"media de los {real}")
    ax.bar(np.arange(5) + W, medias[pred][a:b], W, color=EV.NAR, label=f"media de los {pred}"); ax.set_xticks(range(5)); ax.set_xticklabels(et, fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.set_title("la feature del RIZO (2×2 max del mapa + cuenta)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    # filas 1–2: atribución espacial, rectas (8) y gris (6) + total por grupo
    def celda_a_mapa(v):
        return np.kron(v.reshape(8, 8), np.ones((4, 4)))
    tope = float(np.abs(at[:896]).max()) + 1e-9
    for q in range(8):
        ax = fig.add_subplot(G[1, q]); pintar_digito(ax, img[args.idx]); m = celda_a_mapa(at[q * 64:(q + 1) * 64])
        ax.imshow(m, cmap="RdBu_r", vmin=-tope, vmax=tope, alpha=0.75); ax.set_title(f"{RECTAS_N[q]}\nΣ {at[q*64:(q+1)*64].sum():+.2f}", fontsize=8)
    for q in range(6):
        ax = fig.add_subplot(G[2, q]); pintar_digito(ax, img[args.idx]); m = celda_a_mapa(at[512 + q * 64:512 + (q + 1) * 64])
        ax.imshow(m, cmap="RdBu_r", vmin=-tope, vmax=tope, alpha=0.75); ax.set_title(f"gris: {GRIS_N[q]}\nΣ {at[512+q*64:512+(q+1)*64].sum():+.2f}", fontsize=8)
    ax = fig.add_subplot(G[2, 6:8]); tot = [out["atribucion_por_grupo"][g] for g in GRUPOS]
    ax.bar(range(3), tot, color=[EV.NAR if t > 0 else EV.AZU for t in tot]); ax.set_xticks(range(3)); ax.set_xticklabels(list(GRUPOS)); ax.axhline(0, color="k", lw=.6)
    ax.set_title(f"ATRIBUCIÓN (grad × entrada) de logit {pred} − logit {real}, por grupo\nrojo = empuja hacia {pred} · azul = hacia {real}", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    # fila 3: vecinos
    for q, k in enumerate(orden):
        ax = fig.add_subplot(G[3, q]); pintar_digito(ax, img[tr[k]])
        for c in DV.filtrar(cache[("tr", (0, 0))][k], *DV.VARIANTES["rizo (h0, 8px)"]):
            ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU, ms=7, mew=1.5)
        ax.set_title(f"vecino {q+1}: clase {y[tr][k]} · idx {tr[k]}\ndist {dist[k]:.1f}", fontsize=8, color=EV.NAR if y[tr][k] == pred else (EV.AZU if y[tr][k] == real else "k"))
    ax = fig.add_subplot(G[3, 6:8]); ax.bar(range(10), [dist8[k] for k in range(10)], color=cols); ax.set_xticks(range(10))
    ax.set_title("distancia media a los 18 de cada clase en train\n(features estandarizadas)", fontsize=9); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    fig.suptitle(f"rizo-2 · 20 · RASTREO del idx {args.idx}: por qué el mejor compositor (MLP 256 · rectas + gris + rizo, con aumento; semilla {args.semilla}) "
                 f"lee el {real} como {pred}", fontsize=11)
    fig.savefig(AQUI / "resultados" / f"20-rastrear-{args.idx}.png", dpi=100, bbox_inches="tight")
    print(f"→ resultados/20-rastrear-{args.idx}.png · 20-rastrear-{args.idx}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
