#!/usr/bin/env python3
"""rizo-2 — TODO OBJETOS (118 números: rectas-objeto ×1 + gris-objetos + huecos cerrados y concavidades APARTE) sobre una
ENTRADA ALTERNATIVA: un .npz con un mapa por dígito (5620×32×32, mismo orden que datos.npz) que se binariza con --umbral.
Sirve para medir qué les pasa a estos detectores, TAL COMO ESTÁN calibrados (tinta de 4–6 px), cuando el dígito llega
adelgazado por otro experimento. Sin --entrada corre sobre la tinta original: es la referencia con el MISMO protocolo.

    python nn/todo_objetos_entrada.py --nombre tinta
    python nn/todo_objetos_entrada.py --nombre delgado-k3x3 --entrada /ruta/delgados-k3x3.npz --clave M --umbral 0.5 [--engrosar 2] [--var "medio (h1.5, 14px)"]
        → resultados/40-entrada-<nombre>.json · 40-entrada-<nombre>-muestra.png
    --var v       variante de candidatos de rizo para los huecos (por defecto la de rizo_huecos, rayos de 8 px)
    --engrosar r  dilata la entrada binaria con un disco de radio r ANTES de los detectores (devolver un grosor fijo
                  a un trazo de 1 px es la forma más barata de «ajustar» detectores calibrados para 4–6 px).

Protocolo: 180 de train + 12 copias movidas (±1,±2,±3 px), MLP 256, 3 semillas, val (1617) y ciega (3823).
Los candidatos de rizo (compuesto2) se calculan para la entrada y se cachean en resultados/40-candidatos-<nombre>.pkl
(~0,25 s por imagen: ~35 min la primera vez).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

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
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

GRUPOS = {"rectas-objeto ×1": (0, 44), "gris-objetos": (44, 98), "huecos+concavidades": (98, 118)}


def features(img, C):
    return np.concatenate([RO.rectas_objetos(img, (0,)), GO.gris_objetos(img), RH.huecos_separados(img, C)], 1)


def engrosar(img, r):
    if not r:
        return img
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disco = (xx**2 + yy**2) <= r * r + 0.25
    return np.stack([ndimage.binary_dilation(x > 0, disco) for x in img]).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", required=True); ap.add_argument("--entrada", default=None); ap.add_argument("--clave", default="M")
    ap.add_argument("--umbral", type=float, default=0.5); ap.add_argument("--engrosar", type=int, default=0)
    ap.add_argument("--var", default=RH.VAR, help=f"variante de candidatos para los huecos: {list(DV.VARIANTES)}")
    a = ap.parse_args(); RH.VAR = a.var
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); y, part = d["etiquetas"].astype(int), d["particion"]
    if a.entrada:
        z = np.load(a.entrada); img = (z[a.clave].astype(np.float32) > a.umbral).astype(np.uint8)
        assert img.shape == d["imagenes"].shape and (z["etiquetas"] == d["etiquetas"]).all(), "la entrada no casa con el dataset"
    else:
        img = (d["imagenes"] > 0).astype(np.uint8)
    img = engrosar(img, a.engrosar)
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    cache_f = AQUI / "resultados" / f"40-candidatos-{a.nombre}.pkl"
    if cache_f.exists():
        cache = pickle.loads(cache_f.read_bytes())
    elif not a.entrada and not a.engrosar and R.CACHE.exists() and DV.CACHE.exists():   # tinta original con cachés (no viajan en git)
        cache = pickle.loads(R.CACHE.read_bytes()); cache["todos"] = pickle.loads(DV.CACHE.read_bytes())
    else:
        cache = {"todos": [DV.candidatos(x) for x in img]}
        for m in aum[1:]:
            cache[("tr", m)] = [DV.candidatos(x) for x in R.mover(img[tr], *m)]
        cache_f.write_bytes(pickle.dumps(cache)); print(f"candidatos listos ({time.time() - t0:.0f} s)", flush=True)
    todos = cache["todos"]; cache[("tr", (0, 0))] = [todos[i] for i in tr]
    X = np.concatenate([features(R.mover(img[tr], *m), cache[("tr", m)]) for m in aum]); Y = np.concatenate([y[tr]] * 13)
    Xva, Xex = features(img[va], [todos[i] for i in va]), features(img[ex], [todos[i] for i in ex]); print(f"features listas: {X.shape[1]} ({time.time() - t0:.0f} s)", flush=True)
    r = {"val": [], "ciega": [], "cuota": []}; preds = []
    for sem in D.SEMILLAS:
        red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); wt = (W ** 2).sum()
        pv = RA.logits(red, mu, sd, Xva).argmax(1); pe = RA.logits(red, mu, sd, Xex).argmax(1); preds.append(pv)
        r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean()))
        r["cuota"].append({g: float((W[:, b0:b1] ** 2).sum() / wt) for g, (b0, b1) in GRUPOS.items()})
    p0 = preds[0]; veces = np.sum([p != y[va] for p in preds], 0); f = np.flatnonzero(p0 != y[va])
    conf = {}
    for j in f:
        k = f"{y[va][j]}→{p0[j]}"; conf[k] = conf.get(k, 0) + 1
    # cuántos objetos ve cada grupo, de media por dígito de val (para leer el efecto del adelgazado sin el compositor)
    nobj = {"rectas": float(np.mean([len(RO.objetos(x)[0]) for x in img[va]])),
            "huecos cerrados": float(np.mean(Xva[:, 98 + 9])), "concavidades": float(np.mean(Xva[:, 98 + 19]))}
    out = {"nombre": a.nombre, "entrada": Path(a.entrada).name if a.entrada else None, "umbral": a.umbral, "engrosar": a.engrosar, "var": a.var, "n": int(X.shape[1]),
           "grosor_medio_px": float(np.mean([2 * np.percentile(ndimage.distance_transform_edt(x)[x > 0], 75) for x in img[va] if x.any()])),
           "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4),
           "cuota_peso": {g: round(float(np.mean([c[g] for c in r["cuota"]])), 3) for g in GRUPOS}, "objetos_por_digito_val": nobj,
           "fallos_s0": int(len(f)), "fallan_en_las_3": int((veces == 3).sum()), "confusiones": dict(sorted(conf.items(), key=lambda z: -z[1])),
           "segundos": round(time.time() - t0), "fallos": [{"idx": int(va[j]), "real": int(y[va][j]), "predicho": int(p0[j]), "semillas_que_fallan": int(veces[j])} for j in f]}
    (AQUI / "resultados" / f"40-entrada-{a.nombre}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "fallos"}, ensure_ascii=False))
    # muestra: 12 dígitos de val con lo que ven los detectores (misma figura que todo_objetos)
    import todo_objetos as TO
    ids = va[np.linspace(0, len(va) - 1, 12).astype(int)]
    fig, axs = plt.subplots(2, 6, figsize=(6 * 1.9, 2 * 2.15 + 0.9))
    for ax, i in zip(axs.flat, ids):
        ax.axis("off"); TO.dibujar_objetos(ax, img[i], todos[i]); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
        ax.set_title(f"{y[i]} · idx {i}", fontsize=7.5, pad=2)
    fig.suptitle(f"rizo-2 · 40 · todo objetos sobre entrada «{a.nombre}» (umbral {a.umbral}, engrosar {a.engrosar}): val {out['val']:.4f} · ciega {out['ciega']:.4f}\n"
                 "segmentos = rectas · círculos = curvas del gris por sentido · × = golpe · + = agujero", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / f"40-entrada-{a.nombre}-muestra.png", dpi=100)
    print(f"→ resultados/40-entrada-{a.nombre}.json · 40-entrada-{a.nombre}-muestra.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
