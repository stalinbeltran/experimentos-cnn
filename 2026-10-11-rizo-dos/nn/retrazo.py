#!/usr/bin/env python3
"""rizo-2 § 123 — VALIDAR el adelgazador RE-TRAZANDO: se vuelve a dibujar su línea media con el grosor del dígito y se compara con la
tinta de entrada (el dueño, 2026-10-18: «¿has probado a validar la eficacia del delgado volviendo a retrazar después de adelgazar, para
verificar que el adelgazamiento es confiable, y así darle feedback al aprendizaje?»). No hace falta ninguna verdad hecha a mano.

Por dígito: la línea media (`cresta_curvas.tramos` del delgado, con los nudos cerrados, como en § 120) se dibuja en la MISMA rejilla
con una pluma del grosor medido del propio dígito (§ 118: 2 × distancia media al borde sobre el esqueleto). Con tinta = entrada > 0,5 y
re-trazo = dibujo > 0,5:

  · SIN EXPLICAR = tinta que el re-trazo no cubre / tinta  → falta línea media (la barra perdida del 7)
  · INVENTADO    = re-trazo fuera de la tinta / re-trazo   → sobra línea media (una rama falsa)

Un adelgazador fiable tiene las dos cerca de 0. ⚠ Ni siquiera uno perfecto da 0: una pluma de grosor constante no reproduce un
trazo de grosor variable ni las puntas redondeadas de NIST, así que lo que importa es la COMPARACIÓN entre adelgazadores y los peores.

    ../.venv/bin/python nn/retrazo.py [delgado …]   → resultados/123-retrazo.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, special
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import dataset_alto32 as A32                                              # noqa: E402  (segmentos: la línea media con los nudos cerrados)
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"
NIST, T4 = "uci-optdigits-orig-32px-r20261005", "uci-optdigits-orig-32px-trazo4-r20261018"
VARIANTES = [("NIST + adelgazador binario (hoy)", NIST, "delgados-k3x3.npz", 1),
             ("mismo trazo + gris (§ 119)", T4, f"delgados-k3x3-g0.7-{T4}.npz", 1),
             ("mismo trazo + gris + dos trazos (§ 121)", T4, f"delgados-k3x3-g0.7x-{T4}.npz", 1),
             ("mismo trazo + gris a 64 px (§ 122)", T4, f"delgados-k3x3-g0.7x-64-{T4}-x2.npz", 2)]


def grosor(t):
    if not t.any():
        return float("nan")
    dt = ndimage.distance_transform_edt(t); sk = skeletonize(t); return 2 * float(dt[sk].mean()) if sk.any() else float("nan")


def retrazar(M, pluma, ampl=1, n=32):
    S = A32.segmentos(M.astype(np.float32)) / ampl                      # a coordenadas de 32 px
    if not len(S):
        return np.zeros((n, n), np.float32)
    A, B = S[:, 0], S[:, 1]; yy, xx = np.mgrid[0:n, 0:n] + 0.5; P = np.stack([xx.ravel(), yy.ravel()], 1)[:, None, :]
    AB = (B - A)[None]; t = np.clip(((P - A[None]) * AB).sum(-1) / ((AB ** 2).sum(-1) + 1e-9), 0, 1)
    d = np.sqrt((((A[None] + t[..., None] * AB) - P) ** 2).sum(-1)).min(1)
    return (0.5 * (1 - special.erf((d - pluma / 2) / (0.7 * np.sqrt(2))))).reshape(n, n).astype(np.float32)


def main():
    variantes = [v for v in VARIANTES if (DELG / v[2]).is_file()]
    datos = {ds: np.load(exigir_dataset(ds) / "datos.npz") for ds in {v[1] for v in variantes}}
    y = datos[NIST]["etiquetas"]; n = len(y); res = {}; por = {}
    for nombre, ds, f, ampl in variantes:
        img = datos[ds]["imagenes"].astype(np.float32); tin = img > 0.5; M = np.load(DELG / f)["M"]
        se, inv = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            g = grosor(tin[i])
            if np.isnan(g):
                continue
            r = retrazar(M[i], g, ampl) > 0.5
            se[i] = (tin[i] & ~r).sum() / max(tin[i].sum(), 1); inv[i] = (r & ~tin[i]).sum() / max(r.sum(), 1)
        por[nombre] = (se, inv)
        res[nombre] = {"sin_explicar_med": float(np.nanmedian(se)), "inventado_med": float(np.nanmedian(inv)),
                       "sin_explicar_p90": float(np.nanpercentile(se, 90)), "inventado_p90": float(np.nanpercentile(inv, 90)),
                       "por_clase": {str(c): {"sin_explicar": float(np.nanmedian(se[y == c])), "inventado": float(np.nanmedian(inv[y == c]))} for c in range(10)}}
        print(f"{nombre:42s} sin explicar {100*res[nombre]['sin_explicar_med']:.1f} % (p90 {100*res[nombre]['sin_explicar_p90']:.0f}) · inventado {100*res[nombre]['inventado_med']:.1f} % (p90 {100*res[nombre]['inventado_p90']:.0f})"
              + " · 7: " + f"{100*res[nombre]['por_clase']['7']['sin_explicar']:.0f}/{100*res[nombre]['por_clase']['7']['inventado']:.0f}"
              + " · 1: " + f"{100*res[nombre]['por_clase']['1']['sin_explicar']:.0f}/{100*res[nombre]['por_clase']['1']['inventado']:.0f}", flush=True)

    nv = len(variantes); fig = plt.figure(figsize=(18, 4.2 + 2.3 * nv), layout="constrained")
    arriba, abajo = fig.subfigures(2, 1, height_ratios=[4.2, 2.3 * nv]); a, b = arriba.subplots(1, 2); x = np.arange(10); w = 0.8 / nv
    cv = ["#999", "#c4b5fd", "#9333ea", "#1b7f3b"]
    for k, (nombre, *_r) in enumerate(variantes):
        a.bar(x + (k - nv / 2 + 0.5) * w, [100 * res[nombre]["por_clase"][str(c)]["sin_explicar"] for c in x], w, color=cv[k], label=nombre)
        b.bar(x + (k - nv / 2 + 0.5) * w, [100 * res[nombre]["por_clase"][str(c)]["inventado"] for c in x], w, color=cv[k], label=nombre)
    a.set_xticks(x); a.set_ylabel("% de la tinta (mediana)"); a.legend(fontsize=8); a.set_title("SIN EXPLICAR: tinta que el re-trazo no cubre (falta línea media)", fontsize=10)
    b.set_xticks(x); b.set_ylabel("% del re-trazo (mediana)"); b.set_title("INVENTADO: re-trazo fuera de la tinta (sobra línea media)", fontsize=10)
    # los 7 peor explicados por cada variante: tinta gris, sin explicar en rojo, inventado en azul
    axs = abajo.subplots(nv, 10)
    for f_, (nombre, ds, fich, ampl) in enumerate(variantes):
        img = datos[ds]["imagenes"].astype(np.float32); tin = img > 0.5; M = np.load(DELG / fich)["M"]; se = por[nombre][0]
        sie = np.flatnonzero(y == 7); peores = sie[np.argsort(-np.nan_to_num(se[sie]))][:10]
        for k, i in enumerate(peores):
            r = retrazar(M[i], grosor(tin[i]), ampl) > 0.5; rgb = np.ones((32, 32, 3))
            rgb[tin[i] & r] = (0.7, 0.7, 0.7); rgb[tin[i] & ~r] = (0.85, 0.15, 0.15); rgb[r & ~tin[i]] = (0.2, 0.45, 0.9)
            ax = axs[f_, k]; ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i} · {100*se[i]:.0f} %", fontsize=8)
            if k == 0:
                ax.set_ylabel(nombre.replace(" + ", "\n+ "), fontsize=7, fontweight="bold")
    fig.suptitle("VALIDAR el adelgazador RE-TRAZANDO su línea media con el grosor del dígito · abajo: los 10 sietes PEOR explicados de cada uno "
                 "(gris = explicado · ROJO = tinta sin línea media · AZUL = trazo inventado)", fontsize=12)
    out = AQUI / "resultados" / "123-retrazo.png"; fig.savefig(out, dpi=85); out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
