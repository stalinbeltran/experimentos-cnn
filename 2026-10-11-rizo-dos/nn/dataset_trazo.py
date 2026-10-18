#!/usr/bin/env python3
"""rizo-2 § 118 — un DATASET APARTE con el MISMO TRAZO en todos los dígitos (el dueño, 2026-10-18: «el objetivo es hallar el grosor
real de cada dígito… para un humano esos 1l se ven demasiado extraños. Ajústalos, proporcionalmente, hasta que todos tengan el mismo
trazo. Guárdalo como un dataset aparte. Suavízalos si es necesario (la binarización introduce artifacts). Muéstrame los resultados»).

De dónde sale (§ 117): NIST llevó el ALTO de cada dígito a 32 px con un escalado uniforme, así que el grosor del trazo dice cuánto se
agrandó cada uno (los 1 se escribieron a ~70 % del alto típico y quedaron gruesos). Aquí se deshace: cada dígito se ENCOGE
uniformemente con f = T / grosor, de modo que su trazo pase a medir T px.

  · GROSOR = 2 × la distancia media al borde de la tinta sobre su esqueleto (skimage). Autocontenido: sólo necesita el dataset de
    origen (Spearman 0,974 con el de § 116, que usa la cresta de la CNN; medido 2026-10-18).
  · T = 4,0 px por defecto: el 2,9 % de los dígitos es más fino y NO puede llegar (en 32×32 no cabe agrandar algo que ya mide 32 de
    alto); se quedan a su tamaño (f = 1) y se marcan en `no_llega`. Con T = 4,5 serían el 14 %; con 5,0, el 34 %.
  · SUAVIZADO, sin binarizar: la tinta pasa por una gaussiana de σ = max(SIGMA_MIN, 0,5 / f) px de origen (quita los escalones de la
    binarización y, al encoger, hace de antialias) y se remuestrea con interpolación lineal. Salida en gris [0, 1].
  · CENTRADO: el centro de la caja de la tinta va al centro de la imagen.
  · COMPROBACIÓN: el grosor se vuelve a medir sobre la salida (umbral 0,5) y se guarda en `grosor_final`.

    ../.venv/bin/python nn/dataset_trazo.py [--T 4.0] [--publicar]
        → resultados/118-trazo-comun.{png,json}; con --publicar, el dataset en foveal-vision-data/experimentos-cnn/<NOMBRE>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parent))
from expcnn import exigir_dataset, ruta_datos                             # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

ORIGEN = "uci-optdigits-orig-32px-r20261005"
SIGMA_MIN = 0.8
N = 32


def grosor(t: np.ndarray) -> float:
    """2 × distancia media al borde sobre el esqueleto de la tinta binaria t; NaN sin tinta."""
    t = t > 0
    if not t.any():
        return float("nan")
    dt = ndimage.distance_transform_edt(t); sk = skeletonize(t)
    return 2 * float(dt[sk].mean()) if sk.any() else float("nan")


def ajustar(t: np.ndarray, f: float) -> np.ndarray:
    """La tinta binaria t, suavizada y encogida por f (≤ 1) alrededor del centro de su caja, en una imagen N×N gris [0, 1]."""
    s = ndimage.gaussian_filter(t.astype(np.float32), max(SIGMA_MIN, 0.5 / f), mode="constant")
    fil, col = np.flatnonzero(t.any(1)), np.flatnonzero(t.any(0))
    cy, cx = (fil[0] + fil[-1]) / 2, (col[0] + col[-1]) / 2; c = (N - 1) / 2
    rr, cc = np.mgrid[0:N, 0:N].astype(np.float32)
    return np.clip(ndimage.map_coordinates(s, [(rr - c) / f + cy, (cc - c) / f + cx], order=1, mode="constant"), 0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=float, default=4.0); ap.add_argument("--publicar", action="store_true")
    a = ap.parse_args()
    src = exigir_dataset(ORIGEN); d = np.load(src / "datos.npz"); T0 = d["imagenes"] > 0; y = d["etiquetas"]; n = len(y)
    G0 = np.array([grosor(t) for t in T0]); f = np.minimum(1.0, a.T / G0); no = G0 < a.T
    X = np.stack([ajustar(T0[i], f[i]) for i in range(n)]); G1 = np.array([grosor(x > 0.5) for x in X])
    alto = np.array([np.ptp(np.flatnonzero((x > 0.5).any(1))) + 1 for x in X])
    nombre = f"uci-optdigits-orig-32px-trazo{a.T:g}-r20261018"
    res = {"nombre": nombre, "T": a.T, "sigma_min": SIGMA_MIN, "no_llegan": int(no.sum()), "no_llegan_pct": round(100 * no.mean(), 1),
           "grosor_antes": {"p10": float(np.nanpercentile(G0, 10)), "p50": float(np.nanmedian(G0)), "p90": float(np.nanpercentile(G0, 90))},
           "grosor_despues": {"p10": float(np.nanpercentile(G1, 10)), "p50": float(np.nanmedian(G1)), "p90": float(np.nanpercentile(G1, 90))},
           "por_clase": {str(c): {"grosor_antes": float(np.nanmedian(G0[y == c])), "grosor_despues": float(np.nanmedian(G1[y == c])),
                                  "alto_despues": float(np.median(alto[y == c])), "escala": float(np.median(f[y == c]))} for c in range(10)}}
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # figura: grosor antes/después y una muestra antes/después
    fig = plt.figure(figsize=(17, 10.5), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[1, 0.72])
    h1, h2 = arriba.subplots(1, 2, width_ratios=[1.3, 1]); bins = np.linspace(2.5, 14, 47)
    h1.hist(G0[~np.isnan(G0)], bins, color="#bbb", label=f"antes (10–90 %: {res['grosor_antes']['p10']:.1f}–{res['grosor_antes']['p90']:.1f} px)")
    h1.hist(G1[~np.isnan(G1)], bins, color="#9333ea", alpha=0.7, label=f"después (10–90 %: {res['grosor_despues']['p10']:.1f}–{res['grosor_despues']['p90']:.1f} px)")
    h1.axvline(a.T, color="#c2410c", ls="--"); h1.set_xlabel("grosor del trazo (px)"); h1.set_ylabel("dígitos"); h1.legend(fontsize=9)
    h1.set_title(f"el TRAZO, antes y después (T = {a.T:g} px; {res['no_llegan_pct']} % más finos que T se quedan como estaban)", fontsize=10)
    x = np.arange(10); w = 0.38
    h2.bar(x - w / 2, [res["por_clase"][str(c)]["grosor_antes"] for c in x], w, color="#bbb", label="grosor antes")
    h2.bar(x + w / 2, [res["por_clase"][str(c)]["grosor_despues"] for c in x], w, color="#9333ea", label="grosor después")
    h3 = h2.twinx(); h3.plot(x, [res["por_clase"][str(c)]["alto_despues"] for c in x], "o-", color="#c2410c", label="alto después (px)")
    h3.set_ylim(0, 34); h3.set_ylabel("alto después (px)", color="#c2410c"); h2.set_xticks(x); h2.set_ylabel("grosor (px, mediana)")
    h2.legend(fontsize=8, loc="upper left"); h2.set_title("por clase: el grosor se iguala; el alto de los 1 baja", fontsize=10)
    va = np.flatnonzero(d["particion"] == "val")
    muestra = [736, 2849, 4661, 5538, 1888, 1552] + [int(va[y[va] == c][k]) for c in (0, 2, 3, 4, 5, 6, 8, 9) for k in (0,)]
    axs = abajo.subplots(2, len(muestra))
    for k, i in enumerate(muestra):
        axs[0, k].imshow(T0[i], cmap="Greys", vmin=0, vmax=1.15, interpolation="nearest")
        axs[1, k].imshow(X[i], cmap="Greys", vmin=0, vmax=1.15, interpolation="nearest")
        axs[0, k].set_title(f"#{i} ({y[i]})\ngrosor {G0[i]:.1f} px", fontsize=8); axs[1, k].set_title(f"→ {G1[i]:.1f} px · ×{f[i]:.2f}", fontsize=8)
        for r in (0, 1):
            axs[r, k].set_xticks([]); axs[r, k].set_yticks([])
    axs[0, 0].set_ylabel("ANTES\n(NIST, binario)", fontsize=10, fontweight="bold"); axs[1, 0].set_ylabel("DESPUÉS\n(mismo trazo, gris)", fontsize=10, fontweight="bold")
    fig.suptitle(f"Dataset con el MISMO TRAZO: cada dígito encogido proporcionalmente hasta que su trazo mide {a.T:g} px, y suavizado (sin binarizar)", fontsize=13)
    out = AQUI / "resultados" / "118-trazo-comun.png"; fig.savefig(out, dpi=90); out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(out)

    if a.publicar:
        raiz = ruta_datos(); dst = raiz / "experimentos-cnn" / nombre
        if (dst / "manifiesto.json").exists():                   # regla del repo de datos: dato nuevo = nombre nuevo; nunca se pisa
            sys.exit(f"✗ {nombre} ya está publicado y un dataset publicado no se reescribe. Cambia --T o la fecha del nombre.")
        dst.mkdir(parents=True, exist_ok=True)
        img = X.astype(np.float16)
        np.savez_compressed(dst / "datos.npz", imagenes=img, etiquetas=y, particion=d["particion"], origen=d["origen"],
                            escala=f.astype(np.float32), grosor_original=G0.astype(np.float32), grosor_final=G1.astype(np.float32), no_llega=no)
        hu = lambda v: hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
        man = {"nombre": nombre, "experimento": "rizo-2", "generado": "2026-10-18", "derivado_de": ORIGEN, "n": int(n),
               "generador": "experimentos-cnn/2026-10-11-rizo-dos/nn/dataset_trazo.py", "parametros": {"T": a.T, "sigma_min": SIGMA_MIN},
               "campos": {"imagenes": "(N,32,32) float16 gris [0,1]: suavizado y encogido por `escala`, centrado por su caja",
                          "etiquetas": "0..9 (= origen)", "particion": "= origen", "origen": "= origen",
                          "escala": "f = min(1, T / grosor_original)", "grosor_original": "px, 2 × dist. media al borde sobre el esqueleto",
                          "grosor_final": "px, lo mismo medido sobre imagenes > 0,5", "no_llega": "más fino que T: no se encogió (f = 1)"},
               "huellas": {"imagenes": hu(img), "etiquetas": hu(y)}}
        (dst / "manifiesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
        (dst / "README.md").write_text(f"""# `{nombre}`

Los 5620 dígitos de [`{ORIGEN}`](../{ORIGEN}/) con el **mismo trazo**: cada uno se encogió proporcionalmente (f = T / grosor) hasta
que su trazo mide **T = {a.T:g} px**, y se suavizó en vez de binarizarse (gris [0, 1]). Mismo orden, etiquetas y partición que el origen.

- Por qué (rizo-2 § 117): NIST llevó el ALTO de cada dígito a 32 px con un escalado uniforme, así que los dígitos escritos pequeños
  (sobre todo los 1, a ~70 % del alto típico) quedaron con el trazo grueso. Aquí se deshace.
- {res['no_llegan']} dígitos ({res['no_llegan_pct']} %) son más finos que T y se quedan a su tamaño (`no_llega`).
- Grosor medido sobre el resultado: mediana {res['grosor_despues']['p50']:.2f} px, 10–90 % {res['grosor_despues']['p10']:.2f}–{res['grosor_despues']['p90']:.2f}
  (antes {res['grosor_antes']['p10']:.2f}–{res['grosor_antes']['p90']:.2f}).
- Lo genera `experimentos-cnn/2026-10-11-rizo-dos/nn/dataset_trazo.py --T {a.T:g} --publicar`; detalle en rizo-2 § 118.
""")
        print("publicado en", dst)


if __name__ == "__main__":
    main()
