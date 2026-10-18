#!/usr/bin/env python3
"""rizo-2 § 120 — de vuelta a 32 px de ALTO con el MISMO TRAZO: se escala la GEOMETRÍA, no los píxeles (el dueño, 2026-10-18: «el
agrandamiento permite estandarizar los reconocimientos… necesitamos una forma de escalarlos de vuelta a una altura de 32 px sin que se
pongan más anchos y sin introducir artifacts»).

Agrandar la IMAGEN engorda el trazo (es justo lo que hizo NIST, § 117). Aquí se agranda la LÍNEA MEDIA y se vuelve a dibujar:

  1. la cresta del dígito de MISMO TRAZO (§ 118) según el adelgazador gris k3x3-g0.7 (§ 119): sus tramos y ciclos (polilíneas
     sub-píxel de `cresta_curvas.tramos`), suavizados a lo largo de la curva (gaussiana de SIGMA_CURVA muestras: al agrandar, un
     zigzag de un píxel también se agranda); en cada nudo, cada final de tramo se une al centro del nudo (los tramos acaban en
     píxeles distintos del mismo racimo y dejarían huecos);
  2. se ESCALA uniformemente para que el dígito dibujado mida ALTO px de alto (alto de la cresta × k + PLUMA = ALTO), sin pasar de
     ALTO de ancho, y se centra por su caja;
  3. se DIBUJA con un bolígrafo de PLUMA px fijo y borde suave: gris = ½ · (1 − erf((d − PLUMA/2) / (BORDE·√2))), con d la distancia
     exacta de cada píxel a la polilínea. El trazo no depende de la escala y no hay escalones (no se remuestrea nada).

Los delgados los produce `delgado-cnn` (`exportar_delgados.py --pesos k3x3-g0.7 --dataset uci-optdigits-orig-32px-trazo4-r20261018`)
y aquí se leen como dato.

    ../.venv/bin/python nn/dataset_alto32.py [--muestra] [--publicar]   → resultados/120-alto32.{png,json}; con --publicar, el dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, special
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
from expcnn import exigir_dataset, ruta_datos                             # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

ORIGEN = "uci-optdigits-orig-32px-trazo4-r20261018"
NIST = "uci-optdigits-orig-32px-r20261005"
DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / f"delgados-k3x3-g0.7-{ORIGEN}.npz"
ALTO, PLUMA, BORDE, SIGMA_CURVA = 32, 4.0, 0.7, 1.0
AVISO_G07 = """
⚠ **Defecto CONOCIDO, publicado así por orden del dueño (2026-10-18)**: la línea media sale del adelgazador gris `k3x3-g0.7`, que
**corta la cresta en los cruces**: ~240 sietes pierden la BARRA (7 con algún nudo: 45 %, contra 87 % en NIST) y parte de los 4 pierden
el cruce (78 % contra 95 %). Lo que no está en la línea media no se dibuja. Detalle en rizo-2 § 121.
"""
N = 32


def _suave(p, cerrado):
    if len(p) < 4:
        return p
    return np.stack([ndimage.gaussian_filter1d(p[:, k], SIGMA_CURVA, mode="wrap" if cerrado else "nearest") for k in (0, 1)], 1)


def segmentos(M) -> np.ndarray:
    """(S, 2, 2) segmentos de la línea media de M (coordenadas x, y), con los nudos cerrados."""
    tr, puros = CC.tramos(M); seg = []; finales = {}
    for t in tr:
        p = _suave(np.asarray(t["pts"], float), False)
        seg += [(p[j], p[j + 1]) for j in range(len(p) - 1)]
        for r, q in ((t["a"], p[0]), (t["b"], p[-1])):
            if r > 0:
                finales.setdefault(r, []).append(q)
    for c in puros:
        p = _suave(np.asarray(c["pts"], float), True); seg += [(p[j], p[(j + 1) % len(p)]) for j in range(len(p))]
    for qs in finales.values():                                          # cada final de tramo, al centro de su nudo
        if len(qs) > 1:
            m = np.mean(qs, 0); seg += [(q, m) for q in qs]
    return np.array(seg, float).reshape(-1, 2, 2)


def dibujar(S, k, desp) -> np.ndarray:
    """La polilínea S escalada por k y desplazada, dibujada con PLUMA y borde suave en N×N."""
    A = S[:, 0] * k + desp; B = S[:, 1] * k + desp
    yy, xx = np.mgrid[0:N, 0:N] + 0.5; P = np.stack([xx.ravel(), yy.ravel()], 1)[:, None, :]
    AB = (B - A)[None]; t = np.clip(((P - A[None]) * AB).sum(-1) / ((AB ** 2).sum(-1) + 1e-9), 0, 1)
    d = np.sqrt((((A[None] + t[..., None] * AB) - P) ** 2).sum(-1)).min(1)
    return (0.5 * (1 - special.erf((d - PLUMA / 2) / (BORDE * np.sqrt(2))))).reshape(N, N).astype(np.float32)


def linea_exacta(S, k, desp, sigma=0.8) -> np.ndarray:
    """El DELGADO EXACTO del dígito dibujado: exp(−d²/2σ²) con d la distancia a la MISMA polilínea con la que se dibujó (la forma de la
    verdad de los sintéticos de delgado-cnn, σ 0,8). El revisor lo usa como M de este dataset (§ 124): volver a adelgazar la imagen a
    32 px perdía parte de las barras del 7 (75 % con nudo, contra 89 % de la línea con que se dibujó; medido 2026-10-18)."""
    A = S[:, 0] * k + desp; B = S[:, 1] * k + desp
    yy, xx = np.mgrid[0:N, 0:N] + 0.5; P = np.stack([xx.ravel(), yy.ravel()], 1)[:, None, :]
    AB = (B - A)[None]; t = np.clip(((P - A[None]) * AB).sum(-1) / ((AB ** 2).sum(-1) + 1e-9), 0, 1)
    d = np.sqrt((((A[None] + t[..., None] * AB) - P) ** 2).sum(-1)).min(1)
    return np.exp(-d ** 2 / (2 * sigma ** 2)).reshape(N, N).astype(np.float32)


def alto32(M, exacta=False):
    """(imagen, k) o (None, nan) si no hay cresta. Con `exacta`, (imagen, k, delgado exacto)."""
    S = segmentos(M)
    if not len(S):
        return None, float("nan")
    P = S.reshape(-1, 2); lo, hi = P.min(0), P.max(0); h, w = hi[1] - lo[1], hi[0] - lo[0]
    k = (ALTO - PLUMA) / max(h, 1e-3)
    if w * k + PLUMA > N:                                                # sin pasar de N de ancho
        k = (N - PLUMA) / max(w, 1e-3)
    desp = np.array([N / 2, N / 2]) - (lo + hi) / 2 * k
    if exacta:
        return dibujar(S, k, desp), float(k), linea_exacta(S, k, desp)
    return dibujar(S, k, desp), float(k)


def grosor(t):
    t = t > 0.5
    if not t.any():
        return float("nan")
    dt = ndimage.distance_transform_edt(t); sk = skeletonize(t); return 2 * float(dt[sk].mean()) if sk.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--muestra", action="store_true"); ap.add_argument("--publicar", action="store_true")
    ap.add_argument("--delgado", default=DELG.name, help="fichero de delgado-cnn/resultados (§ 122: el de 64 px)")
    ap.add_argument("--nombre", default="uci-optdigits-orig-32px-trazo4-alto32-r20261018"); ap.add_argument("--fig", default="120-alto32")
    a = ap.parse_args(); delg = DELG.parent / a.delgado
    o = np.load(exigir_dataset(ORIGEN) / "datos.npz"); nist = np.load(exigir_dataset(NIST) / "datos.npz")["imagenes"]
    M = np.load(delg)["M"].astype(np.float32); y = o["etiquetas"]; n = len(y)   # a 32·k px si se amplió: alto32 normaliza la escala
    MUESTRA = [736, 2849, 4661, 5538, 1888, 3517, 1668, 1552, 0, 2, 3, 4, 5, 6, 8, 9]
    ids = MUESTRA if a.muestra else range(n)
    X = np.zeros((n, N, N), np.float32); K = np.full(n, np.nan)
    for i in ids:
        img, k = alto32(M[i])
        if img is not None:
            X[i], K[i] = img, k
    sel = np.array(list(ids)); bin_ = X[sel] > 0.5
    alto = np.array([np.ptp(np.flatnonzero(b.any(1))) + 1 if b.any() else 0 for b in bin_]); ancho = np.array([np.ptp(np.flatnonzero(b.any(0))) + 1 if b.any() else 0 for b in bin_])
    G1 = np.array([grosor(X[i]) for i in sel]); G0 = np.array([grosor(nist[i]) for i in sel])
    res = {"alto": PLUMA and ALTO, "pluma": PLUMA, "borde": BORDE, "sigma_curva": SIGMA_CURVA, "n": int(len(sel)), "sin_cresta": int(np.isnan(K[sel]).sum()),
           "alto_32_pct": float(100 * np.mean(alto >= 31)), "ancho_med": float(np.median(ancho)), "limitados_por_ancho": int(np.sum(ancho >= 31)),
           "grosor_nist": {"p10": float(np.nanpercentile(G0, 10)), "p50": float(np.nanmedian(G0)), "p90": float(np.nanpercentile(G0, 90))},
           "grosor_nuevo": {"p10": float(np.nanpercentile(G1, 10)), "p50": float(np.nanmedian(G1)), "p90": float(np.nanpercentile(G1, 90))},
           "grosor_por_clase": {str(c): {"nist": float(np.nanmedian(G0[y[sel] == c])), "nuevo": float(np.nanmedian(G1[y[sel] == c]))} for c in range(10) if (y[sel] == c).any()}}
    print(json.dumps(res, ensure_ascii=False, indent=1))

    fig = plt.figure(figsize=(1.6 * len(MUESTRA) + 1, 9.5), layout="constrained"); axs = fig.subplots(4, len(MUESTRA))
    filas = [("NIST (binario)", lambda i: nist[i]), ("mismo trazo (§ 118)", lambda i: o["imagenes"][i].astype(np.float32)),
             (f"su cresta\n({delg.name.split('-uci')[0].replace('delgados-', '')})", None), ("NUEVO: 32 px de alto,\npluma 4 px", lambda i: X[i])]
    for f, (nombre, fn) in enumerate(filas):
        for c, i in enumerate(MUESTRA):
            ax = axs[f, c]; ax.set_xticks([]); ax.set_yticks([])
            if fn is None:
                ax.imshow(o["imagenes"][i].astype(np.float32), cmap="Greys", vmin=0, vmax=3, extent=(0, N, N, 0))
                for s in segmentos(M[i]) / (M.shape[-1] // N):              # a coordenadas de 32 px si el delgado es ampliado
                    ax.plot(s[:, 0], s[:, 1], "-", color="#9333ea", lw=1.2)
                ax.set_xlim(0, N); ax.set_ylim(N, 0)
            else:
                ax.imshow(fn(i), cmap="Greys", vmin=0, vmax=1.1, interpolation="nearest")
            if f == 0:
                ax.set_title(f"#{i} ({y[i]})", fontsize=9)
            if f == 3:
                g = grosor(X[i]); ax.set_xlabel(f"×{K[i]:.1f} · trazo {g:.1f} px", fontsize=8)
            if c == 0:
                ax.set_ylabel(nombre, fontsize=9, fontweight="bold")
    fig.suptitle("De vuelta a 32 px de alto con el MISMO trazo: se agranda la LÍNEA MEDIA y se vuelve a dibujar con una pluma de 4 px "
                 "(no se agranda la imagen)", fontsize=12)
    out = AQUI / "resultados" / (f"{a.fig}-muestra.png" if a.muestra else f"{a.fig}.png"); fig.savefig(out, dpi=90); print(out)
    if not a.muestra:
        out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if a.publicar and not a.muestra:
        nombre = a.nombre; dst = ruta_datos() / "experimentos-cnn" / nombre
        if (dst / "manifiesto.json").exists():
            sys.exit(f"✗ {nombre} ya está publicado y un dataset publicado no se reescribe.")
        dst.mkdir(parents=True, exist_ok=True); img = X.astype(np.float16)
        np.savez_compressed(dst / "datos.npz", imagenes=img, etiquetas=y, particion=o["particion"], origen=o["origen"], escala=K.astype(np.float32))
        hu = lambda v: hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
        (dst / "manifiesto.json").write_text(json.dumps({
            "nombre": nombre, "experimento": "rizo-2", "generado": "2026-10-18", "derivado_de": ORIGEN, "n": int(n),
            "generador": "experimentos-cnn/2026-10-11-rizo-dos/nn/dataset_alto32.py",
            "delgado": f"delgado-cnn/resultados/{delg.name}",
            "parametros": {"alto": ALTO, "pluma": PLUMA, "borde": BORDE, "sigma_curva": SIGMA_CURVA},
            "campos": {"imagenes": "(N,32,32) float16 gris [0,1]: la línea media escalada a 32 px de alto y dibujada con pluma fija",
                       "etiquetas": "= origen", "particion": "= origen", "origen": "= origen", "escala": "k aplicado a la línea media (NaN: sin cresta)"},
            "huellas": {"imagenes": hu(img), "etiquetas": hu(y)}}, ensure_ascii=False, indent=1))
        (dst / "README.md").write_text(f"""# `{nombre}`

Los 5620 dígitos de [`{ORIGEN}`](../{ORIGEN}/) (mismo trazo) **de vuelta a 32 px de alto sin engordar el trazo**: se agranda la línea
media (la cresta de `delgado-cnn/resultados/{delg.name}`) y se vuelve a dibujar con una pluma fija de {PLUMA:g} px y borde suave
(σ {BORDE:g}), en vez de agrandar la imagen. Mismo orden, etiquetas y partición. Gris [0, 1].

- Grosor medido sobre el resultado: mediana {res['grosor_nuevo']['p50']:.2f} px, 10–90 % {res['grosor_nuevo']['p10']:.2f}–{res['grosor_nuevo']['p90']:.2f}
  (NIST: {res['grosor_nist']['p10']:.2f}–{res['grosor_nist']['p90']:.2f}).
- {res['alto_32_pct']:.1f} % mide 32 px de alto; {res['limitados_por_ancho']} quedan limitados por el ancho (no caben a 32 de alto).
- Lo genera `experimentos-cnn/2026-10-11-rizo-dos/nn/dataset_alto32.py --delgado {delg.name} --nombre {nombre} --publicar`; detalle en rizo-2 § 120–122.
- Sin cresta (salen en blanco): {[int(i) for i in np.flatnonzero(np.isnan(K))]}.
{"" if delg.name != DELG.name else AVISO_G07}""")
        print("publicado en", dst)




def exportar_linea(delgado: str, dataset: str):
    """Guarda el delgado EXACTO de un dataset de este generador (§ 124) en delgado-cnn/resultados/delgados-exacto-<dataset>.npz,
    con el mismo formato que exportar_delgados.py (M float16). Comprueba que la imagen redibujada es la publicada."""
    M = np.load(DELG.parent / delgado)["M"].astype(np.float32); pub = np.load(exigir_dataset(dataset) / "datos.npz")
    L = np.zeros((len(M), N, N), np.float32); dif = 0.0
    for i in range(len(M)):
        r = alto32(M[i], exacta=True)
        if r[0] is not None:
            L[i] = r[2]; dif = max(dif, float(np.abs(r[0].astype(np.float16).astype(np.float32) - pub["imagenes"][i].astype(np.float32)).max()))
    out = DELG.parent / f"delgados-exacto-{dataset}.npz"
    np.savez_compressed(out, M=L.astype(np.float16), etiquetas=pub["etiquetas"], particion=pub["particion"], dataset=dataset, pesos=f"exacto:{delgado}")
    print(f"→ {out.name}  · diferencia máxima con las imágenes publicadas: {dif:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--exportar-linea":     # --exportar-linea <delgado de 64 px> <dataset publicado>
        exportar_linea(sys.argv[2], sys.argv[3])
    else:
        main()
