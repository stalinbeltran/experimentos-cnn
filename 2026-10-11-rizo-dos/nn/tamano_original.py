#!/usr/bin/env python3
"""rizo-2 § 117 — ¿de qué TAMAÑO eran originalmente los dígitos? (el dueño, 2026-10-18: «sospecho que estos dígitos fueron escalados
distinto que el resto, lo que me dice que originalmente fueron dígitos muy cortos; para estandarizar sus tamaños fueron agrandados y
eso los dejó así. ¿Podrías hacer unas pruebas para ver su verdadero tamaño?»).

Tres pruebas sobre la tinta 32×32 (no hay otra; el dataset no guarda el escritor):
  1. CÓMO se normalizó: alto y ancho de la caja de la tinta por clase.
  2. Si el escalado fue IGUAL en las dos direcciones: grosor de los trazos casi verticales medido a lo ancho contra el de los casi
     horizontales medido a lo alto, en el mismo dígito. Con un estiramiento sólo horizontal, la razón crecería con lo estrecho que era.
  3. El TAMAÑO ORIGINAL estimado: con el alto llevado a 32 px por un factor uniforme s = 32 / alto_original, el grosor del trazo es
     s × bolígrafo. El bolígrafo es del ESCRITOR, no del dígito, y todos escribieron todas las clases: un exceso de grosor que sea de
     UNA clase sólo puede venir de s. Alto original relativo = grosor de referencia (mediana de todo el dataset) / grosor del dígito.
     ⚠ Es relativo (no hay unidades del papel) y supone el mismo bolígrafo para todas las clases; dígito a dígito mezcla el
     bolígrafo de cada escritor con su tamaño.

Grosor = el de § 116 (resultados/116-gruesos.json). Delgado de siempre.

    ../.venv/bin/python nn/tamano_original.py   → resultados/117-tamano-original.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web"))
import cresta_curvas as CC                                               # noqa: E402
import grupos as GR                                                      # noqa: E402
import servidor as S                                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402


def caja(t):
    f, c = np.flatnonzero(t.any(1)), np.flatnonzero(t.any(0))
    return (np.ptp(f) + 1 if len(f) else 0), (np.ptp(c) + 1 if len(c) else 0)


def corrida(t, y, x, dy, dx):
    k = 0
    for s in (1, -1):
        yy, xx = (y, x) if s == 1 else (y - dy, x - dx)
        while 0 <= yy < 32 and 0 <= xx < 32 and t[yy, xx]:
            k += 1; yy += s * dy; xx += s * dx
    return k


def anisotropia(t, M):
    """(grosor de los trazos casi verticales, medido a lo ANCHO; de los casi horizontales, a lo ALTO), medianas, o NaN."""
    H, V = [], []
    for tr in CC.tramos(M)[0]:
        p = np.asarray(tr["pts"], float)
        for j in range(2, len(p) - 2):
            v = p[j + 2] - p[j - 2]; a = abs(np.degrees(np.arctan2(v[1], v[0]))) % 180; y, x = int(p[j, 1]), int(p[j, 0])
            if not t[y, x]:
                continue
            if 70 <= a <= 110:
                H.append(corrida(t, y, x, 0, 1))
            elif a <= 20 or a >= 160:
                V.append(corrida(t, y, x, 1, 0))
    return (float(np.median(H)) if H else np.nan, float(np.median(V)) if V else np.nan)


def main():
    d = S.Datos(); n = len(d.y); T = d.tinta > 0; y = d.y
    J = json.loads((AQUI / "resultados" / "116-gruesos.json").read_text()); G = np.array([J["grosor"][str(i)] for i in range(n)])
    gru = np.zeros(n, bool); gru[J["gruesos"]] = True
    els = GR.elementos(d, "1", S.almacen_revisor()); l1 = np.zeros(n, bool); l1[[e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]] = True
    hw = np.array([caja(T[i]) for i in range(n)]); A = np.array([anisotropia(T[i], d.M[i]) for i in range(n)])
    ref = float(np.nanmedian(G)); tam = ref / G                          # alto original relativo (1 = el típico del dataset)
    ok = ~np.isnan(A).any(1); razon = A[:, 0] / A[:, 1]
    res = {"ref_grosor_px": ref, "alto_32_pct": float(100 * np.mean(hw[:, 0] == 32)), "alto_ge30_pct": float(100 * np.mean(hw[:, 0] >= 30)),
           "ancho_32_pct": float(100 * np.mean(hw[:, 1] == 32)),
           "por_clase": {str(c): {"alto_med": float(np.median(hw[y == c, 0])), "ancho_med": float(np.median(hw[y == c, 1])),
                                  "grosor_med": float(np.nanmedian(G[y == c])), "tam_orig_med": float(np.nanmedian(tam[y == c])),
                                  "razon_anis_med": float(np.nanmedian(razon[ok & (y == c)]))} for c in range(10)},
           "spearman_anis_vs_aspecto": float(stats.spearmanr(razon[ok], (hw[:, 1] / hw[:, 0])[ok])[0]),
           "tam_orig_med": {"1l": float(np.nanmedian(tam[l1])), "gruesos": float(np.nanmedian(tam[gru])), "1_no_gruesos": float(np.nanmedian(tam[(y == 1) & ~gru]))}}
    print(json.dumps(res, ensure_ascii=False, indent=1)[:1600])

    fig = plt.figure(figsize=(16, 13), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    a1, a2, a3 = arriba.subplots(1, 3)
    x = np.arange(10); col = ["#9333ea" if c == 1 else "#999" for c in x]
    a1.bar(x - 0.2, [res["por_clase"][str(c)]["alto_med"] for c in x], 0.4, color="#555", label="alto (mediana)")
    a1.bar(x + 0.2, [res["por_clase"][str(c)]["ancho_med"] for c in x], 0.4, color="#bbb", label="ancho (mediana)")
    a1.axhline(32, color="k", lw=0.8, ls=":"); a1.set_xticks(x); a1.set_ylim(0, 36); a1.set_ylabel("px"); a1.legend(fontsize=8, loc="lower right")
    a1.set_title(f"1 · CÓMO se normalizó: el ALTO se llevó a 32 px\n({res['alto_32_pct']:.0f} % mide 32 de alto, 100 % ≥ 30; ninguno 32 de ancho)", fontsize=10)
    a2.bar(x, [res["por_clase"][str(c)]["razon_anis_med"] for c in x], color=col); a2.axhline(1, color="k", lw=0.8, ls=":")
    a2.set_xticks(x); a2.set_ylim(0, 1.6); a2.set_ylabel("grosor de verticales / de horizontales")
    a2.set_title(f"2 · el escalado fue IGUAL en las dos direcciones\n(razón ≈ 1; no depende de la forma de la caja: ρ = {res['spearman_anis_vs_aspecto']:.2f})", fontsize=10)
    a3.bar(x, [res["por_clase"][str(c)]["tam_orig_med"] for c in x], color=col); a3.axhline(1, color="k", lw=0.8, ls=":")
    for c in x:
        a3.text(c, res["por_clase"][str(c)]["tam_orig_med"] + 0.02, f"{res['por_clase'][str(c)]['tam_orig_med']:.2f}", ha="center", fontsize=8)
    a3.set_xticks(x); a3.set_ylim(0, 1.3); a3.set_ylabel("alto original estimado (1 = típico)")
    a3.set_title(f"3 · el ALTO ORIGINAL estimado = {ref:.2f} px / grosor\n(los 1: {res['por_clase']['1']['tam_orig_med']:.2f}; los gruesos: {res['tam_orig_med']['gruesos']:.2f})", fontsize=10)

    # «en el papel»: cada dígito encogido a su alto original estimado, sobre una línea base común
    ax = abajo.subplots(); ax.set_title("«En el papel»: cada dígito encogido a su alto original estimado, sobre la misma línea base "
                                        "(arriba: tal como está en el dataset, todos de 32 px de alto)", fontsize=11)
    va = np.array([i for i in d.va])
    tip = [int(va[(y[va] == c)][np.argmin(np.abs(G[va][y[va] == c] - np.nanmedian(G[y == c])))]) for c in (0, 3, 5, 7, 8)]
    unos = [int(i) for i in np.flatnonzero(l1)]; unos = sorted(unos, key=lambda i: G[i]); sel1 = [unos[k] for k in np.linspace(0, len(unos) - 1, 7).astype(int)]
    fila = tip[:3] + sel1[:4] + tip[3:] + sel1[4:]
    X = 0
    for i in fila:
        s = min(1.0, tam[i]) if not np.isnan(tam[i]) else 1.0
        img = T[i].astype(float); c = np.flatnonzero(img.any(0)); img = img[:, c.min():c.max() + 1]
        ax.imshow(img, cmap="Greys", vmin=0, vmax=1.4, extent=(X, X + img.shape[1], 70, 38), interpolation="nearest")
        z = ndimage.zoom(img, s, order=1); hz, wz = z.shape
        ax.imshow(z, cmap="Purples" if y[i] == 1 else "Greys", vmin=0, vmax=1.2, extent=(X, X + wz, 32, 32 - hz), interpolation="bilinear")
        ax.text(X + img.shape[1] / 2, 74, f"#{i} ({y[i]})\ngrosor {G[i]:.1f}\nalto ×{tam[i]:.2f}", ha="center", va="top", fontsize=8)
        X += img.shape[1] + 6
    ax.axhline(32, color="#c2410c", lw=1); ax.text(X, 32.5, "línea base", color="#c2410c", fontsize=9, va="top", ha="right")
    ax.set_xlim(-2, X); ax.set_ylim(84, -2); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("¿Eran pequeños los 1? Normalización por ALTO y uniforme ⇒ el grosor dice cuánto se agrandó cada dígito: "
                 f"los 1 se escribieron ~{res['por_clase']['1']['tam_orig_med']:.0%} de alto que el dígito típico", fontsize=13)
    out = AQUI / "resultados" / "117-tamano-original.png"; fig.savefig(out, dpi=90)
    res["fila_papel"] = fila; out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
