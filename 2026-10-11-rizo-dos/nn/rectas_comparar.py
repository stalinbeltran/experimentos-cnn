#!/usr/bin/env python3
"""rizo-2 — ¿detecta bien las RECTAS? (el dueño, 2026-10-13: «mejora las rectas, como hicimos con los otros features»).
Verdad: tramos RECTOS de la línea media real de los trazos sintéticos (|κ| < K_MIN sostenido, ≥ 4 px; misma regla y
mismo muestreo que la cresta). Detectores sobre el mapa M: rectas-OBJETO (rectas_objetos, Gabor par agrupado) y rectas
de la CRESTA (cresta_curvas). Una recta real está cubierta si ≥ FRAC de sus puntos están a ≤ TOL px de una detección
con orientación a ≤ ANG grados; una detección es correcta si ≥ FRAC de sus puntos están a ≤ TOL px de una recta real
con esa orientación.

    python nn/rectas_comparar.py --sinteticos /ruta/sinteticos-verdad-k3x3.npz   → resultados/50-rectas-comparar.{json,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import cresta_curvas as CC                                          # noqa: E402
import curvas_comparar as Q                                         # noqa: E402
import rectas_objetos as RO                                         # noqa: E402

TOL = 1.5; TOL_RO = 2.5; FRAC = 0.5; ANG = 25.0; L_REAL = CC.L_RECTA + 1.0   # el esqueleto pierde ~1 px en cada extremo (skeletonize + suavizado): una recta real de 6 px sale de 4-5 en la cresta


def _orient(pts):
    """Orientación de un conjunto de puntos (eje principal), en grados [0,180)."""
    c = pts - pts.mean(0); ev, vec = np.linalg.eigh(np.cov(c.T) if len(pts) > 2 else np.eye(2)); v = vec[:, 1]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180)


def rectas_reales(p):
    ps = CC._suavizar(Q._remuestrear(p), False); k, nrm = CC.curvatura(ps, False); out = []
    for c, i, j in CC.segmentar(ps, k, nrm):
        if c == 0 and j - i >= CC.L_MIN:
            seg = ps[i:j]; L = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
            if L >= L_REAL:
                out.append({"pts": seg, "largo": L, "ang": _orient(seg)})
    return out


def _dang(a, b):
    d = abs(a - b) % 180; return min(d, 180 - d)


def casar(reales, dets, tol):
    cub = sum(any(_dang(d["ang"], r["ang"]) <= ANG and Q._cerca(r["pts"], d["pts"], tol) >= FRAC for d in dets) for r in reales)
    ok = sum(any(_dang(d["ang"], r["ang"]) <= ANG and Q._cerca(d["pts"], r["pts"], tol) >= FRAC for r in reales) for d in dets)
    return cub, ok


def dets_objeto(x):
    out = []
    for z in RO.objetos(x)[0]:
        d = RO.DIRS[z["o"]]; h = z["largo"] / 2; t = np.linspace(-h, h, max(int(z["largo"]), 2))
        out.append({"pts": np.stack([z["cx"] + d[0] * t, z["cy"] + d[1] * t], 1), "largo": z["largo"], "ang": float(RO.ANG[z["o"]]), "fuerza": min(z["fuerza"] / RO.NORMA, 1.0), "o": z["o"]})
    return out


def dets_cresta(x):
    return [{**r, "pts": np.linspace(r["pts"][0], r["pts"][1], max(int(r["largo"]), 2))} for r in CC.objetos(x)["rectas"]]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sinteticos", required=True); ap.add_argument("--etiqueta", default=""); a = ap.parse_args()
    z = np.load(a.sinteticos); M = z["M"].astype(np.float32); P, ini, tipo = z["puntos"], z["inicio"], z["tipo"]
    st = {"rectas-objeto": [0, 0, 0], "cresta": [0, 0, 0]}; n_reales = 0; ej = []
    for i in range(len(M)):
        reales = rectas_reales(P[ini[i]:ini[i + 1]]); n_reales += len(reales)
        dets = {"rectas-objeto": dets_objeto(M[i]), "cresta": dets_cresta(M[i])}
        for n, d in dets.items():
            cub, ok = casar(reales, d, TOL_RO if n == "rectas-objeto" else TOL); st[n][0] += cub; st[n][1] += ok; st[n][2] += len(d)
        if i < 12:
            ej.append((i, reales, dets))
    res = {"trazos": int(len(M)), "rectas_reales": n_reales, "etiqueta": a.etiqueta}
    for n, (cub, ok, det) in st.items():
        res[n] = {"detecciones": det, "cobertura": round(cub / max(n_reales, 1), 3), "precision": round(ok / max(det, 1), 3), "por_trazo": round(det / len(M), 2)}
    suf = f"-{a.etiqueta}" if a.etiqueta else ""
    (AQUI / "resultados" / f"50-rectas-comparar{suf}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n"); print(json.dumps(res, ensure_ascii=False, indent=1))
    fig, axs = plt.subplots(3, 12, figsize=(12 * 1.55, 3 * 1.8 + 1.0))
    for ax in axs.flat:
        ax.axis("off")
    for col, (i, reales, dets) in enumerate(ej):
        for fila, (tit, objs, colr) in enumerate((("reales", reales, "#111"), ("rectas-objeto", dets["rectas-objeto"], None), ("cresta", dets["cresta"], "#555"))):
            ax = axs[fila, col]; ax.imshow(np.where(M[i] > 0.05, M[i], np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
            for o in objs:
                c = colr or RO.COLORES[o["o"]]; ax.plot(o["pts"][:, 0] - 0.5, o["pts"][:, 1] - 0.5, "-", color=c, lw=1.2 + 2 * o.get("fuerza", 0.6), alpha=0.9 if colr else 0.35 + 0.65 * o["fuerza"])
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"{tipo[i] if fila == 0 else tit}: {len(objs)}", fontsize=7)
    fig.suptitle(f"rizo-2 · 50 · RECTAS contra la verdad en {len(M)} trazos sintéticos ({n_reales} rectas reales ≥ {L_REAL:.0f} px){' · ' + a.etiqueta if a.etiqueta else ''}: "
                 f"rectas-objeto cobertura {res['rectas-objeto']['cobertura']} · precisión {res['rectas-objeto']['precision']} | cresta {res['cresta']['cobertura']} · {res['cresta']['precision']}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / f"50-rectas-comparar{suf}.png", dpi=100); print(f"→ resultados/50-rectas-comparar{suf}.{{json,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
