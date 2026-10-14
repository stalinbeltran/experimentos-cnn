#!/usr/bin/env python3
"""rizo-2 — ¿detecta bien las CONCAVIDADES la cresta? Verdad: en los trazos sintéticos, cada arco REAL de ≥ 60° cuya
abertura esté libre (ningún otro punto del trazo a ≤ 2 px del punto medio de la cuerda) es una concavidad conocida, con
su centroide y la dirección de su abertura. Detectores: cresta (cresta_curvas → concavidades) y rayos v4 (objetos del
mapa de concavidad de huecos_suaves). Casa: centro a ≤ TOL px; para la cresta, además la dirección de abertura a ≤ 45°.

    python nn/concavidades_comparar.py --sinteticos /ruta/sinteticos-verdad-k3x3.npz   → resultados/54-concavidades-comparar.{json,png}
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
from matplotlib.patches import Polygon                              # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                          # noqa: E402
import curvas_comparar as Q                                         # noqa: E402
import huecos_suaves as HS                                          # noqa: E402

TOL = 3.0


def concavidades_reales(p):
    out = []
    for r in Q.arcos_reales(p):
        if r["barrido"] < CC.CONC_MIN:
            continue
        poly = r["pts"]; medio = (poly[0] + poly[-1]) / 2
        if r["barrido"] >= 300 or np.hypot(*(poly[-1] - poly[0])) <= 3.0:   # un arco cerrado (anillo) es un HUECO, no una concavidad
            continue
        otros = np.concatenate([p[:0], p]); d = np.hypot(*(otros - medio).T)
        # ¿la abertura está libre? ningún punto del trazo FUERA del arco a ≤ 2 px del punto medio de la cuerda
        en_arco = np.array([np.hypot(*(q - poly).T).min() <= 1.0 for q in p])
        if (d[~en_arco] <= 2.0).any():
            continue
        cx, cy = poly.mean(0); v = medio - np.array([cx, cy]); n = np.hypot(*v)
        if n < 1e-6:
            continue
        out.append({"cx": float(cx), "cy": float(cy), "dir": v / n, "pts": poly, "barrido": r["barrido"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sinteticos", required=True); a = ap.parse_args()
    HS.configurar(CENTRAR=True, UNO_POR_CELDA=True, SIN_RENDIJAS=True, PONDERAR=True)
    z = np.load(a.sinteticos); M = z["M"].astype(np.float32); P, ini, tipo = z["puntos"], z["inicio"], z["tipo"]
    st = {"cresta": [0, 0, 0], "rayos v4": [0, 0, 0]}; n_reales = 0; ej = []; por_b = {}
    for i in range(len(M)):
        reales = concavidades_reales(P[ini[i]:ini[i + 1]]); n_reales += len(reales)
        ob = CC.objetos(M[i]); _, con = HS.mapas(M[i][None]); HS.M_ACTUAL = M[i]
        dets = {"cresta": ob["concavidades"], "rayos v4": [{**o, "dir": None} for o in HS.objetos(con[0]) if o["fuerza"] >= 0.3]}
        for n, d in dets.items():
            def casa(x, y):
                return np.hypot(x["cx"] - y["cx"], x["cy"] - y["cy"]) <= TOL and (x.get("dir") is None or np.degrees(np.arccos(np.clip(np.dot(x["dir"], y["dir"]), -1, 1))) <= 45)
            cub = sum(any(casa(dd, r) for dd in d) for r in reales); ok = sum(any(casa(dd, r) for r in reales) for dd in d)
            st[n][0] += cub; st[n][1] += ok; st[n][2] += len(d)
            if n == "cresta":
                for r in reales:
                    b = f"{int(min(r['barrido'] // 60, 4) * 60)}°"; por_b.setdefault(b, [0, 0]); por_b[b][1] += 1; por_b[b][0] += any(casa(dd, r) for dd in d)
        if i < 12:
            ej.append((i, reales, dets, ob))
    res = {"trazos": int(len(M)), "concavidades_reales": n_reales, "tol_px": TOL, "cobertura_cresta_por_barrido": {k: f"{v[0]}/{v[1]}" for k, v in sorted(por_b.items())}}
    for n, (cub, ok, det) in st.items():
        res[n] = {"detecciones": det, "cobertura": round(cub / max(n_reales, 1), 3), "precision": round(ok / max(det, 1), 3), "por_trazo": round(det / len(M), 2)}
    (AQUI / "resultados" / "54-concavidades-comparar.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n"); print(json.dumps(res, ensure_ascii=False, indent=1))
    fig, axs = plt.subplots(3, 12, figsize=(12 * 1.55, 3 * 1.8 + 1.0))
    for ax in axs.flat:
        ax.axis("off")
    for col, (i, reales, dets, ob) in enumerate(ej):
        for fila, (tit, objs) in enumerate((("reales", reales), ("cresta", dets["cresta"]), ("rayos v4", dets["rayos v4"]))):
            ax = axs[fila, col]; ax.imshow(np.where(M[i] > 0.05, M[i], np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest", zorder=2)
            for o in objs:
                if "pts" in o and o.get("dir") is not None:
                    ax.add_patch(Polygon(o["pts"] - 0.5, closed=True, facecolor="#1b7f3b", alpha=0.3, edgecolor="none", zorder=1))
                    ax.arrow(o["cx"] - 0.5, o["cy"] - 0.5, 3 * o["dir"][0], 3 * o["dir"][1], color="#0b6b2e", head_width=1.0, lw=1.2, zorder=3)
                else:
                    ax.plot(o["cx"] - 0.5, o["cy"] - 0.5, "+", color="#0b6b2e", ms=4 + 10 * o.get("fuerza", 0.5), mew=2, zorder=3)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"{tipo[i] if fila == 0 else tit}: {len(objs)}", fontsize=7)
    fig.suptitle(f"rizo-2 · 54 · CONCAVIDADES contra la verdad en {len(M)} trazos sintéticos ({n_reales} reales: arcos ≥ 60° con la abertura libre): "
                 f"cresta cobertura {res['cresta']['cobertura']} · precisión {res['cresta']['precision']} | rayos v4 {res['rayos v4']['cobertura']} · {res['rayos v4']['precision']}\n"
                 "verde = región encerrada · flecha = dirección de la abertura · + = objeto de rayos (tamaño = fuerza)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(AQUI / "resultados" / "54-concavidades-comparar.png", dpi=100); print("→ resultados/54-concavidades-comparar.{json,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
