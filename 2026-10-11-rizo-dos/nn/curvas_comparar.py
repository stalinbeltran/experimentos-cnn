#!/usr/bin/env python3
"""rizo-2 — ¿detecta bien las CURVAS el detector de la cresta? (regla del dueño: un detector se juzga por su feature).
Verdad: trazos sintéticos con línea media conocida (el .npz que exporta el experimento de la CNN delgada, por ruta). A la
polilínea real se le aplica la MISMA regla de arcos que usa cresta_curvas (|κ| ≥ K_MIN sostenida, con signo), y cada arco
real tiene centro, sentido (hacia dónde está el centro de curvatura) y ángulo barrido. Se comparan dos detectores sobre
el mapa M de la red: la CRESTA (cresta_curvas.objetos → arcos) y el GRIS (gris_objetos sobre el borde, el detector
anterior). Una detección casa con un arco real si su centro está a ≤ TOL px y tiene el mismo sentido.

    python nn/curvas_comparar.py --sinteticos /ruta/sinteticos-verdad-k3x3.npz   → resultados/48-curvas-comparar.{json,png}
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
import digitos_bordes as DB                                         # noqa: E402
import gris_objetos as GO                                           # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import resistencia as R                                             # noqa: E402

TOL = 1.5; FRAC = 0.5; TOL_GRIS = 2.5; ANG = 60.0   # el gris marca píxeles de BORDE (a ~1,3 px del centro): tolerancia mayor


def _remuestrear(p, paso=1.0):
    """La polilínea real viene a 0,25 px; el esqueleto de la cresta va a ~1 px. Misma regla ⇒ mismo muestreo."""
    d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]); t = np.arange(0, d[-1], paso)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1)


def arcos_reales(p):
    """Los arcos de la polilínea real con la regla de cresta_curvas: [{s, cx, cy, barrido, pts}]."""
    ps = CC._suavizar(_remuestrear(p), False); k, nrm = CC.curvatura(ps, False); n = len(ps)
    out = []
    for c, i, j in CC.segmentar(ps, k, nrm):
        if c != 0 and j - i >= CC.L_MIN:
            seg = ps[i:j]; barrido = float(np.degrees(np.sum(np.abs(k[i:j]) * np.hypot(*np.gradient(seg, axis=0).T)))); nm = nrm[i:j].mean(0)
            if barrido >= 20:                                       # un arco de menos de 20° no es una curva que haya que detectar
                out.append({"s": CC._sentido(nm[0], nm[1]), "nm": nm / (np.hypot(*nm) + 1e-9), "cx": float(seg[:, 0].mean()), "cy": float(seg[:, 1].mean()), "barrido": barrido, "pts": seg})
    return out


def _cerca(pts, otros, tol=TOL):
    """Fracción de `pts` a ≤ tol px de algún punto de `otros` (n,2)."""
    if len(otros) == 0:
        return 0.0
    d = np.hypot(pts[:, None, 0] - otros[None, :, 0], pts[:, None, 1] - otros[None, :, 1]).min(1)
    return float((d <= tol).mean())


def _normal_media(d):
    """Vector unitario medio hacia el centro de curvatura de una detección (de sus puntos, con la misma regla)."""
    if "nm" in d:
        return d["nm"]
    if d.get("borde"):                                               # gris: píxeles de borde sin orden; sólo el cuadrante
        return np.array([(1, -1, 0, 0)[d["s"]], (0, 0, 1, -1)[d["s"]]], float)
    if len(d["pts"]) < 3:
        return np.array([(1, -1, 0, 0)[d["s"]], (0, 0, 1, -1)[d["s"]]], float)
    k, nrm = CC.curvatura(d["pts"], False); nm = (nrm * np.abs(k)[:, None]).sum(0); n = np.hypot(*nm)
    return nm / n if n > 1e-6 else np.array([(1, -1, 0, 0)[d["s"]], (0, 0, 1, -1)[d["s"]]], float)


def _mismo_sentido(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(_normal_media(a), _normal_media(b)), -1, 1))) <= ANG


def casar(reales, dets, lazos=(), tol=TOL):
    """Por PUNTOS: un arco real está cubierto si ≥ FRAC de sus puntos están a ≤ TOL px de una detección del mismo
    sentido (o de un lazo, de cualquier sentido); una detección es correcta si ≥ FRAC de sus puntos están a ≤ TOL px de
    un arco real del mismo sentido. Devuelve (cubiertos, correctos)."""
    cub = 0
    for r in reales:
        ok = any(_mismo_sentido(d, r) and _cerca(r["pts"], d["pts"], tol) >= FRAC for d in dets) or any(_cerca(r["pts"], z["pts"], tol) >= FRAC for z in lazos)
        cub += ok
    correctos = sum(any(_mismo_sentido(d, r) and _cerca(d["pts"], r["pts"], tol) >= FRAC for r in reales) for d in dets)
    return cub, correctos


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sinteticos", required=True); a = ap.parse_args(); GR.configurar(**R.CAL_GRIS)
    z = np.load(a.sinteticos); M = z["M"].astype(np.float32); P, ini, tipo = z["puntos"], z["inicio"], z["tipo"]
    st = {"cresta": [0, 0, 0], "gris": [0, 0, 0]}; n_reales = 0; ejemplos = []   # [cubiertos, correctos, detecciones]
    for i in range(len(M)):
        reales = arcos_reales(P[ini[i]:ini[i + 1]]); n_reales += len(reales)
        ob = CC.objetos(M[i]); gris = GO.objetos(DB.mapas_bordes(M[i], R.GRIS))
        # el gris no guarda píxeles: se reconstruyen de sus mapas (misma regla que gris_objetos.objetos)
        Mb = DB.mapas_bordes(M[i], R.GRIS); K = Mb[1:5]; curvo = K.max(0) > 0; sent = K.argmax(0); gcur = []
        from scipy.ndimage import label
        for s_ in range(4):
            lab, nl = label(curvo & (sent == s_), structure=GO.OCHO)
            for zz in range(1, nl + 1):
                yy_, xx_ = np.nonzero(lab == zz)
                if len(yy_) >= GO.MIN_PX:
                    gcur.append({"s": s_, "pts": np.stack([xx_ + 0.5, yy_ + 0.5], 1), "cx": xx_.mean() + 0.5, "cy": yy_.mean() + 0.5, "largo": len(yy_), "fuerza": 1.0, "borde": True})
        dets = {"cresta": ob["arcos"], "gris": gcur}
        for nombre, d in dets.items():
            cub, ok = casar(reales, d, ob["lazos"] if nombre == "cresta" else (), TOL if nombre == "cresta" else TOL_GRIS); st[nombre][0] += cub; st[nombre][1] += ok; st[nombre][2] += len(d)
        if i < 24:
            ejemplos.append((i, reales, dets))
    res = {"trazos": int(len(M)), "arcos_reales": n_reales, "tolerancia_px": {"cresta": TOL, "gris": TOL_GRIS}, "fraccion": FRAC, "angulo_sentido": ANG}
    for n, (cub, ok, det) in st.items():
        res[n] = {"detecciones": det, "cobertura": round(cub / max(n_reales, 1), 3), "precision": round(ok / max(det, 1), 3), "por_trazo": round(det / len(M), 2)}
    (AQUI / "resultados" / "48-curvas-comparar.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n"); print(json.dumps(res, ensure_ascii=False, indent=1))
    fig, axs = plt.subplots(3, 12, figsize=(12 * 1.55, 3 * 1.8 + 1.0))
    for ax in axs.flat:
        ax.axis("off")
    for col, (i, reales, dets) in enumerate(ejemplos[:12]):
        ax = axs[0, col]; ax.imshow(np.where(M[i] > 0.05, M[i], np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
        for r in reales:
            ax.plot(r["pts"][:, 0] - 0.5, r["pts"][:, 1] - 0.5, "-", color=CC.COLORES[r["s"]], lw=2.2)
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"{tipo[i]} · {len(reales)} arcos reales", fontsize=7)
        ax = axs[1, col]; CC.pintar(ax, M[i]); ax.set_title(f"cresta: {len(dets['cresta'])}", fontsize=7)
        ax = axs[2, col]; ax.imshow(np.where(M[i] > 0.05, M[i], np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
        for c in dets["gris"]:
            ax.plot(c["pts"][:, 0] - 0.5, c["pts"][:, 1] - 0.5, "s", color=GO.COLORES[c["s"]], ms=2.4, mew=0, alpha=0.9)
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"gris: {len(dets['gris'])}", fontsize=7)
    fig.suptitle(f"rizo-2 · 48 · curvas contra la VERDAD en {len(M)} trazos sintéticos ({n_reales} arcos reales): fila 1 arcos reales de la línea media · fila 2 cresta "
                 f"(cobertura {res['cresta']['cobertura']} · precisión {res['cresta']['precision']}) · fila 3 gris (cobertura {res['gris']['cobertura']} · precisión {res['gris']['precision']})\n"
                 "color = centro de curvatura a la → naranja · ← azul · ↓ verde · ↑ morado", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(AQUI / "resultados" / "48-curvas-comparar.png", dpi=100); print("→ resultados/48-curvas-comparar.{json,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
