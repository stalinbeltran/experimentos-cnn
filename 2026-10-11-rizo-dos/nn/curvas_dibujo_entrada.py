#!/usr/bin/env python3
"""rizo-2 — dibuja las CURVAS del detector del gris sobre una entrada como ARCOS en su sitio (pedido por el dueño el
2026-10-13: «una curva de la misma orientación en la misma posición, de modo que sea intuitiva»). Cada objeto de curva
es un grupo de píxeles de borde con curvatura del mismo sentido; se le ajusta un círculo (mínimos cuadrados) y se pinta
el arco que cubren sus píxeles, con el color del sentido y el grosor de la fuerza. Los golpes van como ×.

    python nn/curvas_dibujo_entrada.py --entrada /ruta/delgados-k3x3.npz --de resultados/43-huecos-falsos-<variante>.json
        → resultados/44-curvas-<de>.png     (los mismos dígitos de esa figura)
    python nn/curvas_dibujo_entrada.py --entrada … --idx 985 507 616     → resultados/44-curvas-idx.png
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
import digitos_bordes as DB                                         # noqa: E402
import evaluar as EV                                                # noqa: E402
import gris_objetos as GO                                           # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import resistencia as R                                             # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402
from scipy.ndimage import label                                     # noqa: E402

NOMBRE_SENTIDO = {0: "centro a la derecha (k→)", 1: "centro a la izquierda (k←)", 2: "centro abajo (k↓)", 3: "centro arriba (k↑)"}


def circulo(xs, ys):
    """Ajuste algebraico (Kåsa) de un círculo a los puntos: (cx, cy, r)."""
    A = np.stack([xs, ys, np.ones_like(xs)], 1); b = xs**2 + ys**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None); cx, cy = sol[0] / 2, sol[1] / 2
    return cx, cy, float(np.sqrt(max(sol[2] + cx**2 + cy**2, 1e-6)))


def arcos(x):
    """Los objetos de curva del gris sobre la imagen x, cada uno con su arco: {s, fuerza, largo, pts (n,2)}."""
    M = DB.mapas_bordes(x, R.GRIS); K = M[1:5]; curvo = K.max(0) > 0; sent = K.argmax(0); out = []
    for s in range(4):
        lab, n = label(curvo & (sent == s), structure=GO.OCHO)
        for z in range(1, n + 1):
            yy, xx = np.nonzero(lab == z)
            if len(yy) < GO.MIN_PX:
                continue
            fuerza = float(min(K[s][yy, xx].max() / GO.NORMA_K, 1.0)); xs, ys = xx.astype(float), yy.astype(float)
            if len(yy) >= 4 and (xs.std() > 0.3 or ys.std() > 0.3):
                cx, cy, r = circulo(xs, ys)
                if 2.5 <= r <= 40:                                                # un radio menor con 4–6 píxeles es un «o» sin sentido: va como trazo
                    ang = np.arctan2(ys - cy, xs - cx); a0 = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())   # ángulo medio
                    rel = (ang - a0 + np.pi) % (2 * np.pi) - np.pi; t = np.linspace(rel.min(), rel.max(), 24) + a0
                    pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
                    out.append({"s": s, "fuerza": fuerza, "largo": len(yy), "pts": pts, "r": r}); continue
            # casi recto, muy corto o muy curvado: trazo que sigue sus píxeles, ordenados por el ángulo alrededor del centroide
            # (si es un arco) o por el eje principal (si es casi recto)
            mx, my = xs.mean(), ys.mean(); ang = np.arctan2(ys - my, xs - mx); a0 = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            rel = (ang - a0 + np.pi) % (2 * np.pi) - np.pi
            orden = np.argsort(rel) if np.hypot(xs - mx, ys - my).mean() > 1.0 and np.ptp(rel) > 1.0 else np.argsort(xs if xs.std() >= ys.std() else ys)
            out.append({"s": s, "fuerza": fuerza, "largo": len(yy), "pts": np.stack([xs[orden], ys[orden]], 1), "r": None})
    golpes = []
    lab, n = label(M[5] > 0, structure=GO.OCHO)
    for z in range(1, n + 1):
        yy, xx = np.nonzero(lab == z)
        if len(yy) >= GO.MIN_PX:
            golpes.append((xx.mean(), yy.mean()))
    return out, golpes


def pintar(ax, x, continuo=True):
    if continuo:
        ax.imshow(np.where(x > 0.05, x, np.nan), cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest")
    else:
        ax.imshow(np.where(x > 0, 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest")
    arcs, golpes = arcos(x)
    for a in arcs:
        ax.plot(a["pts"][:, 0], a["pts"][:, 1], "-", color=GO.COLORES[a["s"]], lw=0.8 + 2.6 * a["fuerza"], alpha=0.9, solid_capstyle="round")
    for gx, gy in golpes:
        ax.plot(gx, gy, "x", color="k", ms=6, mew=1.3)
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off")
    return len(arcs), len(golpes)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M")
    ap.add_argument("--de", default=None, help="JSON de 43-huecos-falsos: se dibujan sus «peores»"); ap.add_argument("--idx", type=int, nargs="*", default=None)
    a = ap.parse_args(); GR.configurar(**R.CAL_GRIS)
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); y = d["etiquetas"]; Mz = np.load(a.entrada)[a.clave].astype(np.float32)
    if a.de:
        peores = json.loads(Path(a.de).read_text())["peores"]; ids = list(dict.fromkeys(f["idx"] for f in peores)); nombre = Path(a.de).stem.replace("43-huecos-falsos", "falsos")
    else:
        ids = a.idx; nombre = "idx"
    cols = 12; filas = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.55, filas * 1.8 + 0.9)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, i in zip(axs.flat, ids):
        nc, ng = pintar(ax, Mz[i]); ax.set_title(f"{y[i]} · idx {i} · {nc} curvas", fontsize=7.5)
    fig.suptitle("rizo-2 · 44 · las CURVAS del detector del gris sobre el mapa suave, dibujadas como arcos en su sitio (ajuste de círculo a los píxeles de cada objeto)\n"
                 "color = hacia dónde está el centro de curvatura: naranja → derecha · azul → izquierda · verde → abajo · morado → arriba · grosor = fuerza · × = golpe (cambio brusco)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); out = AQUI / "resultados" / f"44-curvas-{nombre}.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
