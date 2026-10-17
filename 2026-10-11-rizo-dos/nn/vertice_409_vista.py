#!/usr/bin/env python3
"""rizo-2 § 108 — el ÁNGULO que el vértice libre halla en el CENTRO de un 3 (val #409) (el dueño, 2026-10-17: «un detector de
vértices no encuentra el vértice en el centro de un 3, id 409… muéstrame en ese 3 el ángulo hallado, en una imagen»).

Sobre el DELGADO, con el MISMO `vertice_libre` del revisor (sin tocar nada): a la izquierda, el dígito con su cresta coloreada
por el ángulo a 6 px de cada punto, el vértice que SÍ halla (arriba) y el punto más cerrado del centro con sus cuerdas de 6 px
(azul) y 3 px (verde); a la derecha, el ángulo a lo largo de la cresta contra los dos límites (< 90° a 6 px, < 120° a 3 px).

    ../.venv/bin/python nn/vertice_409_vista.py [idx]   → resultados/108-vertice-centro-3-<idx>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import vertice_libre as V                                                # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"


def angulos(p):
    """Por cada punto de la cresta: (ángulo a 6 px, ángulo a 3 px, cuerdas 6, cuerdas 3), o None si un brazo es corto."""
    out = []
    for i in range(len(p)):
        u, lu = V._brazo(p, i, -1, V.BRAZO, False); v, lv = V._brazo(p, i, +1, V.BRAZO, False)
        uc, _ = V._brazo(p, i, -1, V.BRAZO_CORTO, False); vc, _ = V._brazo(p, i, +1, V.BRAZO_CORTO, False)
        ok = lu >= V.BRAZO_MIN and lv >= V.BRAZO_MIN
        out.append((V._angulo(u, v), V._angulo(uc, vc), (u, v), (uc, vc)) if ok else None)
    return out


def arco(a, P, u, v, r, col):
    """El arco del ángulo MENOR entre u y v, en coordenadas del dato (con el eje y invertido, `Arc` pinta el complementario)."""
    t1 = np.arctan2(u[1], u[0]); dt = np.arctan2(u[0] * v[1] - u[1] * v[0], np.dot(u, v))
    t = t1 + np.linspace(0, dt, 30); a.plot(P[0] + r * np.cos(t), P[1] + r * np.sin(t), "-", color=col, lw=2, zorder=5)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 409
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    M = np.load(DELGADOS)["M"][idx].astype(float)
    tr, puros = CC.tramos(M)
    if len(tr) != 1 or puros:
        sys.exit(f"#{idx}: esta figura es para una cresta de UN tramo (tiene {len(tr)} tramos y {len(puros)} ciclos)")
    p = np.asarray(tr[0]["pts"], float); A = angulos(p)
    hallados = V.vertices_libres(M)
    centro = [i for i in range(len(p)) if A[i] and 32 / 3 <= p[i][1] < 64 / 3]          # la fila del medio de la rejilla 3×3
    # el centro de un 3 es la punta que apunta al OESTE (la cintura); en la fila del medio también está el lado derecho de
    # la panza, que apunta al E y no es lo que se busca
    k = min([i for i in centro if V._direccion(*A[i][2]) == "O"], key=lambda i: A[i][0])

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    a.imshow(M, cmap="Greys", vmin=0, vmax=1, alpha=0.45, extent=(0, 32, 32, 0))
    for c in (32 / 3, 64 / 3):
        a.axhline(c, color="#999", lw=0.8, ls=":"); a.axvline(c, color="#999", lw=0.8, ls=":")
    col = [A[i][0] if A[i] else np.nan for i in range(len(p))]
    a.plot(p[:, 0], p[:, 1], "-", color="#555", lw=1)
    sc = a.scatter(p[:, 0], p[:, 1], c=col, cmap="RdYlGn_r", vmin=60, vmax=180, s=28, zorder=3, edgecolors="none")
    fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02, label="ángulo a 6 px (°)")
    P = p[k]; (u, v), (uc, vc) = A[k][2], A[k][3]
    for w in (u, v):
        a.plot([P[0], P[0] + w[0]], [P[1], P[1] + w[1]], "-", color="#2a78d6", lw=2.6, zorder=4)
    for w in (uc, vc):
        a.plot([P[0], P[0] + w[0]], [P[1], P[1] + w[1]], "-", color="#1b7f3b", lw=2.6, zorder=5)
    arco(a, P, u, v, 3.2, "#2a78d6"); arco(a, P, uc, vc, 1.7, "#1b7f3b")
    a.plot(*P, "o", ms=9, mfc="none", mec="#c2410c", mew=2.4, zorder=6)
    a.text(P[0] - 1, P[1] + 0.3, f"centro\n{A[k][0]:.0f}° a 6 px\n{A[k][1]:.0f}° a 3 px\n✗ NO es vértice", ha="right", va="center",
           fontsize=11, color="#c2410c", fontweight="bold", bbox=dict(fc="white", ec="#c2410c", alpha=0.9))
    for h in hallados:
        a.plot(h["cx"], h["cy"], "*", ms=17, color="#1b7f3b", mec="k", zorder=6)
        a.text(h["cx"] + 1.2, h["cy"] - 0.6, f"✓ vértice {h['dir']}\n{h['ang']:.0f}°", fontsize=11, color="#1b7f3b",
               fontweight="bold", va="center", bbox=dict(fc="white", ec="#1b7f3b", alpha=0.9))
    a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"#{idx} (un {d['etiquetas'][idx]}, {d['particion'][idx]}) · delgado y su cresta\nazul = cuerdas de 6 px · verde = de 3 px")

    x = np.arange(len(p)); a6 = np.array(col); a3 = np.array([A[i][1] if A[i] else np.nan for i in range(len(p))])
    b.axvspan(min(centro) - 0.5, max(centro) + 0.5, color="#fde68a", alpha=0.5, label="fila del medio")
    b.plot(x, a6, "-o", color="#2a78d6", ms=4, label="ángulo a 6 px")
    b.plot(x, a3, "-o", color="#1b7f3b", ms=3, alpha=0.7, label="ángulo a 3 px")
    b.axhline(V.A_MAX, color="#2a78d6", ls="--", lw=1.4); b.text(len(p) - 1, V.A_MAX - 4, f"límite a 6 px: < {V.A_MAX:.0f}°", ha="right", va="top", color="#2a78d6")
    b.axhline(V.A_MAX_CORTO, color="#1b7f3b", ls="--", lw=1.4); b.text(len(p) - 1, V.A_MAX_CORTO - 4, f"límite a 3 px: < {V.A_MAX_CORTO:.0f}°", ha="right", va="top", color="#1b7f3b")
    b.annotate(f"centro: {A[k][0]:.0f}°\n(faltan {A[k][0] - V.A_MAX:.0f}° para el límite)", (k, A[k][0]), (k + 6, 75), color="#c2410c",
               fontweight="bold", arrowprops=dict(arrowstyle="->", color="#c2410c"))
    for h in hallados:
        j = int(np.argmin(np.hypot(p[:, 0] - h["cx"], p[:, 1] - h["cy"])))
        b.plot(j, h["ang"], "*", ms=16, color="#1b7f3b", mec="k"); b.annotate(f"vértice {h['ang']:.0f}°", (j, h["ang"]), (j + 0.8, 72), color="#1b7f3b", fontweight="bold")
    b.set_ylim(40, 185); b.set_xlabel("punto de la cresta (de la punta de arriba a la izquierda → la de abajo)"); b.set_ylabel("ángulo (°)")
    b.set_title("el ángulo a lo largo de la cresta: el centro del 3 queda muy por encima del límite"); b.legend(loc="lower left")
    fig.tight_layout()
    out = AQUI / "resultados" / f"108-vertice-centro-3-{idx}.png"; fig.savefig(out, dpi=110); print(out)
    print(f"centro: punto {k} {np.round(P, 1)} · {A[k][0]:.1f}° a 6 px · {A[k][1]:.1f}° a 3 px · hallados: {[(h['dir'], round(h['ang'], 1)) for h in hallados]}")


if __name__ == "__main__":
    main()
