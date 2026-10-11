#!/usr/bin/env python3
"""rizo-2 — rizos SINTÉTICOS con verdad conocida, con cola y distractores; y negativos sin rizo.

Un rizo es un arco de círculo (radio R 2–7, grosor g 2–6 con hueco interior ≥ 1,25 px, cobertura 0,5–1, abertura cualquiera) del que sale una COLA
por uno de sus extremos (como la cabeza del 2 sigue en el trazo diagonal), más 0–2 distractores (rectas, arcos de
menos de media vuelta) que no invaden el hueco. Un negativo son 1–3 trazos sin ningún arco de ≥ media vuelta.

    python nn/sinteticos.py   → resultados/1-sinteticos-muestra.png (24 ejemplos con su verdad)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def arco(cx, cy, R, cob, abertura, g) -> np.ndarray:
    """Arco de círculo que cubre `cob` de la vuelta, con el hueco centrado en `abertura` (grados; 0 = →, 90 = ↓)."""
    d = np.hypot(_XX - cx, _YY - cy); a = np.degrees(np.arctan2(_YY - cy, _XX - cx))
    lejos = np.abs((a - abertura + 180) % 360 - 180) > (1 - cob) * 180
    return (np.abs(d - R) <= g / 2) & lejos


def recta(cx, cy, L, ang, g) -> np.ndarray:
    t = np.deg2rad(ang); px, py = _XX - cx, _YY - cy
    a = px * np.cos(t) + py * np.sin(t); p = -px * np.sin(t) + py * np.cos(t)
    return (np.abs(a) <= L / 2) & (np.abs(p) <= g / 2)


def _distractor(rng, g) -> np.ndarray:
    if rng.random() < 0.5:
        return recta(rng.uniform(4, 28), rng.uniform(4, 28), rng.uniform(6, 16), rng.uniform(0, 180), g)
    return arco(rng.uniform(4, 28), rng.uniform(4, 28), rng.uniform(3, 9), rng.uniform(0.15, 0.45), rng.uniform(0, 360), g)


def rizo(rng) -> dict:
    """Un rizo con cola y distractores, y su verdad."""
    while True:
        R, g, cob, ab = rng.uniform(2, 7), int(rng.integers(2, 7)), rng.uniform(0.5, 1.0), rng.uniform(0, 360)
        if R - g / 2 < 0.5:                                   # sin hueco no hay rizo; los de los 2 reales son rendijas (radio inscrito mediana 2 px, 10 % ≤ 1)
            continue
        m = R + g / 2 + 1
        cx, cy = rng.uniform(m, 32 - m, 2)
        x = arco(cx, cy, R, cob, ab, g)
        hueco = np.hypot(_XX - cx, _YY - cy) < max(R - g / 2 - 0.5, 0.5)
        # la cola: sale de un extremo del arco, siguiendo la tangente ± 50°, 5–14 px
        if cob < 0.98:
            s = rng.choice([-1, 1]); e = ab + s * (1 - cob) * 180
            ex, ey = cx + R * np.cos(np.deg2rad(e)), cy + R * np.sin(np.deg2rad(e))
            tang = e + 90 * s + rng.uniform(-50, 50); L = rng.uniform(5, 14)
            mx, my = ex + L / 2 * np.cos(np.deg2rad(tang)), ey + L / 2 * np.sin(np.deg2rad(tang))
            cola = recta(mx, my, L, tang, g)
            if (cola & hueco).any():
                continue
            x = x | cola
        for _ in range(int(rng.integers(0, 3))):
            d = _distractor(rng, int(rng.integers(2, 7)))
            if not (d & hueco).any():
                x = x | d
        if x.sum() < 12:
            continue
        return {"tipo": "rizo", "x": x.astype(np.uint8), "centro": (float(cx), float(cy)), "radio": float(R),
                "cierre": float((cob - 0.5) / 0.5), "abertura": float(ab), "grosor": g}


def negativo(rng) -> dict:
    x = np.zeros((32, 32), bool)
    for _ in range(int(rng.integers(1, 4))):
        g = int(rng.integers(2, 7))
        if rng.random() < 0.25:                                # una esquina: dos rectas que se tocan
            cx, cy, a = rng.uniform(8, 24), rng.uniform(8, 24), rng.uniform(0, 360)
            for da in (0, rng.uniform(60, 120)):
                L = rng.uniform(6, 14); t = np.deg2rad(a + da)
                x |= recta(cx + L / 2 * np.cos(t), cy + L / 2 * np.sin(t), L, a + da, g)
        else:
            x |= _distractor(rng, g)
    if x.sum() < 12:
        return negativo(rng)
    return {"tipo": "negativo", "x": x.astype(np.uint8)}


def conjunto(n_pos: int, n_neg: int, semilla: int) -> list[dict]:
    rng = np.random.default_rng(semilla)
    return [rizo(rng) for _ in range(n_pos)] + [negativo(rng) for _ in range(n_neg)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ej = conjunto(16, 8, 7)
    fig, axs = plt.subplots(3, 8, figsize=(13, 5.4))
    for ax, f in zip(axs.flat, ej):
        ax.imshow(f["x"], cmap="Greys", vmin=0, vmax=1.8); ax.set_xticks([]); ax.set_yticks([])
        if f["tipo"] == "rizo":
            ax.plot(f["centro"][0] - 0.5, f["centro"][1] - 0.5, "+", color="#2a78d6", ms=10, mew=2)
            ax.set_title(f"R {f['radio']:.1f} g {f['grosor']} cierre {f['cierre']:.2f} ab {f['abertura']:.0f}°", fontsize=7)
        else:
            ax.set_title("negativo", fontsize=7)
    fig.suptitle("rizo-2 · sintéticos con verdad: rizo (arco ≥ media vuelta) + cola + distractores · negativos sin rizo", fontsize=9)
    fig.tight_layout(); (AQUI / "resultados").mkdir(exist_ok=True)
    fig.savefig(AQUI / "resultados" / "1-sinteticos-muestra.png", dpi=90)
    print("→ resultados/1-sinteticos-muestra.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
