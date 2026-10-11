#!/usr/bin/env python3
"""rizo-2 — un COMPOSITOR de doses sintéticos: el rizo superior con la geometría de los 2 REALES (medida en las 48
etiquetas: centro x 11–16 · y 6,5–9,5, radio 3–4,5, hueco 1,5–2,5 px, cobertura 0,5–0,75, abertura hacia abajo-izquierda)
seguido del trazo DIAGONAL y la BASE, con grosor 3–6 y variaciones de posición, tamaño e inclinación. Negativos: formas
sin rizo —un 1, un 7, un 4, una L, rectas— con los mismos grosores.

Segunda vuelta (2026-10-11): la primera, con `sinteticos.py` (rizos genéricos con una cola corta), dio 94 % en
sintéticos y 34 % en los doses reales: la red aprendió un rizo que no se parece al del 2. Esto es el «compositor» que el
dueño dijo que podía hacer falta, usado como GENERADOR de verdad conocida.
"""
from __future__ import annotations

import numpy as np

from sinteticos import arco, recta, _XX, _YY


def _poli(puntos, g) -> np.ndarray:
    x = np.zeros((32, 32), bool)
    for (x0, y0), (x1, y1) in zip(puntos[:-1], puntos[1:]):
        L = np.hypot(x1 - x0, y1 - y0); ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        x |= recta((x0 + x1) / 2, (y0 + y1) / 2, L + g * 0.6, ang, g)
    return x


def dos(rng) -> dict:
    """Un 2: cabeza (arco) + diagonal + base; verdad = la cabeza."""
    while True:
        g = int(rng.integers(3, 8)); R = rng.uniform(2.5, 5.0)
        if R - g / 2 < 0.5:                                    # tercera vuelta: los huecos reales son RENDIJAS (radio inscrito mediana 2 px, 25 % ≤ 1,3)
            continue
        cx, cy = rng.uniform(9, 18), rng.uniform(5.5, 11)
        cob = rng.uniform(0.5, 0.8); ab = rng.uniform(100, 160)
        x = arco(cx, cy, R, cob, ab, g)
        hueco = np.hypot(_XX - cx, _YY - cy) < max(R - g / 2 - 0.3, 0.5)
        # extremo derecho del arco (el de menor ángulo hacia la derecha): ab − (1−cob)·180 … elegir el que queda más a la derecha
        e1, e2 = ab - (1 - cob) * 180, ab + (1 - cob) * 180
        pts = [(cx + R * np.cos(np.deg2rad(e)), cy + R * np.sin(np.deg2rad(e))) for e in (e1, e2)]
        p_der = max(pts, key=lambda p: p[0])
        # la diagonal baja hacia la izquierda, hasta la base; la base va hacia la derecha
        bx = rng.uniform(4, 11); by = rng.uniform(25, 30.5); fin = (min(31, bx + rng.uniform(13, 22)), by + rng.uniform(-1.5, 1.5))
        codo = (p_der[0] + rng.uniform(-1, 3), p_der[1] + rng.uniform(2, 7))      # la diagonal arranca algo más abajo
        x |= _poli([p_der, codo, (bx, by), fin], g)
        if rng.random() < 0.3:                                                     # algunos 2 llevan rizo abajo, como el 7 lazo
            x |= arco(bx + 2, by - 2, rng.uniform(1.8, 3), rng.uniform(0.4, 0.8), rng.uniform(0, 360), g)
        if (x & hueco).mean() > 0.15:                                              # la diagonal se metió en el hueco: otro
            continue
        return {"tipo": "rizo", "x": x.astype(np.uint8), "centro": (float(cx), float(cy)), "radio": float(R),
                "cierre": float((cob - 0.5) / 0.5), "abertura": float(ab), "grosor": g}


def tres(rng) -> dict:
    """Un 3: su cabeza también es un rizo superior (positivo), con una panza debajo."""
    while True:
        g = int(rng.integers(3, 8)); R = rng.uniform(2.5, 4.5)
        if R - g / 2 < 0.5:
            continue
        cx, cy = rng.uniform(11, 19), rng.uniform(5.5, 9.5); cob = rng.uniform(0.5, 0.7); ab = rng.uniform(140, 200)
        x = arco(cx, cy, R, cob, ab, g)
        R2 = rng.uniform(3.5, 6); x |= arco(cx + rng.uniform(-1, 2), cy + R + R2 + rng.uniform(0, 2), R2, rng.uniform(0.5, 0.7), rng.uniform(150, 210), g)
        return {"tipo": "rizo", "x": x.astype(np.uint8), "centro": (float(cx), float(cy)), "radio": float(R),
                "cierre": float((cob - 0.5) / 0.5), "abertura": float(ab), "grosor": g}


def negativo(rng) -> dict:
    g = int(rng.integers(3, 7)); k = rng.integers(0, 5)
    if k == 0:    # un 1
        x = _poli([(rng.uniform(12, 20), 2), (rng.uniform(10, 18), 30)], g)
        if rng.random() < 0.5:
            x |= _poli([(rng.uniform(6, 12), 30), (rng.uniform(20, 26), 30)], g)
    elif k == 1:  # un 7
        x = _poli([(rng.uniform(5, 10), rng.uniform(2, 6)), (rng.uniform(20, 27), rng.uniform(2, 6)), (rng.uniform(8, 16), 30)], g)
    elif k == 2:  # un 4
        x = _poli([(rng.uniform(14, 20), 2), (rng.uniform(4, 10), rng.uniform(16, 22)), (rng.uniform(22, 28), rng.uniform(16, 22))], g)
        x |= _poli([(rng.uniform(16, 22), 6), (rng.uniform(14, 20), 30)], g)
    elif k == 3:  # una L o una esquina suelta
        a = (rng.uniform(6, 14), rng.uniform(4, 10)); x = _poli([a, (a[0], a[1] + rng.uniform(10, 20)), (a[0] + rng.uniform(8, 16), a[1] + rng.uniform(10, 20))], g)
    else:         # una recta o un arco abierto (< media vuelta) cualquiera
        if rng.random() < 0.5:
            x = recta(rng.uniform(6, 26), rng.uniform(6, 26), rng.uniform(10, 24), rng.uniform(0, 180), g)
        else:
            x = arco(rng.uniform(8, 24), rng.uniform(8, 24), rng.uniform(4, 10), rng.uniform(0.2, 0.45), rng.uniform(0, 360), g)
    return {"tipo": "negativo", "x": x.astype(np.uint8)}


def conjunto(n_pos: int, n_neg: int, semilla: int) -> list[dict]:
    rng = np.random.default_rng(semilla)
    pos = [dos(rng) if rng.random() < 0.8 else tres(rng) for _ in range(n_pos)]
    return pos + [negativo(rng) for _ in range(n_neg)]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    ej = conjunto(16, 8, 7)
    fig, axs = plt.subplots(3, 8, figsize=(13, 5.4))
    for ax, f in zip(axs.flat, ej):
        ax.imshow(f["x"], cmap="Greys", vmin=0, vmax=1.8); ax.set_xticks([]); ax.set_yticks([])
        if f["tipo"] == "rizo":
            ax.plot(f["centro"][0] - 0.5, f["centro"][1] - 0.5, "+", color="#2a78d6", ms=10, mew=2)
            ax.set_title(f"R {f['radio']:.1f} g {f['grosor']} cierre {f['cierre']:.2f} ab {f['abertura']:.0f}°", fontsize=7)
        else:
            ax.set_title("negativo", fontsize=7)
    fig.suptitle("rizo-2 · sintéticos del COMPOSITOR de doses (cabeza + diagonal + base; 20 % treses) · negativos: 1, 7, 4, L, rectas", fontsize=9)
    fig.tight_layout(); fig.savefig(Path(__file__).resolve().parents[1] / "resultados" / "1b-sinteticos-dos-muestra.png", dpi=90)
    print("→ resultados/1b-sinteticos-dos-muestra.png")
