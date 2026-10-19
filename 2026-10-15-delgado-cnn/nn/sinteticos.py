#!/usr/bin/env python3
"""delg-cnn — trazos SINTÉTICOS con su línea media conocida.

Primera pieza (2026-10-13): curvas «naturales», como hechas a mano, y su engrosado.

Una curva nace de un modelo de bolígrafo: se avanza a pasos de 0,25 px con un rumbo cuya
curvatura es ruido suavizado (así la curva ondula sin esquinas), con un tirón hacia el centro
cuando se acerca al borde del lienzo de 32×32. La presión (grosor) también varía despacio a lo
largo del trazo, entre `g_min` y `g_max` (3–6 px desde el 2026-10-13: el grosor medido en UCI val es mediana 4,5, p90 6; ni los bloques de los 1 gruesos ni hilos de 2 px, regla del dueño). La verdad es la polilínea de puntos (sub-píxel).

El engrosado es la unión de discos de radio g(t)/2 centrados en cada punto de la polilínea:
la tinta queda centrada en la línea media por construcción, que es lo que la red debe aprender.

    python nn/sinteticos.py   → resultados/0-curvas-muestra.png (tinta + línea media encima)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
N = 32
_YY, _XX = np.mgrid[0:N, 0:N] + 0.5          # centro de cada píxel


def _suave(rng, n: int, escala: float, sigma_pasos: float) -> np.ndarray:
    """Ruido gaussiano filtrado (media móvil gaussiana): señal lenta de desviación ~`escala`."""
    k = int(3 * sigma_pasos)
    t = np.arange(-k, k + 1)
    w = np.exp(-t**2 / (2 * sigma_pasos**2)); w /= w.sum()
    ruido = rng.standard_normal(n + 2 * k)
    s = np.convolve(ruido, w, mode="valid")
    return s / (s.std() + 1e-9) * escala


def curva_natural(rng, largo_px: float | None = None, paso: float = 0.25,
                  g_min: float = 3.0, g_max: float = 6.0, margen: float = 1.5) -> dict:
    """Una curva a mano alzada dentro del lienzo: puntos (n,2) en píxeles y grosor (n,) por punto."""
    L = largo_px if largo_px is not None else rng.uniform(14, 40)
    n = int(L / paso)
    tipo = rng.choice(["recta", "curva", "lazo"], p=[0.3, 0.45, 0.25])
    if tipo == "recta":                                   # casi recta: curvatura ínfima
        kappa = _suave(rng, n, escala=rng.uniform(0.0, 0.03), sigma_pasos=rng.uniform(10, 25))
    elif tipo == "curva":
        kappa = _suave(rng, n, escala=rng.uniform(0.04, 0.18), sigma_pasos=rng.uniform(6, 20))
    else:                                                 # UNA vuelta como mucho (el lazo del 6, el 9, el 2)
        kappa = _suave(rng, n, escala=0.05, sigma_pasos=15) + rng.choice([-1, 1]) * rng.uniform(0.15, 0.3)
    # radio mínimo 3 px: un lazo más cerrado queda macizo con cualquier grosor y no aporta
    kappa = np.clip(kappa, -1 / 3, 1 / 3)
    # y nunca más de ~1,15 vueltas en total: varias vueltas en el mismo sitio no se distinguen
    giro = np.cumsum(np.abs(kappa)) * paso
    kappa[giro > 1.15 * 2 * np.pi] = 0
    g = np.clip(rng.uniform(g_min, g_max) + _suave(rng, n, escala=0.6, sigma_pasos=15), g_min, g_max)

    p = np.array([rng.uniform(6, N - 6), rng.uniform(6, N - 6)])
    th = rng.uniform(0, 2 * np.pi)
    pts = np.empty((n, 2))
    for i in range(n):
        pts[i] = p
        # tirón hacia el centro cuando el punto se acerca al borde (evita salirse sin romper la suavidad)
        hacia = np.array([N / 2, N / 2]) - p
        dist_borde = min(p.min(), N - p.max())
        if dist_borde < 6:
            ang_centro = np.arctan2(hacia[1], hacia[0])
            dth = (ang_centro - th + np.pi) % (2 * np.pi) - np.pi
            th += np.clip(dth, -0.12, 0.12) * (6 - dist_borde) / 6
        th += kappa[i] * paso
        p = p + paso * np.array([np.cos(th), np.sin(th)])
    # recorte duro por si el tirón no bastó: el trazo acaba donde toca el margen
    dentro = (pts.min(1) >= margen) & (pts.max(1) <= N - margen)
    corte = np.argmin(dentro) if not dentro.all() else n
    pts, g = pts[:corte], g[:corte]
    return {"puntos": pts, "grosor": g, "tipo": str(tipo)}


def dos_trazos(rng, c1: dict, margen: float = 1.5) -> dict | None:
    """Un SEGUNDO trazo anclado en el primero (2026-10-18, rizo-2 § 121: el adelgazador sólo había visto una curva por imagen y
    borraba la barra del 7): con prob. ½ empieza en un punto de c1 (una T: la barra o el brazo que sale de un trazo) y si no, lo
    cruza por su mitad (una X). La mitad de las veces es CORTO (5–14 px, como la barra de un 7). None si no cabe en el lienzo."""
    c2 = curva_natural(rng, largo_px=rng.uniform(5, 14) if rng.random() < 0.5 else None)
    p1, p2 = c1["puntos"], c2["puntos"]
    if len(p1) < 8 or len(p2) < 8:
        return None
    q = p1[rng.integers(len(p1) // 6, len(p1) - len(p1) // 6)]               # ni en las puntas
    ancla = 0 if rng.random() < 0.5 else len(p2) // 2
    p2 = p2 - p2[ancla] + q
    dentro = (p2.min(1) >= margen) & (p2.max(1) <= N - margen)
    if dentro.sum() < 8:
        return None
    return {"puntos": p2[dentro], "grosor": c2["grosor"][dentro], "tipo": "T" if ancla == 0 else "X"}


def casi_cerrado(rng, paso: float = 0.25, g_min: float = 3.0, g_max: float = 6.0, margen: float = 1.5) -> dict:
    """Un trazo CASI CERRADO, como un 0 hecho de un tirón (2026-10-19, el dueño: «empiezas en un punto, haces casi todo el trazo como
    un cero pero en vez de cerrar en el punto exacto donde empezaste terminas un poco más arriba o más abajo»).

    Un óvalo con el radio ondulando despacio; se empieza arriba (± 50°) y se da algo menos o algo más de una vuelta (−25° … +60°).
    En el último tercio del trazo el bolígrafo se DESVÍA de la trayectoria cerrada (1,5–3,5 px hacia dentro o hacia fuera, y 0,5–2 px arriba o
    abajo: con 1–3,5 / ±1,5 no se veía, con 2–5 / 1–3 algunos parecían un 6 o una U; el dueño eligió el término medio), así que el final no cae sobre el principio: queda una puntita o un solapamiento corrido, que es la tinta que se ve en
    los 0 reales (#304, #357, #980 de val, rizo-2 § 125). La verdad es la polilínea entera, puntita incluida."""
    a, b = rng.uniform(4.5, 8.5), rng.uniform(8.0, 12.5)                    # semiejes: más alto que ancho, como un 0
    cx, cy = rng.uniform(N / 2 - 2, N / 2 + 2), rng.uniform(N / 2 - 1.5, N / 2 + 1.5)
    giro = rng.choice([-1, 1]); t0 = -np.pi / 2 + np.radians(rng.uniform(-50, 50))   # empieza arriba (y crece hacia abajo)
    total = 2 * np.pi + np.radians(rng.uniform(-25, 60))
    n = int(total * (a + b) / 2 / paso)
    t = t0 + giro * np.linspace(0, total, n)
    r = 1 + _suave(rng, n, escala=0.04, sigma_pasos=40)
    pts = np.stack([cx + a * r * np.cos(t), cy + b * r * np.sin(t)], 1)
    rampa = np.clip((np.arange(n) / n - 2 / 3) * 3, 0, 1) ** 1.5             # 0 hasta 2/3 del trazo, 1 al final
    desvio = rng.uniform(1.5, 3.5) * rng.choice([-1, 1])
    radial = np.stack([np.cos(t), np.sin(t)], 1)                             # + hacia fuera, − hacia dentro
    pts = pts + rampa[:, None] * (desvio * radial + np.array([0, rng.uniform(0.5, 2.0) * rng.choice([-1, 1])]))
    g = np.clip(rng.uniform(g_min, g_max) + _suave(rng, n, escala=0.6, sigma_pasos=15), g_min, g_max)
    dentro = (pts.min(1) >= margen) & (pts.max(1) <= N - margen)
    return {"puntos": pts[dentro], "grosor": g[dentro], "tipo": f"casi-0 {np.degrees(total) - 360:+.0f}° {desvio:+.1f}px",
            "cola": (rampa > 0.3)[dentro]}                                  # los puntos de la puntita desviada: lo que hay que conservar


def engrosar(puntos: np.ndarray, grosor: np.ndarray) -> np.ndarray:
    """Tinta binaria 32×32: unión de discos de radio grosor/2 en cada punto de la polilínea."""
    x = np.zeros((N, N), bool)
    for (px, py), g in zip(puntos, grosor):
        x |= np.hypot(_XX - px, _YY - py) <= g / 2
    return x


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    muestras = []
    while len(muestras) < 20:
        c = curva_natural(rng)
        if len(c["puntos"]) * 0.25 >= 10:            # descarta trazos que el borde dejó cortos
            muestras.append(c)
    fig, axs = plt.subplots(2, 10, figsize=(18, 4.4))
    for ax, c in zip(axs.flat, muestras):
        tinta = engrosar(c["puntos"], c["grosor"])
        ax.imshow(tinta, cmap="Greys", vmin=0, vmax=1.6, extent=(0, N, N, 0), interpolation="nearest")
        ax.plot(c["puntos"][:, 0], c["puntos"][:, 1], "-", color="#e8432b", lw=1.2)
        ax.plot(c["puntos"][0, 0], c["puntos"][0, 1], "o", color="#2a78d6", ms=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{c['tipo']} · L {len(c['puntos']) * 0.25:.0f} px · g {c['grosor'].min():.1f}–{c['grosor'].max():.1f}", fontsize=7)
    fig.suptitle("delg-cnn · curvas a mano alzada (rojo: línea media, punto azul: inicio) y su engrosado (gris) · 32×32", fontsize=9)
    fig.tight_layout(); (AQUI / "resultados").mkdir(exist_ok=True)
    out = AQUI / "resultados" / "0-curvas-muestra.png"
    fig.savefig(out, dpi=110)
    print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
