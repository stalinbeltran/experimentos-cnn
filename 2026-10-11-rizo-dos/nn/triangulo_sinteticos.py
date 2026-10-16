#!/usr/bin/env python3
"""rizo-2 — SINTÉTICOS del TRIÁNGULO del 4 (el dueño, 2026-10-16: «vamos a entrenar un detector del triángulo formado por el 4,
usando sintéticos; muéstrame imágenes de muestras»). Primera pieza: el generador y una figura de muestras; nada se entrena.

Un «4 cerrado» es un VÉRTICE del que bajan dos trazos —el palo, casi vertical, y la raya izquierda, en diagonal— y una BARRA
horizontal que los une por debajo: el triángulo. El palo sigue por debajo de la barra y la barra sobresale a la derecha, como
en un 4 real. Según el grosor y el tamaño, el triángulo queda con hueco (lazo) o macizo (relleno: los 198/225/239/250 de val).

Estilo de tinta: el de delg-cnn (`sinteticos.py`): polilíneas con ondulación suave y grosor 3–6 px, engrosadas como unión de
discos. La VERDAD de cada muestra: los tres vértices (vértice, pie izquierdo, cruce barra-palo), el centro del triángulo y su
máscara 32×32 (el interior del triángulo, incluida la tinta de sus lados).

Negativos, los que de verdad se confunden: (a) el 4 ABIERTO: la raya izquierda no llega al vértice (hueco arriba); (b) el «ч»: la
raya izquierda baja y se junta con el palo A LA ALTURA de la barra, y el palo asoma poco o nada por encima; (c) trazos sueltos
de `curva_natural` (recta, curva, lazo), que es lo que hay en los demás dígitos.

    ../.venv/bin/python nn/triangulo_sinteticos.py   → resultados/92-triangulo-sinteticos.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parent / "2026-10-15-delgado-cnn" / "nn"))
import sinteticos as SN                                                  # noqa: E402  (delg-cnn: _suave, engrosar, curva_natural)

N = 32
_YY, _XX = np.mgrid[0:N, 0:N] + 0.5


def _tramo(rng, a, b, paso=0.25, ondula=0.35):
    """Polilínea de a a b con ondulación suave perpendicular (como un trazo a mano)."""
    a, b = np.asarray(a, float), np.asarray(b, float); L = np.hypot(*(b - a)); n = max(int(L / paso), 2)
    t = np.linspace(0, 1, n)[:, None]; p = a + t * (b - a); d = (b - a) / (L + 1e-9); nrm = np.array([-d[1], d[0]])
    w = SN._suave(rng, n, escala=ondula, sigma_pasos=max(4, n / 6)) * np.sin(np.pi * t[:, 0])   # cero en los extremos
    return p + w[:, None] * nrm


def _grosor(rng, n, g0=None, g_min=3.0, g_max=6.0):
    g0 = rng.uniform(g_min, g_max) if g0 is None else g0; return np.clip(g0 + SN._suave(rng, n, escala=0.5, sigma_pasos=15), g_min, g_max)


def _dentro(poly):
    """Máscara 32×32 del interior de un polígono (par-impar sobre centros de píxel)."""
    poly = np.asarray(poly, float); m = np.zeros((N, N), bool)
    for i in range(len(poly)):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % len(poly)]
        m ^= ((y0 > _YY) != (y1 > _YY)) & (_XX < x0 + (_YY - y0) * (x1 - x0) / (y1 - y0 + 1e-12))
    return m


def cuatro(rng, cerrado: bool = True, tipo_abierto: str = "hueco") -> dict:
    """Un 4 sintético. cerrado=True → triángulo (positivo). cerrado=False → 'hueco' (la raya no llega al vértice) o 'ch' (se junta
    a la altura de la barra). Devuelve tinta, verdad y parámetros."""
    xa = rng.uniform(13, 21); ya = rng.uniform(1.5, 6)                   # vértice
    yb = rng.uniform(13, 21)                                             # altura de la barra
    ancho = rng.uniform(6, 13); xl = max(1.5, xa - ancho)                # pie izquierdo del triángulo
    xr = min(30.5, xa + rng.uniform(3, 9))                               # fin de la barra a la derecha
    yfin = rng.uniform(26, 30.5)                                         # fin del palo
    incl = rng.uniform(-2.5, 2.5)                                        # el palo se inclina un poco
    g0 = rng.uniform(3.0, 6.0)                                           # grosor base del dígito (los tramos varían alrededor)
    tramos = []
    if cerrado:
        A = (xa, ya); L = (xl, yb + rng.uniform(-1, 1)); C = (xa + incl * (yb - ya) / (yfin - ya), yb)
        tramos.append(_tramo(rng, A, (xa + incl, yfin)))                 # palo entero
        tramos.append(_tramo(rng, A, L))                                 # raya izquierda hasta el pie
        tramos.append(_tramo(rng, (xl - rng.uniform(0, 2), L[1]), (xr, yb + rng.uniform(-0.8, 0.8))))   # barra
        verdad = {"vertice": A, "pie": L, "cruce": C, "centro": tuple(np.mean([A, L, C], 0)), "mascara": _dentro([A, L, C])}
    elif tipo_abierto == "hueco":                                        # la raya izquierda empieza más abajo y a la izquierda: no toca el palo
        # ⚠ el hueco se mide DESCONTANDO el grosor: con 3–7 px de separación y 6 px de tinta, la tinta tapaba el hueco y el
        # «abierto» salía cerrado (visto en la primera figura). Separación libre entre tintas ≥ 1,5 px.
        y_ini = ya + rng.uniform(1, 5); x_ini = xa - g0 - rng.uniform(1.5, 5); L = (xl, yb + rng.uniform(-1, 1))
        tramos.append(_tramo(rng, (xa, ya), (xa + incl, yfin)))
        tramos.append(_tramo(rng, (x_ini, y_ini), L))
        tramos.append(_tramo(rng, (xl - rng.uniform(0, 2), L[1]), (xr, yb + rng.uniform(-0.8, 0.8))))
        verdad = {"mascara": np.zeros((N, N), bool)}
    else:                                                                # «ч»: la raya baja en curva y se junta con el palo a la altura de la barra
        y_ini = rng.uniform(1.5, 6); x_ini = xa - rng.uniform(5, 11); asoma = rng.uniform(0, 4)
        tramos.append(_tramo(rng, (xa, yb - asoma), (xa + incl, yfin)))
        cod = np.array([x_ini - 2, (y_ini + yb) / 2 + 3]); t = np.linspace(0, 1, 60)[:, None]
        p0, p2 = np.array([x_ini, y_ini]), np.array([xa, yb]); curva = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * cod + t**2 * p2
        tramos.append(curva)
        tramos.append(_tramo(rng, (xa - rng.uniform(0, 3), yb), (xr, yb + rng.uniform(-0.8, 0.8))))
        verdad = {"mascara": np.zeros((N, N), bool)}
    tinta = np.zeros((N, N), bool)
    for p in tramos:
        tinta |= SN.engrosar(p, _grosor(rng, len(p), g0))
    return {"tinta": tinta, "cerrado": cerrado, "tipo": "triangulo" if cerrado else tipo_abierto, "verdad": verdad}


def suelto(rng) -> dict:
    c = SN.curva_natural(rng); return {"tinta": SN.engrosar(c["puntos"], c["grosor"]), "cerrado": False, "tipo": "suelto:" + c["tipo"], "verdad": {"mascara": np.zeros((N, N), bool)}}


def muestra(rng, n: int):
    """n muestras: 40 % triángulo, 20 % hueco, 20 % ч, 20 % suelto."""
    out = []
    for k in range(n):
        u = rng.uniform()
        out.append(cuatro(rng, True) if u < 0.4 else cuatro(rng, False, "hueco") if u < 0.6 else cuatro(rng, False, "ch") if u < 0.8 else suelto(rng))
    return out


def main() -> int:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import torch, json
    rng = np.random.default_rng(0)
    # delg-cnn para enseñar también cómo quedan en DELGADO (lo que ven los detectores de la cresta)
    sys.path.insert(0, str(AQUI.parent / "2026-10-15-delgado-cnn" / "nn")); from entrenar_local import SoloKernels   # noqa: E402
    pesos = AQUI.parent / "2026-10-15-delgado-cnn" / "nn" / "pesos" / "k3x3"; cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    grupos = [("TRIÁNGULO (positivo)", [cuatro(rng, True) for _ in range(12)]), ("abierto: HUECO arriba", [cuatro(rng, False, "hueco") for _ in range(12)]),
              ("abierto: «ч»", [cuatro(rng, False, "ch") for _ in range(12)]), ("sueltos (curva_natural)", [suelto(rng) for _ in range(12)])]
    fig, axs = plt.subplots(8, 12, figsize=(12 * 1.35, 8 * 1.5))
    for ax in axs.ravel():
        ax.axis("off")
    for f, (tit, ms) in enumerate(grupos):
        X = torch.from_numpy(np.stack([m["tinta"] for m in ms]).astype(np.float32))[:, None]
        with torch.no_grad():
            M = red(X)[:, 0].numpy()
        for k, m in enumerate(ms):
            ax = axs[2 * f, k]; ax.imshow(m["tinta"], cmap="Greys", vmin=0, vmax=1)
            if m["cerrado"]:
                v = m["verdad"]; ax.imshow(np.where(v["mascara"], 1.0, np.nan), cmap="Oranges", vmin=0, vmax=2, alpha=0.45)
                for key, mk in (("vertice", "^"), ("pie", "<"), ("cruce", "x")):
                    ax.plot(v[key][0] - 0.5, v[key][1] - 0.5, mk, color="#1f6fd6", ms=5)
                ax.plot(v["centro"][0] - 0.5, v["centro"][1] - 0.5, "+", color="#b22222", ms=8, mew=1.5)
            ax.set_title(("" if k else tit + " · ") + "tinta", fontsize=6.5, loc="left")
            axs[2 * f + 1, k].imshow(M[k], cmap="Greys", vmin=0, vmax=1); axs[2 * f + 1, k].set_title("delgado (delg-cnn)", fontsize=6.5, loc="left")
    fig.suptitle("rizo-2 · 92 · sintéticos del TRIÁNGULO del 4 (fila impar: tinta con la verdad —naranja = interior del triángulo, ▲ vértice, ◀ pie, × cruce, + centro—; fila par: su delgado)\n"
                 "grosor 3–6 px con ondulación; negativos: 4 abierto con hueco arriba, «ч» que se junta a la altura de la barra, y trazos sueltos", fontsize=9)
    fig.tight_layout(); out = AQUI / "resultados" / "92-triangulo-sinteticos.png"; fig.savefig(out, dpi=110); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
