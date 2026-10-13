#!/usr/bin/env python3
"""rizo-2 — HUECOS SUAVES: detector de huecos cerrados y concavidades para un mapa CONTINUO (p. ej. la línea media de la
CNN delgada, valores 0–1), sin binarizar en ningún paso. Pedido por el dueño el 2026-10-13.

La idea: desde cada píxel se lanzan 36 rayos de hasta R = 14 px y cada rayo devuelve su PARED, la altura máxima del mapa que
encuentra por el camino (0 = no hay nada, 1 = trazo seguro). Entonces, para cada píxel p:
    cerrado(p)    = (1 − M(p)) · percentil 10 de las 36 paredes     → alto sólo si CASI TODOS los rayos chocan con trazo
    concavidad(p) = (1 − M(p)) · percentil 35 · (1 − percentil 10)  → al menos 2/3 de los rayos chocan y ≥ 1/10 escapan
    (la primera versión usaba media − p10 y marcaba el EXTERIOR de cualquier pared: medio cielo de rayos choca y eso ya
    daba 0,4–0,5; con el percentil 35 una pared recta no cuenta, una C sí)
Los objetos son los máximos locales de cada mapa (supresión a 2 px), con su altura como fuerza, y los números del
compositor copian la forma de rizo_huecos.huecos_separados: 3×3 (máximo por celda) + cuenta (suma de fuerzas), 10 por
mapa, 20 en total. Con tinta binaria (0/1) se reduce exactamente a «¿chocan los rayos?», así que sirve también para la
referencia.

    python nn/huecos_suaves.py --entrada /ruta/delgados-k3x3.npz   → resultados/41-huecos-suaves.png (sintéticos + dígitos)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parent))

R_CERRADO = 14.0;   # 14 y no 8: el hueco de un 0 grande queda a 9–10 px de sus paredes y con 8 px no se cerraba (idx 0, medido 2026-10-13)
# N_RAYOS = 36; PASO = 0.5; CELDAS = 3; NMS = 3; MIN_FUERZA = 0.2
_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def paredes(M: np.ndarray, R: float = R_CERRADO) -> np.ndarray:
    """(n,32,32) → (n,36,32,32): para cada píxel y rayo, el máximo de M a lo largo del rayo hasta R px (bilineal)."""
    t = torch.from_numpy(np.asarray(M, np.float32))[:, None]; n = t.shape[0]
    ang = np.deg2rad(np.arange(N_RAYOS) * 360 / N_RAYOS); pasos = np.arange(PASO, R + 1e-6, PASO)
    base_y, base_x = torch.meshgrid(torch.arange(32.0) + 0.5, torch.arange(32.0) + 0.5, indexing="ij")
    out = torch.zeros(n, N_RAYOS, 32, 32)
    for k, a in enumerate(ang):
        best = torch.zeros(n, 32, 32)
        for s in pasos:
            gx = (base_x + s * np.cos(a)) / 16 - 1; gy = (base_y + s * np.sin(a)) / 16 - 1       # a coordenadas [-1,1]
            grid = torch.stack([gx, gy], -1)[None].expand(n, 32, 32, 2)
            v = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
            best = torch.maximum(best, v)
        out[:, k] = best
    return out.numpy()


def mapas(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,32,32) → (cerrado, concavidad), cada uno (n,32,32) en [0,1]."""
    M = np.asarray(M, np.float32); P = paredes(M)
    p10 = np.percentile(P, 10, axis=1); p35 = np.percentile(P, 35, axis=1); libre = 1 - M
    return libre * p10, libre * p35 * (1 - p10)


def objetos(mapa: np.ndarray) -> list[dict]:
    """Máximos locales (ventana 2·NMS+1) con fuerza ≥ MIN_FUERZA: [{cx, cy, fuerza}]."""
    t = torch.from_numpy(mapa)[None, None]; mx = F.max_pool2d(t, 2 * NMS + 1, stride=1, padding=NMS)[0, 0].numpy()
    ys, xs = np.nonzero((mapa >= mx - 1e-6) & (mapa >= MIN_FUERZA))
    obs = sorted([{"cx": x + 0.5, "cy": y + 0.5, "fuerza": float(mapa[y, x])} for y, x in zip(ys, xs)], key=lambda o: -o["fuerza"])
    out = []
    for o in obs:                                                   # supresión de vecinos a ≤ NMS px (mesetas)
        if all(np.hypot(o["cx"] - q["cx"], o["cy"] - q["cy"]) > NMS for q in out):
            out.append(o)
    return out


def caract(M: np.ndarray) -> np.ndarray:
    """(n,32,32) → (n,20): por mapa, 3×3 (máximo por celda) + cuenta suave (suma de fuerzas de los objetos)."""
    cer, con = mapas(M); out = []
    for c, k in zip(cer, con):
        f = []
        for m in (c, k):
            f.extend(F.adaptive_max_pool2d(torch.from_numpy(m)[None], CELDAS)[0].numpy().ravel().tolist())
            f.append(float(sum(o["fuerza"] for o in objetos(m))))
        out.append(f)
    return np.array(out, np.float32)


def _sinteticos():
    def anillo(r, g, cx=16, cy=16):
        d = np.hypot(_XX - cx, _YY - cy); return np.exp(-(d - r) ** 2 / (2 * (g / 2.35) ** 2)).astype(np.float32)
    anillo_c = anillo(6, 2.5)
    ce = anillo(6, 2.5) * (np.abs(np.degrees(np.arctan2(_YY - 16, _XX - 16))) > 35)          # C abierta hacia la derecha
    rendija = anillo(6, 2.5) * (np.abs(np.degrees(np.arctan2(_YY - 16, _XX - 16))) > 6)      # casi cerrada (12° abiertos)
    debil = 0.4 * anillo_c                                                                     # anillo con poca confianza
    recta = np.exp(-((_XX - 16) ** 2) / (2 * 1.0 ** 2)).astype(np.float32) * ((_YY > 4) & (_YY < 28))
    return [("anillo", anillo_c), ("C abierta 70°", ce), ("rendija 12°", rendija), ("anillo débil 0,4", debil), ("recta", recta)]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", default=None); ap.add_argument("--clave", default="M"); a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ejemplos = _sinteticos()
    if a.entrada:
        from expcnn import exigir_dataset
        import evaluar as EV
        z = np.load(a.entrada); d = np.load(exigir_dataset(EV.DATASET) / "datos.npz")
        for i in (816, 890, 898, 1423, 66, 1765, 27, 1562):          # 8 que fallan sin huecos · 6 · 3 · 7 · 9
            ejemplos.append((f"{d['etiquetas'][i]} · idx {i}", z[a.clave][i].astype(np.float32)))
    fig, axs = plt.subplots(3, len(ejemplos), figsize=(1.7 * len(ejemplos), 5.4))
    for j, (nombre, m) in enumerate(ejemplos):
        cer, con = mapas(m[None]); cer, con = cer[0], con[0]
        for k, (img, cmap, tit) in enumerate(((m, "Greys", "mapa"), (cer, "viridis", "cerrado"), (con, "magma", "concavidad"))):
            ax = axs[k, j]; ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if k else 1.6, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if k:
                for o in objetos(img):
                    ax.plot(o["cx"] - 0.5, o["cy"] - 0.5, "+", color="w" if k == 1 else "c", ms=8, mew=1.5)
                ax.set_title(f"{tit}: {', '.join(f'{o['fuerza']:.2f}' for o in objetos(img)[:3]) or '—'}", fontsize=6.5)
            else:
                ax.set_title(nombre, fontsize=7)
    fig.suptitle("rizo-2 · 41 · huecos SUAVES: cerrado = (1−M)·percentil10(paredes de 36 rayos, 14 px) · concavidad = (1−M)·p35·(1−p10) · + = objeto (fuerza)", fontsize=8.5)
    fig.tight_layout(); out = AQUI / "resultados" / "41-huecos-suaves.png"; fig.savefig(out, dpi=120); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
