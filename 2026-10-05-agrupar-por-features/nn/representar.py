#!/usr/bin/env python3
"""Las DESCRIPCIONES de los 5620 dígitos que se agrupan (REGLAS.md §3), limpios y con las tres transformaciones de L4, y los
tres índices de píxeles de L2b. SIN etiquetas: `datos.cargar()` no las devuelve.

  Z32  máximo de cada mapa del banco de 32 en 9 zonas solapadas (ventanas 4×4 celdas, paso 2)   13 × 9 = 117
  P32  máximo de cada mapa (presencia)                                                            13
  M32  los 13 mapas enteros                                                                       832
  Z8   como Z32, con el banco de 8×8                                                              117
  X8   los píxeles: el 8×8 contado (cuentas 4×4 ÷ 16)                                             64
  (Z32-a / Z8-a, sin los 4 arcos, salen de Z quitando columnas: no se guardan aparte)

    python nn/representar.py      → resultados/representaciones.npz (no se commitea: se regenera en minutos)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos                                    # noqa: E402
import detectores as D                          # noqa: E402

RES = AQUI.parent / "resultados"
REPS = RES / "representaciones.npz"
VERSIONES = ("limpio", "desplazar", "engrosar", "adelgazar")
INICIOS = (0, 2, 4)                              # las ventanas 0–3, 2–5, 4–7 en filas y en columnas
ZONAS = ("NW", "N", "NE", "W", "C", "E", "SW", "S", "SE")
BRAZOS = ("Z32", "P32", "M32", "Z8", "X8")
BANCO = {"Z32": "32", "P32": "32", "M32": "32", "Z8": "8", "Z32-a": "32", "Z8-a": "8"}
# COPIADO de `feat-ind32` (nn/componer.py): la misma tupla y la misma semilla, para que la definición sea la misma.
TRANSFORMACIONES = ("desplazar", "ruido", "engrosar", "adelgazar", "ocluir")
SEMILLA_T = 2027


def transformar(x: np.ndarray, que: str) -> np.ndarray:
    """COPIA de `feat-ind32` (nn/componer.py, C6), sólo los tres casos que se usan aquí. x (N,1,32,32) 0/1:
    desplazar 1 celda = 4 px en una de 8 direcciones · engrosar = dilatar 2 px · adelgazar = erosionar 1 px."""
    rng = np.random.default_rng(SEMILLA_T + TRANSFORMACIONES.index(que))
    t = torch.from_numpy(x)
    if que == "desplazar":
        out = np.zeros_like(x)
        dirs = [(dy, dx) for dy in (-4, 0, 4) for dx in (-4, 0, 4) if (dy, dx) != (0, 0)]
        for i, k in enumerate(rng.integers(0, 8, len(x))):
            dy, dx = dirs[k]
            src = x[i, 0, max(0, -dy):32 - max(0, dy), max(0, -dx):32 - max(0, dx)]
            out[i, 0, max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
        return out
    if que == "engrosar":
        return Fn.max_pool2d(t, 5, 1, 2).numpy()
    if que == "adelgazar":
        return (-Fn.max_pool2d(-t, 3, 1, 1)).numpy()
    raise ValueError(que)


def zonas(s: np.ndarray) -> np.ndarray:
    """σ (N, 13, 8, 8) → (N, 13 × 9): el máximo de cada mapa en cada zona, en el orden (feature, zona) con ZONAS."""
    out = np.empty((len(s), s.shape[1], 3, 3), np.float32)
    for i, r in enumerate(INICIOS):
        for j, c in enumerate(INICIOS):
            out[:, :, i, j] = s[:, :, r:r + 4, c:c + 4].max((2, 3))
    return out.reshape(len(s), -1)


def sin_arcos(z: np.ndarray) -> np.ndarray:
    """Z (N, 117) → Z−a (N, 81): fuera las 4 features de arco (las 4 primeras de FAMILIAS)."""
    return z.reshape(len(z), len(D.FAMILIAS), 9)[:, len(D.ARCOS):].reshape(len(z), -1)


def indices(x32: np.ndarray) -> dict:
    """Los tres índices de L2b, en px de 32 y SIN detectores. Con las filas de tinta r0…r1 (alto h) e I(r) la columna de
    tinta más a la izquierda de la fila r:
      bandera     = mediana de I en el tercio central − mínimo de I en el cuarto superior (1 recto ≈ 0, con bandera > 0, «/» < 0)
      grosor      = píxeles de tinta ÷ filas con tinta
      inclinación = grados del eje principal de la tinta respecto de la vertical («/» < 0, «\\» > 0)."""
    n = len(x32); b, g, inc = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        im = x32[i, 0] > 0.5
        filas = np.flatnonzero(im.any(1))
        if len(filas) == 0:
            continue
        r0, r1 = filas[0], filas[-1]; h = r1 - r0 + 1
        izq = {int(r): int(np.flatnonzero(im[r])[0]) for r in filas}
        medio = [izq[r] for r in izq if r0 + h / 3 <= r <= r0 + 2 * h / 3]
        arriba = [izq[r] for r in izq if r <= r0 + h / 4]
        b[i] = float(np.median(medio) - min(arriba)) if medio and arriba else 0.0
        g[i] = im.sum() / len(filas)
        yy, xx = np.nonzero(im)
        if len(yy) >= 2:
            vals, vecs = np.linalg.eigh(np.cov(np.stack([xx, yy]).astype(np.float64)))
            vx, vy = vecs[:, -1]
            ang = float(np.degrees(np.arctan2(vx, vy)))
            inc[i] = ang - 180 if ang > 90 else (ang + 180 if ang <= -90 else ang)
    return {"bandera": b, "grosor": g, "inclinacion": inc}


def main() -> int:
    t0 = time.time()
    d = datos.cargar()
    torch.set_num_threads(2)
    reds = {b: D.cargar(b) for b in D.BANCOS}
    reps = {}
    for v in VERSIONES:
        x32 = d["x32"] if v == "limpio" else transformar(d["x32"], v)
        x8 = datos.reducir(x32)
        s32 = D.mapas("32", x32, reds=reds["32"][0])
        s8 = D.mapas("8", x8, reds=reds["8"][0])
        reps[f"Z32/{v}"] = zonas(s32)
        reps[f"P32/{v}"] = s32.max((2, 3))
        reps[f"M32/{v}"] = s32.reshape(len(s32), -1)
        reps[f"Z8/{v}"] = zonas(s8)
        reps[f"X8/{v}"] = x8.reshape(len(x8), -1)
        print(f"  {v:<10} mapas de los dos bancos en {time.time() - t0:.0f} s", flush=True)
    ind = indices(d["x32"])
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(REPS, **reps, **ind, umbrales32=reds["32"][1], umbrales8=reds["8"][1], origen=d["origen"])
    print(f"→ {REPS.relative_to(AQUI.parent)}: {len(d['origen'])} dígitos × {len(BRAZOS)} brazos × {len(VERSIONES)} versiones "
          f"+ índices de L2b, en {time.time() - t0:.0f} s")
    return 0


def cargar() -> dict:
    if not REPS.is_file():
        raise SystemExit(f"✗ no está {REPS.name}: primero `python nn/representar.py`")
    return dict(np.load(REPS))


def rep(r: dict, brazo: str, version: str = "limpio") -> np.ndarray:
    """La descripción de un brazo; Z32-a y Z8-a salen de Z quitando los arcos."""
    if brazo.endswith("-a"):
        return sin_arcos(r[f"{brazo[:-2]}/{version}"])
    return r[f"{brazo}/{version}"]


if __name__ == "__main__":
    raise SystemExit(main())
