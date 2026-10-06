#!/usr/bin/env python3
"""NORMALIZAR EL GROSOR de un trazo binario de 32×32: esqueleto (adelgazamiento de Zhang y Suen, 1984) y vuelta a engrosar
a un grosor fijo. Un dígito de 4 px y el mismo dígito de 12 px dan (casi) el mismo resultado: el grosor deja de existir
antes de que lo vea ningún detector. Es la solución S3 del estudio (REGLAS.md).

    python nn/normalizar.py      comprueba el adelgazamiento en casos hechos a mano
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as Fn


def esqueleto(x: np.ndarray, max_iter: int = 40) -> np.ndarray:
    """(N, 32, 32) o (N, 1, 32, 32) 0/1 → (N, 32, 32) uint8: el esqueleto de 1 px (Zhang–Suen, vectorizado por lote).
    Fuera del lienzo cuenta como fondo."""
    x = (np.asarray(x) > 0.5).astype(np.uint8)
    if x.ndim == 4:
        x = x[:, 0]
    img = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    for _ in range(max_iter):
        cambio = False
        for paso in (0, 1):
            p2, p3, p4 = img[:, :-2, 1:-1], img[:, :-2, 2:], img[:, 1:-1, 2:]
            p5, p6, p7 = img[:, 2:, 2:], img[:, 2:, 1:-1], img[:, 2:, :-2]
            p8, p9, c = img[:, 1:-1, :-2], img[:, :-2, :-2], img[:, 1:-1, 1:-1]
            vec = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(v.astype(np.int16) for v in vec[:8])
            a = sum(((vec[i] == 0) & (vec[i + 1] == 1)).astype(np.int16) for i in range(8))
            if paso == 0:
                m = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 & p4 & p6) == 0) & ((p4 & p6 & p8) == 0)
            else:
                m = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 & p4 & p8) == 0) & ((p2 & p6 & p8) == 0)
            if m.any():
                cambio = True
                centro = img[:, 1:-1, 1:-1]
                centro[m] = 0
        if not cambio:
            break
    return img[:, 1:-1, 1:-1].copy()


def engrosar(x: np.ndarray, ancho: int) -> np.ndarray:
    """(N, 32, 32) 0/1 → (N, 32, 32) uint8: dilatación con un cuadrado de `ancho` px (1 = nada)."""
    if ancho <= 1:
        return x.astype(np.uint8)
    t = torch.from_numpy(x.astype(np.float32))[:, None]
    izq = (ancho - 1) // 2
    t = Fn.pad(t, (izq, ancho - 1 - izq, izq, ancho - 1 - izq))
    return Fn.max_pool2d(t, ancho, 1, 0)[:, 0].numpy().astype(np.uint8)


def normalizar(x: np.ndarray, ancho: int = 3) -> np.ndarray:
    """El trazo, a `ancho` px: esqueleto y dilatación. (N, 32, 32)/(N, 1, 32, 32) → (N, 1, 32, 32) uint8."""
    return engrosar(esqueleto(x), ancho)[:, None]


def desinclinar(x: np.ndarray) -> np.ndarray:
    """(N,1,32,32) o (N,32,32) 0/1 → (N,1,32,32) uint8, cizallado para que el eje de la tinta quede VERTICAL (la iteración 4).

    α = cov(fila, columna) / var(fila) de la tinta: columnas que se corre por cada fila. La salida en (r, c) toma la entrada en
    (r, c + α·(r − f0)), con f0 la fila del centro de masas, así que la cizalla pasa por él y el dígito no se desplaza en
    vertical. Vecino más cercano: la imagen sigue siendo 0/1 (el esqueleto y los bordes lo necesitan). Lo que la cizalla saca
    del lienzo se pierde; `perdida()` lo cuenta."""
    t = (np.asarray(x).reshape(len(x), 32, 32) > 0.5)
    fil, col = np.mgrid[:32, :32]
    m = t.sum((1, 2)).clip(1)
    f0 = (t * fil).sum((1, 2)) / m; c0 = (t * col).sum((1, 2)) / m
    vf = (t * (fil - f0[:, None, None]) ** 2).sum((1, 2)) / m
    cv = (t * (fil - f0[:, None, None]) * (col - c0[:, None, None])).sum((1, 2)) / m
    a = cv / vf.clip(1e-9)
    src = np.rint(col[None] + a[:, None, None] * (fil[None] - f0[:, None, None])).astype(np.int64)
    ok = (src >= 0) & (src < 32)
    n = np.broadcast_to(np.arange(len(t))[:, None, None], src.shape)
    r = np.broadcast_to(fil[None], src.shape)
    out = np.zeros(t.shape, np.uint8)
    out[ok] = t[n[ok], r[ok], src[ok]]
    return out[:, None]


def perdida(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fracción de la tinta de x que no está en y (la que la cizalla sacó del lienzo, o la que el vecino más cercano fundió)."""
    a = np.asarray(x).reshape(len(x), -1).sum(1); b = np.asarray(y).reshape(len(y), -1).sum(1)
    return 1 - b / np.maximum(1, a)


def comprobar() -> int:
    ok = True

    def mira(que, cond):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FALLA':>5}] {que}"); ok &= bool(cond)

    barra = np.zeros((2, 32, 32), np.uint8); barra[0, 4:28, 10:20] = 1; barra[1, 4:28, 14:16] = 1    # 10 px y 2 px
    e = esqueleto(barra)
    anchos = [(e[i, 8:24].sum(1)).max() for i in range(2)]
    mira("el esqueleto de una barra de 10 px y de una de 2 px tiene 1 px de ancho en el tramo central", anchos == [1, 1])
    cols = [set(np.nonzero(e[i, 8:24])[1].tolist()) for i in range(2)]
    mira("y cae en el centro de la barra (± 1 px)", min(cols[0]) >= 13 and max(cols[0]) <= 16 and min(cols[1]) >= 14)
    n = normalizar(barra, 3)
    mira("normalizada a 3 px, las dos barras miden 3 px de ancho en el centro", [(n[i, 0, 8:24].sum(1)).max() for i in range(2)] == [3, 3])
    anillo = np.zeros((1, 32, 32), np.uint8)
    yy, xx = np.mgrid[:32, :32]; r = np.hypot(yy - 16, xx - 16); anillo[0][(r >= 6) & (r <= 11)] = 1
    ea = esqueleto(anillo)[0]
    mira("un anillo grueso sigue siendo un lazo (su esqueleto no tiene extremos sueltos y no toca el centro)",
         ea.sum() > 20 and ea[14:19, 14:19].sum() == 0)
    inclinada = np.zeros((1, 32, 32), np.uint8)
    for f in range(4, 28):
        c = int(round(16 + 0.4 * (f - 16))); inclinada[0, f, c - 1:c + 2] = 1      # «\»: 0,4 columnas por fila
    d = desinclinar(inclinada)[0, 0]
    cols = [np.flatnonzero(d[f]).mean() for f in range(6, 26)]
    mira("una barra inclinada 0,4 col/fila queda vertical (sus columnas varían ≤ 1 px) y conserva la tinta",
         max(cols) - min(cols) <= 1 and abs(int(d.sum()) - int(inclinada.sum())) <= 3)
    recta = np.zeros((1, 32, 32), np.uint8); recta[0, 4:28, 14:17] = 1
    mira("una barra ya vertical no cambia", np.array_equal(desinclinar(recta)[0, 0], recta[0]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(comprobar())
