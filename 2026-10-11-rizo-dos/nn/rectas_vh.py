#!/usr/bin/env python3
"""rizo-2 § 129 — RECTAS y CURVAS DE RADIO AMPLIO, VERTICALES u HORIZONTALES (el dueño, 2026-10-19: «un feature que indique rectas
y curvas de radio amplio. En una 3×3 que estén las verticales, en otra las horizontales. Deben tolerar otros ángulos similares hasta
cierto rango. No incluyas inclinados a 45 grados»). Detector y números; la vista previa es `nn/rectas_vh_vista.py`.

Sobre la cresta del DELGADO (los tramos de `cresta_curvas`, empalmados en los cruces, y los lazos puros), muestra a muestra:
  · dirección de la tangente (0–180°): VERTICAL si está a ≤ TOL de 90°, HORIZONTAL si está a ≤ TOL de 0°/180°; lo demás (las
    diagonales, 45° ± (45° − TOL)) no cuenta;
  · y sólo si es casi recta: |κ| ≤ K_AMPLIO (radio ≥ 1/K_AMPLIO px). Así entra el lado de un 0 (radio ~10–12 px) y no la panza
    cerrada de un 2 o de un 3.
  · Las muestras seguidas de la misma clase forman una LÍNEA (un objeto) si miden ≥ L_LINEA px.

Rejillas (lo que pidió): por orientación, 3×3 celdas SIN solape (int(x·3/32)), en cada una el largo de línea que cae dentro / (32/3),
acotado a 1. Y la CONTINUIDAD, las dos codificaciones de la propuesta (§ 129):
  · ENLACES: V, entre la fila r y la r+1 de cada columna (2×3 = 6); H, entre la columna c y la c+1 de cada fila (3×2 = 6). Vale 1 si
    UNA MISMA línea pasa por las dos celdas. Dos verticales continuas a izquierda y derecha = los cuatro enlaces de las columnas 0 y 2.
  · RANURAS: por orientación, las 2 líneas más largas ordenadas por su posición transversal (V por x, H por y), cada una con
    (presente, posición transversal, inicio, fin) relativos a la caja de la cresta del dígito (0–1).
"""
from __future__ import annotations

import numpy as np

import cresta_curvas as CC

TOL = 30.0          # grados alrededor de 90° (V) y de 0° (H): entre 30° y 60° es diagonal y no cuenta. Medido en val (§ 129):
                    # con 25° un 15 % de los 4 se quedaba sin ninguna línea (el palo inclinado ~30°); con 30°, el 5 %; con 35°, el 1 %
K_AMPLIO = 1 / 8    # |κ| ≤ 1/8 px⁻¹ = radio ≥ 8 px
L_LINEA = 4.0       # px: una línea más corta no cuenta
ORI = ["V", "H"]


def _muestras(M):
    """(puntos suavizados, dirección 0–180°, |κ|) de cada tramo empalmado y de cada lazo puro."""
    tr, puros = CC.tramos(np.asarray(M, np.float32)); out = []
    for t, lazo in [(t, False) for t in CC.empalmar(tr)] + [(c, True) for c in puros]:
        p = np.asarray(t["pts"], float)
        if len(p) < 3:
            continue
        ps = CC._suavizar(p, lazo); k, _ = CC.curvatura(ps, lazo)
        d = (np.roll(ps, -1, 0) - np.roll(ps, 1, 0)) if lazo else np.gradient(ps, axis=0)
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
        out.append((ps, ang, np.abs(k), lazo))
    return out


def lineas(M) -> list[dict]:
    """[{o: 'V'|'H', pts, cx, cy, largo}]: los runs de muestras casi rectas y casi verticales/horizontales."""
    res = []
    for ps, ang, ak, lazo in _muestras(M):
        cl = np.where(ak > K_AMPLIO, -1, np.where(np.abs(ang - 90) <= TOL, 0, np.where((ang <= TOL) | (ang >= 180 - TOL), 1, -1)))
        n = len(cl); runs, i = [], 0
        while i < n:
            j = i
            while j < n and cl[j] == cl[i]:
                j += 1
            runs.append([int(cl[i]), i, j]); i = j
        if lazo and len(runs) > 1 and runs[0][0] == runs[-1][0]:             # en un lazo, el primer run y el último son el mismo
            c, i0, _ = runs.pop(); runs[0] = [c, i0 - n, runs[0][2]]
        for c, i, j in runs:
            if c < 0:
                continue
            seg = ps[np.arange(i, j) % n]; largo = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
            if largo >= L_LINEA:
                res.append({"o": ORI[c], "pts": seg, "cx": float(seg[:, 0].mean()), "cy": float(seg[:, 1].mean()), "largo": largo})
    return res


def _celda(x, y, celdas):
    return min(celdas - 1, max(0, int(y * celdas / 32))), min(celdas - 1, max(0, int(x * celdas / 32)))


def rejillas(ls, celdas: int = 3) -> dict:
    """{'V': (celdas, celdas), 'H': …}: largo de línea en cada celda / lado de la celda, acotado a 1."""
    g = {o: np.zeros((celdas, celdas)) for o in ORI}
    for l in ls:
        p = l["pts"]; paso = np.r_[np.hypot(*np.diff(p, axis=0).T), 0]
        for (x, y), s in zip(p, paso):
            g[l["o"]][_celda(x, y, celdas)] += s
    return {o: np.minimum(v / (32 / celdas), 1.0) for o, v in g.items()}


def enlaces(ls, celdas: int = 3) -> dict:
    """V: (celdas−1, celdas) — la fila r enlazada con la r+1 en la columna c; H: (celdas, celdas−1). 1 = una MISMA línea las cruza."""
    e = {"V": np.zeros((celdas - 1, celdas)), "H": np.zeros((celdas, celdas - 1))}
    for l in ls:
        cs = [_celda(x, y, celdas) for x, y in l["pts"]]
        for (r0, c0), (r1, c1) in zip(cs[:-1], cs[1:]):
            if l["o"] == "V" and c0 == c1 and abs(r1 - r0) == 1:
                e["V"][min(r0, r1), c0] = 1
            if l["o"] == "H" and r0 == r1 and abs(c1 - c0) == 1:
                e["H"][r0, min(c0, c1)] = 1
    return e


def caja(M):
    sk = np.argwhere(CC.skeletonize(np.asarray(M, np.float32) > CC.UMBRAL))
    if not len(sk):
        return 0.0, 0.0, 32.0, 32.0
    (y0, x0), (y1, x1) = sk.min(0), sk.max(0); return float(x0), float(y0), float(max(x1 - x0, 1)), float(max(y1 - y0, 1))


def ranuras(ls, M, k: int = 2) -> dict:
    """Por orientación, las k líneas más largas ordenadas por posición transversal: [(presente, transversal, inicio, fin)], 0–1
    relativos a la caja de la cresta del dígito (V: x, y-arriba, y-abajo; H: y, x-izquierda, x-derecha)."""
    x0, y0, w, h = caja(M); out = {}
    for o in ORI:
        sel = sorted([l for l in ls if l["o"] == o], key=lambda l: -l["largo"])[:k]
        filas = []
        for l in sel:
            p = l["pts"]
            if o == "V":
                filas.append((1.0, (l["cx"] - x0) / w, (p[:, 1].min() - y0) / h, (p[:, 1].max() - y0) / h))
            else:
                filas.append((1.0, (l["cy"] - y0) / h, (p[:, 0].min() - x0) / w, (p[:, 0].max() - x0) / w))
        filas.sort(key=lambda f: f[1]); filas += [(0.0, 0.0, 0.0, 0.0)] * (k - len(filas))
        out[o] = [tuple(float(np.clip(v, 0, 1)) for v in f) for f in filas]
    return out
