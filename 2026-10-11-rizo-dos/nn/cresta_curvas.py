#!/usr/bin/env python3
"""rizo-2 — CURVAS SOBRE LA CRESTA del mapa suave (2026-10-13, «procede» del dueño): en vez de medir la curvatura en el
BORDE del dígito (dos bordes por trazo, cada uno con su objeto y el centro al lado contrario), se mide sobre la línea
media que devuelve la CNN delgada. Una panza = UN arco con UN sentido; un trazo recto = una recta; un cambio brusco = un
golpe; y un CICLO del grafo de la cresta = un LAZO cerrado, o sea un hueco deducido de las curvas y no de rayos.

Cómo:
  1. esqueleto (1 px) de M > UMBRAL; cada píxel de la cresta se afina a sub-píxel con el centroide de M en su 3×3;
  2. grafo 8-conexo de la cresta, partido en los nodos de grado ≥ 3 (cruces) y los de grado 1 (extremos): tramos;
     un tramo cuyo principio y fin coinciden, o una componente sin extremos ni cruces, es un LAZO;
  3. cada tramo se suaviza (gaussiana σ = SIGMA muestras; circular en los lazos) y se calcula la curvatura con signo
     κ = (x'y'' − y'x'') / |v|³; el vector hacia el centro es sign(κ)·(−y', x')/|v|;
  4. se parte en runs: |κ| ≥ K_MIN con el mismo signo y ≥ L_MIN muestras → ARCO (sentido = cuadrante del vector medio
     hacia el centro: → ← ↓ ↑ en coordenadas de imagen, y hacia abajo); |κ| < K_MIN y ≥ L_MIN → RECTA; un pico |κ| ≥
     K_GOLPE → GOLPE en ese punto.
  fuerza de un arco = ángulo barrido / 180° (acotado a 1); de un lazo, 1; de una recta, largo / 32.

Números para el compositor (caract): arcos 4 sentidos × (3×3 del máximo de fuerza + cuenta + largo/32) = 44,
golpes 3×3 + cuenta = 10, LAZOS 3×3 (fuerza = 1, campana en el centro) + cuenta = 10, y desde el 2026-10-13 RECTAS
4 orientaciones (0/45/90/135°) × (3×3 + cuenta + largo/32) = 44 → 108. caract(M)[:, :64] es lo de antes.

    python nn/cresta_curvas.py --entrada /ruta/delgados-k3x3.npz --de resultados/43-huecos-falsos-<variante>.json
        → resultados/45-cresta-<de>.png      (los mismos 48 dígitos, con arcos, rectas, golpes y lazos)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))

UMBRAL = 0.4; SIGMA = 1.5; K_MIN = 0.07; K_GOLPE = 0.45; L_MIN = 4; CELDAS = 3
ANG_EMPALME = 30.0   # 2026-10-14 (el dueño: «donde empieza un cruce termina una recta»): en un cruce, dos tramos que
                     # llegan colineales (≤ 30° entre sus direcciones) se EMPALMAN en uno antes de buscar rectas y arcos.
                     # Medido antes: el 82 % de las rectas detectadas acababan en un cruce (400 dígitos de val)
BARRIDO_MIN = 20.0
CONC_MIN = 60.0      # 2026-10-14 (el dueño: «me gusta tu definición de concavidad»): una concavidad es la región que
                     # encierra un arco de ≥ 60° de barrido —el polígono arco + cuerda—, salvo que el arco forme parte
                     # de un lazo (entonces la abertura está cerrada por otro tramo: es un hueco, no una concavidad)
BRAZO_ESQ = 10    # 2026-10-15 (el dueño: «una concavidad… ignorando si está rodeada por rectas o curvas»): en una ESQUINA, cada
                  # brazo del polígono se toma hasta 10 px desde el nodo (y nunca más que el brazo corto): es la región del entrante
L_RECTA = 5.0        # una recta más corta no es una recta: el 89 % de las falsas medían < 6 px (sobre 600 sintéticos, 2026-10-13)
# BARRIDO_MIN (2026-10-13, mejora de rectas): un run curvo que barre < 20° no es un arco; se funde con las rectas vecinas
SENTIDOS = ["k→", "k←", "k↓", "k↑"]; COLORES = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]; COLOR_LAZO = "#111111"; COLOR_RECTA = "#8a8a8a"
_YY, _XX = np.mgrid[0:32, 0:32] + 0.5
VEC8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _afinar(M, yy, xx):
    """Posición sub-píxel de cada píxel de la cresta: centroide de M en su 3×3 (en coordenadas de centro de píxel)."""
    P = np.pad(M, 1); pts = []
    for y, x in zip(yy, xx):
        w = P[y:y + 3, x:x + 3]; s = w.sum()
        dy, dx = (np.arange(3)[:, None] * w).sum() / s - 1, (np.arange(3)[None] * w).sum() / s - 1
        pts.append((x + 0.5 + dx, y + 0.5 + dy))
    return pts


def tramos(M: np.ndarray) -> tuple[list[dict], list[dict]]:
    """La cresta de M como grafo: nodos = RACIMOS de píxeles de grado ≠ 2 (un cruce del esqueleto son varios píxeles de
    grado 3 pegados; se agrupan dilatando 1 px), aristas = tramos entre racimos [{a, b, pts}]. Devuelve (tramos, ciclos
    puros): una componente sin ningún nodo es un ciclo puro (un 0 limpio)."""
    esq = skeletonize(M > UMBRAL); yy, xx = np.nonzero(esq)
    if len(yy) == 0:
        return [], []
    idx = {(y, x): k for k, (y, x) in enumerate(zip(yy, xx))}; pos = _afinar(M, yy, xx)
    vec = [[idx[(y + dy, x + dx)] for dy, dx in VEC8 if (y + dy, x + dx) in idx] for y, x in zip(yy, xx)]
    grado = np.array([len(v) for v in vec]); nodo = grado != 2
    # racimos: componentes 8-conexas de la máscara de nodos dilatada 1 px (sólo sobre la cresta)
    mn = np.zeros(esq.shape, bool); mn[yy[nodo], xx[nodo]] = True
    lab, _ = ndimage.label(ndimage.binary_dilation(mn, np.ones((3, 3))) & esq, structure=np.ones((3, 3)))
    racimo = np.array([lab[y, x] for y, x in zip(yy, xx)])            # 0 = no es nodo ni está pegado a uno
    es_nodo = racimo > 0
    out, vistos = [], set()
    for a in np.flatnonzero(es_nodo):
        for b in vec[a]:
            if es_nodo[b]:
                continue
            cam = [a, b]; prev, cur = a, b
            while not es_nodo[cur]:
                sig = [v for v in vec[cur] if v != prev]
                if not sig:
                    break
                prev, cur = cur, sig[0]; cam.append(cur)
            clave = frozenset(cam[1:-1]) if len(cam) > 2 else frozenset(cam)
            if clave in vistos:
                continue
            vistos.add(clave)
            out.append({"a": int(racimo[cam[0]]), "b": int(racimo[cam[-1]]) if es_nodo[cam[-1]] else -len(out) - 1, "pts": np.array([pos[k] for k in cam])})
    ciclos = []
    # ciclos puros: componentes de la cresta sin ningún nodo
    comp, nc = ndimage.label(esq, structure=np.ones((3, 3)))
    for c in range(1, nc + 1):
        ks = [k for k, (y, x) in enumerate(zip(yy, xx)) if comp[y, x] == c]
        if len(ks) >= 4 and not any(es_nodo[k] for k in ks):
            cam = [ks[0]]; prev, cur = None, ks[0]
            while True:
                sig = [v for v in vec[cur] if v != prev]
                if not sig:
                    break
                prev, cur = cur, sig[0]
                if cur == ks[0]:
                    break
                cam.append(cur)
            ciclos.append({"pts": np.array([pos[k] for k in cam])})
    return out, ciclos


def _direccion(pts, en_inicio, n=4):
    """Dirección unitaria del tramo en uno de sus extremos, SALIENDO del tramo (hacia fuera)."""
    q = pts[:n] if en_inicio else pts[-n:][::-1]                     # q[0] es el extremo, q[-1] hacia dentro
    v = q[0] - q[-1]; return v / (np.hypot(*v) + 1e-9)


def empalmar(tr: list[dict]) -> list[dict]:
    """Une, a través de cada racimo de cruce, los pares de tramos más colineales (ángulo entre la dirección con la que
    uno LLEGA y la que el otro SALE ≤ ANG_EMPALME). Se repite hasta que no queda nada que unir, así una recta que
    atraviesa dos cruces sale entera. Los tramos de un racimo a sí mismo (lazos) no se tocan."""
    tr = [dict(t) for t in tr]
    while True:
        extremos = {}                                                 # racimo → [(índice de tramo, en_inicio)]
        for i, t in enumerate(tr):
            if t["a"] == t["b"]:
                continue
            if t["a"] > 0:
                extremos.setdefault(t["a"], []).append((i, True))
            if t["b"] > 0:
                extremos.setdefault(t["b"], []).append((i, False))
        mejor = None
        for rac, lista in extremos.items():
            for x in range(len(lista)):
                for y in range(x + 1, len(lista)):
                    (i, ii), (j, ji) = lista[x], lista[y]
                    if i == j:
                        continue
                    # salir de i hacia el cruce y seguir por j: colineal ⇔ dir_salida(i) ≈ −dir_salida(j)
                    ang = np.degrees(np.arccos(np.clip(-np.dot(_direccion(tr[i]["pts"], ii), _direccion(tr[j]["pts"], ji)), -1, 1)))
                    if ang <= ANG_EMPALME and (mejor is None or ang < mejor[0]):
                        mejor = (ang, rac, i, ii, j, ji)
        if mejor is None:
            return tr
        _, rac, i, ii, j, ji = mejor
        pi = tr[i]["pts"] if not ii else tr[i]["pts"][::-1]          # pi acaba en el cruce
        pj = tr[j]["pts"] if ji else tr[j]["pts"][::-1]              # pj empieza en el cruce
        a = tr[i]["b"] if not ii else tr[i]["a"]; a = tr[i]["a"] if not ii else tr[i]["b"]
        nuevo = {"a": (tr[i]["a"] if not ii else tr[i]["b"]), "b": (tr[j]["b"] if ji else tr[j]["a"]), "pts": np.concatenate([pi, pj[1:]])}
        tr = [t for k, t in enumerate(tr) if k not in (i, j)] + [nuevo]


def ciclos_del_grafo(tr: list[dict]) -> list[np.ndarray]:
    """Ciclos fundamentales del grafo de racimos: árbol de expansión por BFS y una arista sobrante por ciclo. Cada ciclo
    vuelve como polígono (n,2). Un tramo de un racimo a sí mismo es un ciclo por sí solo."""
    ady = {}
    for i, t in enumerate(tr):
        ady.setdefault(t["a"], []).append((t["b"], i)); ady.setdefault(t["b"], []).append((t["a"], i))
    ciclos, en_arbol, padre = [], set(), {}
    for raiz in ady:
        if raiz in padre:
            continue
        padre[raiz] = (None, None); cola = [raiz]
        while cola:
            u = cola.pop(0)
            for v, i in ady[u]:
                if i in en_arbol or (u == v):
                    continue
                if v not in padre:
                    padre[v] = (u, i); en_arbol.add(i); cola.append(v)
    def camino(u):
        c = []
        while padre[u][0] is not None:
            c.append(padre[u][1]); u = padre[u][0]
        return c
    def orientar(pts, desde):
        return pts if np.hypot(*(pts[0] - desde)) <= np.hypot(*(pts[-1] - desde)) else pts[::-1]
    for i, t in enumerate(tr):
        if t["a"] == t["b"]:
            if len(t["pts"]) > 3:
                ciclos.append(t["pts"])
            continue
        if i in en_arbol or t["b"] < 0 or t["a"] < 0:
            continue
        ca, cb = camino(t["a"]), camino(t["b"])                      # caminos a la raíz; se quita el tronco común
        while ca and cb and ca[-1] == cb[-1]:
            ca.pop(); cb.pop()
        aristas = [i] + cb + ca[::-1]                                 # b→…→ancestro común→…→a, y el tramo i cierra
        poly = [t["pts"]]; fin = t["pts"][-1]
        for j in aristas[1:]:
            seg = orientar(tr[j]["pts"], fin); poly.append(seg); fin = seg[-1]
        ciclos.append(np.concatenate(poly))
    return ciclos


def _suavizar(p, lazo):
    n = len(p); k = int(3 * SIGMA); t = np.arange(-k, k + 1); w = np.exp(-t**2 / (2 * SIGMA**2)); w /= w.sum()
    if lazo:
        q = np.concatenate([p[-k:], p, p[:k]]) if n > k else np.concatenate([p] * (2 * k // n + 3))[: n + 2 * k]
    else:
        q = np.pad(p, ((k, k), (0, 0)), mode="reflect") if n > k else np.pad(p, ((k, k), (0, 0)), mode="edge")
    return np.stack([np.convolve(q[:, 0], w, "valid"), np.convolve(q[:, 1], w, "valid")], 1)


def curvatura(p, lazo):
    """κ con signo y el vector unitario hacia el centro, por muestra."""
    if lazo:
        d1 = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2; d2 = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
    else:
        d1 = np.gradient(p, axis=0); d2 = np.gradient(d1, axis=0)
    v = np.hypot(d1[:, 0], d1[:, 1]) + 1e-6
    k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / v**3
    normal = np.stack([-d1[:, 1], d1[:, 0]], 1) / v[:, None] * np.sign(k)[:, None]
    return k, normal


def segmentar(ps, k, nrm):
    """Parte un tramo suavizado en runs [(clase, i, j)] con clase +1/−1 (arco) o 0 (recta): primero por el signo de κ
    con |κ| ≥ K_MIN, luego los runs curvos que barren < BARRIDO_MIN se reclasifican como recta y se funden con sus
    vecinos rectos. La MISMA función se aplica a la verdad sintética (curvas_comparar / rectas_comparar)."""
    n = len(ps); cl = np.where(np.abs(k) >= K_MIN, np.sign(k), 0).astype(int)
    paso = np.hypot(*np.gradient(ps, axis=0).T)
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and cl[j] == cl[i]:
            j += 1
        runs.append([int(cl[i]), i, j]); i = j
    for r in runs:                                                  # arcos demasiado cortos → recta
        if r[0] != 0 and np.degrees(np.sum(np.abs(k[r[1]:r[2]]) * paso[r[1]:r[2]])) < BARRIDO_MIN:
            r[0] = 0
    # ⚠ Probado y DESCARTADO el 2026-10-14 (idx 947): reclasificar como arco una «recta» que acumula ≥ 20° de giro con
    # signo consistente. En 600 sintéticos bajó rectas 0,913/0,799 → 0,892/0,743 y curvas 0,801/0,833 → 0,782/0,823, y
    # el 947 no cambió: su giro de 41° es el ZIGZAG de la doble cresta de un 1 grueso (signos alternos), un fallo de
    # la ENTRADA (delg-cnn, pendiente 1), no de esta regla.
    out = []
    for r in runs:                                                  # fundir rectas consecutivas
        if out and out[-1][0] == 0 and r[0] == 0:
            out[-1][2] = r[2]
        else:
            out.append(r)
    return [tuple(r) for r in out]


def _sentido(nx, ny):
    return (0 if nx > 0 else 1) if abs(nx) >= abs(ny) else (2 if ny > 0 else 3)   # y crece hacia abajo


def objetos(M: np.ndarray) -> dict:
    """{'arcos': [{s, cx, cy, largo, fuerza, pts}], 'rectas': [...], 'golpes': [{cx, cy}], 'lazos': [{cx, cy, largo, area, pts}]}"""
    arcos, rectas, golpes, lazos, restos = [], [], [], [], []
    tr, puros = tramos(M)

    def resto(seg):
        """2026-10-15 (el dueño: «corrige para que funcione con todo tramo»): lo que no llega a recta ni a arco no se tira,
        es un RESTO con centro, largo y orientación (como una recta): un tramo de la cresta sin objeto era invisible para el
        compositor (medido: el 17 % de la cresta en val, § 71–73). Las rectas y arcos medidos contra la verdad no cambian."""
        seg = np.asarray(seg, np.float32)
        if len(seg) < 2:
            return
        largo = float(np.sum(np.hypot(*np.diff(seg, axis=0).T))); m = seg.mean(0); v = seg[-1] - seg[0]
        if largo < 0.5:
            return
        ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180)
        restos.append({"cx": float(m[0]), "cy": float(m[1]), "largo": largo, "ang": ang, "o": int(round(ang / 45)) % 4, "fuerza": min(largo / 32, 1.0), "pts": seg})
    for poly in [c["pts"] for c in puros] + ciclos_del_grafo(tr):
        if len(poly) < 4:
            continue
        ps = _suavizar(poly, True)
        area = 0.5 * abs(np.sum(ps[:, 0] * np.roll(ps[:, 1], -1) - np.roll(ps[:, 0], -1) * ps[:, 1]))
        if area < 2.0:
            continue
        cx, cy = ps.mean(0); lazos.append({"cx": float(cx), "cy": float(cy), "largo": len(ps), "area": float(area), "pts": ps})
    for t in empalmar(tr):
        p = t["pts"]
        if len(p) < 3:
            resto(p); continue
        ps = _suavizar(p, False); k, nrm = curvatura(ps, False); n = len(ps)
        # golpes: picos de |κ|
        for i in np.flatnonzero((np.abs(k) >= K_GOLPE) & (np.abs(k) >= np.maximum(np.roll(np.abs(k), 1), np.roll(np.abs(k), -1)))):
            golpes.append({"cx": float(ps[i, 0]), "cy": float(ps[i, 1])})
        for c, i, j in segmentar(ps, k, nrm):
            if j - i < L_MIN:
                resto(ps[i:j]); continue
            seg = ps[i:j]; largo = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
            if c == 0:
                if largo < L_RECTA:
                    resto(seg); continue
                # la recta se representa por su AJUSTE: segmento sobre el eje principal, con su orientación en grados
                m = seg.mean(0); ev, vec = np.linalg.eigh(np.cov((seg - m).T)); v = vec[:, 1]; t = (seg - m) @ v
                ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180); pts = np.stack([m + t.min() * v, m + t.max() * v])
                rectas.append({"cx": float(m[0]), "cy": float(m[1]), "largo": largo, "ang": ang, "o": int(round(ang / 45)) % 4, "fuerza": min(largo / 32, 1.0), "pts": pts})
            else:
                barrido = float(np.degrees(np.sum(np.abs(k[i:j]) * np.hypot(*np.gradient(seg, axis=0).T)))); nm = nrm[i:j].mean(0)
                arcos.append({"s": _sentido(nm[0], nm[1]), "cx": float(seg[:, 0].mean()), "cy": float(seg[:, 1].mean()), "largo": largo, "barrido": barrido,
                              "fuerza": min(barrido / 180, 1.0), "pts": seg})
    # 2026-10-15, segunda puerta (medido: tramos de 5 px que no dejaban ni recta ni resto, idx 191/198/225 de val): lo que
    # quede a > 1 px de TODO objeto se convierte en resto por construcción, trozo a trozo, así la cobertura no depende de
    # qué regla lo descartó. Mínimo 2 puntos seguidos.
    Q = [a["pts"] for a in arcos] + [z["pts"] for z in lazos] + [r["pts"] for r in restos]
    Q += [np.linspace(r["pts"][0], r["pts"][1], int(np.hypot(*(r["pts"][1] - r["pts"][0]))) + 2) for r in rectas]
    Q = np.concatenate(Q) if Q else np.zeros((0, 2), np.float32)
    for t in tr:
        p = np.asarray(t["pts"], np.float32)
        if len(p) < 2:
            continue
        lejos = np.ones(len(p), bool) if len(Q) == 0 else np.sqrt(((p[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(1) > 1.0
        i = 0
        while i < len(p):
            if not lejos[i]:
                i += 1; continue
            j = i
            while j < len(p) and lejos[j]:
                j += 1
            if j - i >= 2:
                resto(p[i:j])
            i = j
    conc = concavidades(arcos, lazos); esq = esquinas(tr, lazos); crz = cruces(tr)   # ⚠ sobre los tramos SIN empalmar: el empalmado pasa POR el nodo y la barra se queda sin pareja
    return {"arcos": arcos, "rectas": rectas, "golpes": golpes, "lazos": lazos, "concavidades": conc, "esquinas": esq, "entrantes": entrantes(conc, esq), "cruces": crz, "restos": restos}


def _dentro(p, poly):
    """¿El punto p está dentro del polígono poly (n,2)? (regla par-impar)"""
    x, y = p; dentro = False; n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-12):
            dentro = not dentro
    return dentro


def concavidades(arcos, lazos):
    """[{cx, cy, area, abertura, dir (vector unitario hacia la abertura), o (cuadrante de la abertura: → ← ↓ ↑),
    barrido, s, pts (polígono)}] para los arcos de ≥ CONC_MIN° que no estén dentro de un lazo."""
    out = []
    for a in arcos:
        if a["barrido"] < CONC_MIN:
            continue
        poly = a["pts"]; x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area < 2.0:
            continue
        cx, cy = poly.mean(0); medio = (poly[0] + poly[-1]) / 2; v = medio - np.array([cx, cy]); n = np.hypot(*v)
        if n < 1e-6:
            continue
        v = v / n
        if any(_dentro((cx, cy), z["pts"]) for z in lazos):
            continue
        o = (0 if v[0] > 0 else 1) if abs(v[0]) >= abs(v[1]) else (2 if v[1] > 0 else 3)
        out.append({"cx": float(cx), "cy": float(cy), "area": float(area), "abertura": float(np.hypot(*(poly[-1] - poly[0]))), "dir": v, "o": o,
                    "barrido": a["barrido"], "s": a["s"], "fuerza": min(area / 60.0, 1.0), "pts": poly})
    return out


def _entrante(poly, barrido, s, lazos):
    """Objeto concavidad/esquina a partir de su polígono (abierto: el último lado, la cuerda, lo cierra np.roll)."""
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area < 2.0:
        return None
    cx, cy = poly.mean(0); medio = (poly[0] + poly[-1]) / 2; v = medio - np.array([cx, cy]); n = np.hypot(*v)
    if n < 1e-6 or any(_dentro((cx, cy), z["pts"]) for z in lazos):
        return None
    v = v / n; o = (0 if v[0] > 0 else 1) if abs(v[0]) >= abs(v[1]) else (2 if v[1] > 0 else 3)
    return {"cx": float(cx), "cy": float(cy), "area": float(area), "abertura": float(np.hypot(*(poly[-1] - poly[0]))), "dir": v, "o": o,
            "barrido": float(barrido), "s": s, "fuerza": min(area / 60.0, 1.0), "pts": poly}


def esquinas(tr: list[dict], lazos) -> list[dict]:
    """Concavidades de ESQUINA (2026-10-15): en cada racimo, cada par de tramos (ya empalmados) que se juntan con un giro
    de ≥ CONC_MIN° (ángulo interior ≤ 180 − CONC_MIN) encierra un entrante: polígono = brazo de uno (≤ BRAZO_ESQ px
    desde el nodo, nunca más largo que el brazo corto) + brazo del otro + cuerda. Mismo objeto que `concavidades`,
    con el giro como `barrido` y `s = -1` (no tiene sentido de curvatura). Recibe los tramos SIN empalmar (medido el
    2026-10-15 con una T sintética: el tramo empalmado pasa por dentro del nodo y la barra se quedaba sin pareja → 0
    esquinas); dos tramos colineales dan giro ≈ 0 y se descartan solos."""
    extremos = {}
    for i, t in enumerate(tr):
        if t["a"] == t["b"] or len(t["pts"]) < 3:
            continue
        if t["a"] > 0:
            extremos.setdefault(t["a"], []).append((i, True))
        if t["b"] > 0:
            extremos.setdefault(t["b"], []).append((i, False))
    out = []
    for rac, lista in extremos.items():
        for x in range(len(lista)):
            for y in range(x + 1, len(lista)):
                (i, ii), (j, ji) = lista[x], lista[y]
                if i == j:
                    continue
                di, dj = _direccion(tr[i]["pts"], ii), _direccion(tr[j]["pts"], ji)      # las dos SALEN del nodo
                interior = np.degrees(np.arccos(np.clip(np.dot(di, dj), -1, 1))); giro = 180.0 - interior
                if giro < CONC_MIN:
                    continue
                bi = tr[i]["pts"] if ii else tr[i]["pts"][::-1]; bj = tr[j]["pts"] if ji else tr[j]["pts"][::-1]   # empiezan en el nodo
                L = min(BRAZO_ESQ, len(bi), len(bj))
                poly = np.concatenate([bi[:L][::-1], bj[1:L]])
                e = _entrante(poly, giro, -1, lazos)
                if e is not None:
                    out.append(e)
    return out


def entrantes(conc: list[dict], esq: list[dict]) -> list[dict]:
    """Concavidades de arco ∪ esquinas, sin contar dos veces el mismo entrante: si el centro de uno cae dentro del
    polígono del otro, se queda el de más área."""
    todos = sorted(conc + esq, key=lambda e: -e["area"]); out = []
    for e in todos:
        if any(_dentro((e["cx"], e["cy"]), q["pts"]) or _dentro((q["cx"], q["cy"]), e["pts"]) for q in out):
            continue
        out.append(e)
    return out


CONO_H = 60.0   # 2026-10-15 (el dueño): un entrante es HORIZONTAL si su abertura está a ±60° de la horizontal; si no, vertical


def cono(q) -> int:
    """Clase de abertura con cono ancho: 0 → derecha (|θ| ≤ 60°), 1 ← izquierda (|θ| ≥ 120°), 2 ↓ abajo, 3 ↑ arriba (θ en
    grados desde la derecha, positivo hacia abajo). Contra `o`, que parte a 45°: aquí un entrante a +50° es → y no ↓."""
    ang = np.degrees(np.arctan2(q["dir"][1], q["dir"][0]))
    if abs(ang) <= CONO_H:
        return 0
    if abs(ang) >= 180 - CONO_H:
        return 1
    return 2 if ang > 0 else 3



ANG_CRUZ = 40.0    # 2026-10-15 (el dueño: «dónde se cruzan 2 rectas»): dos brazos opuestos son la MISMA recta si al salir del nodo
                   # difieren de la colinealidad en ≤ 40°; una cruz son dos de esos pares en el mismo nodo
BRAZO_CRUZ = 2     # brazo mínimo (px de cresta) para que el cruce no sea un extremo: parámetro a customizar (el dueño, 2026-10-15)
PUENTE_CRUZ = 3    # dos nodos de 3 brazos unidos por un tramo ≤ 3 px son UNA cruz (así parte el esqueleto una X de trazo grueso)


def cruces(tr: list[dict], brazo_min: float = None, puente: float = None, ang_max: float = None) -> list[dict]:
    """[{cx, cy, ang (entre las dos rectas, 0–90°), brazos [4 largos en px], brazo_min, dirs [2 vectores unitarios de las rectas]}]
    Un nodo del grafo (tramos SIN empalmar) con ≥ 4 brazos entre los que hay DOS pares casi colineales (≤ ANG_CRUZ° de la
    colinealidad) es una cruz; también lo es un par de nodos de 3 brazos unidos por un tramo ≤ PUENTE_CRUZ px, cuyos 4
    brazos externos formen esos dos pares. Cada brazo tiene que medir ≥ brazo_min px: si no, el cruce está en un extremo
    y no cuenta (el parámetro que el dueño quiere customizar). Los tres parámetros se barren en nn/cruces_barrido.py."""
    brazo_min = BRAZO_CRUZ if brazo_min is None else brazo_min; puente = PUENTE_CRUZ if puente is None else puente; ang_max = ANG_CRUZ if ang_max is None else ang_max
    ext = {}
    for i, t in enumerate(tr):
        if t["a"] == t["b"] or len(t["pts"]) < 2:
            continue
        if t["a"] > 0:
            ext.setdefault(t["a"], []).append((i, True))
        if t["b"] > 0:
            ext.setdefault(t["b"], []).append((i, False))

    def brazo(i, ini):
        p = tr[i]["pts"] if ini else tr[i]["pts"][::-1]              # empieza en el nodo
        return p, float(np.sum(np.hypot(*np.diff(p, axis=0).T))), _direccion(tr[i]["pts"], ini)

    def pares(brazos):
        """brazos = [(pts, largo, dir)] → las dos parejas colineales, o None."""
        n = len(brazos); cand = []
        for x in range(n):
            for y in range(x + 1, n):
                ang = np.degrees(np.arccos(np.clip(-np.dot(brazos[x][2], brazos[y][2]), -1, 1)))
                if ang <= ang_max:
                    cand.append((ang, x, y))
        cand.sort()
        for a1, x1, y1 in cand:
            for a2, x2, y2 in cand:
                if len({x1, y1, x2, y2}) == 4:
                    return (x1, y1), (x2, y2)
        return None

    def armar(centro, brazos, par):
        (x1, y1), (x2, y2) = par
        if min(b[1] for b in brazos) < brazo_min:
            return None
        d1 = brazos[x1][2] - brazos[y1][2]; d1 /= np.hypot(*d1) + 1e-9
        d2 = brazos[x2][2] - brazos[y2][2]; d2 /= np.hypot(*d2) + 1e-9
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), -1, 1))))
        return {"cx": float(centro[0]), "cy": float(centro[1]), "ang": ang, "brazos": [round(brazos[k][1], 1) for k in (x1, y1, x2, y2)],
                "brazo_min": float(min(b[1] for b in brazos)), "dirs": [d1, d2], "pts": [brazos[k][0] for k in (x1, y1, x2, y2)]}

    out, usados = [], set()
    for rac, lista in ext.items():                                   # 1) nodos con ≥ 4 brazos
        if len(lista) < 4:
            continue
        brazos = [brazo(i, ini) for i, ini in lista]; par = pares(brazos)
        if par is None:
            continue
        centro = np.mean([b[0][0] for b in brazos], 0); c = armar(centro, brazos, par)
        if c is not None:
            out.append(c); usados.add(rac)
    for i, t in enumerate(tr):                                       # 2) dos nodos de 3 brazos unidos por un puente corto
        a, b = t["a"], t["b"]
        if a <= 0 or b <= 0 or a == b or a in usados or b in usados or len(ext.get(a, [])) != 3 or len(ext.get(b, [])) != 3:
            continue
        if float(np.sum(np.hypot(*np.diff(t["pts"], axis=0).T))) > puente:
            continue
        brazos = [brazo(j, ini) for rac in (a, b) for j, ini in ext[rac] if j != i]
        if len(brazos) != 4:
            continue
        par = pares(brazos)
        if par is None:
            continue
        centro = (t["pts"][0] + t["pts"][-1]) / 2; c = armar(centro, brazos, par)
        if c is not None:
            c["puente"] = True; out.append(c); usados.update((a, b))
    return out


def _mapa(objs, altura, sig=1.5):
    m = np.zeros((32, 32), np.float32)
    for o in objs:
        m = np.maximum(m, (np.exp(-((_XX - o["cx"]) ** 2 + (_YY - o["cy"]) ** 2) / (2 * sig**2)) * altura(o)).astype(np.float32))
    return m


def bloque_cruces(crz) -> list[float]:
    """10 números: 3×3 de PRESENCIA (gaussiana de altura 1, σ = 2 px, máximo por celda) + cuenta."""
    return F.adaptive_max_pool2d(torch.from_numpy(_mapa(crz, lambda q: 1.0, 2.0))[None], 3)[0].numpy().ravel().tolist() + [float(len(crz))]


def bloques(celdas: int = CELDAS) -> dict:
    """Dónde empieza y acaba cada grupo en caract(·, celdas): con 3×3 son 374 (44·10·10·44·44·44·4·4·36·36·10·44·44); con 4×4, 522. Los 152 primeros son los de antes."""
    c = celdas * celdas; n = 0; out = {}
    for g, k in (("arcos", 4 * (c + 2)), ("golpes", c + 1), ("lazos", c + 1), ("rectas", 4 * (c + 2)), ("concavidades", 4 * (c + 2)), ("entrantes", 4 * (c + 2)), ("entrantes4", 4), ("entrantes4n", 4), ("entrantes60", 36), ("entrantes60f", 36), ("cruces", 10), ("restos", 4 * (c + 2)), ("rectas_todo", 4 * (c + 2))):
        out[g] = (n, n + k); n += k
    return out


def caract(M: np.ndarray, celdas: int = CELDAS) -> np.ndarray:
    """(n,32,32) → (n,204) con 3×3: arcos 44 · golpes 10 · lazos 10 · rectas 44 · concavidades 44 · entrantes 44 (concavidades de arco ∪ esquinas, 2026-10-15; por dirección de la abertura)
    · entrantes4 4 (el dueño, 2026-10-15: «reduce los entrantes a 4 posibilidades: arriba, abajo, derecha e izq.»: la fuerza máxima, área/60, por dirección → ← ↓ ↑) · entrantes4n 4 (la cuenta por dirección)
    · entrantes60 36 (el dueño, 2026-10-15: CONOS de ±60° alrededor de la horizontal —derecha, izquierda— y el resto vertical —abajo, arriba—; 3×3 de PRESENCIA por cono, sin magnitud; siempre 3×3)
    · entrantes60f 36 (ídem con la fuerza máxima por celda en vez de presencia) · cruces 10 (3×3 de presencia + cuenta; el dueño, 2026-10-15)
    · restos 44 (los trozos de cresta que no son recta ni arco, como rectas: 4 orient. × (3×3 + cuenta + largo/32)) · rectas_todo 44 (rectas ∪ restos). 2026-10-15.
    `celdas` es el lado de la rejilla (2026-10-15: 4×4 contra 3×3); los cortes los da bloques(celdas)."""
    out = []
    for m in np.asarray(M, np.float32):
        ob = objetos(m); f = []
        for s in range(4):
            cs = [a for a in ob["arcos"] if a["s"] == s]
            f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(cs, lambda o: o["fuerza"]))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(cs), sum(a["largo"] for a in cs) / 32])
        f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(ob["golpes"], lambda o: 1.0))[None], celdas)[0].numpy().ravel().tolist()); f.append(len(ob["golpes"]))
        f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(ob["lazos"], lambda o: 1.0, 2.0))[None], celdas)[0].numpy().ravel().tolist()); f.append(len(ob["lazos"]))
        for o in range(4):
            rs = [r for r in ob["rectas"] if r["o"] == o]
            f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(rs, lambda q: q["fuerza"]))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(rs), sum(r["largo"] for r in rs) / 32])
        for o in range(4):                                          # concavidades por dirección de la ABERTURA: 3×3 (fuerza = área/60) + cuenta + abertura/32
            cs = [c for c in ob["concavidades"] if c["o"] == o]
            f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(cs, lambda q: q["fuerza"], 2.0))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(cs), sum(c["abertura"] for c in cs) / 32])
        for o in range(4):                                          # entrantes (arco ∪ esquina), misma representación que las concavidades
            cs = [c for c in ob["entrantes"] if c["o"] == o]
            f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(cs, lambda q: q["fuerza"], 2.0))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(cs), sum(c["abertura"] for c in cs) / 32])
        f.extend([max([c["fuerza"] for c in ob["entrantes"] if c["o"] == o], default=0.0) for o in range(4)])   # entrantes4: → ← ↓ ↑
        f.extend([float(sum(c["o"] == o for c in ob["entrantes"])) for o in range(4)])                           # entrantes4n
        for altura in (lambda q: 1.0, lambda q: q["fuerza"]):                                                     # entrantes60 · entrantes60f
            for o in range(4):
                cs = [c for c in ob["entrantes"] if cono(c) == o]
                f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(cs, altura, 2.0))[None], 3)[0].numpy().ravel().tolist())
        f.extend(bloque_cruces(ob["cruces"]))
        for lista in (ob["restos"], ob["rectas"] + ob["restos"]):                                               # restos · rectas_todo
            for o in range(4):
                rs = [r for r in lista if r["o"] == o]
                f.extend(F.adaptive_max_pool2d(torch.from_numpy(_mapa(rs, lambda q: q["fuerza"]))[None], celdas)[0].numpy().ravel().tolist()); f.extend([len(rs), sum(r["largo"] for r in rs) / 32])
        out.append(f)
    return np.array(out, np.float32)


def pintar(ax, m):
    ax.imshow(np.where(m > 0.05, m, np.nan), cmap="Greys", vmin=0, vmax=2.4, interpolation="nearest")
    ob = objetos(m)
    for r in ob["rectas"]:
        ax.plot(r["pts"][:, 0] - 0.5, r["pts"][:, 1] - 0.5, "-", color=COLOR_RECTA, lw=1.2, alpha=0.8)
    for r in ob["restos"]:
        ax.plot(r["pts"][:, 0] - 0.5, r["pts"][:, 1] - 0.5, "-", color="#d6a21a", lw=1.2, alpha=0.9)        # restos: ocre
    for a in ob["arcos"]:
        ax.plot(a["pts"][:, 0] - 0.5, a["pts"][:, 1] - 0.5, "-", color=COLORES[a["s"]], lw=1.2 + 2.4 * a["fuerza"], solid_capstyle="round")
    for z in ob["lazos"]:
        ax.plot(np.append(z["pts"][:, 0], z["pts"][0, 0]) - 0.5, np.append(z["pts"][:, 1], z["pts"][0, 1]) - 0.5, "-", color=COLOR_LAZO, lw=2.2)
        ax.plot(z["cx"] - 0.5, z["cy"] - 0.5, "+", color=COLOR_LAZO, ms=9, mew=2)
    for g in ob["golpes"]:
        ax.plot(g["cx"] - 0.5, g["cy"] - 0.5, "x", color="k", ms=6, mew=1.3)
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.axis("off")
    return ob


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M")
    ap.add_argument("--de", default=None); ap.add_argument("--idx", type=int, nargs="*", default=None); a = ap.parse_args()
    import evaluar as EV
    from expcnn import exigir_dataset
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
        ob = pintar(ax, Mz[i]); ax.set_title(f"{y[i]} · idx {i} · {len(ob['arcos'])}a {len(ob['rectas'])}r {len(ob['lazos'])}L {len(ob['golpes'])}g", fontsize=7)
    fig.suptitle("rizo-2 · 45 · curvas sobre la CRESTA del mapa suave: arcos (color = centro a la → naranja · ← azul · ↓ verde · ↑ morado; grosor = ángulo barrido) · rectas gris · "
                 "LAZOS cerrados negro con + · golpes ×\ntítulo: nº de arcos · rectas · Lazos · golpes", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); out = AQUI / "resultados" / f"45-cresta-{nombre}.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
