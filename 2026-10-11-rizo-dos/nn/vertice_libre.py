#!/usr/bin/env python3
"""rizo-2 — el VÉRTICE LIBRE como FEATURE del revisor: detector, redondez y bloque (el dueño, 2026-10-17: «métELOS en el app»;
«al entrenar compositores yo debo poder elegir cuál entreno»). Nace de los prototipos de § 102–104, de los que se MUEVE aquí el
núcleo (los scripts de figuras lo importan: dos copias de un detector divergen).

ENTRADA: el DELGADO (`M`, delg-cnn) — la regla del repo, «todo sobre el delgado». Se busca en su CRESTA (`cresta_curvas.tramos`).

Un vértice = un punto de la cresta del que salen dos brazos ≥ BRAZO_MIN px con < A_MAX° a BRAZO px (y < A_MAX_CORTO° a
BRAZO_CORTO px: esquina, no curva), a mitad de un tramo o en un nudo, sin contar los cruces. Su DIRECCIÓN es hacia dónde apunta
la punta, en 4 sectores de 90° (N E S O, sin diagonales); su POSICIÓN, UNA celda (sin mancha); su AMPLITUD, ángulo/180; y su
REDONDEZ con las tres definiciones de § 104: despegue (0–4 px), se abre de cerca (0–90°) y radio (0–6 px), escaladas a 0–1.

El BLOQUE (`bloque`) da, por dirección N, E, S, O: rejilla de presencia + rejilla de amplitud + rejillas de despegue, cerca y radio
(celdas×celdas cada una) + cuenta. Con 3×3: 4 × (5·9 + 1) = 184 números. En el revisor cada una es una PIEZA que se elige al
entrenar. Si dos vértices de la misma dirección caen en la misma celda, la celda lleva los del más agudo (la cuenta dice cuántos).

⚠ Se calcula FUERA de `cresta_curvas.py` a propósito: tocar ese fichero cambia su hash y con él la huella de TODOS los
compositores guardados. Este módulo lleva su propio hash (`huella_modulo`), que sólo entra en la huella de los que lo usan.
⚠ Parámetros SIN calibrar contra ninguna verdad.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import cresta_curvas as CC                                               # noqa: E402

# Lo PÚBLICO (R5: el revisor importa de aquí; esto es el contrato y lo prueban sus tests en web/probar_revisor.py)
__all__ = ["vertices_libres", "medir", "escala", "bloque", "PIEZAS", "ORDEN", "huella_modulo", "celda", "TOPE",
           "BRAZO", "BRAZO_CORTO", "BRAZO_MIN", "A_MAX", "A_MAX_CORTO", "SUPR", "ANG_SIGUE", "ALCANCE"]

N = 32
BRAZO = 6.0          # px de cresta a los que se mide la amplitud (cuerda desde el vértice)
BRAZO_CORTO = 3.0    # px a los que se comprueba que es una esquina y no una curva
BRAZO_MIN = 4.0      # un brazo más corto (un espolón del esqueleto) no cuenta
A_MAX = 90.0         # el dueño, 2026-10-16: «por supuesto debe ser menor de 90°»
A_MAX_CORTO = 120.0
SUPR = 3.0           # px: dos candidatos más cerca son el mismo vértice
ANG_SIGUE = 40.0     # en un nudo, un brazo «sigue de largo» si otro sale a ≥ 140° de él (= cresta_curvas.ANG_CRUZ). Si los DOS
                     # brazos de un par siguen de largo por trazos distintos, es un cruce (la X del 8), no un vértice: medido en la
                     # 1.ª pasada, sin esta regla el 8 daba 168 N y 152 S en val
def _brazo(p, i, paso, L, cerrado):
    """Cuerda desde p[i] hasta el punto a L px de cresta en el sentido `paso` (±1). (vector, largo recorrido)."""
    n = len(p); acc, j = 0.0, i
    while acc < L:
        k = j + paso
        if cerrado:
            k %= n
        elif k < 0 or k >= n:
            break
        acc += float(np.hypot(*(p[k] - p[j]))); j = k
        if cerrado and j == i:
            break
    return p[j] - p[i], acc


def _camino(p, i, paso, L, cerrado) -> np.ndarray:
    """Los puntos de la cresta desde p[i] (incluido) hasta L px de recorrido en el sentido `paso`: el BRAZO entero, no sólo su
    cuerda. Lo usan las medidas de REDONDEZ (`redondez_vista.py`, 2026-10-17); no interviene en la detección."""
    n = len(p); acc, j = 0.0, i; out = [p[i]]
    while acc < L:
        k = j + paso
        if cerrado:
            k %= n
        elif k < 0 or k >= n:
            break
        acc += float(np.hypot(*(p[k] - p[j]))); j = k; out.append(p[j])
        if cerrado and j == i:
            break
    return np.asarray(out)


ALCANCE = 9.0        # px de brazo que se guardan con cada vértice (para medir su redondez)


def _angulo(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.hypot(*u) * np.hypot(*v) + 1e-9), -1, 1))))


def _direccion(u, v) -> str:
    """El sector de 90° del rumbo (ver `_rumbo`): N [315°, 45°), E [45°, 135°), S [135°, 225°), O [225°, 315°). Sale del MISMO
    rumbo para que las fronteras sean exactamente las escritas (2026-10-17: la versión anterior empataba distinto en 135° y 315°;
    en val no pasaba nunca, 0 de 1416)."""
    return "NESO"[int(((_rumbo(u, v) + 45) % 360) // 90)]


def _rumbo(u, v) -> float:
    """Hacia dónde apunta la punta, como una brújula: 0° = N, 90° = E, 180° = S, 270° = O. `_direccion` lo parte en 4
    sectores de 90° (±45° alrededor de cada uno): no hay NE; un vértice a 30° es N y uno a 60° es E (el dueño, 2026-10-17)."""
    b = -(u / (np.hypot(*u) + 1e-9) + v / (np.hypot(*v) + 1e-9))
    return float((np.degrees(np.arctan2(b[1], b[0])) + 90) % 360)


def _candidato(x, y, u, v, uc, vc):
    a, ac = _angulo(u, v), _angulo(uc, vc)
    if a < A_MAX and ac < A_MAX_CORTO:
        return {"cx": float(x), "cy": float(y), "ang": a, "dir": _direccion(u, v), "rumbo": _rumbo(u, v)}
    return None


def vertices_libres(M) -> list[dict]:
    tr, puros = CC.tramos(M)
    cand = []
    # (1) a mitad de un tramo, o de un ciclo puro: mínimos locales del ángulo
    for p, cerrado in [(np.asarray(t["pts"], float), False) for t in tr] + [(np.asarray(c["pts"], float), True) for c in puros]:
        n = len(p); angs = np.full(n, 180.0); res = [None] * n
        for i in range(n):
            u, lu = _brazo(p, i, -1, BRAZO, cerrado); v, lv = _brazo(p, i, +1, BRAZO, cerrado)
            if lu < BRAZO_MIN or lv < BRAZO_MIN:
                continue
            uc, _ = _brazo(p, i, -1, BRAZO_CORTO, cerrado); vc, _ = _brazo(p, i, +1, BRAZO_CORTO, cerrado)
            angs[i] = _angulo(u, v); res[i] = _candidato(*p[i], u, v, uc, vc)
            if res[i] is not None:
                res[i]["brazos"] = (_camino(p, i, -1, ALCANCE, cerrado), _camino(p, i, +1, ALCANCE, cerrado))
        for i in range(n):
            vec = [angs[(i - 1) % n], angs[(i + 1) % n]] if cerrado else [angs[i - 1] if i > 0 else 180, angs[i + 1] if i < n - 1 else 180]
            if res[i] is not None and angs[i] <= min(vec):
                cand.append(res[i])
    # (2) en un racimo: cada par de tramos que salen de él (un lazo de un racimo a sí mismo aporta sus dos extremos)
    ext = {}
    for t in tr:
        p = np.asarray(t["pts"], float)
        if t["a"] > 0:
            ext.setdefault(t["a"], []).append(p)
        if t["b"] > 0:
            ext.setdefault(t["b"], []).append(p[::-1])                   # siempre empezando en el racimo
    for brazos in ext.values():
        largo = [_brazo(b, 0, +1, BRAZO, False) for b in brazos]
        def opuesto(k):                                                  # el brazo casi opuesto a k (su trazo PASA por el nudo), o None
            c = [(_angulo(largo[k][0], uj), j) for j, (uj, lj) in enumerate(largo) if j != k and lj >= BRAZO_MIN]
            c = [t for t in c if t[0] >= 180 - ANG_SIGUE]
            return max(c)[1] if c else None
        par = [opuesto(k) for k in range(len(brazos))]
        for x in range(len(brazos)):
            for y in range(x + 1, len(brazos)):
                p, q = brazos[x], brazos[y]
                (u, lu), (v, lv) = largo[x], largo[y]
                # un CRUCE (§ 69) es cuando los DOS siguen de largo por trazos DISTINTOS (la X del 8); si los dos se oponen
                # al MISMO brazo es una λ (la punta del 4 con el palo asomando por arriba, val #390): eso sí es un vértice
                if lu < BRAZO_MIN or lv < BRAZO_MIN or (par[x] is not None and par[y] is not None and par[x] != par[y]):
                    continue
                uc, _ = _brazo(p, 0, +1, BRAZO_CORTO, False); vc, _ = _brazo(q, 0, +1, BRAZO_CORTO, False)
                c = _candidato(*((p[0] + q[0]) / 2), u, v, uc, vc)
                if c is not None:
                    c["brazos"] = (_camino(p, 0, +1, ALCANCE, False), _camino(q, 0, +1, ALCANCE, False)); cand.append(c)
    out = []
    for c in sorted(cand, key=lambda c: c["ang"]):                       # supresión: el más agudo de cada sitio
        if all(np.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) >= SUPR for o in out):
            out.append(c)
    return out


def celda(v) -> int:
    return min(int(v["cy"] * 3 / N), 2) * 3 + min(int(v["cx"] * 3 / N), 2)



TOPE = {"a": 4.0, "b": 90.0, "c": 6.0}
def _largo(b):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(b, axis=0).T))])


def _en(b, L):
    """El punto del brazo a L px de recorrido (interpolado)."""
    s = _largo(b)
    if L >= s[-1]:
        return b[-1]
    k = int(np.searchsorted(s, L)); t = (L - s[k - 1]) / (s[k] - s[k - 1] + 1e-9); return b[k - 1] + t * (b[k] - b[k - 1])


def _recta(b, desde=2.0, hasta=6.0):
    """Recta ajustada al tramo del brazo entre `desde` y `hasta` px de la punta: (punto, dirección unitaria hacia FUERA).
    Primera versión: de 3 a 9 px, y en brazos curvos (el lazo de un 6) las rectas se cruzaban a 112 px de la punta."""
    s = _largo(b); q = b[(s >= desde) & (s <= hasta)]
    if len(q) < 2:
        q = b[len(b) // 2:]
    m = q.mean(0); _, vec = np.linalg.eigh(np.cov((q - m).T)); d = vec[:, 1]
    if np.dot(d, b[-1] - b[0]) < 0:
        d = -d
    return m, d


def medir(v) -> dict:
    b1, b2 = v["brazos"]; P = (b1[0] + b2[0]) / 2
    # (a) despegue
    (m1, d1), (m2, d2) = _recta(b1), _recta(b2)
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-3:
        X, da = None, TOPE["a"]
    else:
        t = np.linalg.solve(A, m2 - m1); X = m1 + t[0] * d1; da = float(np.hypot(*(X - P)))
    # (b) se abre de cerca
    def ang(L):
        return _angulo(_en(b1, L) - b1[0], _en(b2, L) - b2[0])
    db = max(0.0, float(ang(2.0) - ang(6.0)))   # en un cruce raro salía negativo (de cerca MÁS cerrado): eso no es redondez
    # (c) radio: círculo ajustado (mínimos cuadrados, Kåsa) a los puntos de la cresta a ≤ 3 px de la punta por cada brazo, SIN
    # suavizar. Primera versión: 1/curvatura máxima de la cresta suavizada; el suavizado (σ 1,5) movía la punta y el radio
    # medía sobre todo el suavizado (casi todos entre 2 y 4 px, círculos que no tocaban la punta).
    q = np.concatenate([b1[_largo(b1) <= 3.0][::-1], b2[_largo(b2) <= 3.0][1:]])
    A3 = np.c_[2 * q, np.ones(len(q))]; rhs = (q ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A3, rhs, rcond=None); centro = sol[:2]; r = float(np.sqrt(max(sol[2] + (centro ** 2).sum(), 0)))
    except np.linalg.LinAlgError:
        centro, r = P, 99.0
    if not np.isfinite(r) or len(q) < 3:
        centro, r = P, 99.0
    return {"a": da, "b": db, "c": r, "cruce": X, "P": P, "rectas": ((m1, d1), (m2, d2)), "cerca": (_en(b1, 2.0), _en(b2, 2.0)),
            "lejos": (_en(b1, 6.0), _en(b2, 6.0)), "circulo": (centro, r), "puntos_c": q}


def escala(k, x):
    return float(np.clip(x / TOPE[k], 0, 1))


ORDEN = ["N", "E", "S", "O"]                                             # el de la brújula, en el sentido del reloj (§ 103)
PIEZAS = ["rejilla", "amplitud", "despegue", "cerca", "radio", "cuenta"]   # por dirección, en este orden de columnas


def bloque(M, celdas: int = 3) -> list[float]:
    """Los números del bloque `vertice_libre` de UN dígito (delgado M), en el orden de `PIEZAS` y por dirección `ORDEN`."""
    c2 = celdas * celdas; g = {d: np.zeros((5, c2)) for d in ORDEN}; ang = {d: np.full(c2, 999.0) for d in ORDEN}; n = dict.fromkeys(ORDEN, 0)
    for v in vertices_libres(M):
        d = v["dir"]; k = min(celdas - 1, int(v["cy"] * celdas / N)) * celdas + min(celdas - 1, int(v["cx"] * celdas / N)); n[d] += 1
        if v["ang"] < ang[d][k]:                                          # el más agudo de la celda manda
            m = medir(v); ang[d][k] = v["ang"]
            g[d][:, k] = [1.0, v["ang"] / 180.0, escala("a", m["a"]), escala("b", m["b"]), escala("c", m["c"])]
    out = []
    for d in ORDEN:
        out += g[d].ravel().tolist() + [float(n[d])]
    return out


def huella_modulo() -> str:
    """El hash de lo que calcula este bloque: este fichero y `cresta_curvas.py` (de donde sale la cresta)."""
    h = hashlib.sha1((AQUI / "nn" / "vertice_libre.py").read_bytes() + (AQUI / "nn" / "cresta_curvas.py").read_bytes())
    return h.hexdigest()[:10]
