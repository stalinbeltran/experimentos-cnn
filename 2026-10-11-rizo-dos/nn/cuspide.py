#!/usr/bin/env python3
"""rizo-2 — la CÚSPIDE como FEATURE del revisor: detector y bloque (el dueño, 2026-10-17: «lo que busco es un cambio en la
derivada: 2 curvas abiertas en la misma dirección que se encuentran en una punta»; tras ver § 110, «procede»). Nace del prototipo
de § 110, del que se MUEVE aquí el núcleo (`cuspide_vista.py` lo importa: dos copias de un detector divergen).

ENTRADA: el DELGADO (`M`, delg-cnn), regla del repo. Se mira su CRESTA con la MISMA curvatura con signo κ de `cresta_curvas`
(suavizado σ = CC.SIGMA, umbral CC.K_MIN). Al recorrer la cintura de un 3 la tangente gira hacia un lado en la curva de arriba (+),
da un giro corto AL REVÉS en la punta (−) y vuelve a girar hacia el primer lado en la curva de abajo (+). Esa firma + − + es la
cúspide:

  · un run de κ de un signo con |giro| ≥ G_PUNTA y ≤ L_PUNTA px de largo (la punta),
  · con ≥ G_ARCO de giro del signo CONTRARIO en los VENT px de cada lado (las dos curvas, que por eso abren hacia el mismo lado).
  · Una S (2, 5) es + −, sin el segundo +: no la da. Un vértice de la cresta (vertice_libre) es UN giro, sin curvas a los lados.

Se mira en cada tramo empalmado y, para la punta que cae en un NUDO, en cada par de brazos del nudo de dos maneras: (a) los dos
brazos unidos, con la misma firma en κ (así sale la cintura del 8, en su X); y (b) por GEOMETRÍA (`en_nudo`), cuando los demás
brazos son cortos (la lengüeta de un 3). La (b) existe porque en el nudo el esqueleto deja un HUECO de ~3 px (el racimo), y unir y
suavizar diluye el giro: medido el 2026-10-17 con un 3 sintético espejado, una punta de ~150° salía como 17° y no se veía. En (b) el
giro de la punta es la dirección con que el trazo llega al nudo contra la dirección con que sale (a CUERDA px), y el de cada curva
se mide sobre su brazo solo.
DIRECCIÓN = hacia dónde apunta la punta (contraria al centro de su giro), en 4 sectores de 90° (N E S O, regla del repo: § 103).
Medido en val el 2026-10-17 (§ 110): 3 → 80 % (todas O), 8 → 69 % (la cintura, E y O), 9 → 3 %, el resto 0 %.

El BLOQUE (`bloque`) da, por dirección N, E, S, O: rejilla de presencia (UNA celda, sin mancha) + rejilla del GIRO de la punta
(grados/90, acotado a 1) + rejilla de las CURVAS (el menor de los dos giros de al lado, grados/180, acotado a 1) + cuenta.
Con 3×3: 4 × (3·9 + 1) = 112 números. Si dos cúspides de la misma dirección caen en la misma celda, manda la de más giro.

⚠ Fuera de `cresta_curvas.py` a propósito (como `vertice_libre.py`): tocar ese fichero cambia la huella de TODOS los
compositores guardados. Este módulo lleva su propio hash (`huella_modulo`), que sólo entra en la huella de los que lo usan.
⚠ Parámetros SIN calibrar contra ninguna verdad (los de § 110).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import cresta_curvas as CC                                               # noqa: E402

# Lo PÚBLICO (R5: el revisor importa de aquí; lo prueban sus tests en web/probar_revisor.py)
__all__ = ["cuspides", "bloque", "celda", "PIEZAS", "ORDEN", "huella_modulo", "G_ARCO", "G_PUNTA", "L_PUNTA", "VENT", "SUPR", "TOPE",
           "CUERDA", "LENGUA"]

N = 32
G_ARCO = 45.0      # giro mínimo de cada curva, en sus VENT px junto a la punta
G_PUNTA = 15.0     # giro mínimo de la punta (al revés)
L_PUNTA = 12.0     # largo máximo de la punta (px de cresta); con 8 px el 3 baja del 80 al 70 % (§ 110)
VENT = 12.0        # px de cada lado en los que se suma el giro de las curvas
SUPR = 3.0         # px: dos candidatas más cerca son la misma cúspide
CUERDA = 3.0       # px: en un nudo, a qué distancia se mide la dirección de llegada y de salida del trazo
LENGUA = 6.0       # px: para medir por geometría (`en_nudo`), cualquier OTRO brazo del nudo tiene que ser así de corto (la lengüeta
                   # del 3). Sin esta regla, medido el 2026-10-17 en val: la unión del lazo y la cola del 9 daba la firma (9: 3 % → 33 %)
TOPE = {"giro": 90.0, "curvas": 180.0}
ORDEN = ["N", "E", "S", "O"]                                             # el de la brújula, en el sentido del reloj (§ 103)
PIEZAS = ["rejilla", "giro", "curvas", "cuenta"]                         # por dirección, en este orden de columnas


def _rumbo(b) -> float:
    """Brújula del vector b (en coordenadas de imagen, y hacia abajo): 0° = N, 90° = E, 180° = S, 270° = O."""
    return float(np.degrees(np.arctan2(b[0], -b[1])) % 360)


def _direccion(r: float) -> str:
    return "NESO"[int(((r + 45) % 360) // 90)]


def _a(p, L):
    """El punto del camino p (empezando en p[0]) a L px de recorrido."""
    acc = 0.0
    for j in range(1, len(p)):
        acc += float(np.hypot(*(p[j] - p[j - 1])))
        if acc >= L:
            return p[j]
    return p[-1]


def _giro_curva(p, L) -> float:
    """Giro con signo (°, la convención de κ) de los primeros L px del camino p, sobre el camino SOLO (sin el hueco del nudo)."""
    if len(p) < 5:
        return 0.0
    ps = CC._suavizar(p, False); k, _ = CC.curvatura(ps, False); paso = np.hypot(*np.gradient(ps, axis=0).T)
    s = np.concatenate([[0], np.cumsum(paso)])[:-1]
    return float(np.degrees(np.sum((k * paso)[s <= L])))


def en_nudo(bx, by) -> dict | None:
    """La cúspide en un nudo entre dos brazos que SALEN de él (bx[0], by[0] en el nudo), o None. El trazo llega por bx y sale por
    by: la punta es el giro en el nudo; las curvas, el giro de cada brazo en sus VENT px, del signo CONTRARIO."""
    if len(bx) < 3 or len(by) < 3:
        return None
    nx, ny = bx[0], by[0]; u = _a(bx, CUERDA) - nx; v = _a(by, CUERDA) - ny
    hin, hout = -u, v                                                   # llega en contra de bx, sale a lo largo de by
    giro = float(np.degrees(np.arctan2(hin[0] * hout[1] - hin[1] * hout[0], np.dot(hin, hout))))
    if abs(giro) < G_PUNTA:
        return None
    c = np.sign(giro)
    gi = -c * _giro_curva(bx, VENT) * -1                                # bx recorrido hacia el nudo: al revés, el giro cambia de signo
    gd = -c * _giro_curva(by, VENT)
    if gi < G_ARCO or gd < G_ARCO:
        return None
    b = -(u / (np.hypot(*u) + 1e-9) + v / (np.hypot(*v) + 1e-9)); r = _rumbo(b); P = (nx + ny) / 2
    return {"cx": float(P[0]), "cy": float(P[1]), "giro": abs(giro), "curvas": (float(gi), float(gd)), "rumbo": r, "dir": _direccion(r),
            "pts": np.concatenate([bx[:8][::-1], by[:8]])}


def en_camino(p, junta=None) -> list[dict]:
    """Las cúspides de UN camino de cresta (puntos en orden). Con `junta`, sólo las que caen a ≤ SUPR de ese punto (un nudo)."""
    p = np.asarray(p, float)
    if len(p) < 5:
        return []
    ps = CC._suavizar(p, False); k, nrm = CC.curvatura(ps, False); paso = np.hypot(*np.gradient(ps, axis=0).T)
    s = np.concatenate([[0], np.cumsum(paso)]); cl = np.where(np.abs(k) >= CC.K_MIN, np.sign(k), 0).astype(int)
    out = []; i = 0; n = len(ps)
    while i < n:
        j = i
        while j < n and cl[j] == cl[i]:
            j += 1
        c = cl[i]
        if c != 0:
            giro = np.degrees(np.sum(k[i:j] * paso[i:j])); largo = s[j] - s[i]
            izq = s[i] - s[:i] <= VENT; der = s[j:n] - s[j] <= VENT
            gi = -c * np.degrees(np.sum((k[:i] * paso[:i])[izq])); gd = -c * np.degrees(np.sum((k[j:] * paso[j:])[der]))
            if abs(giro) >= G_PUNTA and largo <= L_PUNTA and gi >= G_ARCO and gd >= G_ARCO:
                m = i + int(np.argmax(np.abs(k[i:j]))); r = _rumbo(-nrm[i:j].mean(0))
                if junta is not None and np.hypot(*(ps[m] - junta)) > SUPR:
                    i = j; continue
                out.append({"cx": float(ps[m, 0]), "cy": float(ps[m, 1]), "giro": float(abs(giro)), "curvas": (float(gi), float(gd)),
                            "rumbo": r, "dir": _direccion(r), "pts": ps[max(0, i - 6):min(n, j + 6)]})
        i = j
    return out


def cuspides(M) -> list[dict]:
    """[{cx, cy, giro (°, la punta), curvas (°, °), rumbo, dir (N E S O), pts (la cresta alrededor)}] de un dígito (delgado M)."""
    tr, _ = CC.tramos(M); cand = []
    for t in CC.empalmar(tr):
        cand += en_camino(t["pts"])
    ext = {}
    for t in tr:
        p = np.asarray(t["pts"], float)
        if t["a"] > 0:
            ext.setdefault(t["a"], []).append(p)
        if t["b"] > 0:
            ext.setdefault(t["b"], []).append(p[::-1])                   # siempre empezando en el nudo
    largo = lambda b: float(np.sum(np.hypot(*np.diff(b, axis=0).T))) if len(b) > 1 else 0.0
    for br in ext.values():                                               # la punta EN un nudo: cada par de brazos, de DOS maneras
        for x in range(len(br)):
            for y in range(x + 1, len(br)):
                # (a) los dos brazos UNIDOS y la firma en κ (§ 110, lo que aprobó el dueño): da la cintura del 8 en su X
                cand += en_camino(np.concatenate([br[x][::-1], br[y][1:]]), junta=br[x][0])
                # (b) por GEOMETRÍA (`en_nudo`), sólo si cualquier OTRO brazo es corto (la lengüeta del 3): el hueco del racimo
                # diluye (a). Con otro brazo largo, medido en val, (b) daba la unión del lazo y la cola del 9 (3 % → 33 %)
                if any(largo(br[z]) > LENGUA for z in range(len(br)) if z not in (x, y)):
                    continue
                c = en_nudo(br[x], br[y])
                if c is not None:
                    cand.append(c)
    out = []
    for c in sorted(cand, key=lambda c: -c["giro"]):                      # supresión: la de más giro de cada sitio
        if all(np.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) >= SUPR for o in out):
            out.append(c)
    return out


def celda(c, celdas: int = 3) -> int:
    return min(celdas - 1, int(c["cy"] * celdas / N)) * celdas + min(celdas - 1, int(c["cx"] * celdas / N))


def bloque(M, celdas: int = 3) -> list[float]:
    """Los números del bloque `cuspide` de UN dígito (delgado M), en el orden de `PIEZAS` y por dirección `ORDEN`."""
    c2 = celdas * celdas; g = {d: np.zeros((3, c2)) for d in ORDEN}; n = dict.fromkeys(ORDEN, 0)
    for c in cuspides(M):
        d = c["dir"]; k = celda(c, celdas); n[d] += 1
        if c["giro"] / TOPE["giro"] > g[d][1, k] or g[d][0, k] == 0:     # la de más giro de la celda manda
            g[d][:, k] = [1.0, min(c["giro"] / TOPE["giro"], 1.0), min(min(c["curvas"]) / TOPE["curvas"], 1.0)]
    out = []
    for d in ORDEN:
        out += g[d].ravel().tolist() + [float(n[d])]
    return out


def huella_modulo() -> str:
    """El hash de lo que calcula este bloque: este fichero y `cresta_curvas.py` (de donde salen la cresta y κ)."""
    h = hashlib.sha1((AQUI / "nn" / "cuspide.py").read_bytes() + (AQUI / "nn" / "cresta_curvas.py").read_bytes())
    return h.hexdigest()[:10]
