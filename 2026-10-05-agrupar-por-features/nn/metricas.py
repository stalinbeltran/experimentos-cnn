#!/usr/bin/env python3
"""Las piezas numéricas de `feat-agr`, en numpy (aquí no hay scipy ni scikit-learn, y no se instalan para no tocar el venv
que comparten los demás experimentos). Cada una se prueba contra un caso hecho a mano:

    python nn/metricas.py --comprobar

  kmeans       k-means euclídeo, siembra k-means++ (la del obtenedor de `feat-ind`, corrida 6, pasada a distancia euclídea)
  asignar      el centroide más cercano
  medoides     el miembro más cercano a cada centroide
  ward         árbol de Ward sobre centroides pesados por tamaño, y el orden de sus hojas
  pureza, nmi, ari          la partición contra otra (NMI con normalización aritmética, ARI de Hubert y Arabie)
  auc, auc_sd  Mann–Whitney con empates a medias; auc_sd = max(AUC, 1 − AUC), sin dirección
  kappa        consistencia corregida por azar: (c − Σp²) / (1 − Σp²)
  separacion   suma de cuadrados ENTRE grupos, por dimensión (lo que reparte L7)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def dist2(X: np.ndarray, C: np.ndarray, x2: np.ndarray | None = None) -> np.ndarray:
    """(n, K) distancias euclídeas al cuadrado, ≥ 0."""
    x2 = (X * X).sum(1) if x2 is None else x2
    return np.maximum(x2[:, None] - 2 * (X @ C.T) + (C * C).sum(1)[None], 0)


def asignar(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return dist2(X, C).argmin(1)


def kmeans(X: np.ndarray, K: int, semilla: int, iters: int = 300) -> dict:
    """Lloyd con siembra k-means++, hasta que ninguna asignación cambie o `iters`. Un grupo vacío se re-siembra en el punto
    más lejano de su centroide. Devuelve grupos = el centroide más cercano a los centroides finales."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = len(X); rng = np.random.default_rng(semilla); x2 = (X * X).sum(1)
    C = np.empty((K, X.shape[1]), np.float32)
    i0 = int(rng.integers(n)); C[0] = X[i0]
    d = dist2(X, C[:1], x2)[:, 0]
    for k in range(1, K):
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1 / n)
        i = int(rng.choice(n, p=p)); C[k] = X[i]
        d = np.minimum(d, dist2(X, C[k:k + 1], x2)[:, 0])
    prev, convergio, it = None, False, 0
    for it in range(1, iters + 1):
        D = dist2(X, C, x2); lab = D.argmin(1)
        for k in np.flatnonzero(np.bincount(lab, minlength=K) == 0):
            lejos = int(D[np.arange(n), lab].argmax())
            C[k] = X[lejos]; D[:, k] = dist2(X, C[k:k + 1], x2)[:, 0]; lab = D.argmin(1)
        if prev is not None and np.array_equal(lab, prev):
            convergio = True
            break
        prev = lab
        uno = np.zeros((n, K), np.float32); uno[np.arange(n), lab] = 1
        C = ((uno.T @ X) / np.maximum(uno.sum(0), 1)[:, None]).astype(np.float32)
    D = dist2(X, C, x2); lab = D.argmin(1)
    return {"grupos": lab.astype(np.int16), "centroides": C, "inercia": float(D[np.arange(n), lab].sum(dtype=np.float64)),
            "iteraciones": it, "convergio": convergio}


def medoides(X: np.ndarray, C: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """(K,) el índice del miembro más cercano a cada centroide (−1 si un grupo quedara vacío)."""
    D = dist2(np.ascontiguousarray(X, dtype=np.float32), C)
    out = np.full(len(C), -1, np.int64)
    for k in range(len(C)):
        m = np.flatnonzero(lab == k)
        if len(m):
            out[k] = m[D[m, k].argmin()]
    return out


def ward(C: np.ndarray, tam: np.ndarray) -> list:
    """Árbol de Ward sobre K centroides con sus tamaños. Devuelve las K − 1 uniones [a, b, altura, tamaño], con los ids de
    scipy (0..K−1 las hojas, K.. los nodos). Coste de unir A y B: nA·nB/(nA + nB)·‖cA − cB‖²; altura = √(2·coste)."""
    act = {i: (np.asarray(C[i], np.float64), float(tam[i])) for i in range(len(C))}
    Z, nuevo = [], len(C)
    while len(act) > 1:
        ids = sorted(act); mejor = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                (ca, na), (cb, nb) = act[ids[x]], act[ids[y]]
                coste = na * nb / (na + nb) * float(((ca - cb) ** 2).sum())
                if mejor is None or coste < mejor[0]:
                    mejor = (coste, ids[x], ids[y])
        coste, a, b = mejor
        (ca, na), (cb, nb) = act.pop(a), act.pop(b)
        act[nuevo] = ((na * ca + nb * cb) / (na + nb), na + nb)
        Z.append([a, b, float(np.sqrt(2 * coste)), na + nb]); nuevo += 1
    return Z


def orden_hojas(Z: list, K: int) -> list:
    """Las hojas de izquierda a derecha: lo parecido queda junto."""
    if K == 1:
        return [0]
    hijos = {K + i: (int(a), int(b)) for i, (a, b, _, _) in enumerate(Z)}

    def bajar(n):
        return [n] if n < K else bajar(hijos[n][0]) + bajar(hijos[n][1])
    return bajar(K + len(Z) - 1)


def _contingencia(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ia = np.unique(a, return_inverse=True); _, ib = np.unique(b, return_inverse=True)
    t = np.zeros((ia.max() + 1, ib.max() + 1), np.int64)
    np.add.at(t, (ia, ib), 1)
    return t


def pureza(g: np.ndarray, y: np.ndarray) -> float:
    """Fracción de dígitos cuya etiqueta es la mayoritaria de su grupo."""
    return float(_contingencia(g, y).max(1).sum() / len(y))


def _entropia(n: np.ndarray) -> float:
    p = n[n > 0] / n.sum()
    return float(-(p * np.log(p)).sum())


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Información mutua normalizada, 2·I / (H(a) + H(b)) (la «arithmetic» de scikit-learn)."""
    t = _contingencia(a, b).astype(np.float64); n = t.sum()
    ha, hb = _entropia(t.sum(1)), _entropia(t.sum(0))
    if ha + hb == 0:
        return 1.0
    pa, pb, pij = t.sum(1) / n, t.sum(0) / n, t / n
    nz = pij > 0
    i = float((pij[nz] * np.log(pij[nz] / (pa[:, None] * pb[None])[nz])).sum())
    return 2 * i / (ha + hb)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Índice de Rand ajustado (Hubert y Arabie)."""
    t = _contingencia(a, b).astype(np.float64); n = t.sum()

    def c2(x):
        return x * (x - 1) / 2
    s, sa, sb = c2(t).sum(), c2(t.sum(1)).sum(), c2(t.sum(0)).sum()
    esp = sa * sb / c2(n); mx = (sa + sb) / 2
    return 1.0 if mx == esp else float((s - esp) / (mx - esp))


def _rangos(x: np.ndarray) -> np.ndarray:
    """Rangos 1..n con los empates a la media."""
    o = np.argsort(x, kind="mergesort"); xs = x[o]
    r = np.empty(len(x), np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(un valor de a > uno de b), con los empates a medias (Mann–Whitney)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = _rangos(np.concatenate([a, b]))
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(u / (len(a) * len(b)))


def auc_sd(a, b) -> float:
    v = auc(a, b)
    return max(v, 1 - v)


def kappa(c: float, lab: np.ndarray, K: int) -> float:
    """(c − Σp²) / (1 − Σp²), con p la fracción de dígitos de cada grupo: lo que c tendría por azar con esos tamaños."""
    p = np.bincount(lab, minlength=K) / len(lab)
    e = float((p * p).sum())
    return (c - e) / (1 - e)


def separacion(X: np.ndarray, lab: np.ndarray, K: int) -> np.ndarray:
    """(D,) suma de cuadrados ENTRE grupos por dimensión: Σ_g n_g (c_gd − μ_d)². Sumada da la parte de la varianza total que
    explica la partición; repartida por dimensión dice qué la manda (L7)."""
    X = np.asarray(X, np.float64); mu = X.mean(0)
    out = np.zeros(X.shape[1])
    for k in range(K):
        m = lab == k
        if m.any():
            out += m.sum() * (X[m].mean(0) - mu) ** 2
    return out


def comprobar() -> int:
    ok = True

    def mira(nombre, cond, detalle=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FALLA':>5}] {nombre}" + (f" — {detalle}" if detalle else "")); ok &= bool(cond)

    a, b = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2])
    mira("ARI([0,0,1,1], [0,0,1,2]) = 4/7, hecho a mano", abs(ari(a, b) - 4 / 7) < 1e-12, f"{ari(a, b):.6f}")
    mira("ARI y NMI de una partición consigo misma (renombrada) = 1", ari(a, 1 - a) == 1 and abs(nmi(a, 1 - a) - 1) < 1e-12)
    mira("NMI de dos particiones independientes = 0", abs(nmi(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))) < 1e-12)
    mira("pureza([0,0,1,1], [0,1,1,1]) = 3/4", pureza(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == 0.75)
    mira("AUC([1,2,3], [0]) = 1 y AUC([0], [0]) = 0,5", auc([1, 2, 3], [0]) == 1 and auc([0], [0]) == 0.5)
    mira("AUC([1,3], [2,2]) = 0,5 y auc_sd([0], [1]) = 1", auc([1, 3], [2, 2]) == 0.5 and auc_sd([0], [1]) == 1)
    lab = np.array([0, 0, 0, 1])
    mira("kappa: c = Σp² da 0 y c = 1 da 1", abs(kappa(10 / 16, lab, 2)) < 1e-12 and abs(kappa(1.0, lab, 2) - 1) < 1e-12)
    rng = np.random.default_rng(0)
    centros = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    X = np.concatenate([c + rng.normal(0, 0.5, (50, 2)) for c in centros]).astype(np.float32)
    verdad = np.repeat([0, 1, 2], 50)
    km = kmeans(X, 3, semilla=1)
    mira("k-means separa tres nubes lejanas (ARI = 1) y converge", ari(km["grupos"], verdad) == 1 and km["convergio"],
         f"{km['iteraciones']} iteraciones")
    med = medoides(X, km["centroides"], km["grupos"])
    mira("cada medoide es de su grupo", all(km["grupos"][m] == k for k, m in enumerate(med)))
    sep = separacion(X, km["grupos"], 3)
    tot = ((X - X.mean(0)) ** 2).sum(0)
    w = sum(((X[km["grupos"] == k] - X[km["grupos"] == k].mean(0)) ** 2).sum(0) for k in range(3))
    mira("separación = total − dentro, por dimensión", np.allclose(sep, tot - w, rtol=1e-6))
    C4 = np.array([[0.0], [1.0], [10.0], [11.0]]); Z = ward(C4, np.ones(4))
    mira("Ward une primero los pares cercanos y luego los dos pares",
         {tuple(sorted(Z[0][:2])), tuple(sorted(Z[1][:2]))} == {(0, 1), (2, 3)} and sorted(Z[2][:2]) == [4, 5])
    mira("el orden de las hojas deja juntos los pares", orden_hojas(Z, 4) in ([0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 3, 2],
                                                                              [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 0, 1],
                                                                              [2, 3, 1, 0], [3, 2, 1, 0]))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    sys.exit(main())
