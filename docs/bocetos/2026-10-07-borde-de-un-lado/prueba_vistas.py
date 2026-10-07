#!/usr/bin/env python3
"""Prueba rápida (0 $, en el dev): ¿cuánta información pierde el compositor si ve UNA vista de bordes de un lado, dos,
cuatro u ocho? ¿Y un «compositor de compositores» (uno por vista, y otro encima) frente a uno que ve todo junto?

NO es el experimento: los detectores son los kernels FIJOS de ver_proceso.py (no los de features entrenados) y el
compositor es una regresión logística sobre el mapa de cada vista reducido a 8×8 (media por bloques 4×4). Mide la
información que queda en cada vista, no lo que haría la red de verdad.

Datos: uci-optdigits-orig-32px-r20261005. Dos regímenes de entrenamiento: 3823 dígitos de otros 30 escritores
(`extra`), y los 180 de `train` (el régimen de los compositores de feat-ind32). Prueba: los 1617 de `val`, tal cual y
ENGROSADOS artificialmente (dilatación de 1 y 2 px: un estímulo de prueba, no un pre-proceso de la cadena), y el cuartil
de dígitos con más tinta de cada clase (gruesos de verdad).

    /tmp/vizenv/bin/python prueba_vistas.py --datos <dir> [--pool media|max5|max-bloque]   # resultados-vistas[-pool].json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from ver_proceso import detectar, DIRS

AQUI = Path(__file__).resolve().parent
NOMBRE = {g: f for f, g in DIRS}


POOL = "media"


def pool8(m):                                   # 32×32 → 8×8
    if POOL == "max5":                          # tolera desplazamientos de ±2 px antes de reducir
        m = maximum_filter(m, size=5)
    if POOL == "max-bloque":                    # el máximo de cada bloque 4×4, no la media
        return m.reshape(8, 4, 8, 4).max((1, 3)).ravel()
    return m.reshape(8, 4, 8, 4).mean((1, 3)).ravel()


def vistas(X):
    """{dirección: (N, 64)} y la tinta (N, 64)."""
    out = {g: np.stack([pool8(detectar(x, g)) for x in X]) for _, g in DIRS}
    out["tinta"] = np.stack([pool8(x.astype(float)) for x in X])
    return out


def engrosar(X, r):
    if r == 0: return X
    est = np.ones((2 * r + 1, 2 * r + 1), bool)
    return np.stack([binary_dilation(x > 0, est) for x in X]).astype(np.uint8)


def logreg():
    return LogisticRegression(max_iter=3000, C=1.0)


def peores_pares(y, p, k=3):
    c = confusion_matrix(y, p, labels=range(10)); np.fill_diagonal(c, 0)
    s = c + c.T; pares = [(int(s[i, j]), f"{i}↔{j}") for i in range(10) for j in range(i + 1, 10)]
    return [f"{n_}{'' if n == 0 else ''} ({n})" for n, n_ in sorted(pares, reverse=True)[:k]]


def main():
    a = argparse.ArgumentParser(); a.add_argument("--datos", required=True)
    a.add_argument("--pool", choices=("media", "max5", "max-bloque"), default="media"); a = a.parse_args()
    global POOL; POOL = a.pool
    d = np.load(Path(a.datos) / "datos.npz"); X, y, part = d["imagenes"], d["etiquetas"], d["particion"]
    te = part == "val"
    tinta = X.reshape(len(X), -1).sum(1)
    gruesos = np.zeros(len(X), bool)
    for k in range(10):
        m = te & (y == k); gruesos[m] = tinta[m] >= np.quantile(tinta[m], .75)
    pruebas = {"normal": (X[te], y[te]), "+1px": (engrosar(X[te], 1), y[te]), "+2px": (engrosar(X[te], 2), y[te]),
               "gruesos-reales": (X[gruesos], y[gruesos])}
    F_test = {n: (vistas(Xp), yp) for n, (Xp, yp) in pruebas.items()}
    brazos = {"tinta": ["tinta"]}
    brazos |= {f"1 vista {NOMBRE[g]}": [g] for _, g in DIRS}
    brazos |= {"2 vistas → ↓": [0, 90], "4 vistas → ↓ ← ↑": [0, 90, 180, 270], "8 vistas": [g for _, g in DIRS]}
    res = {}
    for reg, mtr in (("3823 de otros escritores", part == "extra"), ("180 de train", part == "train")):
        Ftr, ytr = vistas(X[mtr]), y[mtr]
        r = res[reg] = {}
        for nombre, vs in brazos.items():            # UN compositor con las vistas concatenadas
            m = logreg().fit(np.hstack([Ftr[v] for v in vs]), ytr)
            r[nombre + " (juntas)" if len(vs) > 1 else nombre] = {
                n: round(float((m.predict(np.hstack([F[v] for v in vs])) == yp).mean()), 4) for n, (F, yp) in F_test.items()}
        for nombre, vs in (("2 vistas → ↓", [0, 90]), ("4 vistas → ↓ ← ↑", [0, 90, 180, 270]), ("8 vistas", [g for _, g in DIRS])):
            # compositor de compositores: uno por vista (probabilidades de las 10 clases) y uno encima (apilado)
            base = [logreg().fit(Ftr[v], ytr) for v in vs]
            oof = np.hstack([cross_val_predict(logreg(), Ftr[v], ytr, cv=5, method="predict_proba") for v in vs])
            meta = logreg().fit(oof, ytr)
            r[nombre + " (compositor de compositores)"] = {}
            r[nombre + " (media de compositores)"] = {}
            for n, (F, yp) in F_test.items():
                P = [b.predict_proba(F[v]) for b, v in zip(base, vs)]
                r[nombre + " (compositor de compositores)"][n] = round(float((meta.predict(np.hstack(P)) == yp).mean()), 4)
                r[nombre + " (media de compositores)"][n] = round(float((np.mean(P, 0).argmax(1) == yp).mean()), 4)
        print(f"\n== entrenado con {reg} · acierto en val (1617) ==")
        print(f"{'brazo':52s}" + "".join(f"{n:>16s}" for n in pruebas))
        for b, v in r.items(): print(f"{b:52s}" + "".join(f"{v[n]:>16.3f}" for n in pruebas))
    (AQUI / f"resultados-vistas{'' if POOL == 'media' else '-' + POOL}.json").write_text(json.dumps(
        {"pool": POOL, "cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_gruesos_reales": int(gruesos.sum()), "acierto": res},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
