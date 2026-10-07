#!/usr/bin/env python3
"""Prueba rápida (0 $, en el dev), 2026-10-07 (tarde): ¿qué COMPOSITOR aguanta mejor el grosor y el desplazamiento?

Sigue a prueba_vistas.py (mismos kernels FIJOS de ver_proceso.py, misma regresión logística, mismo mapa 8×8 por vista)
y compara compositores que se diferencian SÓLO en cómo ven la posición:

    pos            la media de cada bloque 4×4: la posición tal cual (el «compositor de antes»)
    max5           máximo 5×5 sobre el mapa 32×32 antes de reducir (tolera ±2 px)
    …+desp≤k       el mismo compositor entrenado además con su ENTRADA 8×8 desplazada 1..k CELDAS en las 8
                   direcciones (relleno con ceros): 1 + 8k copias de cada ejemplo. Una celda del 8×8 son 4 px.
    …+px≤k         lo mismo, pero desplazando la IMAGEN 1..k PÍXELES (32×32) antes de detectar y reducir: el
                   desplazamiento fino, de menos de una celda, que la entrada 8×8 no puede expresar.
                   La prueba es siempre sin desplazar: es aumentación de entrenamiento, no de prueba.

Dos medidas de «grueso», las dos del pendiente del 2026-10-07:

  A. Dígitos UCI (uci-optdigits-orig-32px-r20261005). Se entrena con los 3823 de otros escritores (`extra`) o con los
     180 de `train`; se prueba en los 1617 de `val`: tal cual, engrosados 1 y 2 px por dilatación (el estímulo de ayer,
     que funde trazos), el cuartil con más tinta de cada clase (gruesos REALES), y desplazados 2 px en una dirección al
     azar por dígito (el desplazamiento que el compositor debería tolerar).
  B. Trazos sintéticos (feat-bor-sinteticas-grueso-32px-r20261006): 13 familias de feature + vacío, dibujadas de 2 a
     12 px. Se entrena con las de 2–4 px (70 %, semilla 0) y se prueba con el 30 % restante de 2–4 px y con cada grosor
     NO visto: 6, 8, 10 y 12 px. Clasifica la FAMILIA de la principal, esté donde esté (la ancla recorre todo el 8×8).
     Es un sustituto de la pregunta, no dígitos.

    /tmp/vizenv/bin/python prueba_compositores.py --uci <dir con datos.npz> --grueso <dir con datos.npz> [--partes …]
    → resultados-compositores.json (se reescribe al acabar CADA parte) y la tabla por stdout
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, shift as desplazar_img
from sklearn.linear_model import LogisticRegression
from ver_proceso import detectar, DIRS

AQUI = Path(__file__).resolve().parent
BRAZOS = {"tinta": ["tinta"], "2 vistas → ↓": [0, 90], "8 vistas": [g for _, g in DIRS]}
DESP = (0, 1, 2, 3, 4)


def mapas(X, pool):
    """{vista: (N, 8, 8)} para las 8 direcciones y la tinta."""
    def red(m):
        if pool == "max5":
            m = maximum_filter(m, size=5)
        return m.reshape(8, 4, 8, 4).mean((1, 3))
    out = {g: np.stack([red(detectar(x, g)) for x in X]) for _, g in DIRS}
    out["tinta"] = np.stack([red(x.astype(float)) for x in X])
    return out


def mover(M, dy, dx):
    """Desplaza (N, n, n) dy, dx casillas, rellenando con ceros."""
    n = M.shape[-1]; R = np.zeros_like(M)
    ys, yd = (slice(0, n - dy), slice(dy, n)) if dy >= 0 else (slice(-dy, n), slice(0, n + dy))
    xs, xd = (slice(0, n - dx), slice(dx, n)) if dx >= 0 else (slice(-dx, n), slice(0, n + dx))
    R[:, yd, xd] = M[:, ys, xs]
    return R


def movimientos(k):
    """(0, 0) y los desplazamientos de 1..k en las 8 direcciones: 1 + 8k."""
    return [(0, 0)] + [(s * a, s * b) for s in range(1, k + 1) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def aumentar(F, vs, y, k):
    """Entrada concatenada de las vistas vs, con las copias desplazadas 1..k celdas en las 8 direcciones."""
    movs = movimientos(k)
    X = np.vstack([np.hstack([mover(F[v], dy, dx).reshape(len(y), -1) for v in vs]) for dy, dx in movs])
    return X.astype(np.float32), np.tile(y, len(movs))


def juntar(F, vs):
    return np.hstack([F[v].reshape(len(F[v]), -1) for v in vs])


def engrosar(X, r):
    est = np.ones((2 * r + 1, 2 * r + 1), bool)
    return np.stack([binary_dilation(x > 0, est) for x in X]).astype(np.uint8)


def desplazados(X, px, semilla=0):
    """Cada imagen movida px píxeles en una de las 8 direcciones, al azar (fija por semilla)."""
    rng = np.random.default_rng(semilla)
    dirs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    return np.stack([desplazar_img(x, np.array(dirs[rng.integers(8)]) * px, order=0, cval=0) for x in X])


def ajustar(res, nombre, A, b, vs, Fte, etiqueta):
    t = time.time()
    m = LogisticRegression(max_iter=3000, C=1.0).fit(A, b)
    res[nombre] = {n: round(float((m.predict(juntar(F, vs)) == yp).mean()), 4) for n, (F, yp) in Fte.items()}
    res[nombre]["_iter"] = int(m.n_iter_.max())
    print(f"  [{etiqueta}] {nombre:32s} {len(b):>7d} ej. {time.time() - t:6.1f} s  "
          + " ".join(f"{v:.3f}" for n, v in res[nombre].items() if n != "_iter"), flush=True)


def medir(Xtr, ytr, pruebas, etiqueta):
    """Para cada pool × brazo × desplazamiento de entrenamiento (celdas y píxeles), el acierto en cada prueba."""
    res = {}
    for pool in ("pos", "max5"):
        Ftr = mapas(Xtr, pool)
        Fte = {n: (mapas(Xp, pool), yp) for n, (Xp, yp) in pruebas.items()}
        for k in DESP:                                   # desplazar la ENTRADA 8×8 k celdas
            for brazo, vs in BRAZOS.items():
                A, b = aumentar(Ftr, vs, ytr, k)
                ajustar(res, f"{brazo} · {pool}" + (f"+desp≤{k}" if k else ""), A, b, vs, Fte, etiqueta)
        for k in DESP[1:]:                               # desplazar la IMAGEN k píxeles, antes de detectar
            movs = movimientos(k)
            Fa = mapas(np.vstack([mover(Xtr, dy, dx) for dy, dx in movs]), pool)
            for brazo, vs in BRAZOS.items():
                ajustar(res, f"{brazo} · {pool}+px≤{k}", juntar(Fa, vs).astype(np.float32), np.tile(ytr, len(movs)),
                        vs, Fte, etiqueta)
    return res


def tabla(titulo, res, columnas):
    print(f"\n== {titulo} ==")
    print(f"{'compositor':34s}" + "".join(f"{c:>15s}" for c in columnas))
    for b, v in res.items():
        print(f"{b:34s}" + "".join(f"{v[c]:>15.3f}" for c in columnas))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--uci", required=True); a.add_argument("--grueso", required=True)
    a.add_argument("--partes", default="uci3823,uci180,sinteticas"); a = a.parse_args()
    partes = a.partes.split(",")
    destino = AQUI / "resultados-compositores.json"
    salida = {"cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "desp": list(DESP), "A_uci": {}}
    guardar = lambda: destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1))

    d = np.load(Path(a.uci) / "datos.npz"); X, y, part = d["imagenes"], d["etiquetas"], d["particion"]
    te = part == "val"
    tinta = X.reshape(len(X), -1).sum(1)
    gruesos = np.zeros(len(X), bool)
    for c in range(10):
        m = te & (y == c); gruesos[m] = tinta[m] >= np.quantile(tinta[m], .75)
    pruebas = {"normal": (X[te], y[te]), "+1px": (engrosar(X[te], 1), y[te]), "+2px": (engrosar(X[te], 2), y[te]),
               "gruesos-reales": (X[gruesos], y[gruesos]), "desplazado-2px": (desplazados(X[te], 2), y[te])}
    salida["A_uci_n_gruesos_reales"] = int(gruesos.sum())
    for clave, reg, mtr in (("uci3823", "3823 de otros escritores", part == "extra"), ("uci180", "180 de train", part == "train")):
        if clave in partes:
            salida["A_uci"][reg] = medir(X[mtr], y[mtr], pruebas, f"UCI {reg}"); guardar()

    g = np.load(Path(a.grueso) / "datos.npz"); Xg, yg, gr = g["imagenes"], g["principal"].astype(int), g["grosor"]
    finos = np.flatnonzero(gr <= 4)                                  # incluye las vacías (grosor −1)
    rng = np.random.default_rng(0); rng.shuffle(finos)
    ntr = int(.7 * len(finos)); tr, va = finos[:ntr], finos[ntr:]
    pruebas_g = {"2–4px (val)": (Xg[va], yg[va])} | {f"{p}px": (Xg[gr == p], yg[gr == p]) for p in (6, 8, 10, 12)}
    salida["B_n"] = {"entrena": len(tr)} | {n: len(v[1]) for n, v in pruebas_g.items()}
    if "sinteticas" in partes:
        salida["B_sinteticas"] = medir(Xg[tr], yg[tr], pruebas_g, "sintéticas"); guardar()

    for reg, r in salida["A_uci"].items():
        tabla(f"A · dígitos UCI · entrenado con {reg} · acierto", r, list(pruebas))
    if "B_sinteticas" in salida:
        tabla(f"B · trazos sintéticos · entrenado con {len(tr)} de 2–4 px · acierto (14 clases)", salida["B_sinteticas"],
              list(pruebas_g))


if __name__ == "__main__":
    main()
