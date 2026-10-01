#!/usr/bin/env python3
"""Las VENTANAS de `bor-ae` (copiadas de `bor-k` el 2026-10-01), recortadas del dataset publicado. No se publica nada nuevo.

    python nn/datos.py            resume las ventanas de cada particion y comprueba sus garantias

El dato de entrada es `parrafos1000-pagina1024-r4-r20261001` (lo produce `bor-p4`): paginas
de 256 x 256 (la SUMA de cada bloque 4x4 de un render de 1024) y la caja de tinta de cada
parrafo. Aqui se recortan ventanas de esas paginas, en memoria y siempre las mismas
(semilla fija, independiente de la del entrenamiento): dos semillas de entrenamiento ven
EXACTAMENTE las mismas ventanas y solo difieren en la inicializacion y el orden.

LA VENTANA, Y DE DONDE SALE CADA NUMERO (todo en px de la pagina GUARDADA)
=========================================================================

  LADO = 55, MEDIO = 27
  SALIDA = [9, 45]  (37 px)  -- la region que la red LEE, la MISMA para todo k
      La convolucion `valid` de un k <= 19 se recorta despues a 37 x 37: k=19 no recorta
      nada, k=3 recorta 8 por lado. Es la regla del §6.2 del banco (descarte fijo de 9 px)
      y por la misma razon: si cada k leyera su propio campo, los brazos verian DISTINTA
      cantidad de imagen y la comparacion mezclaria kernel con campo de vision.

  VAIVEN = 12
      La ventana NO se centra exactamente en el borde: se desplaza perpendicular a el un
      u ~ U[-12, 12]. Centrada siempre en el borde, predecir "el centro" daria error 0 sin
      aprender nada. Con el vaiven el borde cae entre los px 15 y 39 de la ventana, dentro
      de la SALIDA con 6 px de holgura.

  MEDIO + VAIVEN = 39  <=  (ventana_limpia_min - 1) / 2
      `bor-p4` GARANTIZA por construccion que una ventana de 79 x 79 centrada en
      CUALQUIER punto de un borde no sale del papel ni ve tinta de otro parrafo. Una de
      55 desplazada 12 cae dentro de esa de 79, asi que tambien es limpia: cada ventana
      positiva ve UN parrafo. Se COMPRUEBA al construir, no se supone.

QUE ES UNA VENTANA Y QUE ETIQUETA LLEVA
======================================
  positivas  por cada parrafo y cada uno de sus 4 bordes, una ventana centrada en un
             punto sorteado UNIFORME a lo largo del borde, mas el vaiven. 4 por parrafo,
             sin filtrar ninguno (sortear-y-rechazar sesgaria hacia parrafos pequenos).
  al azar    por cada pagina, tantas como parrafos tiene, centradas al azar en el papel.
             Son los negativos (fondo e interior) y los bordes "de refilon". Una ventana
             al azar que toque DOS parrafos se vuelve a sortear: ahi si se rechaza, porque
             es un negativo y no hay tamano que sesgar.

  Etiqueta, por borde t en (izq, der, sup, inf): `existe_t` y `coord_t` (px de la
  ventana, continuo). Un borde EXISTE en la ventana si su linea cae en la SALIDA con 1 px
  de holgura ([10, 44]) y el segmento cubre al menos MIN_TRAMO = 6 px de la SALIDA a lo
  largo. Puede haber varios bordes en una ventana (los parrafos pequenos caben enteros);
  de cada tipo, como mucho uno, porque la ventana ve un solo parrafo.

LA ENTRADA DE LA RED: TINTA
    x = 1 - suma / 4080, en [0, 1]: papel 0, tinta 1. Asi el papel no aporta nada a la
    convolucion y un kernel sin bias puede quedarse en 0 donde no hay nada que ver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))

from expcnn import exigir_dataset  # noqa: E402

DATASET = "parrafos1000-pagina1024-r4-r20261001"
REDUCCION = 4
BLANCO = 255 * REDUCCION ** 2          # 4080: la suma de un bloque de papel puro

LADO = 55
MEDIO = (LADO - 1) // 2                # 27
DESCARTE = 9                           # la SALIDA empieza aqui, para todo k
SALIDA = LADO - 2 * DESCARTE           # 37
VAIVEN = 12
HOLGURA = 1
MIN_TRAMO = 6
SEMILLA_VENTANAS = 0

BORDES = ("izq", "der", "sup", "inf")
PARTES = {"train": 0, "val": 1, "eval": 2}


def _cargar_dataset():
    p = exigir_dataset(DATASET)
    d = np.load(p / "paginas.npz")
    c = np.load(p / "cajas.npz")
    return d["paginas"], d["particion"], c["cajas"], c["derivadas"]


def etiqueta(caja_r4, x0: int, y0: int):
    """Las 4 etiquetas de UNA caja (izq, der, sup, inf en px guardados) en la ventana
    cuya esquina superior-izquierda es (x0, y0). Devuelve (existe[4], coord[4]).

    ⚠ La coordenada va en el espacio de INDICES de pixel (el centro del pixel i es i), no
    en el continuo de la caja (donde el pixel i ocupa [i, i+1)): por eso el -0,5. Un borde
    de tinta cae ENTRE dos pixeles, y es ahi donde una derivada responde (lo comprueba
    `nn/modelo.py`). Sin esto, la cabeza -- que no tiene sesgo de coordenada -- arrastraria
    medio pixel de error fijo en todos los brazos."""
    izq, der, sup, inf = (caja_r4[0] - x0 - 0.5, caja_r4[1] - x0 - 0.5,
                          caja_r4[2] - y0 - 0.5, caja_r4[3] - y0 - 0.5)
    lo, hi = DESCARTE + HOLGURA, DESCARTE + SALIDA - 1 - HOLGURA       # [10, 44]
    s_lo, s_hi = DESCARTE, DESCARTE + SALIDA - 1                       # [9, 45]
    tramo_v = max(0.0, min(inf, s_hi) - max(sup, s_lo))   # lo que cubre en vertical
    tramo_h = max(0.0, min(der, s_hi) - max(izq, s_lo))   # y en horizontal
    existe = np.zeros(4, np.float32)
    coord = np.array([izq, der, sup, inf], np.float32)
    for i, (linea, tramo) in enumerate(((izq, tramo_v), (der, tramo_v),
                                        (sup, tramo_h), (inf, tramo_h))):
        existe[i] = float(lo <= linea <= hi and tramo >= MIN_TRAMO)
    return existe, coord


def _toca(caja_r4, x0, y0) -> bool:
    return not (caja_r4[1] < x0 or caja_r4[0] > x0 + LADO - 1
                or caja_r4[3] < y0 or caja_r4[2] > y0 + LADO - 1)


def ventanas(parte: str):
    """(x, existe, coord, info) de una particion. x: (N, 1, 55, 55) float32 en TINTA.

    `info` (N, 4) int32: pagina, parrafo (-1 si es al azar), borde objetivo (-1 si es al
    azar), x0. Solo para el desglose y las figuras: la red no lo ve."""
    paginas, particion, cajas, derivadas = _cargar_dataset()
    rng = np.random.default_rng(SEMILLA_VENTANAS + 1000 * PARTES[parte])
    lado_pag = paginas.shape[-1]
    sel_pags = np.flatnonzero(particion == PARTES[parte])
    xs, ex, co, info = [], [], [], []

    def anyadir(pag, x0, y0, idx_par, borde):
        sub = paginas[pag, y0:y0 + LADO, x0:x0 + LADO]
        assert sub.shape == (LADO, LADO), (pag, x0, y0)
        xs.append(1.0 - sub.astype(np.float32) / BLANCO)
        e_tot = np.zeros(4, np.float32)
        c_tot = np.zeros(4, np.float32)
        for j in np.flatnonzero(cajas[:, 0] == pag):
            r4 = cajas[j, 1:] / REDUCCION
            if not _toca(r4, x0, y0):
                continue
            e, c = etiqueta(r4, x0, y0)
            for t in range(4):
                if e[t]:
                    assert not e_tot[t], f"dos bordes {BORDES[t]} en una ventana (pag {pag})"
                    e_tot[t], c_tot[t] = 1.0, c[t]
        ex.append(e_tot); co.append(c_tot); info.append((pag, idx_par, borde, x0))

    for pag in sel_pags:
        mias = np.flatnonzero(cajas[:, 0] == pag)
        for j in mias:
            izq, der, sup, inf = cajas[j, 1:] / REDUCCION
            limpia = derivadas[j, 2]
            assert MEDIO + VAIVEN <= (limpia - 1) / 2, (
                f"parrafo {j}: ventana limpia {limpia} < {2 * (MEDIO + VAIVEN) + 1}: "
                f"el dataset no da la garantia que esta ventana necesita")
            for t, borde in enumerate(BORDES):
                u = rng.uniform(-VAIVEN, VAIVEN)
                if borde in ("izq", "der"):
                    cx = (izq if borde == "izq" else der) + u
                    cy = rng.uniform(sup, inf)
                else:
                    cx = rng.uniform(izq, der)
                    cy = (sup if borde == "sup" else inf) + u
                x0 = int(round(cx)) - MEDIO
                y0 = int(round(cy)) - MEDIO
                anyadir(pag, x0, y0, int(j), t)
        hechas = 0
        while hechas < len(mias):
            x0, y0 = (int(v) for v in rng.integers(0, lado_pag - LADO + 1, size=2))
            if sum(_toca(cajas[j, 1:] / REDUCCION, x0, y0) for j in mias) >= 2:
                continue
            anyadir(pag, x0, y0, -1, -1)
            hechas += 1

    x = np.stack(xs)[:, None]
    return (x.astype(np.float32), np.stack(ex), np.stack(co),
            np.array(info, dtype=np.int32))


def _resumen() -> int:
    for parte in PARTES:
        x, e, c, info = ventanas(parte)
        pos = info[:, 2] >= 0
        print(f"{parte:>5}: {len(x):>5} ventanas ({pos.sum()} de borde, {(~pos).sum()} al azar) · "
              + " · ".join(f"{b} {int(e[:, i].sum())}" for i, b in enumerate(BORDES))
              + f" · sin ningun borde {int((e.sum(1) == 0).sum())}")
        objetivo_presente = e[pos, info[pos, 2]].mean()
        print(f"        el borde objetivo existe en el {100 * objetivo_presente:.1f} % de las de borde "
              f"(tiene que ser 100)")
        if objetivo_presente < 1:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_resumen())
