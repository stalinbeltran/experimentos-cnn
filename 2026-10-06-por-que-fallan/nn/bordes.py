#!/usr/bin/env python3
"""COPIA del `nn/bordes.py` de `feat-bor` (2026-10-06). Los BORDES de una imagen binaria de 32×32. El resto —el vocabulario, el sorteo, la red,
el entrenamiento— es el de `feat-ind32`; aquí sólo cambia lo que VE el detector.

  lineas     1 canal: la imagen tal cual (el trazo relleno). Es la referencia: lo que vieron los detectores de feat-ind32
  contorno   1 canal: la tinta con algún vecino-4 de fondo (el borde interior, 1 px), SIN signo. Es lo pedido, literal
  signo      4 canales: la tinta con fondo a la IZQUIERDA · a la DERECHA · ARRIBA · ABAJO. Cada lado del trazo, aparte

Fuera del lienzo cuenta como fondo. Entra (N, 32, 32) o (N, 1, 32, 32), 0/1; sale (N, C, 32, 32) uint8 0/1.

⚠ Por qué dos bordes y no uno (medido el 2026-10-06 con `nn/sonda_grosor.py`, antes de entrenar): un trazo grueso tiene
DOS bordes, uno a cada lado, y se separan al engrosarlo. El contorno sin signo convierte «grueso» en «dos líneas»; el
borde con signo pone cada lado en su canal, y cada canal conserva la forma (desplazada medio grosor).

    python nn/bordes.py      comprueba las dos transformaciones en casos hechos a mano
"""

from __future__ import annotations

import sys

import numpy as np

REPRESENTACIONES = ("lineas", "contorno", "signo")
CANALES = {"lineas": 1, "contorno": 1, "signo": 4}
LADOS = ("izquierda", "derecha", "arriba", "abajo")     # los 4 canales de `signo`, en este orden


def bordes(x: np.ndarray, modo: str) -> np.ndarray:
    x = (np.asarray(x) > 0.5).astype(np.uint8)
    if x.ndim == 4:
        x = x[:, 0]
    if modo == "lineas":
        return x[:, None]
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    c = p[:, 1:-1, 1:-1]
    izq, der, arr, aba = p[:, 1:-1, :-2], p[:, 1:-1, 2:], p[:, :-2, 1:-1], p[:, 2:, 1:-1]
    if modo == "contorno":
        return (c & (1 - (izq & der & arr & aba)))[:, None]
    if modo == "signo":
        return np.stack([c & (1 - izq), c & (1 - der), c & (1 - arr), c & (1 - aba)], 1)
    raise ValueError(f"representación '{modo}' desconocida: {REPRESENTACIONES}")


def comprobar() -> int:
    ok = True

    def mira(que, cond):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FALLA':>5}] {que}"); ok &= bool(cond)

    barra = np.zeros((1, 32, 32), np.uint8); barra[0, 4:28, 10:16] = 1          # barra vertical de 6 px de ancho
    c = bordes(barra, "contorno")[0, 0]; s = bordes(barra, "signo")[0]
    mira("contorno de una barra de 6 px: dos columnas (10 y 15) más las tapas", c[10:22, 10].all() and c[10:22, 15].all()
         and not c[10:22, 11:15].any() and c[4, 10:16].all() and c[27, 10:16].all())
    mira("signo: el canal izquierda es SÓLO la columna 10, el derecha SÓLO la 15", (s[0].sum(0) > 0).nonzero()[0].tolist() == [10]
         and (s[1].sum(0) > 0).nonzero()[0].tolist() == [15])
    mira("signo: arriba es la fila 4 y abajo la 27", (s[2].sum(1) > 0).nonzero()[0].tolist() == [4]
         and (s[3].sum(1) > 0).nonzero()[0].tolist() == [27])
    fina = np.zeros((1, 32, 32), np.uint8); fina[0, 4:28, 12] = 1                 # barra de 1 px
    mira("una línea de 1 px: su contorno es ella misma, y está en izquierda y en derecha a la vez",
         (bordes(fina, "contorno")[0, 0] == fina[0]).all() and (bordes(fina, "signo")[0, 0] == fina[0]).all()
         and (bordes(fina, "signo")[0, 1] == fina[0]).all())
    borde = np.zeros((1, 32, 32), np.uint8); borde[0, :, 0] = 1                   # tinta en la columna 0: fuera es fondo
    mira("fuera del lienzo cuenta como fondo", bordes(borde, "signo")[0, 0, :, 0].all())
    mira("lineas devuelve la imagen tal cual, con su canal", (bordes(barra, "lineas")[:, 0] == barra).all())
    mira("salida uint8 0/1 con los canales de CANALES", all(bordes(barra, m).shape[1] == CANALES[m] and bordes(barra, m).dtype == np.uint8
                                                             and bordes(barra, m).max() <= 1 for m in REPRESENTACIONES))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(comprobar())
