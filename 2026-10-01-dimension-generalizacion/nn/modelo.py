#!/usr/bin/env python3
"""La red de `dim-gen`, AUTOCONTENIDA (no importa nada del repo): `L` capas `valid` con kernel
`n = f * W` y una `Linear(L*L*C -> 4)`.

    python nn/modelo.py        la tabla de mapas y parametros por W, y un forward

La regla (ESPECIFICACION.md §3.1): con f = 1/L el lado final es W - L*(W/L - 1) = L para todo W
multiplo de L, asi que el mapa final es L x L y la cabeza es CONSTANTE entre W. Lo que cambia
con W son los pixeles -- y los parametros de las convoluciones, que crecen con n^2.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

L = 4
F = 0.25            # = 1/L. Decision 1 del dueno (2026-10-02): «mejor criterio» -> lo recomendado
C = 8               # decision 2: 8 canales por capa, constante entre W
W_TODOS = (128, 64, 32, 16, 8)
SEMILLAS = (1, 2, 3, 4, 5)
# brazo -> (W de la red, es el control)
BRAZOS = {"w128": (128, False), "w064": (64, False), "w032": (32, False), "w016": (16, False),
          "w008": (8, False), "w128-de16": (128, True)}
_ID = re.compile(r"^(?P<brazo>w\d{3}(?:-de16)?)-s(?P<s>\d+)$")


def parsear(ident: str) -> tuple[str, int, bool, int]:
    """'w128-de16-s3' -> ('w128-de16', 128, True, 3). Se niega con cualquier otra forma."""
    m = _ID.match(ident)
    if not m or m["brazo"] not in BRAZOS:
        raise ValueError(f"brazo '{ident}' desconocido; los validos son "
                         f"{sorted(BRAZOS)} con sufijo -s<semilla>")
    W, control = BRAZOS[m["brazo"]]
    return m["brazo"], W, control, int(m["s"])


def kernel_de(W: int, L_: int = L, f: float = F) -> int:
    n = f * W
    if abs(n - round(n)) > 1e-9 or round(n) < 2:
        raise ValueError(f"n = f*W = {n} no es entero >= 2 para W={W}, f={f}")
    return int(round(n))


def mapas(W: int, L_: int = L, f: float = F) -> list[int]:
    n, lados, w = kernel_de(W, L_, f), [], W
    for _ in range(L_):
        w = w - n + 1
        lados.append(w)
    return lados


class Red(nn.Module):
    def __init__(self, W: int, C_: int = C, L_: int = L, f: float = F):
        super().__init__()
        n = kernel_de(W, L_, f)
        lados = mapas(W, L_, f)
        if lados[-1] != L_:
            raise ValueError(f"W={W}: el mapa final es {lados[-1]}, no {L_}: f no es 1/L")
        capas, cin = [], 1
        for _ in range(L_):
            capas += [nn.Conv2d(cin, C_, n), nn.ReLU()]   # sin padding, sin stride: `valid`
            cin = C_
        self.tronco = nn.Sequential(*capas)
        self.cabeza = nn.Linear(lados[-1] * lados[-1] * C_, 4)
        self.W, self.n, self.lados = W, n, lados

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(torch.flatten(self.tronco(x), 1))

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_parametros_cabeza(self) -> int:
        return sum(p.numel() for p in self.cabeza.parameters())


def construir(W: int, semilla: int, C_: int = C) -> Red:
    """La red de un brazo, con la inicializacion por defecto de torch (escala con 1/sqrt(fan_in),
    o sea con el tamano del kernel) sembrada con la semilla del brazo."""
    torch.manual_seed(semilla)
    return Red(W, C_)


def tabla() -> list[dict]:
    filas = []
    for W in W_TODOS:
        red = Red(W)
        filas.append({"W": W, "n": red.n, "mapas": red.lados, "parametros": red.n_parametros(),
                      "cabeza": red.n_parametros_cabeza()})
    return filas


def main() -> int:
    print(f"L={L} f={F} C={C}")
    for fila in tabla():
        print(f"  W={fila['W']:>3} n={fila['n']:>2} mapas {'->'.join(map(str, fila['mapas'])):<14} "
              f"parametros {fila['parametros']:>7} (cabeza {fila['cabeza']})")
    red = construir(32, 1)
    y = red(torch.rand(2, 1, 32, 32))
    print(f"forward W=32: salida {tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
