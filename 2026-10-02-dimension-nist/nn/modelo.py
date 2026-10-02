#!/usr/bin/env python3
"""La red de `dim-nist`, AUTOCONTENIDA: L = 2 capas `valid` con kernel n = ⌈W/2⌉ (f = 1/L = 0,5),
promedio global (GAP) y una `Linear(C -> 10)`.

    python nn/modelo.py        la tabla de mapas y parametros por W, y un forward

Por que L = 2 y no las 4 de dim-gen: con imagenes de 8 px, cuatro capas sin padding quitan al
menos 4 px (n >= 2) y a W = 4 el mapa se agota. Con L = 2 la misma regla f = 1/L da n = W/2; para
W impar n se redondea hacia arriba, el mapa final es 2x2 (W par) o 1x1 (W impar) y por eso la
cabeza es un promedio global en vez de un `Flatten`: asi es CONSTANTE entre W (C -> 10) y la red
ve la imagen entera en todos los W (campo receptivo 2n - 1 >= W - 1).
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

L = 2
F = 0.5             # = 1/L
C = 8               # canales por capa, constante entre W (como dim-gen)
CLASES = 10
W_TODOS = (8, 7, 6, 5, 4)
SEMILLAS = (1, 2, 3, 4, 5)
BRAZOS = {"w8": (8, False), "w7": (7, False), "w6": (6, False), "w5": (5, False), "w4": (4, False),
          "w8-de4": (8, True)}
_ID = re.compile(r"^(?P<brazo>w\d(?:-de4)?)-s(?P<s>\d+)$")


def parsear(ident: str) -> tuple[str, int, bool, int]:
    m = _ID.match(ident)
    if not m or m["brazo"] not in BRAZOS:
        raise ValueError(f"brazo '{ident}' desconocido; los validos son {sorted(BRAZOS)} con sufijo -s<semilla>")
    W, control = BRAZOS[m["brazo"]]
    return m["brazo"], W, control, int(m["s"])


def kernel_de(W: int, L_: int = L, f: float = F) -> int:
    n = math.ceil(f * W - 1e-9)
    if n < 2:
        raise ValueError(f"W={W}: n={n} no es una convolucion")
    return n


def mapas(W: int, L_: int = L, f: float = F) -> list[int]:
    n, lados, w = kernel_de(W, L_, f), [], W
    for _ in range(L_):
        w = w - n + 1
        if w < 1:
            raise ValueError(f"W={W}: el mapa se agota con n={n}")
        lados.append(w)
    return lados


class Red(nn.Module):
    def __init__(self, W: int, C_: int = C, L_: int = L, f: float = F):
        super().__init__()
        n = kernel_de(W, L_, f)
        lados = mapas(W, L_, f)
        capas, cin = [], 1
        for _ in range(L_):
            capas += [nn.Conv2d(cin, C_, n), nn.ReLU()]      # sin padding, sin stride: `valid`
            cin = C_
        self.tronco = nn.Sequential(*capas)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cabeza = nn.Linear(C_, CLASES)
        self.W, self.n, self.lados = W, n, lados

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(torch.flatten(self.gap(self.tronco(x)), 1))

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_parametros_cabeza(self) -> int:
        return sum(p.numel() for p in self.cabeza.parameters())


def construir(W: int, semilla: int, C_: int = C) -> Red:
    torch.manual_seed(semilla)
    return Red(W, C_)


def tabla() -> list[dict]:
    return [{"W": W, "n": (r := Red(W)).n, "mapas": r.lados, "parametros": r.n_parametros(),
             "cabeza": r.n_parametros_cabeza(), "campo_receptivo": 2 * r.n - 1} for W in W_TODOS]


def main() -> int:
    print(f"L={L} f={F} C={C}")
    for fila in tabla():
        print(f"  W={fila['W']} n={fila['n']} mapas {'->'.join(map(str, fila['mapas'])):<6} RF {fila['campo_receptivo']} "
              f"parametros {fila['parametros']:>5} (cabeza {fila['cabeza']})")
    red = construir(5, 1)
    print(f"forward W=5: salida {tuple(red(torch.rand(2, 1, 5, 5)).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
