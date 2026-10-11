"""rizo-2 — la CNN de la estrategia C, AUTOCONTENIDA (no importa nada del repo).

Totalmente convolucional, 32×32 → 5 mapas: [0] logit del mapa de calor del centro del rizo, y en cada píxel [1] radio/7,
[2] cierre, [3] cos(abertura), [4] sin(abertura). Las convoluciones dilatadas dan un campo receptivo de ±10 px, que
cubre un rizo de radio 7 con su trazo.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BRAZOS = {"c-base": {"c": (32, 32, 64, 64, 32)}, "c-ancha": {"c": (48, 48, 96, 96, 48)}}


class Rizo(nn.Module):
    def __init__(self, c=(32, 32, 64, 64, 32)):
        super().__init__()
        dil = (1, 1, 2, 4, 1); capas = []; prev = 1
        for ch, d in zip(c, dil):
            capas += [nn.Conv2d(prev, ch, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]; prev = ch
        self.cuerpo = nn.Sequential(*capas)
        self.cabeza = nn.Conv2d(prev, 5, 1)

    def forward(self, x):
        return self.cabeza(self.cuerpo(x))

    def parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())
