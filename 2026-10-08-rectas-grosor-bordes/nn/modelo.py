#!/usr/bin/env python3
"""El detector de `rect-lin`: un kernel LINEAL (célula simple de V1), AUTOCONTENIDO.

    2 kernels k×k aprendidos: K0 (—) y K45 (\\).  K90 = rot90(K0) (|), K135 = rot90(K45) (/): giro EXACTO en la rejilla.
    para cada nivel de la pirámide (la imagen, ½, ¼ por media 2×2):  r_o = K_o ⋆ x  →  max espacial
    s_o = max entre niveles;   logit_o = a·s_o + c        (a, c escalares compartidos; c hace de umbral)

Parámetros: 2k² + 2 (52 con k=5, 100 con k=7, 164 con k=9).

`Gabor` es la referencia SIN entrenar: el mismo esquema con kernels puestos a mano (Gabor par, alargado a lo largo de
la recta) y sólo el umbral calibrado sobre negativos.

    python nn/modelo.py --comprobar
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def piramide(x: torch.Tensor, niveles: int) -> list[torch.Tensor]:
    return [x] + [F.avg_pool2d(x, 2 ** l) for l in range(1, niveles)]


def respuesta(x: torch.Tensor, kernels: torch.Tensor, niveles: int) -> torch.Tensor:
    """(N,1,32,32), kernels (4,1,k,k) → (N,4): max sobre posiciones y niveles de K⋆x.

    Sin ReLU a propósito: max(ReLU(z − t)) = ReLU(max(z) − t), así que la rectificación sólo recorta lo que ya está
    bajo el umbral, y el umbral lo lleva `c`. Con un umbral aprendido DENTRO de la ReLU el entrenamiento se moría
    (medido 2026-10-08: t subía hasta apagar todas las respuestas → gradiente cero, recall 0 en todo)."""
    k = kernels.shape[-1]
    s = None
    for xl in piramide(x, niveles):
        r = F.conv2d(xl, kernels, padding=k // 2).amax(dim=(-2, -1))
        s = r if s is None else torch.maximum(s, r)
    return s


class Lineal(nn.Module):
    def __init__(self, k: int, niveles: int, semilla: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(semilla)
        self.k, self.niveles = k, niveles
        self.K = nn.Parameter(torch.randn(2, 1, k, k, generator=g) * (1.0 / k))
        self.a = nn.Parameter(torch.ones(()))
        self.c = nn.Parameter(torch.full((), -1.0))

    def kernels(self) -> torch.Tensor:
        K0, K45 = self.K[0:1], self.K[1:2]
        return torch.cat([K0, K45, torch.rot90(K0, 1, (-2, -1)), torch.rot90(K45, 1, (-2, -1))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * respuesta(x, self.kernels(), self.niveles) + self.c

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def kernel_gabor(k: int, ang: float, lam: float = 6.0) -> np.ndarray:
    """Gabor PAR: positivo en la línea media, negativo a los lados (λ = 6 px: trazo de ~3 px), alargado a lo largo.
    Media cero, norma 1. Misma convención de ángulo que los datos (y hacia abajo)."""
    r = (k - 1) / 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.deg2rad(ang)
    u = xx * np.cos(t) + yy * np.sin(t)              # a lo largo
    v = -xx * np.sin(t) + yy * np.cos(t)             # perpendicular
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * v / lam)
    g -= g.mean()
    return (g / np.linalg.norm(g)).astype(np.float32)


class Gabor(nn.Module):
    """La referencia a mano. logit_o = s_o − umbral; el umbral se fija con `calibrar` sobre NEGATIVOS (no ve rectas)."""

    def __init__(self, k: int, niveles: int):
        super().__init__()
        self.k, self.niveles = k, niveles
        self.register_buffer("Ks", torch.from_numpy(np.stack([kernel_gabor(k, a) for a in (0, 45, 90, 135)])[:, None]))
        self.umbral = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return respuesta(x, self.Ks, self.niveles) - self.umbral

    @torch.no_grad()
    def calibrar(self, x_neg: torch.Tensor, fp: float = 0.05) -> float:
        self.umbral = 0.0
        self.umbral = float(torch.quantile(self(x_neg).amax(1), 1 - fp))
        return self.umbral


def comprobar() -> int:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import datos as D  # noqa: PLC0415
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien); print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    for k in (5, 7, 9):
        m = Lineal(k, 3)
        Ks = m.kernels()
        prueba(f"k={k}: K90 = rot90(K0) y K135 = rot90(K45), exactos",
               torch.equal(Ks[2], torch.rot90(Ks[0], 1, (-2, -1))) and torch.equal(Ks[3], torch.rot90(Ks[1], 1, (-2, -1))),
               f"{m.n_parametros()} parámetros")
    prueba("la salida es (N, 4)", Lineal(7, 2)(torch.rand(3, 1, 32, 32)).shape == (3, 4))
    prueba("la pirámide de 3 niveles da 32, 16 y 8", [t.shape[-1] for t in piramide(torch.rand(1, 1, 32, 32), 3)] == [32, 16, 8])
    # el Gabor, sin entrenar, orienta bien una recta fina en los 4 centros
    g = Gabor(9, 1)
    x = torch.from_numpy(np.stack([D.recta(16, 16, 20, a, 3) for a in (0, 45, 90, 135)]).astype(np.float32)[:, None])
    prueba("Gabor 9×9: — \\ | / caen en su orientación", g(x).argmax(1).tolist() == [0, 1, 2, 3], str(g(x).argmax(1).tolist()))
    # y el giro de rot90 casa con la convención de los datos: 45° (\) girado da 135° (/)
    prueba("rot90 del Gabor de 45° es el de 135°",
           np.allclose(np.rot90(kernel_gabor(9, 45)), kernel_gabor(9, 135), atol=1e-6) or
           np.allclose(np.rot90(kernel_gabor(9, 45), -1), kernel_gabor(9, 135), atol=1e-6))
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    raise SystemExit(comprobar() if p.parse_args().comprobar else (print(__doc__) or 0))
