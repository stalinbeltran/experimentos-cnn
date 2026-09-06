#!/usr/bin/env python3
"""La red: UNA convolucion y una cabeza de 3 parametros.

    entrada (1, 32, 32)
       -> conv 1 kernel k x k, stride 1, SIN padding, SIN bias      k^2 parametros
       -> mapa M de m x m,  m = 32 - k + 1
       -> cabeza C1:  p = softmax(beta * M)
                      (x, y) = esperanza de la posicion bajo p      0 parametros
                      existe = sigmoide(a * logsumexp(beta*M)/beta + b)   2 parametros
       -> [existe, x, y]

POR QUE NO HAY ReLU ENTRE LA CONV Y LA CABEZA
    Medido el 2026-09-06 sobre un mapa sintetico 22x22 con un pico sobre fondo
    negativo (que es lo que produce un filtro de cuadrante). Error del centroide
    en pixeles del mapa:

        beta      directo   con ReLU
           1         8,85       9,07
           2         4,85       8,28
           3         0,47       5,02
           5         0,00       0,20

    El fondo rectificado queda EMPATADO en 0, asi que su masa nunca se apaga: con
    ReLU hacen falta beta ~5-8 para clavar la esquina y sin ella basta ~3.
    Y el argumento que no depende de beta: con un filtro de cuadrante casi todo el
    mapa es negativo, y la ReLU pone su gradiente a CERO -- justo donde el kernel
    aprende a NO responder. Ademas, con una sola convolucion no hay segunda capa
    que hacer no lineal: la no linealidad util ya esta en el softmax.

POR QUE LA CONV NO LLEVA BIAS
    El softmax es invariante a sumar una constante al mapa, y en `existe` esa
    constante sale del logsumexp y la absorbe `b`. O sea que el bias es
    REDUNDANTE -- pero solo mientras no haya ReLU. Las dos decisiones van juntas.

BETA SE APRENDE, Y NO ARRANCA EN 1
    A beta = 1 la lectura colapsa al centro del mapa (tabla de arriba) y no hay
    nada que aprender. Arranca en ~3,5, que es donde la lectura se engancha, y
    va como exp(log_beta) para que no pueda cruzar cero.

    python nn/modelo.py            # comprueba las cinco estructuras
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

VENTANA = 32
BRAZOS = {"k03": 3, "k05": 5, "k07": 7, "k09": 9, "k11": 11}
BETA0 = 3.5


class EsquinaUnKernel(nn.Module):
    def __init__(self, k: int, ventana: int = VENTANA, beta0: float = BETA0):
        super().__init__()
        if (k - 1) % 2:
            raise ValueError(f"k tiene que ser impar para que el campo receptivo tenga centro; es {k}")
        self.k, self.ventana = k, ventana
        self.m = ventana - k + 1
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        # La posicion i del mapa mira el centro de su campo receptivo, que en
        # coordenadas de la VENTANA cae en i + (k-1)/2. Sin esto, la esquina
        # predicha saldria desplazada (k-1)/2 px y nadie lo notaria hasta el final.
        self.register_buffer("coords", torch.arange(self.m, dtype=torch.float32) + (k - 1) / 2)

    def forward(self, x: torch.Tensor):
        """x: (B, 1, N, N) -> logit_existe (B,), x (B,), y (B,), mapa (B, m, m)"""
        mapa = self.conv(x).squeeze(1)
        beta = self.log_beta.exp()
        plano = (beta * mapa).flatten(1)
        p = torch.softmax(plano, dim=1).view_as(mapa)
        py = p.sum(dim=2)                      # marginal en y
        px = p.sum(dim=1)                      # marginal en x
        y = (py * self.coords).sum(dim=1)
        x_ = (px * self.coords).sum(dim=1)
        lse = torch.logsumexp(plano, dim=1) / beta
        return self.a * lse + self.b, x_, y, mapa

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def construir(brazo: str, semilla: int = 1) -> EsquinaUnKernel:
    """La red de un brazo, con inicializacion REPRODUCIBLE.

    La semilla se fija aqui y no fuera: 'sin entrenar' tiene que ser el mismo
    'sin entrenar' cada vez que se dibuje la figura de las muestras."""
    if brazo not in BRAZOS:
        raise KeyError(f"brazo '{brazo}' no existe; hay {sorted(BRAZOS)}")
    torch.manual_seed(semilla)
    return EsquinaUnKernel(BRAZOS[brazo])


if __name__ == "__main__":
    print(f"{'brazo':>6} {'k':>3} {'mapa':>7} {'kernel':>7} {'cabeza':>7} {'total':>6}")
    for brazo, k in BRAZOS.items():
        red = construir(brazo)
        x = torch.randn(2, 1, VENTANA, VENTANA)
        logit, px, py, mapa = red(x)
        assert logit.shape == (2,) and px.shape == (2,) and mapa.shape == (2, red.m, red.m)
        # la coordenada leida SIEMPRE cae dentro del rango representable
        lo, hi = float(red.coords[0]), float(red.coords[-1])
        px, py = px.detach(), py.detach()
        assert lo <= float(px.min()) and float(px.max()) <= hi, "la lectura se sale del mapa"
        print(f"{brazo:>6} {k:>3} {str(red.m)+'x'+str(red.m):>7} {k*k:>7} {3:>7} {red.n_parametros():>6}")
    print(f"\nmargen ciego = (k-1)/2 px: la esquina no se puede senalar mas cerca del borde")
    print("todas construyen, la lectura cae dentro del mapa y el bias no existe.")
