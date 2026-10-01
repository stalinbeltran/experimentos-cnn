#!/usr/bin/env python3
"""La red de `bor-ae`: un AUTOENCODER DISPERSO de UN solo filtro.

    python nn/modelo.py            comprueba la estructura (formas, region, norma del decodificador)

AUTONOMO: no importa nada del repo.

LA ESTRUCTURA
=============
    z  = ReLU(conv_k(x))                 el CODIFICADOR: sin bias, sin padding. ESTE es el
                                         kernel que va al banco (`conv.weight`, 1x1xkxk)
    x^ = convT_k(z, d / |d|) + b         el DECODIFICADOR, con su filtro de NORMA 1
    perdida = MSE(x^, x) en la region R  +  lambda * media(z) en la region O

POR QUE EL DECODIFICADOR TIENE NORMA 1 (y no es un detalle)
    Sin esa restriccion, lambda no significa nada: escalar el codificador por s y el
    decodificador por 1/s deja la reconstruccion igual y divide |z| por s, asi que con
    s -> 0 la penalizacion desaparece para CUALQUIER lambda. Es la degeneracion conocida
    del autoencoder disperso, y con norma 1 en el diccionario la escala la tiene que
    pagar el codigo.

LAS DOS REGIONES, IGUALES PARA TODO k (en px de la ventana de 55)
    R = [18, 36] (19 x 19): donde se mide la reconstruccion. Es la region donde el campo
        receptivo ENTERO de codificador + decodificador (k - 1 <= 18 px a cada lado) cae
        dentro de la ventana para todo k <= 19. Fuera de ahi un k grande reconstruiria
        con menos soporte que uno pequeno, y la comparacion mediria eso.
    O = [9, 45] (37 x 37): donde se cuenta la dispersion. Es la SALIDA de `bor-k`: las
        posiciones de codigo que alimentan R.

EL RIESGO, escrito antes de entrenar
    Con lambda = 0 el optimo es la IDENTIDAD: codificador y decodificador deltas
    reconstruyen perfecto (R^2 = 1). Lo midio la sonda L1 de foveal-vision en 2026-09-03.
    Con lambda > 0 la delta encogida sigue siendo candidata. Por eso lambda NO se fija a
    ojo: se elige con el tanteo de `nn/tanteo_lambda.py` y la regla escrita ANTES en
    `instrucciones/02-criterio.md`, que mide lo DELTA que es el kernel (`delta()`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LADO = 55
R = (18, 37)        # [18, 36]: reconstruccion
O = (9, 46)         # [9, 45]: dispersion
K_BARRIDO = (3, 5, 7, 9, 11, 13, 15, 17, 19)
SEMILLAS = (1, 2, 3)


def nombre(k: int, semilla: int) -> str:
    return f"k{k:02d}-s{semilla}"


BRAZOS = {nombre(k, s): (k, s) for k in K_BARRIDO for s in SEMILLAS}


class AutoencoderUnFiltro(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        if k % 2 == 0 or not 3 <= k <= 19:
            raise ValueError(f"k tiene que ser impar y estar en [3, 19]; es {k}")
        self.k = k
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.dec = nn.Parameter(torch.randn(1, 1, k, k) / k)
        self.b = nn.Parameter(torch.zeros(()))

    def kernel(self) -> torch.Tensor:
        return self.conv.weight[0, 0]

    def decodificador(self) -> torch.Tensor:
        return self.dec / self.dec.norm()

    def forward(self, x: torch.Tensor):
        """x (N,1,55,55) -> (x^ (N,1,55,55), z (N,1,55-k+1,55-k+1))."""
        z = F.relu(F.conv2d(x, self.conv.weight))
        xh = F.conv_transpose2d(z, self.decodificador()) + self.b
        return xh, z

    def z_en_O(self, z: torch.Tensor) -> torch.Tensor:
        """El codigo en las posiciones que miran a O, que son las mismas para todo k."""
        r = O[0] - (self.k - 1) // 2
        lado = O[1] - O[0]
        return z[:, :, r:r + lado, r:r + lado]

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def construir(k: int, semilla: int) -> AutoencoderUnFiltro:
    torch.manual_seed(semilla)
    return AutoencoderUnFiltro(k)


def delta(w: torch.Tensor) -> float:
    """Que fraccion de la energia del kernel esta en su pixel mas fuerte. 1 = una delta
    (el kernel no hace nada: es la condicion `identidad` del banco); 1/k^2 = repartido."""
    e = (w.detach().float() ** 2).flatten()
    return float(e.max() / e.sum()) if e.sum() > 0 else float("nan")


def _comprobar() -> int:
    fallos = 0
    x = torch.rand(2, 1, LADO, LADO)
    for k in (3, 9, 19):
        red = construir(k, 1)
        xh, z = red(x)
        zo = red.z_en_O(z)
        bien = (xh.shape == x.shape and z.shape[-1] == LADO - k + 1
                and zo.shape[-1] == O[1] - O[0]
                and abs(float(red.decodificador().detach().norm()) - 1) < 1e-6)
        # con codificador y decodificador deltas, la reconstruccion en R es EXACTA:
        # es la identidad que el criterio tiene que poder detectar
        with torch.no_grad():
            red.conv.weight.zero_(); red.conv.weight[0, 0, k // 2, k // 2] = 1.0
            red.dec.zero_(); red.dec[0, 0, k // 2, k // 2] = 1.0
        xh, _ = red(x)
        ident = float((xh - x)[..., R[0]:R[1], R[0]:R[1]].abs().max())
        bien &= ident < 1e-5 and delta(red.kernel()) == 1.0
        fallos += not bien
        print(f"  k={k:>2}: x^ {tuple(xh.shape)} · z {tuple(z.shape)} · z en O {tuple(zo.shape)} · "
              f"identidad exacta en R ({ident:.1e}) · delta={delta(red.kernel()):.2f} "
              f"{'ok' if bien else 'FALLA'}")
    print(f"\n{'la estructura cumple' if not fallos else str(fallos) + ' FALLO(S)'}")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_comprobar())
