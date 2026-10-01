#!/usr/bin/env python3
"""La red de `bor-k`: UN kernel k x k, y una cabeza de 9 parametros que lee CUATRO bordes.

    python nn/modelo.py            comprueba la estructura sobre bordes sinteticos

AUTONOMO: no importa nada del repo. Es lo que garantiza poder cargar los pesos dentro de
un ano (regla 1 de `CLAUDE.md` § «La forma de un experimento»).

LA ESTRUCTURA
=============
    M = conv_k(x)            sin bias, sin padding; recortada a 37 x 37 (la SALIDA, igual
                             para todo k: ver `nn/datos.py`)
    c = media de M por filas     -> un perfil a lo largo de x (37)
    r = media de M por columnas  -> un perfil a lo largo de y (37)

    izq = softargmax(+beta * c)      der = softargmax(-beta * c)
    sup = softargmax(+beta * r)      inf = softargmax(-beta * r)

    existe_t = a_t * logsumexp(+-beta * perfil) / beta + b_t     (un logit por borde)

    Cabeza: beta + 4 a + 4 b = 9 parametros. Todo lo demas es el kernel.

POR QUE ASI
    Es la lectura de `esq-2d` (maximo y minimo de UN mapa) llevada a los dos ejes. Un
    filtro lineal no puede tener cuatro orientaciones a la vez; lo que si puede es
    responder con SIGNO OPUESTO en bordes enfrentados -- una derivada responde + al
    entrar en la tinta y - al salir --, y eso es justo lo que el maximo y el minimo de un
    perfil leen. El mismo kernel sirve a los dos ejes porque una derivada diagonal (o la
    suma de una horizontal y una vertical) responde en los dos.

    El riesgo, escrito antes de entrenar: `esq-2d` midio que un kernel se queda con la
    esquina barata (tl 90 %, br 23 %). Aqui puede quedarse con UN EJE. Por eso se reporta
    por borde, nunca el promedio.

LA COORDENADA
    La posicion i del perfil mira el CENTRO de su campo receptivo, que en px de la ventana
    es i + 9 (la SALIDA empieza en 9 para todo k). Sin ese desplazamiento el borde
    predicho saldria corrido y nadie lo notaria hasta el final.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LADO = 55
DESCARTE = 9
SALIDA = LADO - 2 * DESCARTE        # 37
BETA0 = 3.5
K_BARRIDO = (3, 5, 7, 9, 11, 13, 15, 17, 19)
SEMILLAS = (1, 2, 3)
BORDES = ("izq", "der", "sup", "inf")


def nombre(k: int, semilla: int) -> str:
    return f"k{k:02d}-s{semilla}"


BRAZOS = {nombre(k, s): (k, s) for k in K_BARRIDO for s in SEMILLAS}


class UnKernelCuatroBordes(nn.Module):
    def __init__(self, k: int, beta0: float = BETA0):
        super().__init__()
        if k % 2 == 0 or not 3 <= k <= 19:
            raise ValueError(f"k tiene que ser impar y estar en [3, 19]; es {k}")
        self.k = k
        self.recorte = DESCARTE - (k - 1) // 2
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        self.a = nn.Parameter(torch.ones(4))
        self.b = nn.Parameter(torch.zeros(4))
        self.register_buffer("coords", torch.arange(SALIDA, dtype=torch.float32) + DESCARTE)

    def kernel(self) -> torch.Tensor:
        return self.conv.weight[0, 0]

    def mapa(self, x: torch.Tensor) -> torch.Tensor:
        m = F.conv2d(x, self.conv.weight)[:, 0]
        r = self.recorte
        return m[:, r:r + SALIDA, r:r + SALIDA]

    def forward(self, x: torch.Tensor):
        """x (N, 1, 55, 55) -> (logits (N, 4), coords (N, 4)) en el orden de BORDES."""
        m = self.mapa(x)
        perfil_x = m.mean(dim=1)              # (N, 37): a lo largo de x
        perfil_y = m.mean(dim=2)              # (N, 37): a lo largo de y
        beta = self.log_beta.exp()
        logits, coords = [], []
        for i, (perfil, signo) in enumerate(((perfil_x, 1.0), (perfil_x, -1.0),
                                             (perfil_y, 1.0), (perfil_y, -1.0))):
            z = signo * beta * perfil
            coords.append((torch.softmax(z, dim=1) * self.coords).sum(dim=1))
            logits.append(self.a[i] * torch.logsumexp(z, dim=1) / beta + self.b[i])
        return torch.stack(logits, dim=1), torch.stack(coords, dim=1)

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def construir(k: int, semilla: int) -> UnKernelCuatroBordes:
    torch.manual_seed(semilla)
    return UnKernelCuatroBordes(k)


def _comprobar() -> int:
    """Con un kernel de derivada PUESTO A MANO, la cabeza tiene que encontrar los cuatro
    bordes de un rectangulo de tinta. Si esto falla, ningun entrenamiento significa nada."""
    fallos = 0
    for k in (3, 9, 19):
        red = UnKernelCuatroBordes(k, beta0=50.0)
        w = torch.zeros(k, k)
        c = k // 2
        # derivada horizontal + vertical: + al entrar en la tinta, - al salir.
        # (conv2d de torch es CORRELACION: el peso de la derecha mira a la derecha)
        w[c, c + 1:] = 1.0; w[c, :c] = -1.0
        w[c + 1:, c] += 1.0; w[:c, c] += -1.0
        with torch.no_grad():
            red.conv.weight.copy_(w.view(1, 1, k, k))
        x = torch.zeros(1, 1, LADO, LADO)
        x[0, 0, 15:38, 20:40] = 1.0                    # tinta: filas 15..37, columnas 20..39
        _, coords = red(x)
        esperado = torch.tensor([19.5, 39.5, 14.5, 37.5])   # el borde cae ENTRE dos px
        err = (coords[0] - esperado).abs().max().item()
        bien = err < 1.01
        fallos += not bien
        print(f"  k={k:>2}: izq/der/sup/inf = {[round(v, 1) for v in coords[0].tolist()]} "
              f"(esperado {esperado.tolist()}) · error max {err:.2f} px · "
              f"{red.n_parametros() - k * k} parametros de cabeza "
              f"{'ok' if bien else 'FALLA'}")
    print(f"\n{'la estructura lee los cuatro bordes' if not fallos else str(fallos) + ' FALLO(S)'}")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(_comprobar())
