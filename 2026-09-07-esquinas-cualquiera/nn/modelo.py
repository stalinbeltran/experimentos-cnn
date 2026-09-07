#!/usr/bin/env python3
"""Las ESTRUCTURAS: un solo kernel, y UNA esquina cualquiera de las dos.

    python nn/modelo.py            # comprueba las estructuras e imprime la tabla

EL PROBLEMA, Y POR QUE ES OTRO
    En `esq-2d` habia que decir CUAL de las dos esquinas era, y eso es lo que lo
    hundio. Aqui no: «Si detecta una esquina, cualquiera... el resultado es
    valido» (encargo del dueno, 2026-09-07). La salida vuelve a ser UNA:
    (existe, x, y), como en `esq-k`.

    Y quitar esa exigencia quita EXACTAMENTE la tension que fallo. Con la
    descomposicion bajo giro de 180 grados, W = S + A, y para el mismo parche:

        respuesta_tl = <S,P> + <A,P>        respuesta_br = <S,P> - <A,P>

    `esq-2d` necesitaba que las dos fueran DISTINTAS y en extremos opuestos (una
    el maximo y otra el minimo), o sea que mandara A. Aqui hace falta lo
    contrario: que las dos sean el MISMO tipo de extremo. Eso pasa cuando
    <A,P> ~ 0, es decir cuando el kernel es SIMETRICO bajo el giro -- y entonces
    las dos esquinas dan literalmente la misma respuesta.

    Ahi salen las tres estructuras, y ninguna es un capricho: son las tres formas
    de conseguir que las dos esquinas sean el mismo extremo.

      dejar que el gradiente lo encuentre solo            `libre`
      forzar S = W, o sea A = 0 por construccion          `sim`
      no tocar el kernel y doblar el SIGNO en la lectura  `abs`

    ⚠ La tercera es distinta en especie: no pide que el kernel sea simetrico,
    pide que la CABEZA no distinga el signo. Un kernel antisimetrico da +v en una
    esquina y -v en la otra, y |M| convierte las dos en un pico.

LO QUE NO CAMBIA RESPECTO DE `esq-k`, Y AHORA SI ES LITERAL
    conv 1 kernel k x k sin padding y SIN bias · entrada TINTA /255 · ventana
    32x32 · cabeza C1 de TRES parametros (beta aprendida desde 3,5, y `existe`
    de logsumexp) · sin ReLU. En `esq-2d` la cabeza tuvo que pasar a 5 porque
    habia dos cosas que detectar; aqui vuelve a haber una, asi que la red es la
    de `esq-k` con otra etiqueta. Ni un parametro de diferencia.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VENTANA = 32
BETA0 = 3.5
K_BARRIDO = (5, 7, 9, 11, 13)

# LA ESTRUCTURA QUE SE CORRE ES UNA, y lo unico que varia es `k`. Las otras dos
# quedan implementadas y comprobadas, sin armar: es la forma menos ambigua de
# anotarlas, y ponerlas en marcha es anadir su linea a `BRAZOS`. Ver
# `instrucciones/03-alternativas-anotadas.md`.
ESTRUCTURA = "libre"
ESTRUCTURAS = ("libre", "sim", "abs")


def _nombre(k: int) -> str:
    return f"k{k:02d}"


BRAZOS = {_nombre(_k): (ESTRUCTURA, _k) for _k in K_BARRIDO}


class EsquinaCualquiera(nn.Module):
    """UNA convolucion k x k sin bias, y la cabeza C1 de 3 parametros de `esq-k`.

        libre  el kernel es libre. Es la red de `esq-k` sin tocar: el gradiente
               decide si le conviene volverse simetrico.
        sim    el kernel se PROYECTA simetrico bajo giro de 180 (W = (V+rot180(V))/2)
               en cada paso. Entonces las dos esquinas dan la MISMA respuesta por
               construccion, no por suerte. Precio: (k^2+1)/2 grados de libertad.
        abs    el kernel es libre y la cabeza lee |M| en vez de M. Un pico
               negativo cuenta igual que uno positivo, asi que un kernel
               ORIENTADO puede servir para las dos esquinas sin ser simetrico.
    """

    def __init__(self, estructura: str, k: int, ventana: int = VENTANA,
                 beta0: float = BETA0):
        super().__init__()
        if (k - 1) % 2:
            raise ValueError(f"k tiene que ser impar para que el campo receptivo tenga centro; es {k}")
        if estructura not in ESTRUCTURAS:
            raise ValueError(f"estructura '{estructura}' no existe; hay {ESTRUCTURAS}")
        self.estructura, self.k, self.ventana = estructura, k, ventana
        self.m = ventana - k + 1
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        # La posicion i del mapa mira el CENTRO de su campo receptivo, que en
        # coordenadas de la ventana cae en i + (k-1)/2. Sin esto la esquina
        # predicha saldria desplazada (k-1)/2 px y nadie lo notaria hasta el final.
        self.register_buffer("coords", torch.arange(self.m, dtype=torch.float32) + (k - 1) / 2)

    def kernel(self) -> torch.Tensor:
        """El kernel EFECTIVO. En `sim` se proyecta simetrico en cada paso.

        ⚠ La proyeccion va en el forward y no en un `no_grad` despues del paso
        del optimizador: asi la parte antisimetrica de `V` no recibe gradiente
        (el modelo no depende de ella) en vez de recibirlo y ser borrada despues,
        que es la version que se ve igual y entrena otra cosa."""
        w = self.conv.weight
        if self.estructura == "sim":
            w = (w + torch.flip(w, dims=(-2, -1))) / 2
        return w

    def forward(self, x: torch.Tensor):
        """x: (B,1,N,N) -> (logit_existe, x, y, mapa)"""
        mapa = F.conv2d(x, self.kernel()).squeeze(1)
        # ⚠ `abs` lee la MAGNITUD: un pico negativo cuenta igual que uno
        # positivo. El mapa que se devuelve para dibujar sigue siendo el crudo,
        # con su signo, porque es lo que hay que mirar para entender el filtro.
        lectura = mapa.abs() if self.estructura == "abs" else mapa
        beta = self.log_beta.exp()
        plano = (beta * lectura).flatten(1)
        p = torch.softmax(plano, dim=1).view_as(mapa)
        y = (p.sum(dim=2) * self.coords).sum(dim=1)
        x_ = (p.sum(dim=1) * self.coords).sum(dim=1)
        pico = torch.logsumexp(plano, dim=1) / beta
        return self.a * pico + self.b, x_, y, mapa

    def n_kernel(self) -> int:
        """Grados de libertad REALES del kernel: en `sim` no son los guardados,
        porque la parte antisimetrica no recibe gradiente y no existe para el
        modelo. Contarla seria inflar el precio del brazo mas barato."""
        return (self.k ** 2 + 1) // 2 if self.estructura == "sim" else self.k ** 2

    def n_cabeza(self) -> int:
        return 3

    def n_parametros(self) -> int:
        return self.n_kernel() + self.n_cabeza()


def construir_suelta(estructura: str, k: int, semilla: int = 1) -> EsquinaCualquiera:
    """Una estructura CUALQUIERA, este armada o no en `BRAZOS`.

    Existe para que las alternativas anotadas se sigan comprobando: una
    estructura que se guarda "por si acaso" y deja de ejecutarse se pudre en
    silencio, y el dia que se arme habria que depurarla desde cero."""
    torch.manual_seed(semilla)
    return EsquinaCualquiera(estructura, k)


def construir(brazo: str, semilla: int = 1) -> EsquinaCualquiera:
    """La red de un brazo, con inicializacion REPRODUCIBLE.

    La semilla se fija aqui y no fuera: "sin entrenar" tiene que ser el mismo
    "sin entrenar" cada vez que se dibuje la figura de las muestras."""
    if brazo not in BRAZOS:
        raise KeyError(f"brazo '{brazo}' no existe; hay {sorted(BRAZOS)}")
    estructura, k = BRAZOS[brazo]
    return construir_suelta(estructura, k, semilla)


def simetria(w: torch.Tensor) -> tuple[float, float]:
    """Que fraccion de la ENERGIA del kernel es simetrica bajo giro de 180.

    Es LA magnitud de este experimento --la que dice si el kernel se ha vuelto
    simetrico solo-- asi que se mide en cada epoca y no a mano al final."""
    r = torch.flip(w, dims=(-2, -1))
    s, a = (w + r) / 2, (w - r) / 2
    tot = float((w ** 2).sum().detach()) or 1.0
    return float((s ** 2).sum().detach()) / tot, float((a ** 2).sum().detach()) / tot


if __name__ == "__main__":
    print(f"LOS {len(BRAZOS)} BRAZOS QUE SE CORREN (estructura `{ESTRUCTURA}`, "
          f"y lo unico que varia es k):\n")
    print(f"{'brazo':>6} {'estruct':>8} {'k':>3} {'mapa':>7} {'kernel':>7} "
          f"{'cabeza':>7} {'total':>6}   (esq-k)")
    for brazo in BRAZOS:
        red = construir(brazo)
        logit, px, py, mapa = red(torch.randn(2, 1, VENTANA, VENTANA))
        assert logit.shape == (2,) and px.shape == (2,) and mapa.shape == (2, red.m, red.m)
        lo, hi = float(red.coords[0]), float(red.coords[-1])
        px, py = px.detach(), py.detach()
        assert lo <= float(px.min()) and float(px.max()) <= hi, "la lectura se sale del mapa"
        print(f"{brazo:>6} {red.estructura:>8} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {red.n_kernel():>7} {red.n_cabeza():>7} "
              f"{red.n_parametros():>6}   {red.k**2 + 3:>6}")

    print("\nLAS ALTERNATIVAS ANOTADAS Y NO ARMADAS (ver instrucciones/03-...):\n")
    print(f"{'estruct':>8} {'k':>3} {'mapa':>7} {'kernel':>7} {'cabeza':>7} {'total':>6}")
    for _e in ("sim", "abs"):
        red = construir_suelta(_e, 7)
        red(torch.randn(2, 1, VENTANA, VENTANA))
        print(f"{_e:>8} {red.k:>3} {str(red.m)+'x'+str(red.m):>7} {red.n_kernel():>7} "
              f"{red.n_cabeza():>7} {red.n_parametros():>6}")

    # `sim` tiene que ser simetrico EXACTO: las dos esquinas dan la misma
    # respuesta por construccion. Si esto falla, el brazo no prueba lo que dice.
    red = construir_suelta("sim", 7)
    w = red.kernel()
    assert torch.allclose(w, torch.flip(w, dims=(-2, -1)), atol=1e-7)
    sim, anti = simetria(w)
    assert sim > 0.999, f"sim no es simetrico: {sim}"
    p = torch.randn(1, 1, VENTANA, VENTANA)
    m1, m2 = F.conv2d(p, red.kernel()), F.conv2d(torch.flip(p, dims=(2, 3)), red.kernel())
    assert torch.allclose(m1, torch.flip(m2, dims=(2, 3)), atol=1e-5)
    print("\nsim: el kernel es simetrico exacto y responde IGUAL a un parche y a su giro")

    # `abs` tiene que encontrar un pico NEGATIVO, que es justo lo que `libre` no
    # puede: es toda la diferencia entre las dos y por eso se comprueba.
    for est, esperado in (("abs", (4, 9)), ("libre", (15, 3))):
        red = construir_suelta(est, 7)
        mapa = torch.zeros(1, red.m, red.m)
        mapa[0, 4, 9] = -20.0                    # pico NEGATIVO, mas fuerte
        mapa[0, 15, 3] = +12.0                   # pico positivo, mas debil
        lectura = mapa.abs() if est == "abs" else mapa
        beta = red.log_beta.exp()
        pl = (beta * lectura).flatten(1)
        pr = torch.softmax(pl, 1).view_as(mapa)
        with torch.no_grad():
            yy = float((pr.sum(2) * red.coords).sum(1))
            xx = float((pr.sum(1) * red.coords).sum(1))
        off = (red.k - 1) / 2
        assert abs(xx - (esperado[1] + off)) < 0.01 and abs(yy - (esperado[0] + off)) < 0.01, \
            f"{est}: leyo ({xx:.1f},{yy:.1f})"
    print("abs: lee el pico mas fuerte AUNQUE SEA NEGATIVO; libre solo ve el positivo")
    print("todas construyen, la lectura cae dentro del mapa y ninguna conv lleva bias.")
