#!/usr/bin/env python3
"""Las ESTRUCTURAS: un solo kernel por red, y DOS esquinas en diagonal.

    python nn/modelo.py            # comprueba las estructuras e imprime la tabla

EL PROBLEMA, EN UNA LINEA
    Un filtro lineal que pica en una esquina superior-izquierda tiene forma de
    CUADRANTE -- positivo donde espera tinta, negativo donde espera fondo -- y esa
    forma es ORIENTADA. La esquina inferior-derecha es esa misma forma girada 180
    grados. Con UN solo kernel, las dos esquinas no son dos problemas: son el
    mismo problema visto al reves, y de ahi salen las estructuras de abajo.

LA DESCOMPOSICION QUE LAS ORDENA, Y QUE DECIDE CUAL PUEDE FUNCIONAR
    Todo kernel W se parte bajo giro de 180 grados en su parte simetrica y su
    antisimetrica:

        W = S + A       S = (W + rot180(W))/2       A = (W - rot180(W))/2

    Si el parche de una esquina br es el giro del de una tl (y lo es, salvo el
    texto concreto que caiga dentro), entonces para el MISMO parche P:

        respuesta_tl = <S,P> + <A,P>
        respuesta_br = <S,P> - <A,P>

    Las dos respuestas son SIMETRICAS respecto de <S,P>, y toda la diferencia
    entre esquinas vive en A. De ahi las tres lecturas que se prueban:

      S ~ 0, manda A   ->  tl es el MAXIMO del mapa y br el MINIMO        `sig`
      se FUERZA A = W  ->  lo mismo, pero con S = 0 por construccion: la
                           igualdad respuesta_br = -respuesta_tl deja de ser
                           una esperanza y pasa a ser exacta               `ant`
      ninguna de las   ->  no se comparte el MAPA sino la VISTA: el mismo
      dos              ->  kernel sobre la entrada GIRADA 180 grados es,
                           literalmente, un detector de br                 `rot`

    ⚠ Y UNA CUARTA QUE SE DESCARTA EN PAPEL, no midiendo: "las dos esquinas son
    maximos y se distinguen por el VALOR" (si manda S). No es expresable con
    esta cabeza. `existe` es una funcion LINEAL de un solo escalar (el pico), o
    sea MONOTONA; y esa hipotesis pide justo lo contrario -- fondo 0, br medio,
    tl alto -> "br" seria una BANDA [t1, t2], que ninguna recta separa. Habria
    que darle dos escalares (el maximo y el minimo), y eso ya es `sig`. Pagar
    dos minutos por demostrar un argumento de papel es lo unico que un tanteo no
    debe hacer.

    ⚠ MEDIDO el 2026-09-07 sobre el kernel ganador de `esq-k`, que solo vio
    esquinas tl: el 74,0 % de su energia esta en S y el 26,0 % en A, y su suma
    vale -73,68 -- o sea que es sobre todo un SUPRESOR DE TINTA. Consecuencia
    directa, medida sobre las 10 paginas enteras de aquel experimento: su MINIMO
    cae en la mancha de tinta, NO en la esquina br (0/10, mediana 91 px). Asi que
    `sig` no sale gratis de lo ya entrenado: pide un kernel que gaste menos en S,
    y eso es justo lo que este experimento mide. El comando esta en el encargo.

LO QUE NO CAMBIA RESPECTO DE `esq-k`, Y POR QUE
    conv 1 kernel k x k, stride 1, SIN padding, SIN bias · entrada TINTA /255 ·
    ventana 32x32 · lectura por esperanza bajo softmax(beta*M) con beta aprendida
    (BETA0 = 3,5) · sin ReLU entre la conv y la cabeza. Los porques estan medidos
    en `esq-k` y no se vuelven a pagar aqui.

EL DISENO ES DE DOS EJES, y eso hay que decirlo porque cambia como se lee
    ESTRUCTURA (rot · sig · ant, mas el control ind) x KERNEL (5 · 7 · 9 · 11 ·
    13). Son 25 brazos. El eje que contesta la PREGUNTA es el de la estructura;
    el del kernel esta para que "esta lectura no puede" no se confunda nunca con
    "con este kernel no cabe" -- que es la duda que en `esq-k` dejo abierta el
    borde del rango.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VENTANA = 32
BETA0 = 3.5

# EL EJE DEL KERNEL, por orden del dueno (2026-09-07): los mismos de `esq-k`
# MENOS el 3x3, y uno mas grande en su lugar.
#
# ⚠ El 3 se cae con un dato, no por gusto: alli se quedo en el 8,3 % de acierto,
# que es EXACTAMENTE su suelo sin entrenar. No aprendio poco: no aprendio. Con 12
# parametros y 3 px de campo receptivo no hay esquina que ver, y aqui la tarea es
# mas dificil (dos esquinas), asi que repetirlo seria pagar por re-confirmar al
# perdedor. El 13 entra en su sitio porque el eje NO estaba acotado por arriba:
# el 11 seguia mejorando y era el borde del rango.
#
# ⚠ Y 13 es el ultimo k que este dataset admite SIN regenerarlo: las esquinas se
# sortean con el punto entre los pixeles 8 y 23 de la ventana, y el mapa de un
# kernel k solo representa de (k-1)/2 a 31-(k-1)/2. Con k = 17 los dos rangos
# coinciden exactamente; con k = 19 ya habria esquinas que el mapa no puede
# senalar. Cabe 13, cabe 15, y a partir de 19 hay que rehacer el dato.
K_BARRIDO = (5, 7, 9, 11, 13)
ESTRUCTURAS = ("rot", "sig", "ant")
ESQUINAS = ("tl", "br")


def _nombre(estructura: str, k: int, esquina: str | None = None) -> str:
    return f"{estructura}{'-' + esquina if esquina else ''}-k{k:02d}"


# nombre -> (estructura, k, esquina). `esquina` solo lo usa el control `ind`,
# que entrena UNA red por esquina y por tanto no comparte nada.
BRAZOS = {}
for _e in ESTRUCTURAS:
    for _k in K_BARRIDO:
        BRAZOS[_nombre(_e, _k)] = (_e, _k, None)
for _c in ESQUINAS:
    for _k in K_BARRIDO:
        BRAZOS[_nombre("ind", _k, _c)] = ("ind", _k, _c)


class DosEsquinasUnKernel(nn.Module):
    """UNA convolucion k x k sin bias, y una cabeza de 3 a 5 parametros.

    `estructura` decide como se leen DOS esquinas de un solo mapa:

        rot   el mismo kernel sobre x y sobre girar180(x). La cabeza tambien es
              la misma para las dos, porque por simetria son la MISMA tarea:
              3 parametros de cabeza, los mismos que en `esq-k` con una esquina.
        sig   un mapa; tl = maximo, br = minimo. Dos `existe` (5 parametros).
        ant   igual que `sig`, pero el kernel se PROYECTA antisimetrico en cada
              paso: W = (V - rot180(V))/2. Asi S = 0 por construccion y
              respuesta_br = -respuesta_tl es exacta, no una esperanza. El
              precio es la mitad de grados de libertad -- (k^2-1)/2 -- y perder
              la parte del kernel que apaga el INTERIOR del parrafo, que es
              justo lo que el kernel de `esq-k` gastaba el 74% de su energia en
              hacer. Es la hipotesis `sig` llevada a su extremo, y sirve para
              separar "esta lectura no funciona" de "el optimizador no llega".
        ind   una sola esquina, como en `esq-k`. Es el CONTROL: dos redes
              independientes, sin compartir kernel. No compite: es el techo.
    """

    def __init__(self, estructura: str, k: int, esquina: str | None = None,
                 ventana: int = VENTANA, beta0: float = BETA0):
        super().__init__()
        if (k - 1) % 2:
            raise ValueError(f"k tiene que ser impar para que el campo receptivo tenga centro; es {k}")
        if estructura not in ("rot", "sig", "ant", "ind"):
            raise ValueError(f"estructura '{estructura}' no existe")
        self.estructura, self.k, self.ventana = estructura, k, ventana
        self.m = ventana - k + 1
        self.esquinas = (esquina,) if estructura == "ind" else ESQUINAS
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        # `rot` comparte la cabeza entera entre las dos esquinas (son la misma
        # tarea girada); las demas necesitan un `existe` por esquina.
        n_cabezas = 1 if estructura in ("rot", "ind") else 2
        self.a = nn.Parameter(torch.ones(n_cabezas))
        self.b = nn.Parameter(torch.zeros(n_cabezas))
        # La posicion i del mapa mira el CENTRO de su campo receptivo, que en
        # coordenadas de la ventana cae en i + (k-1)/2. Sin esto la esquina
        # predicha saldria desplazada (k-1)/2 px y nadie lo notaria hasta el final.
        self.register_buffer("coords", torch.arange(self.m, dtype=torch.float32) + (k - 1) / 2)

    def kernel(self) -> torch.Tensor:
        """El kernel EFECTIVO. En `ant` se proyecta antisimetrico en cada paso.

        ⚠ La proyeccion va en el forward y no en un `no_grad` despues del paso
        del optimizador: asi la parte simetrica de `V` no recibe gradiente (el
        modelo no depende de ella) en vez de recibirlo y ser borrada despues,
        que es la version que se ve igual y entrena otra cosa."""
        w = self.conv.weight
        if self.estructura == "ant":
            w = (w - torch.flip(w, dims=(-2, -1))) / 2
        return w

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel()).squeeze(1)

    # --- la lectura C1, con signo ------------------------------------------
    def _leer(self, mapa: torch.Tensor, signo: float, cabeza: int):
        """(logit_existe, x, y) leyendo el MAXIMO (signo +1) o el MINIMO (-1)."""
        beta = self.log_beta.exp()
        plano = (signo * beta * mapa).flatten(1)
        p = torch.softmax(plano, dim=1).view_as(mapa)
        y = (p.sum(dim=2) * self.coords).sum(dim=1)
        x = (p.sum(dim=1) * self.coords).sum(dim=1)
        # ~ max(signo*M): el signo lo absorbe `a`, que se aprende
        pico = torch.logsumexp(plano, dim=1) / beta
        return self.a[cabeza] * pico + self.b[cabeza], x, y

    def forward(self, x: torch.Tensor):
        """x: (B,1,N,N) -> ({esquina: (logit, x, y)}, mapa)"""
        mapa = self._conv(x)
        if self.estructura == "ind":
            return {self.esquinas[0]: self._leer(mapa, +1.0, 0)}, mapa
        if self.estructura in ("sig", "ant"):
            return {"tl": self._leer(mapa, +1.0, 0),
                    "br": self._leer(mapa, -1.0, 1)}, mapa
        # rot: el MISMO kernel sobre la entrada girada 180 grados. Una esquina br
        # de la entrada es una esquina tl de la vista girada, asi que la cabeza
        # es la misma. Las coordenadas se devuelven al marco original.
        mapa_g = self._conv(torch.flip(x, dims=(2, 3)))
        logit_tl, px, py = self._leer(mapa, +1.0, 0)
        logit_br, gx, gy = self._leer(mapa_g, +1.0, 0)
        borde = self.ventana - 1
        return {"tl": (logit_tl, px, py), "br": (logit_br, borde - gx, borde - gy)}, mapa

    def n_parametros(self) -> int:
        """Los GRADOS DE LIBERTAD reales, que en `ant` no son los tensores
        guardados: la parte simetrica de V no recibe gradiente y no existe para
        el modelo. Contarla seria inflar el precio del brazo mas barato."""
        return self.n_kernel() + self.n_cabeza()

    def n_kernel(self) -> int:
        return (self.k ** 2 - 1) // 2 if self.estructura == "ant" else self.k ** 2

    def n_cabeza(self) -> int:
        return 1 + 2 * self.a.numel()


def construir(brazo: str, semilla: int = 1) -> DosEsquinasUnKernel:
    """La red de un brazo, con inicializacion REPRODUCIBLE.

    La semilla se fija aqui y no fuera: "sin entrenar" tiene que ser el mismo
    "sin entrenar" cada vez que se dibuje la figura de las muestras."""
    if brazo not in BRAZOS:
        raise KeyError(f"brazo '{brazo}' no existe; hay {sorted(BRAZOS)}")
    estructura, k, esquina = BRAZOS[brazo]
    torch.manual_seed(semilla)
    return DosEsquinasUnKernel(estructura, k, esquina)


def simetria(w: torch.Tensor) -> tuple[float, float]:
    """Que fraccion de la ENERGIA del kernel es simetrica bajo giro de 180.

    Es la magnitud que ordena las estructuras (ver cabecera), asi que se mide
    sola en cada epoca y no a mano al final."""
    r = torch.flip(w, dims=(-2, -1))
    s, a = (w + r) / 2, (w - r) / 2
    tot = float((w ** 2).sum().detach()) or 1.0
    return float((s ** 2).sum().detach()) / tot, float((a ** 2).sum().detach()) / tot


if __name__ == "__main__":
    print(f"{'brazo':>11} {'estruct':>8} {'k':>3} {'mapa':>7} {'esquinas':>10} "
          f"{'kernel':>7} {'cabeza':>7} {'total':>6}")
    for brazo in BRAZOS:
        red = construir(brazo)
        x = torch.randn(2, 1, VENTANA, VENTANA)
        salida, mapa = red(x)
        assert mapa.shape == (2, red.m, red.m)
        lo, hi = float(red.coords[0]), float(red.coords[-1])
        for esq, (logit, px, py) in salida.items():
            assert logit.shape == (2,) and px.shape == (2,)
            px, py = px.detach(), py.detach()
            # la lectura SIEMPRE cae dentro del rango representable, tambien
            # despues de deshacer el giro de `rot`
            assert lo <= float(px.min()) and float(px.max()) <= hi, \
                f"{brazo}/{esq}: la lectura se sale del mapa"
        assert set(salida) == set(red.esquinas)
        print(f"{brazo:>11} {red.estructura:>8} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {'+'.join(red.esquinas):>10} "
              f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    # `rot` tiene que ser EXACTAMENTE equivariante: girar la entrada 180 grados
    # intercambia las dos esquinas predichas. Si esto falla, el brazo no es lo
    # que dice ser, y se veria como "aprende peor" en vez de como un fallo.
    # `ant` tiene que ser antisimetrico EXACTO, y por tanto responder a un
    # parche y a su giro con el mismo numero cambiado de signo. Si esto falla,
    # el brazo no prueba la hipotesis que dice probar.
    red = construir(_nombre("ant", 7))
    w = red.kernel()
    assert torch.allclose(w, -torch.flip(w, dims=(-2, -1)), atol=1e-7)
    sim, anti = simetria(w)
    assert anti > 0.999, f"ant no es antisimetrico: {anti}"
    p = torch.randn(1, 1, VENTANA, VENTANA)
    m1, m2 = red._conv(p), red._conv(torch.flip(p, dims=(2, 3)))
    assert torch.allclose(m1, -torch.flip(m2, dims=(1, 2)), atol=1e-5)
    print("\nant: el kernel es antisimetrico exacto y responde al giro con el signo cambiado")

    red = construir(_nombre("rot", 7))
    x = torch.randn(3, 1, VENTANA, VENTANA)
    s1, _ = red(x)
    s2, _ = red(torch.flip(x, dims=(2, 3)))
    borde = VENTANA - 1
    assert torch.allclose(s1["tl"][0], s2["br"][0], atol=1e-5)
    assert torch.allclose(s1["tl"][1], borde - s2["br"][1], atol=1e-4)
    assert torch.allclose(s1["br"][2], borde - s2["tl"][2], atol=1e-4)
    print("rot: girar la entrada 180 grados intercambia las dos esquinas (equivariante)")
    sim, anti = simetria(red.kernel())
    print(f"kernel sin entrenar: {100*sim:.1f}% simetrico · {100*anti:.1f}% antisimetrico")
    print("todas construyen, la lectura cae dentro del mapa y ninguna conv lleva bias.")
