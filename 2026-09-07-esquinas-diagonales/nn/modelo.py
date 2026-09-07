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

    ⚠ HUBO UNA TERCERA --aplicar el mismo kernel a la entrada GIRADA 180 grados--
    y el dueno la DESCARTO el 2026-09-07: «No queremos girar el kernel». No se
    implementa, a proposito: una alternativa descartada que sigue en el codigo se
    acaba armando por error. Queda anotada en
    `instrucciones/03-alternativas-anotadas.md`, que es donde se mira antes de
    volver a proponerla.

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

EL DISENO ES DE UN SOLO EJE: EL KERNEL
    5 brazos, k en {5, 7, 9, 11, 13}, todos con la MISMA estructura. Por orden
    del dueno (2026-09-07): «Debe ser identica al exper anterior, solo que en vez
    de 1 esquina van a ser 2».

    Y lo es, salvo en un punto que no puede serlo y hay que decir cual: la cabeza
    pasa de 3 parametros a 5. La conv es la misma, la lectura C1 es la misma, y
    lo unico que se duplica es el `existe` (a, b), porque ahora hay DOS cosas que
    detectar. La beta sigue siendo UNA, compartida.

    ⚠ Y la posicion de la segunda esquina sale del MINIMO del mapa porque no hay
    otro sitio de donde sacarla. Con una sola conv y una cabeza minima, las
    unicas dos lecturas distintas de un mapa son su maximo y su minimo:
    cualquier otra pediria pesos POR POSICION (m^2 de ellos), y eso rompe la
    restriccion que el dueno puso en `esq-k` -- «si la cabeza es grande, el
    kernel no aprende nada». No es una eleccion entre varias: es la unica que
    cabe.

    Lo que cuesta, y hay que decirlo: si el barrido sale mal, este diseno NO
    puede distinguir "un kernel no da para las dos esquinas" de "esta lectura no
    es la buena". Esa pregunta es la que contestarian las alternativas anotadas.
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
ESQUINAS = ("tl", "br")

# LA ESTRUCTURA QUE SE CORRE ES UNA, Y LO UNICO QUE VARIA ES `k`.
#
# Por orden del dueno (2026-09-07): «Anotalas pero no vamos a ejecutarlas. Solo
# variamos los kernels». Las otras tres --`sig`, `ant` y el control `ind`-- SIGUEN
# IMPLEMENTADAS aqui abajo a proposito: una estructura implementada es la forma
# menos ambigua de anotarla, y ponerla en marcha es anadir su linea a `BRAZOS`.
# Estan descritas, con lo que cada una contestaria y lo que cuesta, en
# `instrucciones/03-alternativas-anotadas.md`.
#
ESTRUCTURA = "sig"
ESTRUCTURAS = ("sig", "ant")             # implementadas; solo se corre ESTRUCTURA

# ⚠ LOS BRAZOS SE LLAMAN COMO LOS DE `esq-k` -- k05, k07... -- y no `sig-k05`.
# Con una sola estructura, el nombre solo tiene que decir lo unico que varia; y
# asi las dos tablas de resultados se leen una al lado de la otra sin traducir.
def _nombre(k: int, esquina: str | None = None) -> str:
    return f"k{k:02d}{'-' + esquina if esquina else ''}"


# nombre -> (estructura, k, esquina). `esquina` solo lo usa el control `ind`,
# que entrena UNA red por esquina y por tanto no comparte nada.
BRAZOS = {_nombre(_k): (ESTRUCTURA, _k, None) for _k in K_BARRIDO}


class DosEsquinasUnKernel(nn.Module):
    """UNA convolucion k x k sin bias, y una cabeza de 3 a 5 parametros.

    `estructura` decide como se leen DOS esquinas de un solo mapa:

        sig   LA QUE SE CORRE. Un mapa; tl = maximo, br = minimo. Una beta
              compartida y un `existe` por esquina: 5 parametros de cabeza,
              contra los 3 de `esq-k`, que tenia una sola esquina que detectar.
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
        if estructura not in ("sig", "ant", "ind"):
            raise ValueError(f"estructura '{estructura}' no existe")
        self.estructura, self.k, self.ventana = estructura, k, ventana
        self.m = ventana - k + 1
        self.esquinas = (esquina,) if estructura == "ind" else ESQUINAS
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        # El control `ind` solo detecta UNA esquina; las demas necesitan un
        # `existe` por esquina. La beta es siempre una, compartida.
        n_cabezas = 1 if estructura == "ind" else 2
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
        # sig / ant: UN solo mapa, y las dos esquinas en sus dos extremos.
        return {"tl": self._leer(mapa, +1.0, 0),
                "br": self._leer(mapa, -1.0, 1)}, mapa

    def n_parametros(self) -> int:
        """Los GRADOS DE LIBERTAD reales, que en `ant` no son los tensores
        guardados: la parte simetrica de V no recibe gradiente y no existe para
        el modelo. Contarla seria inflar el precio del brazo mas barato."""
        return self.n_kernel() + self.n_cabeza()

    def n_kernel(self) -> int:
        return (self.k ** 2 - 1) // 2 if self.estructura == "ant" else self.k ** 2

    def n_cabeza(self) -> int:
        return 1 + 2 * self.a.numel()


def construir_suelta(estructura: str, k: int, esquina: str | None = None,
                     semilla: int = 1) -> DosEsquinasUnKernel:
    """Una estructura CUALQUIERA, este armada o no en `BRAZOS`.

    Existe para que las alternativas anotadas se sigan comprobando: una
    estructura que se guarda "por si acaso" y deja de ejecutarse nunca se pudre
    en silencio, y el dia que se arme habria que depurarla desde cero."""
    torch.manual_seed(semilla)
    return DosEsquinasUnKernel(estructura, k, esquina)


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
    print(f"LOS {len(BRAZOS)} BRAZOS QUE SE CORREN (estructura `{ESTRUCTURA}`, "
          f"y lo unico que varia es k):\n")
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
            # la lectura SIEMPRE cae dentro del rango representable
            assert lo <= float(px.min()) and float(px.max()) <= hi, \
                f"{brazo}/{esq}: la lectura se sale del mapa"
        assert set(salida) == set(red.esquinas)
        print(f"{brazo:>11} {red.estructura:>8} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {'+'.join(red.esquinas):>10} "
              f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    print("\nLAS ALTERNATIVAS ANOTADAS Y NO ARMADAS (ver instrucciones/03-...):\n")
    print(f"{'estruct':>11} {'k':>3} {'mapa':>7} {'esquinas':>10} "
          f"{'kernel':>7} {'cabeza':>7} {'total':>6}")
    for _e, _esq in (("ant", None), ("ind", "tl"), ("ind", "br")):
        red = construir_suelta(_e, 7, _esq)
        salida, mapa = red(torch.randn(2, 1, VENTANA, VENTANA))
        assert set(salida) == set(red.esquinas)
        print(f"{_e + ('-' + _esq if _esq else ''):>11} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {'+'.join(red.esquinas):>10} "
              f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    # `ant` tiene que ser antisimetrico EXACTO, y por tanto responder a un
    # parche y a su giro con el mismo numero cambiado de signo. Si esto falla,
    # el brazo no prueba la hipotesis que dice probar.
    red = construir_suelta("ant", 7)
    w = red.kernel()
    assert torch.allclose(w, -torch.flip(w, dims=(-2, -1)), atol=1e-7)
    sim, anti = simetria(w)
    assert anti > 0.999, f"ant no es antisimetrico: {anti}"
    p = torch.randn(1, 1, VENTANA, VENTANA)
    m1, m2 = red._conv(p), red._conv(torch.flip(p, dims=(2, 3)))
    assert torch.allclose(m1, -torch.flip(m2, dims=(1, 2)), atol=1e-5)
    print("\nant: el kernel es antisimetrico exacto y responde al giro con el signo cambiado")

    # La lectura de `br` tiene que ser LA DEL MINIMO, no otra cosa: sobre un mapa
    # con un pico negativo claro, la esquina leida cae donde esta ese pico. Sin
    # esto, "br = minimo" seria una intencion escrita en un comentario.
    red = construir(_nombre(7))
    mapa = torch.full((1, red.m, red.m), 0.0)
    mapa[0, 4, 9] = -20.0                       # un unico pico NEGATIVO
    mapa[0, 15, 3] = +20.0                      # y uno positivo en otro sitio
    with torch.no_grad():
        _, bx, by = red._leer(mapa, -1.0, 1)
        _, tx, ty = red._leer(mapa, +1.0, 0)
    off = (red.k - 1) / 2
    assert abs(float(bx) - (9 + off)) < 0.01 and abs(float(by) - (4 + off)) < 0.01
    assert abs(float(tx) - (3 + off)) < 0.01 and abs(float(ty) - (15 + off)) < 0.01
    print("\nsig: `tl` lee el MAXIMO del mapa y `br` el MINIMO, cada uno en su sitio")
    sim, anti = simetria(red.kernel())
    print(f"kernel sin entrenar: {100*sim:.1f}% simetrico · {100*anti:.1f}% antisimetrico")
    print("todas construyen, la lectura cae dentro del mapa y ninguna conv lleva bias.")
