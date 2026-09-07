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
# ⚠ EL EJE LLEGA HASTA 17, Y ESO ES UNA DECISION CON UNA MEDIDA DETRAS.
# Las esquinas se sortean con el punto entre los pixeles 8 y 23 de la ventana, y
# el mapa de un kernel k solo representa de (k-1)/2 a 31-(k-1)/2. Con k = 17 los
# dos rangos coinciden EXACTAMENTE (8..23); con k = 19 ya habria esquinas que el
# mapa no puede senalar. O sea que 17 es el techo del dataset, no una redondez.
#
# ⚠⚠ Y por que se extiende de 13 a 17, que en `esq-2d` no se hizo: medido el
# 2026-09-07, la tinta mas cercana a `br` esta a 8,1 px de mediana, y el RADIO
# del campo receptivo es (k-1)/2 -- o sea 2/3/4/5/6 px para k = 5..13. NINGUN
# brazo del rango viejo puede VER la tinta de `br`. El primero que llega es
# k = 17 (radio 8). Con {5..13} el resultado en `br` estaba escrito de antemano.
K_BARRIDO = (5, 7, 9, 11, 13, 15, 17)

# ⚠ ESQUINAS YA NO SON "LAS SALIDAS": son de cuales se DERIVA el unico positivo.
# En `esq-2d` habia una salida por esquina; aqui hay UNA, y `tl`/`br` solo dicen
# de donde sale la etiqueta (ver `datos.objetivo`). El desglose por esquina
# verdadera se calcula al medir, no lo produce la red -- que es justo el punto:
# la red no sabe cual es.
LIMITES = ("sup", "inf")
# Las cuatro celdas de esquina siguen existiendo en el DATO y se usan para el
# desglose al medir -- son lo unico comparable columna a columna con `esq-cq`.
# La red no las ve: para ella solo hay "hay limite y esta a esta altura".
CELDAS = ("tl", "tr", "bl", "br")

# LA ESTRUCTURA QUE SE CORRE ES UNA, Y LO UNICO QUE VARIA ES `k`.
#
# Por orden del dueno (2026-09-07): «Anotalas pero no vamos a ejecutarlas. Solo
# variamos los kernels». Las otras tres --`sig`, `ant` y el control `ind`-- SIGUEN
# IMPLEMENTADAS aqui abajo a proposito: una estructura implementada es la forma
# menos ambigua de anotarla, y ponerla en marcha es anadir su linea a `BRAZOS`.
# Estan descritas, con lo que cada una contestaria y lo que cuesta, en
# `instrucciones/03-alternativas-anotadas.md`.
#
ESTRUCTURA = "max"
ESTRUCTURAS = ("max", "sim")             # implementadas; solo se corre ESTRUCTURA

# ⚠ LOS BRAZOS SE LLAMAN COMO LOS DE `esq-k` -- k05, k07... -- y no `sig-k05`.
# Con una sola estructura, el nombre solo tiene que decir lo unico que varia; y
# asi las dos tablas de resultados se leen una al lado de la otra sin traducir.
def _nombre(k: int, esquina: str | None = None) -> str:
    return f"k{k:02d}{'-' + esquina if esquina else ''}"


# nombre -> (estructura, k, esquina). `esquina` solo lo usa el control `ind`,
# que entrena UNA red por esquina y por tanto no comparte nada.
BRAZOS = {_nombre(_k): (ESTRUCTURA, _k, None) for _k in K_BARRIDO}


class LimiteHorizontalUnKernel(nn.Module):
    """UNA convolucion k x k sin bias, y una cabeza de 3 parametros.

    UNA sola salida: `existe` y UNA ALTURA (`y`), sin decir si el limite es el
    superior o el inferior. Por orden del dueno (2026-09-07): «se van a probar
    los mismos valores de kernel, pero ahora seran limites en vez de esquinas»,
    y «la altura tomala de la pos y de tl y bl (borde alto y bajo)».

    ⚠ NO HAY `x`: un limite horizontal es una LINEA. Y no cuesta parametros
    quitarla -- `x` e `y` salian de las dos marginales del MISMO softmax 2-D, sin
    pesos propios. La cabeza sigue siendo (beta, a, b) = 3 parametros y los
    totales siguen siendo k^2 + 3, o sea los de `esq-k` y `esq-cq`.

    `estructura` decide QUE se le exige al kernel:

        max   LA QUE SE CORRE. La altura se lee del MAXIMO del mapa, y punto.
              Cabeza de 3 parametros (beta, a, b): exactamente la de `esq-k`.
        sim   igual que `max`, pero el kernel se PROYECTA simetrico bajo VOLTEO
              VERTICAL en cada paso: W = (V + flipud(V))/2. Asi la red es
              EQUIVARIANTE exacta al dar la vuelta a la imagen de arriba abajo.

    ⚠⚠ LA SIMETRIA QUE IMPORTA AQUI ES OTRA, Y ESO NO ES UN DETALLE.
    En `esq-cq` el par (tl, br) era giro de 180 grados uno del otro, asi que la
    prediccion se escribia sobre `rot180`. Aqui el par es (borde alto, borde
    bajo), que son reflejo uno del otro por VOLTEO VERTICAL:

        conv(flipud(x), W) = flipud(conv(x, W))   <=>   W[a,b] = W[k-1-a, b]

    `simetria()` devuelve por eso DOS fracciones: la vertical -- que es la que
    predice el algebra de ESTE experimento -- y la de 180 grados, que se conserva
    solo porque es la unica columna comparable con `esq-cq` y `esq-k`. Reportar
    una sola y llamarla "S %" haria que las dos tablas parecieran hablar de lo
    mismo.

    ⚠ Y AQUI LA PREMISA SI SE SOSTIENE, al reves que en `esq-cq`. Alli el algebra
    suponia que el parche de una `br` era el giro del de una `tl`, y estaba
    MEDIDO FALSO (tinta a 1,0 px contra 8,1 px). El borde alto y el bajo si son
    reflejo vertical uno del otro -- salvo en el tramo derecho del bajo, donde la
    ultima linea del parrafo es corta (medido 2026-09-07: 7,00 px de mediana en
    la celda `br` contra 1,00-2,00 en las otras tres).

    ⚠⚠ Y EL SUPUESTO SOBRE EL QUE DESCANSA ESA ALGEBRA ES FALSO, medido el
    2026-09-07 sobre las 389 ventanas del `val.npz` publicado. `esq-2d` la
    construia sobre «el parche de una br es el giro del de una tl, salvo el texto
    concreto». No lo es: la distancia del punto etiquetado al pixel de TINTA mas
    cercano es de 1,0 px en `tl` y de 8,1 px en `br` (p90 12,2), y el cuadrante
    propio de `br` (radio 6 px) esta VACIO en el 82 % de los casos. `br` no es una
    esquina de tinta: es un vertice de la caja del layout, y la ultima linea del
    parrafo es corta. Simetrizar ayuda, pero no puede cerrar la tarea -- el kernel
    tendria que responder alto a dos patrones que NO son giro uno del otro.
    Consecuencia directa para el eje: el radio del campo receptivo es 2/3/4/5/6 px
    para k = 5..13, asi que NINGUN brazo del rango alcanza los 8,1 px donde
    empieza la tinta de `br`. El primer k con radio >= 8 es 17.
    """

    def __init__(self, estructura: str, k: int, esquina: str | None = None,
                 ventana: int = VENTANA, beta0: float = BETA0):
        super().__init__()
        if (k - 1) % 2:
            raise ValueError(f"k tiene que ser impar para que el campo receptivo tenga centro; es {k}")
        if estructura not in ESTRUCTURAS:
            raise ValueError(f"estructura '{estructura}' no existe; hay {ESTRUCTURAS}")
        self.estructura, self.k, self.ventana = estructura, k, ventana
        self.m = ventana - k + 1
        # De donde se deriva el positivo. La red NO sabe cual es, y ese es el punto.
        self.limites = LIMITES
        self.conv = nn.Conv2d(1, 1, k, stride=1, padding=0, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)))
        # UNA sola cabeza: hay una sola cosa que detectar. `esq-2d` tenia dos
        # porque tenia dos salidas; al colapsarlas, la cabeza vuelve a ser
        # exactamente la de `esq-k`: beta + (a, b) = 3 parametros.
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        # La posicion i del mapa mira el CENTRO de su campo receptivo, que en
        # coordenadas de la ventana cae en i + (k-1)/2. Sin esto la esquina
        # predicha saldria desplazada (k-1)/2 px y nadie lo notaria hasta el final.
        self.register_buffer("coords", torch.arange(self.m, dtype=torch.float32) + (k - 1) / 2)

    def kernel(self) -> torch.Tensor:
        """El kernel EFECTIVO. En `sim` se proyecta simetrico en cada paso.

        ⚠ La proyeccion va en el forward y no en un `no_grad` despues del paso
        del optimizador: asi la parte simetrica de `V` no recibe gradiente (el
        modelo no depende de ella) en vez de recibirlo y ser borrada despues,
        que es la version que se ve igual y entrena otra cosa."""
        w = self.conv.weight
        if self.estructura == "sim":
            # VOLTEO VERTICAL (dims=-2), no rot180: el par que esta red tiene que
            # tratar igual es (borde alto, borde bajo), y esos son reflejo uno del
            # otro al dar la vuelta a la imagen de arriba abajo.
            w = (w + torch.flip(w, dims=(-2,))) / 2
        return w

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel()).squeeze(1)

    # --- la lectura C1, con signo ------------------------------------------
    def _leer(self, mapa: torch.Tensor):
        """(logit_existe, y) leyendo el MAXIMO del mapa. SOLO la altura.

        ⚠ `x` se calcula pero NO se devuelve como salida: se expone aparte, en
        `diagnostico()`, porque contesta "¿el maximo se pega a algun sitio de la
        linea?" y eso es util de mirar -- pero no entra en la perdida ni en el
        criterio, porque para un limite horizontal no significa nada.

        El marginal en `y` esta ademas MEJOR CONDICIONADO que el de `x`: el mapa
        de un limite es una CRESTA, no un pico, asi que sumar sobre columnas
        concentra la senal mientras que sumar sobre filas da el centroide de la
        cresta, que es ruido."""
        beta = self.log_beta.exp()
        plano = (beta * mapa).flatten(1)
        p = torch.softmax(plano, dim=1).view_as(mapa)
        y = (p.sum(dim=2) * self.coords).sum(dim=1)
        # ~ max(M). `existe` es una funcion LINEAL del pico, o sea MONOTONA --
        # y con una sola clase ("hay esquina") eso es exactamente lo que hace
        # falta: fondo bajo, esquina alta, una recta separa. Con las dos salidas
        # de `esq-2d` no bastaba, porque "br" habria sido una BANDA.
        pico = torch.logsumexp(plano, dim=1) / beta
        return self.a[0] * pico + self.b[0], y

    def forward(self, x: torch.Tensor):
        """x: (B,1,N,N) -> ((logit, y), mapa). UNA salida, y solo la ALTURA."""
        mapa = self._conv(x)
        return self._leer(mapa), mapa

    def diagnostico(self, mapa: torch.Tensor) -> torch.Tensor:
        """La `x` del maximo. NO es una salida: no entra en la perdida ni en el
        criterio. Se imprime para ver si el maximo se pega a un extremo de la
        linea o se queda en el centro, que es informacion barata."""
        p = torch.softmax((self.log_beta.exp() * mapa).flatten(1), dim=1).view_as(mapa)
        return (p.sum(dim=1) * self.coords).sum(dim=1)

    def n_parametros(self) -> int:
        """Los GRADOS DE LIBERTAD reales, que en `sim` no son los tensores
        guardados: la parte antisimetrica de V no recibe gradiente y no existe
        para el modelo. Contarla seria inflar el precio de ese brazo."""
        return self.n_kernel() + self.n_cabeza()

    def n_kernel(self) -> int:
        # Simetrico bajo VOLTEO VERTICAL: las filas i y k-1-i comparten valor, y
        # la fila central es la suya propia -> k * (k+1) / 2 grados de libertad.
        return self.k * (self.k + 1) // 2 if self.estructura == "sim" else self.k ** 2

    def n_cabeza(self) -> int:
        """3: beta + (a, b). La misma cabeza de `esq-k`."""
        return 1 + 2 * self.a.numel()


def construir_suelta(estructura: str, k: int, esquina: str | None = None,
                     semilla: int = 1) -> LimiteHorizontalUnKernel:
    """Una estructura CUALQUIERA, este armada o no en `BRAZOS`.

    Existe para que las alternativas anotadas se sigan comprobando: una
    estructura que se guarda "por si acaso" y deja de ejecutarse nunca se pudre
    en silencio, y el dia que se arme habria que depurarla desde cero."""
    torch.manual_seed(semilla)
    return LimiteHorizontalUnKernel(estructura, k, esquina)


def construir(brazo: str, semilla: int = 1) -> LimiteHorizontalUnKernel:
    """La red de un brazo, con inicializacion REPRODUCIBLE.

    La semilla se fija aqui y no fuera: "sin entrenar" tiene que ser el mismo
    "sin entrenar" cada vez que se dibuje la figura de las muestras."""
    if brazo not in BRAZOS:
        raise KeyError(f"brazo '{brazo}' no existe; hay {sorted(BRAZOS)}")
    estructura, k, esquina = BRAZOS[brazo]
    torch.manual_seed(semilla)
    return LimiteHorizontalUnKernel(estructura, k, esquina)


def _frac_sim(w: torch.Tensor, dims) -> float:
    r = torch.flip(w, dims=dims)
    s = (w + r) / 2
    tot = float((w ** 2).sum().detach()) or 1.0
    return float((s ** 2).sum().detach()) / tot


def simetria(w: torch.Tensor) -> tuple[float, float]:
    """(fraccion simetrica bajo VOLTEO VERTICAL, fraccion simetrica bajo ROT180).

    ⚠ SON DOS Y HAY QUE REPORTAR LAS DOS. La primera es la que predice el algebra
    de ESTE experimento: el par (borde alto, borde bajo) es reflejo vertical. La
    segunda no dice nada aqui, y se conserva porque es la UNICA columna comparable
    con la `S %` de `esq-cq` y `esq-k`. Dar solo una y llamarla "S %" haria que las
    tres tablas pareciesen hablar de lo mismo."""
    return _frac_sim(w, (-2,)), _frac_sim(w, (-2, -1))


if __name__ == "__main__":
    print(f"LOS {len(BRAZOS)} BRAZOS QUE SE CORREN (estructura `{ESTRUCTURA}`, "
          f"y lo unico que varia es k):\n")
    print(f"{'brazo':>11} {'estruct':>8} {'k':>3} {'mapa':>7} {'radio':>6} "
          f"{'kernel':>7} {'cabeza':>7} {'total':>6}")
    for brazo in BRAZOS:
        red = construir(brazo)
        x = torch.randn(2, 1, VENTANA, VENTANA)
        (logit, py), mapa = red(x)
        assert mapa.shape == (2, red.m, red.m)
        # UNA salida y SOLO la altura: si esto devuelve tres cosas, el cambio de
        # esquinas a limites no se aplico.
        assert logit.shape == (2,) and py.shape == (2,)
        lo, hi = float(red.coords[0]), float(red.coords[-1])
        py = py.detach()
        assert lo <= float(py.min()) and float(py.max()) <= hi, \
            f"{brazo}: la altura leida se sale del mapa"
        assert red.n_cabeza() == 3, f"{brazo}: cabeza de {red.n_cabeza()}, no 3"
        assert red.n_parametros() == red.k ** 2 + 3
        print(f"{brazo:>11} {red.estructura:>8} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {(red.k-1)//2:>5} px "
              f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    print("\nLA ALTERNATIVA ANOTADA Y NO ARMADA (ver instrucciones/03-...):\n")
    print(f"{'estruct':>11} {'k':>3} {'mapa':>7} {'kernel':>7} {'cabeza':>7} {'total':>6}")
    red = construir_suelta("sim", 7)
    print(f"{'sim':>11} {red.k:>3} {str(red.m)+'x'+str(red.m):>7} "
          f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    # --- INVARIANTE 1: la ALTURA sale del maximo, y de ningun otro sitio ------
    red = construir(_nombre(7))
    mapa = torch.zeros(1, red.m, red.m)
    mapa[0, 4, 9] = -20.0                       # pico negativo: no se lee
    mapa[0, 15, 3] = +20.0                      # pico positivo: este es
    with torch.no_grad():
        _, py = red._leer(mapa)
    off = (red.k - 1) / 2
    assert abs(float(py) - (15 + off)) < 0.01, "la altura no sale del maximo"
    print("\nmax: la ALTURA sale del maximo del mapa (y no se devuelve ninguna x)")

    # --- INVARIANTE 2: EQUIVARIANZA bajo VOLTEO VERTICAL ---------------------
    # ⚠ Es EL invariante de este experimento, y NO es el de `esq-cq`. Alli el par
    # era (tl, br), giro de 180; aqui es (borde alto, borde bajo), reflejo
    # vertical. Dar la vuelta a la imagen de arriba abajo tiene que devolver la
    # misma altura, medida desde el otro lado.
    red = construir_suelta("sim", 7)
    w = red.kernel()
    assert torch.allclose(w, torch.flip(w, dims=(-2,)), atol=1e-7), "sim no es simetrico vertical"
    sv, s180 = simetria(w)
    assert sv > 0.999, f"sim no es simetrico bajo volteo vertical: {sv}"
    x = torch.randn(1, 1, VENTANA, VENTANA)
    with torch.no_grad():
        (_, y1), _ = red(x)
        (_, y2), _ = red(torch.flip(x, dims=(2,)))     # volteo VERTICAL solo
    dy = abs(float(y1) - ((VENTANA - 1) - float(y2)))
    assert dy < 0.01, f"sim NO es equivariante bajo volteo vertical: dy={dy:.3f}"
    print("sim: EQUIVARIANTE exacta bajo VOLTEO VERTICAL -- la misma altura, del otro lado")

    # --- INVARIANTE 3: las dos simetrias son DISTINTAS ------------------------
    # Si alguien vuelve a poner rot180 donde va el volteo vertical, esto lo pilla:
    # un kernel simetrico vertical NO tiene por que serlo bajo giro de 180.
    assert s180 < 0.999, ("simetria() devuelve dos veces lo mismo: el kernel `sim` "
                          "es simetrico vertical, y bajo rot180 no tiene por que")
    print(f"     y bajo rot180 solo {100*s180:.0f}% -- son DOS medidas distintas")

    sv0, s1800 = simetria(construir(_nombre(7)).kernel())
    print(f"\nkernel sin entrenar (max): {100*sv0:.1f}% simetrico VERTICAL · "
          f"{100*s1800:.1f}% simetrico rot180")
    print("⚠ PREDICCION REGISTRADA: la fraccion VERTICAL debe SUBIR al entrenar.")
    print("  (la de rot180 se reporta solo para poder mirarla junto a `esq-cq`)")
    print("todas construyen, la altura cae dentro del mapa y ninguna conv lleva bias.")
