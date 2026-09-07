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

# ⚠ ESQUINAS YA NO SON "LAS SALIDAS": son de cuales se DERIVA el unico positivo.
# En `esq-2d` habia una salida por esquina; aqui hay UNA, y `tl`/`br` solo dicen
# de donde sale la etiqueta (ver `datos.objetivo`). El desglose por esquina
# verdadera se calcula al medir, no lo produce la red -- que es justo el punto:
# la red no sabe cual es.
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


class CualquierEsquinaUnKernel(nn.Module):
    """UNA convolucion k x k sin bias, y una cabeza de 3 parametros.

    UNA sola salida: `existe` y UNA posicion, sin decir si la esquina es `tl` o
    `br`. Por orden del dueno (2026-09-07): «no se quieren detectar las 2
    esquinas por separado, sino cualquiera de ellas [...] Decimos "hay esquina"
    si la hay, y la posicion de ella, sin importar si es tl o br».

    `estructura` decide QUE se le exige al kernel:

        max   LA QUE SE CORRE. La posicion se lee del MAXIMO del mapa, y punto.
              Cabeza de 3 parametros (beta, a, b): exactamente la de `esq-k`.
        sim   igual que `max`, pero el kernel se PROYECTA simetrico en cada paso:
              W = (V + rot180(V))/2. Asi A = 0 por construccion y la red es
              EQUIVARIANTE exacta bajo giro de 180 grados. El precio es algo mas
              de la mitad de grados de libertad -- (k^2+1)/2 --. Sirve para
              separar "el maximo no basta" de "el gradiente no llego a la
              simetria".

    ⚠ POR QUE `sim` Y NO `ant`, QUE ERA LA CONTINUACION DECLARADA DE `esq-2d`.
    Con dos salidas, toda la diferencia entre esquinas vivia en la parte
    ANTISIMETRICA: respuesta_tl = <S,P> + <A,P> y respuesta_br = <S,P> - <A,P>.
    Con UNA sola salida leida del maximo hay que pedir las DOS altas, y eso exige
    <S,P> >> |<A,P>|, o sea A -> 0. Es la inversa exacta. `ant` no es "la otra
    opcion": con esta lectura es INCAPAZ por construccion, porque responde a una
    esquina con el signo cambiado de la otra y solo una de las dos puede ser el
    maximo. Por eso se BORRA en vez de quedarse anotada -- es la regla del propio
    experimento: «una alternativa descartada que se queda en el repo se acaba
    armando por error».

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
        self.esquinas = ESQUINAS
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
            w = (w + torch.flip(w, dims=(-2, -1))) / 2
        return w

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel()).squeeze(1)

    # --- la lectura C1, con signo ------------------------------------------
    def _leer(self, mapa: torch.Tensor):
        """(logit_existe, x, y) leyendo el MAXIMO del mapa.

        Ya no hay parametro `signo`: con una sola salida hay un solo extremo del
        que leer. En `esq-2d` el signo existia porque `br` se leia del MINIMO."""
        beta = self.log_beta.exp()
        plano = (beta * mapa).flatten(1)
        p = torch.softmax(plano, dim=1).view_as(mapa)
        y = (p.sum(dim=2) * self.coords).sum(dim=1)
        x = (p.sum(dim=1) * self.coords).sum(dim=1)
        # ~ max(M). `existe` es una funcion LINEAL del pico, o sea MONOTONA --
        # y con una sola clase ("hay esquina") eso es exactamente lo que hace
        # falta: fondo bajo, esquina alta, una recta separa. Con las dos salidas
        # de `esq-2d` no bastaba, porque "br" habria sido una BANDA.
        pico = torch.logsumexp(plano, dim=1) / beta
        return self.a[0] * pico + self.b[0], x, y

    def forward(self, x: torch.Tensor):
        """x: (B,1,N,N) -> ((logit, x, y), mapa). UNA salida, no un dict."""
        mapa = self._conv(x)
        return self._leer(mapa), mapa

    def n_parametros(self) -> int:
        """Los GRADOS DE LIBERTAD reales, que en `sim` no son los tensores
        guardados: la parte antisimetrica de V no recibe gradiente y no existe
        para el modelo. Contarla seria inflar el precio de ese brazo."""
        return self.n_kernel() + self.n_cabeza()

    def n_kernel(self) -> int:
        # Un kernel simetrico bajo giro de 180 tiene (k^2+1)/2 libres: los pares
        # (i, j) y su reflejo comparten valor, y el centro es su propio reflejo.
        return (self.k ** 2 + 1) // 2 if self.estructura == "sim" else self.k ** 2

    def n_cabeza(self) -> int:
        """3: beta + (a, b). La misma cabeza de `esq-k`."""
        return 1 + 2 * self.a.numel()


def construir_suelta(estructura: str, k: int, esquina: str | None = None,
                     semilla: int = 1) -> CualquierEsquinaUnKernel:
    """Una estructura CUALQUIERA, este armada o no en `BRAZOS`.

    Existe para que las alternativas anotadas se sigan comprobando: una
    estructura que se guarda "por si acaso" y deja de ejecutarse nunca se pudre
    en silencio, y el dia que se arme habria que depurarla desde cero."""
    torch.manual_seed(semilla)
    return CualquierEsquinaUnKernel(estructura, k, esquina)


def construir(brazo: str, semilla: int = 1) -> CualquierEsquinaUnKernel:
    """La red de un brazo, con inicializacion REPRODUCIBLE.

    La semilla se fija aqui y no fuera: "sin entrenar" tiene que ser el mismo
    "sin entrenar" cada vez que se dibuje la figura de las muestras."""
    if brazo not in BRAZOS:
        raise KeyError(f"brazo '{brazo}' no existe; hay {sorted(BRAZOS)}")
    estructura, k, esquina = BRAZOS[brazo]
    torch.manual_seed(semilla)
    return CualquierEsquinaUnKernel(estructura, k, esquina)


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
    print(f"{'brazo':>11} {'estruct':>8} {'k':>3} {'mapa':>7} {'radio':>6} "
          f"{'kernel':>7} {'cabeza':>7} {'total':>6}")
    for brazo in BRAZOS:
        red = construir(brazo)
        x = torch.randn(2, 1, VENTANA, VENTANA)
        (logit, px, py), mapa = red(x)
        assert mapa.shape == (2, red.m, red.m)
        assert logit.shape == (2,) and px.shape == (2,) and py.shape == (2,)
        lo, hi = float(red.coords[0]), float(red.coords[-1])
        px, py = px.detach(), py.detach()
        assert lo <= float(px.min()) and float(px.max()) <= hi, \
            f"{brazo}: la lectura se sale del mapa"
        # LA CABEZA ES DE 3, Y ESO ES LA COMPARABILIDAD CON `esq-k`, no una cifra
        # bonita: alli la red tenia exactamente estos parametros.
        assert red.n_cabeza() == 3, f"{brazo}: cabeza de {red.n_cabeza()}, no 3"
        assert red.n_parametros() == red.k ** 2 + 3
        print(f"{brazo:>11} {red.estructura:>8} {red.k:>3} "
              f"{str(red.m)+'x'+str(red.m):>7} {(red.k-1)//2:>5} px "
              f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    print("\nLA ALTERNATIVA ANOTADA Y NO ARMADA (ver instrucciones/03-...):\n")
    print(f"{'estruct':>11} {'k':>3} {'mapa':>7} {'kernel':>7} {'cabeza':>7} {'total':>6}")
    red = construir_suelta("sim", 7)
    (logit, _, _), _ = red(torch.randn(2, 1, VENTANA, VENTANA))
    print(f"{'sim':>11} {red.k:>3} {str(red.m)+'x'+str(red.m):>7} "
          f"{red.n_kernel():>7} {red.n_cabeza():>7} {red.n_parametros():>6}")

    # --- INVARIANTE 1: la posicion se lee del MAXIMO, y de ningun otro sitio ---
    # Sobre un mapa con un pico positivo y otro negativo en sitios distintos, la
    # lectura tiene que caer en el POSITIVO. En `esq-2d` este mismo mapa daba dos
    # respuestas (una por extremo); aqui solo hay una, y es la del maximo.
    red = construir(_nombre(7))
    mapa = torch.zeros(1, red.m, red.m)
    mapa[0, 4, 9] = -20.0                       # pico NEGATIVO: ya no se lee
    mapa[0, 15, 3] = +20.0                      # pico POSITIVO: este es
    with torch.no_grad():
        _, px, py = red._leer(mapa)
    off = (red.k - 1) / 2
    assert abs(float(px) - (3 + off)) < 0.01 and abs(float(py) - (15 + off)) < 0.01, \
        "la lectura no cae en el maximo del mapa"
    print("\nmax: la posicion sale del MAXIMO del mapa (el minimo ya no se lee)")

    # --- INVARIANTE 2: EQUIVARIANZA bajo giro de 180 grados -------------------
    # Es EL invariante de esta pregunta y hasta hoy no existia en ninguna parte:
    # "sin importar si es tl o br" significa que girar la entrada 180 grados tiene
    # que devolver el MISMO punto, llevado a las coordenadas originales. Con `sim`
    # es exacto por construccion; con un kernel libre, no tiene por que.
    red = construir_suelta("sim", 7)
    w = red.kernel()
    assert torch.allclose(w, torch.flip(w, dims=(-2, -1)), atol=1e-7), "sim no es simetrico"
    s, a = simetria(w)
    assert s > 0.999, f"sim no es simetrico: {s}"
    x = torch.randn(1, 1, VENTANA, VENTANA)
    with torch.no_grad():
        (_, px1, py1), _ = red(x)
        (_, px2, py2), _ = red(torch.flip(x, dims=(2, 3)))
    # el giro manda el punto p al punto (VENTANA-1) - p
    dx = abs(float(px1) - ((VENTANA - 1) - float(px2)))
    dy = abs(float(py1) - ((VENTANA - 1) - float(py2)))
    assert dx < 0.01 and dy < 0.01, f"sim NO es equivariante bajo giro: dx={dx:.3f} dy={dy:.3f}"
    print("sim: EQUIVARIANTE exacta bajo giro de 180 -- la misma posicion, girada")

    sim_, anti_ = simetria(construir(_nombre(7)).kernel())
    print(f"\nkernel sin entrenar (max): {100*sim_:.1f}% simetrico · {100*anti_:.1f}% antisimetrico")
    print("⚠ PREDICCION REGISTRADA: con esta lectura la fraccion SIMETRICA debe")
    print("  SUBIR al entrenar. `esq-k` acabo en 74,0 % y `esq-2d` en 38-53 %.")
    print("todas construyen, la lectura cae dentro del mapa y ninguna conv lleva bias.")
