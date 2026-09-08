#!/usr/bin/env python3
"""La CNN de referencia del banco `banco-k` (especificacion §7), AUTONOMA.

    python nn/modelo.py            imprime la cadena de dimensiones y comprueba los
                                   invariantes: 5.812 parametros, 68 en la cabeza,
                                   rejilla 16 x 16, y las dos lecturas del padding

⚠ NO IMPORTA NADA de este repo ni de `foveal-vision`. Es lo unico que garantiza poder
cargar estos pesos dentro de un anyo. Solo `torch`.

POR QUE ESTA RED ES TAN PEQUENYA, QUE ES LO QUE MAS SORPRENDE AL LEERLA
----------------------------------------------------------------------
No es una red que se quiera buena: es un INSTRUMENTO DE MEDIDA. El criterio unico de
la especificacion (§1.2) es «maximizar la sensibilidad de la medicion al kernel», y de
ahi sale todo: la cabeza de 68 parametros es deliberadamente INCAPAZ de compensar un
kernel deficiente (§7.2). Una cabeza densa grande comprimiria las diferencias entre
condiciones hasta hacerlas indistinguibles del ruido entre semillas, y con 100 muestras
de entrenamiento ese efecto es severo. O sea: darle capacidad a la red REDUCE la
evidencia que el banco puede producir. No se «mejora» esta red.

✅ EL PADDING DEL TRONCO ES `same`, Y LO DICE LA v1.2 EXPLICITAMENTE
------------------------------------------------------------------
La v1.0 se contradecia: escribia la cadena `128 -> 64 -> 32 -> 16` y a la vez decia
«todas las convoluciones son `valid`». Se reporto, y la v1.2 lo cierra:

    «Todas las convoluciones del tronco usan padding `same`. La cadena de rejillas
     es 128 -> 64 -> 32 -> 16.»                                            (§7.1)

Y da el motivo, que es mejor que el que se habia deducido -- ALCANZABILIDAD. No es
una cuestion de gusto ni de cuadrar una tabla: con `valid` los centros de campo
receptivo de la rejilla caen en los pixeles de entrada 10 a 114, mientras que las
etiquetas llegan hasta los extremos del marco. O sea que un borde cerca del pixel 8
o del 119 seria INALCANZABLE POR CONSTRUCCION para el soft-argmax -- la misma clase
de error irreducible que §3.3 existe para evitar.

⚠ DISCREPANCIA ARITMETICA CON LA v1.2, anotada y no heredada. El §7.1 escribe que
«§3.3 permite bordes de parrafo en el rango [8, 120]», tres veces (§7.1 y §7.5). Pero
la aritmetica de la propia especificacion da 119, no 120:

    §3.3 acota la caja a [68, 512] en el marco de 584
    §6.4 transforma  coord_final = (coord_584 / 4) - 9
    ->  68/4 - 9 = 8   y   512/4 - 9 = 119

Aqui se usa 119, que es lo que sale de §3.3 + §6.4. NO cambia ninguna conclusion --
8 y 119 quedan los dos fuera del span 10..114 de `valid`, y los dos dentro del 0..120
de `same` --, asi que el argumento de alcanzabilidad se sostiene igual. Lo que cambia
es el MARGEN por arriba: 1 px (119 contra 120), no 0. Sigue siendo ajustado, y por eso
§7.5 manda comprobarlo CONTRA EL DATASET REAL y no contra el rango teorico.

Comprobado aqui, y es lo que verifica `--span` (medido el 2026-09-08):

    same    rejilla=16   centros 0 … 120  paso 8   CUBRE  las etiquetas [8, 119]
    valid   rejilla=14   centros 10 … 114 paso 8   NO cubre: [8,10) y (114,119]

⚠ El recuento de parametros NO distingue las dos lecturas (5.812 en las dos: el
padding no anyade pesos), asi que la v1.2 obliga a verificar sobre LAS DIMENSIONES:
«La implementacion debe imprimir la cadena de dimensiones al ejecutar y compararla
con 128/64/32/16 como asercion» (§7.1). Es lo que hace `main()`, y por eso
`PADDING_SAME` sigue siendo una constante visible y no un `padding=2` enterrado en
la definicion de las capas: la trampa tiene su propia fila en el §14 («Tronco con
`valid` en vez de `same` … el recuento de parametros no lo detecta»).

⚠ Esto NO contradice el §6, que si elimina el padding: lo que §6 quita es un
artefacto DEPENDIENTE DEL KERNEL, confundible con la variable en estudio. El padding
del tronco es identico en todas las condiciones -- degrada a todas por igual y no
afecta la comparabilidad (§7.1, «Relacion con §6»).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# §7.1 de la v1.2: el tronco usa `same`. NO es una eleccion de esta implementacion.
# Se deja como constante visible porque el recuento de parametros no detecta el error
# (5.812 en los dos casos) y la comprobacion tiene que ser sobre las DIMENSIONES.
PADDING_SAME = True
CADENA_ESPERADA = (64, 32, 16, 16)     # §7.1: 128 -> 64 -> 32 -> 16
SPAN_ESPERADO = (0, 120)               # §7.1/§7.5: centros de la rejilla en la entrada
ETIQUETAS_EN_128 = (8, 119)            # §3.3 [68,512] + §6.4 (/4, -9). La spec
#                                      escribe 120; su aritmetica da 119 (ver docstring)

REJILLA = 16 if PADDING_SAME else 14   # el lado del mapa que llega a la cabeza
TAU = 1.0                              # temperatura del soft-argmax (§7.3), fija
ENTRADA = 128                          # §12: invariante
PARAMS_ESPERADOS = 5812                # §7.1
PARAMS_CABEZA_ESPERADOS = 68           # §7.1: la conv 1x1 ES la cabeza completa


class BancoCNN(nn.Module):
    """4 convoluciones y un soft-argmax. Sin cabeza densa, a proposito (§7.2)."""

    def __init__(self) -> None:
        super().__init__()
        p5 = 2 if PADDING_SAME else 0
        p3 = 1 if PADDING_SAME else 0
        self.rasgos = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=p5), nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=p5), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=p3), nn.ReLU(),
        )
        # La cabeza entera: 16*4 + 4 = 68 parametros. UN CANAL POR BORDE (§7.4).
        self.cabeza = nn.Conv2d(16, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,128,128) -> (B,4) con las 4 coordenadas normalizadas a [0,1]."""
        return leer_coordenadas(self.cabeza(self.rasgos(x)))


def leer_coordenadas(mapas: torch.Tensor) -> torch.Tensor:
    """El soft-argmax sobre marginales 1D del §7.3. (B,4,R,R) -> (B,4).

    Orden de salida: izq, der, sup, inf (§3.4).

    Vive FUERA de la clase a proposito: es la pieza que el §7.4 declara critica, asi
    que tiene que poder comprobarse con mapas fabricados a mano, sin depender de los
    pesos ni de la inicializacion de la red. Una comprobacion que solo se puede hacer
    pasando una imagen por las 4 convoluciones no comprueba esta funcion: comprueba
    los sesgos del init (medido el 2026-09-08: entrada de ceros da 0,4988 y no 0,5,
    porque la conv1 tiene bias y el mapa que llega a la cabeza no es nulo).
    """
    # §7.3: los canales 0 y 1 se leen sobre COLUMNAS (suma sobre filas, dim -2);
    # los canales 2 y 3 sobre FILAS (suma sobre columnas, dim -1).
    #
    # ⚠ §7.4: un canal DEDICADO por borde. Con un solo mapa de columnas para izq y
    # der la marginal es bimodal y el soft-argmax devuelve la esperanza de los dos
    # modos -- el CENTRO del parrafo -- colapsando las dos predicciones al mismo
    # valor. Por eso son 4 canales y no 2, y por eso esto no se «simplifica».
    # `main()` lo DEMUESTRA ejecutandolo, en vez de solo advertirlo.
    n = mapas.shape[-1]
    col = mapas.sum(dim=-2)                      # (B, 4, R) sobre columnas
    fil = mapas.sum(dim=-1)                      # (B, 4, R) sobre filas
    marginales = torch.stack(
        [col[:, 0], col[:, 1], fil[:, 2], fil[:, 3]], dim=1)   # (B, 4, R)

    p = torch.softmax(marginales / TAU, dim=-1)
    idx = torch.arange(n, dtype=p.dtype, device=p.device)
    # coord = indice / (n-1): normaliza a [0,1] (§7.3)
    return (p * idx).sum(dim=-1) / (n - 1)


def _cadena(padding_same: bool) -> tuple[list[tuple[int, int, int]], int, int]:
    """La cadena de dimensiones y los recuentos, para una lectura del padding."""
    p5 = 2 if padding_same else 0
    p3 = 1 if padding_same else 0
    capas = nn.Sequential(
        nn.Conv2d(1, 8, 5, stride=2, padding=p5),
        nn.Conv2d(8, 16, 5, stride=2, padding=p5),
        nn.Conv2d(16, 16, 3, stride=2, padding=p3),
        nn.Conv2d(16, 4, 1),
    )
    x, dims = torch.zeros(1, 1, ENTRADA, ENTRADA), []
    with torch.no_grad():
        for c in capas:
            x = c(x)
            dims.append(tuple(x.shape[1:]))
    total = sum(q.numel() for q in capas.parameters())
    cabeza = sum(q.numel() for q in capas[-1].parameters())
    return dims, total, cabeza


def centros_rejilla(padding_same: bool) -> list[int]:
    """Los centros de campo receptivo de la rejilla final, en coordenadas de ENTRADA.

    §7.5 de la v1.2 lo hace OBLIGATORIO: «el span de centros de la rejilla debe
    contener el rango completo de coordenadas de etiqueta presentes en el dataset».
    Si no lo contiene, esas muestras son INALCANZABLES para el soft-argmax y aportan
    un error irreducible -- y no falla por ningun lado: sale como que todos los
    kernels son un poco malos.

    Es aritmetica pura, sin torch: se puede comprobar antes de tener dataset."""
    capas = [(5, 2, 2), (5, 2, 2), (3, 2, 1)] if padding_same else [(5, 2, 0), (5, 2, 0), (3, 2, 0)]
    n_in = ENTRADA
    c = list(range(n_in))
    for k, paso, pad in capas:
        n_out = (n_in + 2 * pad - k) // paso + 1
        c = [c[j * paso - pad + (k - 1) // 2] for j in range(n_out)]
        n_in = n_out
    return c


def main() -> int:
    print(f"\nBancoCNN — especificacion §7 · PADDING_SAME = {PADDING_SAME}\n")
    fallos = []

    for etiqueta, same in (("padding `same` (dimensiones ESCRITAS)", True),
                           ("padding `valid` (la PALABRA del §7.1)", False)):
        dims, total, cabeza = _cadena(same)
        marca = "<-- implementado" if same == PADDING_SAME else ""
        print(f"  {etiqueta:38} {[d[1] for d in dims]}  params={total} "
              f"cabeza={cabeza} {marca}")
    print()

    m = BancoCNN()
    total = sum(q.numel() for q in m.parameters())
    cabeza = sum(q.numel() for q in m.cabeza.parameters())
    x = torch.zeros(2, 1, ENTRADA, ENTRADA)
    with torch.no_grad():
        rasgos = m.rasgos(x)
        salida = m(x)

    def comprobar(que: str, obtenido, esperado) -> None:
        ok = obtenido == esperado
        print(f"  {'ok ' if ok else 'FALLA'}  {que:34} {obtenido}"
              f"{'' if ok else f'  (esperado {esperado})'}")
        if not ok:
            fallos.append(que)

    comprobar("parametros totales (§7.1)", total, PARAMS_ESPERADOS)
    comprobar("parametros de la cabeza (§7.1)", cabeza, PARAMS_CABEZA_ESPERADOS)
    comprobar("rejilla que llega a la cabeza", rasgos.shape[-1], REJILLA)
    comprobar("canales de la cabeza (§7.4)", m.cabeza.out_channels, 4)
    comprobar("forma de la salida", tuple(salida.shape), (2, 4))
    # --- §7.1: la cadena ENTERA, no solo la rejilla final. El recuento de
    # parametros no detecta un tronco con `valid` (5.812 en los dos casos), asi que
    # la v1.2 obliga a verificar las DIMENSIONES. Es una fila del §14.
    dims_impl, _, _ = _cadena(PADDING_SAME)
    comprobar("cadena de rejillas (§7.1)", tuple(d[1] for d in dims_impl), CADENA_ESPERADA)

    # --- §7.5: el span de centros tiene que CUBRIR el rango de etiquetas ---
    c = centros_rejilla(PADDING_SAME)
    comprobar("span de centros en la entrada (§7.5)", (c[0], c[-1]), SPAN_ESPERADO)
    lo, hi = ETIQUETAS_EN_128
    alcanzable = c[0] <= lo and hi <= c[-1]
    comprobar("el span CUBRE las etiquetas (§7.5)", alcanzable, True)
    print(f"         etiquetas posibles [{lo}, {hi}] · centros [{c[0]}, {c[-1]}] "
          f"paso {c[1] - c[0]} px"
          + ("" if alcanzable else "  <-- HAY BORDES INALCANZABLES"))
    # ⚠ El extremo superior es AJUSTADO -- 1 px: centros hasta 120, etiquetas hasta
    # 119 -- y la v1.2 avisa de que lo es (aunque ella escribe 120 donde su propia
    # aritmetica da 119; ver el docstring). Por eso §7.5 pide repetir esto CONTRA EL
    # DATASET REAL y no contra el rango teorico: si el generador saca un borde por
    # encima de 120, esas muestras son inalcanzables aunque esto salga en verde.

    # --- el soft-argmax, con mapas FABRICADOS: no depende de los pesos (§7.3) ---
    nulos = torch.zeros(1, 4, REJILLA, REJILLA)
    comprobar("mapa nulo -> centro exacto", [round(float(v), 4) for v in leer_coordenadas(nulos)[0]],
              [0.5, 0.5, 0.5, 0.5])

    # Un pico agudo en la columna j tiene que leerse como j/(R-1).
    j = 3
    pico = torch.zeros(1, 4, REJILLA, REJILLA)
    pico[0, 0, :, j] = 50.0                       # canal 0 = borde izquierdo
    comprobar(f"pico en la columna {j} -> {j}/{REJILLA - 1}",
              round(float(leer_coordenadas(pico)[0, 0]), 3), round(j / (REJILLA - 1), 3))

    # El canal 2 lee FILAS: el mismo pico en una fila tiene que mover `sup`, no `izq`.
    fila = torch.zeros(1, 4, REJILLA, REJILLA)
    fila[0, 2, j, :] = 50.0                       # canal 2 = borde superior
    comprobar(f"pico en la fila {j} -> sup = {j}/{REJILLA - 1}",
              round(float(leer_coordenadas(fila)[0, 2]), 3), round(j / (REJILLA - 1), 3))

    # --- §7.4 DEMOSTRADO, no advertido: dos bordes en UN canal colapsan al centro ---
    izq, der = 2, 13
    dos = torch.zeros(1, 4, REJILLA, REJILLA)
    dos[0, 0, :, izq] = 50.0                      # los DOS bordes en el MISMO canal
    dos[0, 0, :, der] = 50.0
    colapso = float(leer_coordenadas(dos)[0, 0])
    centro = (izq + der) / 2 / (REJILLA - 1)
    print(f"\n  §7.4 — dos bordes ({izq} y {der}) leidos de UN canal dan "
          f"{colapso:.3f}, que es el CENTRO ({centro:.3f}),")
    print(f"         no {izq}/{REJILLA - 1}={izq / (REJILLA - 1):.3f} ni "
          f"{der}/{REJILLA - 1}={der / (REJILLA - 1):.3f}. Por eso hay UN CANAL POR BORDE.")
    comprobar("§7.4 el colapso es real (= centro)", round(colapso, 3), round(centro, 3))

    print()
    if fallos:
        print(f"✗ {len(fallos)} invariante(s) del §7 que NO se cumplen: "
              + ", ".join(fallos) + "\n")
        return 1
    print("Los invariantes del §7 se cumplen.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
