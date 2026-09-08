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

⚠⚠ LA CONTRADICCION DEL §7.1, SIN RESOLVER Y CONMUTABLE
-------------------------------------------------------
El §7.1 escribe la cadena `128 -> 64 -> 32 -> 16` y a la vez dice «todas las
convoluciones son `valid`». LAS DOS COSAS NO PUEDEN SER CIERTAS:

    valid, stride 2:  (128-5)//2+1 = 62 -> (62-5)//2+1 = 29 -> (29-3)//2+1 = 14
    padding "same":   128 -> 64 -> 32 -> 16          <- las dimensiones ESCRITAS

El recuento de parametros NO desempata: 5.812 en los dos casos (el padding no anyade
pesos), asi que la cifra del §7.1 es compatible con ambas lecturas. Lo que si desempata
son TRES cosas del propio documento, y las tres apuntan a `same`:

  1. §7.3 lee «cada marginal de 16 elementos» y divide por 15 para normalizar a [0,1];
  2. §7.5 dice «la rejilla de 16 x 16 corresponde a 8 px por celda en el marco de 128»,
     y 128/16 = 8 EXACTO (con 14 seria 9,14, que no es lo escrito);
  3. §7.5 dice que el arreglo diagnostico es quitar el stride de la tercera conv
     «pasando a 32 x 32», que es lo que sale de 64 -> 32 -> 32, no de 62 -> 29 -> 29.

Asi que aqui se implementa `same` (PADDING_SAME = True), que es la lectura que
reproduce las dimensiones escritas y las tres corroboraciones. Queda como INTERRUPTOR
de una linea, no cableado, porque el §12 declara la arquitectura INVARIANTE: si el
duenyo confirma que lo que manda es la palabra `valid`, se cambia la constante y el
banco se congela con 14 x 14 -- pero entonces el §7.5 hay que reescribirlo tambien.

⚠ Esto NO es lo mismo que el `valid` del §6, que si es `valid` y esta bien: alli la
convolucion del KERNEL sobre la imagen de 146 es `valid` a proposito, y el recorte a
128 la compensa (§6.2). Son dos convoluciones distintas en dos sitios distintos.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# La lectura del §7.1 que se implementa. Ver el docstring: `True` reproduce las
# dimensiones ESCRITAS (64/32/16); `False` obedece a la palabra `valid` y da 62/29/14.
PADDING_SAME = True

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
