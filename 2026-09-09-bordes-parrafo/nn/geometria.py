#!/usr/bin/env python3
"""La geometria de una pagina: reparto en celdas, colocacion y las ASERCIONES.

    python nn/geometria.py          corre sus 21 comprobaciones y sale 0 o 1

Este modulo NO renderiza y NO importa nada del repo: es aritmetica pura sobre
cajas. Asi se puede comprobar la parte que decide si el dataset sirve --
separacion, margen y ventana limpia -- en milisegundos y sin navegador.

DE DONDE SALEN LOS NUMEROS (todos derivados de k=19, ninguno de gusto)
=====================================================================

El encargo fija UN numero: el kernel mas grande que se va a entrenar es 19 px.
Todo lo demas se deriva de el.

  RADIO = (19 - 1) / 2 = 9
      Una convolucion k x k centrada en un pixel mira RADIO px a cada lado.

  SEPARACION_DURA = 2 * RADIO + 1 = 19
      Separacion L-infinito MINIMA entre la tinta de dos parrafos. Con esto, la
      ventana 19x19 centrada en el borde de A y la centrada en el borde de B no
      se solapan ENTRE SI, y ninguna de las dos ve tinta del otro parrafo. Es el
      "suficiente espacio entre parrafos para aplicar un kernel de 19 px".

      ⚠ La metrica es L-INFINITO, no euclidea, y no es un detalle: una ventana
      cuadrada de lado 2r+1 centrada en p contiene a q si |px-qx|<=r Y |py-qy|<=r.
      Con distancia euclidea, dos cajas separadas 14 px en diagonal (euclidea
      19,8) pasarian el filtro y la ventana SI las mezclaria. Tiene su prueba.

  MARGEN_DURO = RADIO = 9
      Distancia minima de la tinta al borde de la pagina. Una convolucion `valid`
      con k=19 sobre una pagina de L px devuelve L-18 px, y su pixel (i,j) mira
      al (i+9, j+9) de la entrada: o sea que la salida solo cubre las coordenadas
      [9, L-10]. Un borde a menos de 9 px del papel NO EXISTE en la imagen
      resultante -- es el "que la imagen resultante aun vea algo del borde".

Y dos numeros que NO son minimos sino la HOLGURA con la que se construye, mas
generosos a proposito porque las ventanas de entrenamiento se recortan despues:

  SEPARACION = 40   se construye con esto; el minimo duro es 19
  MARGEN     = 32   se construye con esto; el minimo duro es 9

      MARGEN = 32 tiene su propio motivo: es lo que hace que una ventana de
      65 x 65 centrada en CUALQUIER punto del borde de un parrafo quepa entera
      dentro del papel. Con el minimo duro de 9 solo cabrian ventanas de 19.

POR QUE SE CONSTRUYE Y NO SE SORTEA-Y-RECHAZA
=============================================
El reparto en celdas GARANTIZA la separacion: dos parrafos de celdas distintas
estan separados por al menos una linea de corte, y cada uno se mete SEPARACION/2
hacia dentro de su celda, asi que la suma es SEPARACION. No hace falta rechazar
nada, y por eso el descarte que pide el encargo acaba siendo la red de seguridad
y no el mecanismo.

Sortear posiciones libres y rechazar las que chocan tiene un sesgo conocido y
caro: elimina selectivamente los parrafos GRANDES (los que menos sitio libre
encuentran), y el tamano es justo el factor que hay que variar. Aqui el tamano
se sortea DENTRO de la celda ya reservada, asi que un parrafo grande no compite
con nadie por su hueco.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------- constantes
K_MAX = 19
RADIO = (K_MAX - 1) // 2                  # 9
SEPARACION_DURA = 2 * RADIO + 1           # 19
MARGEN_DURO = RADIO                       # 9

LIENZO = 1024
SEPARACION = 40                           # holgura de construccion
MARGEN = 32                               # holgura de construccion

ANCHO_MIN = 120.0                         # un parrafo mas estrecho no tiene borde util
ALTO_MIN = 45.0
CELDA_MIN_W = ANCHO_MIN + SEPARACION      # 160: la celda tiene que dar para el
CELDA_MIN_H = ALTO_MIN + SEPARACION       # 85   parrafo minimo MAS su medio hueco
CORTE = (0.35, 0.65)                      # fraccion del lado por donde se parte


# --------------------------------------------------------------- primitivas
def solapan(a, b) -> bool:
    """¿Se tocan o se cruzan las dos cajas (x0, y0, x1, y1)?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def separacion(a, b) -> float:
    """Distancia L-INFINITO minima entre dos cajas. 0 si se solapan.

    Es la metrica correcta para una ventana CUADRADA: una de lado 2r+1 centrada
    en un punto de `a` alcanza un punto de `b` exactamente cuando esta distancia
    es <= r. Ver la cabecera."""
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return max(dx, dy)


def margen(caja, lienzo: int = LIENZO) -> float:
    """Distancia de la caja al borde del papel, por el lado mas justo."""
    return min(caja[0], caja[1], lienzo - caja[2], lienzo - caja[3])


def ventana_limpia(caja, otras, lienzo: int = LIENZO) -> int:
    """El lado IMPAR de la ventana mas grande que se puede centrar en CUALQUIER
    punto del borde de `caja` sin salirse del papel y sin ver tinta de `otras`.

    Es lo que contesta «¿cuanto se puede recortar?» sin tener que mirar la
    imagen: media ventana no puede pasar del margen (o se sale del papel) ni
    llegar al vecino mas cercano (o lo mete dentro)."""
    h = margen(caja, lienzo)
    for o in otras:
        h = min(h, separacion(caja, o) - 1)
    return max(0, 2 * int(math.floor(h)) + 1)


# --------------------------------------------------------------- reparto
def _puede_partirse(c) -> bool:
    return (c[2] - c[0]) >= 2 * CELDA_MIN_W or (c[3] - c[1]) >= 2 * CELDA_MIN_H


def _partir(c, rng):
    """Un corte de guillotina. Parte por el lado con mas holgura relativa."""
    w, h = c[2] - c[0], c[3] - c[1]
    puede_v = w >= 2 * CELDA_MIN_W
    puede_h = h >= 2 * CELDA_MIN_H
    if puede_v and puede_h:
        vertical = (w / CELDA_MIN_W) >= (h / CELDA_MIN_H)
    else:
        vertical = puede_v
    if vertical:
        lo, hi = c[0] + max(CELDA_MIN_W, w * CORTE[0]), c[0] + min(w - CELDA_MIN_W, w * CORTE[1])
        x = rng.uniform(lo, hi) if hi > lo else (lo + hi) / 2
        return (c[0], c[1], x, c[3]), (x, c[1], c[2], c[3])
    lo, hi = c[1] + max(CELDA_MIN_H, h * CORTE[0]), c[1] + min(h - CELDA_MIN_H, h * CORTE[1])
    y = rng.uniform(lo, hi) if hi > lo else (lo + hi) / 2
    return (c[0], c[1], c[2], y), (c[0], y, c[2], c[3])


def celdas(n: int, rng, lienzo: int = LIENZO):
    """Reparte el area util del papel en <= n celdas disjuntas de guillotina.

    Devuelve menos de `n` si el papel no da para mas SIN bajar del parrafo
    minimo. Devolver celdas impartibles seria peor: parrafos que no caben."""
    out = [(float(MARGEN), float(MARGEN), float(lienzo - MARGEN), float(lienzo - MARGEN))]
    while len(out) < n:
        cand = [i for i, c in enumerate(out) if _puede_partirse(c)]
        if not cand:
            break
        # se parte la mas grande: mantiene las celdas parecidas y evita que un
        # corte temprano deje una tira inservible
        i = max(cand, key=lambda j: (out[j][2] - out[j][0]) * (out[j][3] - out[j][1]))
        a, b = _partir(out[i], rng)
        out[i : i + 1] = [a, b]
    return out


def ranura(celda):
    """La celda menos SEPARACION/2 por cada lado: donde puede caer la TINTA.

    Aqui esta la garantia entera del modulo. Dos celdas distintas de un reparto
    de guillotina estan separadas por alguna linea de corte; si cada parrafo se
    queda a SEPARACION/2 de todos los bordes de SU celda, dos parrafos quedan a
    SEPARACION o mas. No hay que comprobarlo por pares: sale de la construccion.
    """
    m = SEPARACION / 2
    return (celda[0] + m, celda[1] + m, celda[2] - m, celda[3] - m)


def colocar(ranura_, w: float, h: float, ux: float, uy: float):
    """Caja de tinta de tamano (w, h) dentro de `ranura_`, en la fraccion (ux, uy).

    Devuelve None si no cabe: quien llama tiene que encoger el parrafo, no
    moverlo fuera."""
    libre_x = (ranura_[2] - ranura_[0]) - w
    libre_y = (ranura_[3] - ranura_[1]) - h
    if libre_x < 0 or libre_y < 0:
        return None
    x = ranura_[0] + ux * libre_x
    y = ranura_[1] + uy * libre_y
    return (x, y, x + w, y + h)


# --------------------------------------------------------------- aserciones
def revisar(cajas, lienzo: int = LIENZO):
    """TODO lo que una pagina tiene que cumplir. Devuelve (fallos, medidas).

    `fallos` vacio = la pagina se guarda. Se comprueba sobre la tinta REAL
    medida en el render, nunca sobre la prevista: la prevista es la que se
    quiso, y lo que contamina una medicion es la que salio."""
    fallos = []
    for i, c in enumerate(cajas):
        if c[2] - c[0] < ANCHO_MIN - 0.5:
            fallos.append(f"p{i}: ancho {c[2]-c[0]:.1f} < {ANCHO_MIN}")
        if c[3] - c[1] < ALTO_MIN - 0.5:
            fallos.append(f"p{i}: alto {c[3]-c[1]:.1f} < {ALTO_MIN}")
        m = margen(c, lienzo)
        if m < MARGEN_DURO:
            fallos.append(f"p{i}: margen {m:.1f} < {MARGEN_DURO} (borde inalcanzable tras k={K_MAX})")
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            if solapan(cajas[i], cajas[j]):
                fallos.append(f"p{i}-p{j}: SE SOLAPAN")
                continue
            s = separacion(cajas[i], cajas[j])
            if s < SEPARACION_DURA:
                fallos.append(f"p{i}-p{j}: separacion {s:.1f} < {SEPARACION_DURA} (k={K_MAX} los mezcla)")
    medidas = {
        "separacion_min": round(min(
            (separacion(cajas[i], cajas[j])
             for i in range(len(cajas)) for j in range(i + 1, len(cajas))),
            default=float(lienzo)), 2),
        "margen_min": round(min(margen(c, lienzo) for c in cajas), 2),
        "ventana_limpia_min": min(
            (ventana_limpia(c, [o for k, o in enumerate(cajas) if k != i], lienzo)
             for i, c in enumerate(cajas)), default=0),
    }
    return fallos, medidas


# --------------------------------------------------------------- comprobacion
def _comprobar() -> int:
    import random
    ok = fail = 0

    def es(cond, que):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  ✗ {que}")

    # --- las constantes salen de k=19, no de gusto
    es(RADIO == 9, "RADIO = (19-1)/2 = 9")
    es(SEPARACION_DURA == 19, "SEPARACION_DURA = 2*9+1 = 19")
    es(MARGEN_DURO == 9, "MARGEN_DURO = 9")
    es(SEPARACION >= SEPARACION_DURA, "la holgura de construccion no baja del minimo duro")
    es(MARGEN >= MARGEN_DURO, "el margen de construccion no baja del minimo duro")

    # --- separacion: L-infinito, no euclidea. La prueba que distingue las dos.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (24.0, 24.0, 40.0, 40.0)              # dx = dy = 14
    es(abs(separacion(a, b) - 14.0) < 1e-9, "diagonal: L-inf = max(14,14) = 14")
    es(math.hypot(14, 14) > SEPARACION_DURA, "...y su euclidea (19,8) SI pasaria de 19")
    es(separacion(a, b) < SEPARACION_DURA, "...asi que L-inf la rechaza y la euclidea no")
    es(separacion((0, 0, 10, 10), (30, 0, 40, 10)) == 20.0, "solo horizontal: 20")
    es(separacion((0, 0, 10, 10), (0, 30, 10, 40)) == 20.0, "solo vertical: 20")
    es(separacion((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0, "solapadas: 0")

    # --- solape
    es(solapan((0, 0, 10, 10), (5, 5, 15, 15)), "solape detectado")
    es(not solapan((0, 0, 10, 10), (10, 0, 20, 10)), "pegadas no es solape")

    # --- margen y ventana limpia
    es(margen((32, 32, 992, 992), 1024) == 32.0, "margen = 32 con el papel de 1024")
    es(ventana_limpia((32, 32, 200, 200), [], 1024) == 65, "sin vecinos: 2*32+1 = 65")
    es(ventana_limpia((100, 100, 200, 200), [(240, 100, 400, 200)], 1024) == 79,
       "vecino a 40 y lejos del papel: manda el vecino, media ventana 39 -> 79")
    es(ventana_limpia((32, 32, 200, 200), [(240, 32, 400, 200)], 1024) == 65,
       "el mismo vecino pero pegado al papel: manda el margen, 32 -> 65")
    es(ventana_limpia((9, 9, 200, 200), [], 1024) == 19,
       "con el margen duro justo cabe la ventana de k=19 y nada mas")

    # --- reparto: la garantia estructural, sobre 400 paginas al azar
    rng = random.Random(7)
    peor_sep, peor_mar, celdas_pedidas, celdas_dadas = 1e9, 1e9, 0, 0
    for _ in range(400):
        n = rng.choice([2, 3, 4, 5])
        cs = celdas(n, rng)
        celdas_pedidas += n
        celdas_dadas += len(cs)
        cajas = []
        for c in cs:
            r = ranura(c)
            w = max(ANCHO_MIN, (r[2] - r[0]) * rng.uniform(0.45, 1.0))
            h = max(ALTO_MIN, (r[3] - r[1]) * rng.uniform(0.30, 1.0))
            caja = colocar(r, w, h, rng.random(), rng.random())
            if caja is not None:
                cajas.append(caja)
        f, m = revisar(cajas)
        if f:
            fail += 1
            print(f"  ✗ reparto n={n}: {f[:2]}")
            break
        peor_sep = min(peor_sep, m["separacion_min"])
        peor_mar = min(peor_mar, m["margen_min"])
    else:
        ok += 1
    es(peor_sep >= SEPARACION - 1e-6,
       f"400 paginas: separacion real siempre >= {SEPARACION} (peor: {peor_sep:.1f})")
    es(peor_mar >= MARGEN - 1e-6,
       f"400 paginas: margen real siempre >= {MARGEN} (peor: {peor_mar:.1f})")
    es(celdas_dadas == celdas_pedidas,
       f"el papel da para todas las celdas pedidas ({celdas_dadas}/{celdas_pedidas})")

    # --- revisar() tiene que CAZAR lo que no cumple, no solo aprobar lo que si
    es(revisar([(32, 32, 200, 200), (100, 100, 300, 300)])[0], "caza un solape")
    es(revisar([(32, 32, 200, 200), (210, 32, 400, 200)])[0],
       "caza una separacion de 10 < 19")
    es(revisar([(2, 2, 200, 200)])[0], "caza un margen de 2 < 9")
    es(revisar([(32, 32, 100, 200)])[0], f"caza un ancho de 68 < {ANCHO_MIN}")
    es(not revisar([(32, 32, 200, 200), (240, 32, 400, 200)])[0],
       "aprueba una pagina que si cumple")

    print(f"\n{ok} bien · {fail} mal")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_comprobar())
