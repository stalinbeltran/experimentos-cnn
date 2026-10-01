#!/usr/bin/env python3
"""La geometria de una pagina de `bor-p4`: reparto en celdas, colocacion y las ASERCIONES.

    python nn/geometria.py          corre sus comprobaciones y sale 0 o 1

Este modulo NO renderiza y NO importa nada del repo: es aritmetica pura sobre
cajas. Asi se puede comprobar la parte que decide si el dataset sirve --
separacion, margen y ventana limpia -- en milisegundos y sin navegador.

Copiado de `bor-p` el 2026-10-01 y CAMBIADO a proposito en una cosa: aqui hay DOS
escalas. La pagina se RINDE a 1024 y se GUARDA reducida /4 (256), porque el banco
`banco-k` aplica el kernel despues de reducir /4 y ahi es donde tiene que ser
coherente la escala de la tinta (cuerpo 11-30 px en render -> 2,75-7,5 px en lo
que ve el kernel, igual que en el banco).

DE DONDE SALEN LOS NUMEROS (todos derivados de k=19 A LA ESCALA GUARDADA)
=========================================================================

El kernel mas grande que se va a entrenar es 19 px, y corre sobre la pagina
GUARDADA (/4). Todo lo demas se deriva de eso, y se multiplica por la reduccion
para llevarlo a pixeles de render, que es donde se construye la pagina:

  RADIO = (19 - 1) / 2 = 9                         (px guardados)

  SEPARACION_DURA_R4 = 2 * RADIO + 1 = 19          (px guardados)
  SEPARACION_DURA    = 19 * 4 = 76                 (px de render)
      Separacion L-infinito MINIMA entre la tinta de dos parrafos: la ventana
      19x19 centrada en el borde de A y la centrada en el borde de B no se
      solapan, y ninguna ve tinta del otro parrafo.

      ⚠ La metrica es L-INFINITO, no euclidea: una ventana cuadrada de lado 2r+1
      centrada en p contiene a q si |px-qx|<=r Y |py-qy|<=r. Tiene su prueba.

  MARGEN_DURO_R4 = RADIO = 9                       (px guardados)
  MARGEN_DURO    = 9 * 4 = 36                      (px de render)
      Una convolucion `valid` con k=19 sobre la pagina guardada de 256 solo
      cubre [9, 246]: un borde mas cerca del papel NO EXISTE en la salida.

Y dos numeros que NO son minimos sino la HOLGURA con la que se construye. Son los
de `bor-p` (40 y 32) por cuatro, que es lo que el plan del 2026-10-01 llama
«geometria x4»:

  SEPARACION = 160   en render (40 guardados); el minimo duro es 76 (19)
  MARGEN     = 128   en render (32 guardados); el minimo duro es 36 (9)

      Con eso, la tinta queda a >= MARGEN + SEPARACION/2 = 208 px del papel (52
      guardados) y a >= 160 de otro parrafo (40 guardados), asi que la ventana
      limpia mas pequena del dataset sale de 2*39+1 = 79 px GUARDADOS: es el
      lado de ventana que se puede recortar alrededor de cualquier borde sin
      filtrar ningun parrafo.

LO QUE NO SE MULTIPLICA, y por que
==================================
ANCHO_MIN = 120 y ALTO_MIN = 45 se quedan en px de RENDER, como en `bor-p`. No
son del kernel sino de la TIPOGRAFIA: un parrafo mas estrecho no tiene lineas que
justificar. Guardados son 30 x 11 px, que esta dentro de lo que ve el banco (sus
parrafos van de 20 a 105 px de ancho a /4; leido de su manifiesto el 2026-10-01).

POR QUE SE CONSTRUYE Y NO SE SORTEA-Y-RECHAZA
=============================================
Igual que en `bor-p`: el reparto en celdas GARANTIZA la separacion, asi que no
hace falta rechazar nada, y rechazar eliminaria selectivamente los parrafos
GRANDES -- justo el factor que hay que variar.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------- constantes
REDUCCION = 4
LIENZO = 1024                             # px de render
GUARDADO = LIENZO // REDUCCION            # 256: la pagina que se guarda

# a la escala GUARDADA, que es donde corre el kernel
K_MAX = 19
RADIO = (K_MAX - 1) // 2                  # 9
SEPARACION_DURA_R4 = 2 * RADIO + 1        # 19
MARGEN_DURO_R4 = RADIO                    # 9

# los mismos minimos en px de RENDER, que es donde se construye la pagina
SEPARACION_DURA = SEPARACION_DURA_R4 * REDUCCION   # 76
MARGEN_DURO = MARGEN_DURO_R4 * REDUCCION           # 36

SEPARACION = 160                          # holgura de construccion (render)
MARGEN = 128                              # holgura de construccion (render)

ANCHO_MIN = 120.0                         # render: tipografia, no kernel
ALTO_MIN = 45.0
CELDA_MIN_W = ANCHO_MIN + SEPARACION      # 280: la celda tiene que dar para el
CELDA_MIN_H = ALTO_MIN + SEPARACION       # 205  parrafo minimo MAS su medio hueco
CORTE = (0.35, 0.65)                      # fraccion del lado por donde se parte


# --------------------------------------------------------------- primitivas
def solapan(a, b) -> bool:
    """¿Se tocan o se cruzan las dos cajas (x0, y0, x1, y1)?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def separacion(a, b) -> float:
    """Distancia L-INFINITO minima entre dos cajas. 0 si se solapan."""
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return max(dx, dy)


def margen(caja, lienzo: int = LIENZO) -> float:
    """Distancia de la caja al borde del papel, por el lado mas justo."""
    return min(caja[0], caja[1], lienzo - caja[2], lienzo - caja[3])


def ventana_limpia(caja, otras, lienzo: int = LIENZO) -> int:
    """El lado IMPAR de la ventana mas grande que se puede centrar en CUALQUIER
    punto del borde de `caja` sin salirse del papel y sin ver tinta de `otras`.

    Vale en cualquier escala: se le pasan las cajas y el lienzo en la MISMA. Para
    el dataset se calcula en la GUARDADA (`a_guardado`), que es donde se recorta."""
    h = margen(caja, lienzo)
    for o in otras:
        h = min(h, separacion(caja, o) - 1)
    return max(0, 2 * int(math.floor(h)) + 1)


def a_guardado(caja):
    """Caja de render -> caja en px de la pagina guardada. La MISMA aritmetica que
    usa el banco para sus etiquetas (coord / 4): sin redondeo, porque el borde es
    una coordenada continua y redondearla aqui seria decidir por quien la lea."""
    return tuple(float(v) / REDUCCION for v in caja)


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
        i = max(cand, key=lambda j: (out[j][2] - out[j][0]) * (out[j][3] - out[j][1]))
        a, b = _partir(out[i], rng)
        out[i : i + 1] = [a, b]
    return out


def ranura(celda):
    """La celda menos SEPARACION/2 por cada lado: donde puede caer la TINTA.

    Dos celdas distintas de un reparto de guillotina estan separadas por alguna
    linea de corte; si cada parrafo se queda a SEPARACION/2 de todos los bordes de
    SU celda, dos parrafos quedan a SEPARACION o mas. Sale de la construccion."""
    m = SEPARACION / 2
    return (celda[0] + m, celda[1] + m, celda[2] - m, celda[3] - m)


def colocar(ranura_, w: float, h: float, ux: float, uy: float):
    """Caja de tinta de tamano (w, h) dentro de `ranura_`, en la fraccion (ux, uy).
    None si no cabe: quien llama tiene que encoger el parrafo, no moverlo fuera."""
    libre_x = (ranura_[2] - ranura_[0]) - w
    libre_y = (ranura_[3] - ranura_[1]) - h
    if libre_x < 0 or libre_y < 0:
        return None
    x = ranura_[0] + ux * libre_x
    y = ranura_[1] + uy * libre_y
    return (x, y, x + w, y + h)


# --------------------------------------------------------------- aserciones
def revisar(cajas, lienzo: int = LIENZO):
    """TODO lo que una pagina tiene que cumplir, con las cajas en px de RENDER.
    Devuelve (fallos, medidas); `fallos` vacio = la pagina se guarda.

    Las medidas salen en px GUARDADOS (/4), que es la escala donde se recortan las
    ventanas: un `ventana_limpia_min` en px de render se leeria cuatro veces mas
    grande de lo que es."""
    fallos = []
    for i, c in enumerate(cajas):
        if c[2] - c[0] < ANCHO_MIN - 0.5:
            fallos.append(f"p{i}: ancho {c[2]-c[0]:.1f} < {ANCHO_MIN}")
        if c[3] - c[1] < ALTO_MIN - 0.5:
            fallos.append(f"p{i}: alto {c[3]-c[1]:.1f} < {ALTO_MIN}")
        m = margen(c, lienzo)
        if m < MARGEN_DURO:
            fallos.append(f"p{i}: margen {m:.1f} < {MARGEN_DURO} (borde inalcanzable "
                          f"tras k={K_MAX} a /{REDUCCION})")
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            if solapan(cajas[i], cajas[j]):
                fallos.append(f"p{i}-p{j}: SE SOLAPAN")
                continue
            s = separacion(cajas[i], cajas[j])
            if s < SEPARACION_DURA:
                fallos.append(f"p{i}-p{j}: separacion {s:.1f} < {SEPARACION_DURA} "
                              f"(k={K_MAX} a /{REDUCCION} los mezcla)")
    g = [a_guardado(c) for c in cajas]
    gl = lienzo / REDUCCION
    medidas = {
        "separacion_min_r4": round(min(
            (separacion(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))),
            default=float(gl)), 2),
        "margen_min_r4": round(min(margen(c, gl) for c in g), 2),
        "ventana_limpia_min_r4": min(
            (ventana_limpia(c, [o for k, o in enumerate(g) if k != i], gl)
             for i, c in enumerate(g)), default=0),
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

    # --- las constantes salen de k=19 a la escala guardada, no de gusto
    es(GUARDADO == 256, "1024 / 4 = 256")
    es(RADIO == 9, "RADIO = (19-1)/2 = 9")
    es(SEPARACION_DURA_R4 == 19 and SEPARACION_DURA == 76, "separacion dura 19 guardados = 76 render")
    es(MARGEN_DURO_R4 == 9 and MARGEN_DURO == 36, "margen duro 9 guardados = 36 render")
    es(SEPARACION >= SEPARACION_DURA, "la holgura de construccion no baja del minimo duro")
    es(MARGEN >= MARGEN_DURO, "el margen de construccion no baja del minimo duro")
    es(SEPARACION == 4 * 40 and MARGEN == 4 * 32, "geometria x4 de la de bor-p (40, 32)")

    # --- separacion: L-infinito, no euclidea
    a = (0.0, 0.0, 10.0, 10.0)
    b = (24.0, 24.0, 40.0, 40.0)              # dx = dy = 14
    es(abs(separacion(a, b) - 14.0) < 1e-9, "diagonal: L-inf = max(14,14) = 14")
    es(math.hypot(14, 14) > SEPARACION_DURA_R4, "...y su euclidea (19,8) SI pasaria de 19")
    es(separacion(a, b) < SEPARACION_DURA_R4, "...asi que L-inf la rechaza y la euclidea no")
    es(separacion((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0, "solapadas: 0")
    es(solapan((0, 0, 10, 10), (5, 5, 15, 15)), "solape detectado")
    es(not solapan((0, 0, 10, 10), (10, 0, 20, 10)), "pegadas no es solape")

    # --- las dos escalas
    es(a_guardado((36, 40, 100, 1000)) == (9.0, 10.0, 25.0, 250.0), "a_guardado divide por 4 sin redondear")
    es(margen(a_guardado((36, 36, 988, 988)), GUARDADO) == 9.0,
       "el margen duro de render (36) es exactamente el de la guardada (9)")
    es(ventana_limpia(a_guardado((36, 36, 400, 400)), [], GUARDADO) == 19,
       "con el margen duro justo, a /4 cabe la ventana de k=19 y nada mas")
    es(ventana_limpia((52, 52, 100, 100), [(140, 52, 200, 100)], GUARDADO) == 79,
       "lo que da la construccion: vecino a 40 guardados -> media ventana 39 -> 79")

    # --- reparto: la garantia estructural, sobre 400 paginas al azar
    rng = random.Random(7)
    peor_sep, peor_mar, peor_ven = 1e9, 1e9, 10 ** 9
    celdas_pedidas = celdas_dadas = 0
    for _ in range(400):
        n = rng.choice([2, 3, 4])
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
        peor_sep = min(peor_sep, m["separacion_min_r4"])
        peor_mar = min(peor_mar, m["margen_min_r4"])
        peor_ven = min(peor_ven, m["ventana_limpia_min_r4"])
    else:
        ok += 1
    es(peor_sep >= SEPARACION / REDUCCION - 1e-6,
       f"400 paginas: separacion guardada siempre >= {SEPARACION // REDUCCION} (peor: {peor_sep:.1f})")
    es(peor_mar >= (MARGEN + SEPARACION / 2) / REDUCCION - 1e-6,
       f"400 paginas: margen guardado siempre >= 52 (peor: {peor_mar:.1f})")
    es(peor_ven >= 79, f"400 paginas: ventana limpia guardada siempre >= 79 (peor: {peor_ven})")
    es(celdas_dadas == celdas_pedidas,
       f"con 2-4 parrafos el papel da para todas las celdas ({celdas_dadas}/{celdas_pedidas})")

    # --- revisar() tiene que CAZAR lo que no cumple, no solo aprobar lo que si
    es(revisar([(200, 200, 400, 400), (300, 300, 500, 500)])[0], "caza un solape")
    es(revisar([(200, 200, 400, 400), (450, 200, 700, 400)])[0],
       "caza una separacion de 50 < 76 (que a 1024 sin reducir SI valdria)")
    es(revisar([(20, 200, 400, 400)])[0], "caza un margen de 20 < 36")
    es(revisar([(200, 200, 300, 400)])[0], f"caza un ancho de 100 < {ANCHO_MIN}")
    es(not revisar([(208, 208, 400, 400), (560, 208, 800, 400)])[0],
       "aprueba una pagina que si cumple")

    print(f"\n{ok} bien · {fail} mal")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_comprobar())
