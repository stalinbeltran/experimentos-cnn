#!/usr/bin/env python3
"""El dataset de `bor-p4`: las paginas de `bor-p` A LA ESCALA DEL BANCO (/4), y la caja de cada parrafo.

Copiado de `bor-p` el 2026-10-01. Lo que cambia, y es todo a proposito:
  - la geometria es x4 (separacion 160, margen 128 en render) porque el kernel corre
    sobre la pagina REDUCIDA: ver `nn/geometria.py`;
  - la pagina se GUARDA reducida /4 como uint16 con la SUMA de cada bloque 4x4, que es
    el metodo del §3.8 de `banco-k` (exacto, sin cuantizar);
  - 2-4 parrafos por pagina, no 2-5: con la geometria x4, 5 parrafos no caben en el 22 %
    de las paginas (medido el 2026-10-01 sobre 2000 repartos simulados), y bajar la
    densidad en silencio dejaria el dataset por debajo de los 1000 parrafos del nombre.

    python nn/generar_paginas.py --parrafos 1000    genera en la etapa local (datos/)
    python nn/generar_paginas.py --publicar         lo publica en el repo de datos (una vez)
    python nn/generar_paginas.py --comprobar        casa las huellas del publicado
    python nn/generar_paginas.py --rederivar N      rehace las N primeras y compara
    python nn/generar_paginas.py --muestras N       figura PNG con N paginas

⚠ SE LLAMA ASI PARA EL FRENO, no por estetica. `telegram-coordinator/scripts/
cerrable.mjs` casa `generar_paginas\\.py` en su lista declarada TRABAJOS: son ~20 min
de renders, y sin ese nombre el veredicto «¿se puede apagar este server?» diria «nada
corriendo» a mitad de la generacion. La lista se DECLARA, no se deduce.

LO QUE HAY QUE ENTENDER ANTES DE TOCAR ESTO
===========================================

1. LA SEPARACION SE CONSTRUYE, NO SE SORTEA-Y-RECHAZA. `nn/geometria.py` reparte el
   papel en celdas de guillotina y mete cada parrafo SEPARACION/2 hacia dentro de la
   suya, asi que dos parrafos quedan a SEPARACION o mas POR CONSTRUCCION. El descarte
   que pide el encargo existe igual, pero como red de seguridad: se comprueba la tinta
   REAL de cada pagina renderizada y la que falle se tira entera.

   Sortear y rechazar habria sido mas corto y esta descartado con motivo: elimina
   selectivamente los parrafos GRANDES, que son los que menos hueco libre encuentran,
   y el tamano es justo el factor que el encargo manda variar.

2. EL `avoid_overlap` DEL GENERADOR NO SE USA, Y NO ES DESCONFIANZA: su propio codigo
   dice que el margen es «a preference, not a hard constraint» (resolver.py:394) y que
   cuando no cabe nada conserva «el reparto menos malo en vez de fallar»
   (SAMPLE_FORMAT.md §5: ~1% de solape). El encargo pide que NUNCA se solapen.

3. EL ALTO DE UN PARRAFO NO SE PUEDE PEDIR: emerge del ancho, la fuente, el cuerpo, el
   interlineado y el numero de palabras. Se sortea el alto OBJETIVO y se DERIVA el
   numero de palabras con

       lineas   = alto / (cuerpo * interlineado)        <- EXACTO, medido
       palabras = lineas * C(fuente) * ancho / cuerpo   <- aproximado

   El paso de linea resulto ser exactamente `cuerpo * interlineado` (medido el
   2026-09-09: 12x1.25 -> 15.00 px, 18x1.25 -> 22.50, 28x1.25 -> 35.00). C se midio el
   mismo dia, 3 anchos x 4 fuentes, y es estable dentro de cada fuente (CONST_PALABRA).

4. POR ESO CADA PAGINA SE RENDERIZA DOS VECES, y no es desperdicio:
     pase A  los N parrafos APILADOS en (0,0) -> se mide la caja real de cada uno
     pase B  cada uno en la posicion calculada PARA esa caja -> es la pagina que se guarda
   El pase A puede apilarlos porque las etiquetas salen del layout, no de los pixeles:
   medido el 2026-09-09, un bloque apilado sobre otro reporta EXACTAMENTE la misma caja
   que separado, y el lienzo tampoco recorta la etiqueta. Asi el pase A es UN render por
   pagina en vez de uno por parrafo.

5. MOVER UN PARRAFO NO CAMBIA SU TINTA, y de eso depende que el doble pase sea exacto y
   no una aproximacion. Medido el 2026-09-09: el mismo bloque desplazado (+37, +53) da
   dw = dh = 0.00. Por eso lo medido en el pase A vale tal cual en el pase B.

6. LA ETIQUETA ES LA CAJA DE TINTA, no la de maquetacion: la union de las cajas de
   PALABRA. Y por eso `align: justify`, que es una decision de ESTE experimento con su
   motivo: con alineacion a la izquierda el borde derecho lo marca el final ragged de
   cada linea y la tinta NO llega al borde de la caja, asi que ese borde no tendria
   evidencia visual y el kernel buscaria algo que no esta ahi.

7. LAS PAGINAS SE GUARDAN REDUCIDAS, como un array (P, 256, 256) uint16 dentro del
   .npz. A /4 caben de sobra en RAM (~45 MB para 350 paginas), asi que ya no hace falta
   el truco de los PNG concatenados de `bor-p`. Cada pagina se escribe a disco (`.npy`)
   EN CUANTO se rinde: una caida a los 12 minutos no pierde lo hecho hasta ahi.

8. LAS CAJAS SE GUARDAN EN PX DE RENDER (1024), enteras, como en `bor-p` y como el banco
   guarda las suyas en su marco de 584. En la pagina guardada valen `coord / 4`, sin
   redondear (`geometria.a_guardado`). Las DERIVADAS (margen, separacion, ventana limpia)
   en cambio van en px GUARDADOS, porque es ahi donde se recortan las ventanas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(AQUI))

import geometria as G  # noqa: E402

from expcnn import (SUBDIR_DATASETS, exigir_datos, exigir_generador,  # noqa: E402
                    ruta_dataset)

# ---------------------------------------------------------------- constantes
NOMBRE = "parrafos1000-pagina1024-r4-r20261001"
SEMILLA = 1
N_PARRAFOS = 1000
PARRAFOS_POR_PAGINA = (2, 3, 4)
REPARTO = {"train": 0.70, "val": 0.10, "eval": 0.20}
DATOS = EXP / "datos"
PAGINAS = DATOS / "paginas"

RECETA = json.loads((AQUI / "receta.json").read_text(encoding="utf-8"))
FUENTES = RECETA["fonts"]
CUERPO = tuple(RECETA["blocks"][0]["typography"]["font_size"]["range"])
INTERLINEADO = tuple(RECETA["blocks"][0]["typography"]["line_height"]["range"])
# Topes de la receta. `ancho` y `palabras` se DERIVAN de la celda, asi que estos son
# el tope contra el que se recorta -- y se comprueban al empaquetar. La primera
# version declaraba ancho <= 460 y se generaron de 887: un rango que no lee nadie es
# decoracion que se lee como especificacion (R15).
ANCHO_RECETA = tuple(RECETA["blocks"][0]["width"]["range"])
PALABRAS_RECETA = tuple(RECETA["blocks"][0]["content"]["words"]["range"])

# §3.7 de `banco-k`: reservado para el banco, PROHIBIDO aqui. Se comprueba, no se
# supone: la receta es un fichero editable y un descuido aqui no falla por ningun
# lado -- solo hace optimista cualquier kernel que salga de este dataset.
RESERVA_FUENTE = "LiberationMono"
RESERVA_INTERLINEADO = (1.45, 1.60)

# Palabras por linea normalizadas: wpl * cuerpo / ancho. MEDIDO el 2026-09-09 con
# 3 anchos (200/300/440) x 4 fuentes; estable dentro de cada fuente (0,273-0,288 en
# DejaVuSans, 0,360-0,364 en LiberationSerif). Es un PREDICTOR: el pase A mide la
# caja de verdad y el reparo corrige lo que haga falta.
CONST_PALABRA = {"DejaVuSans": 0.283, "DejaVuSerif": 0.276,
                 "LiberationSans": 0.324, "LiberationSerif": 0.361}

FRAC_ANCHO = (0.45, 1.00)      # del ancho de la ranura
FRAC_ALTO = (0.30, 0.92)       # del alto; el tope deja aire al error del predictor
MAX_REPAROS = 5
# 10 y no 3 (2026-10-01): con 3, el primer render perdio 6 paginas enteras y el dataset
# se quedaba por debajo de los 1000 parrafos de su nombre. Los intentos son
# deterministas (la semilla sale del indice de pagina y del intento), asi que subirlo
# no cambia ninguna pagina que ya saliera antes.
MAX_REINTENTOS_PAGINA = 10


# ---------------------------------------------------------------- la reserva
def comprobar_reserva() -> list[str]:
    """§3.7: que la receta NO alcance lo que `banco-k` se reservo.

    Omitir una de las dos claves es filtrar, no es neutro: sin `fonts` el generador
    sortea TODAS las familias registradas (incluida la reservada) y sin
    `line_height` su defecto es Range(1.15, 1.6), que solapa con [1.45, 1.60]."""
    malos = []
    fuentes = RECETA.get("fonts")
    if not fuentes:
        malos.append("receta.json no declara `fonts`: el generador sortearia TODAS "
                     f"las familias, incluida {RESERVA_FUENTE}")
    elif RESERVA_FUENTE in fuentes:
        malos.append(f"receta.json usa {RESERVA_FUENTE}, que es de `banco-k` (§3.7)")
    lh = RECETA["blocks"][0]["typography"].get("line_height")
    if not isinstance(lh, dict) or "range" not in lh:
        malos.append("receta.json no declara `line_height` como rango: el defecto del "
                     "generador es (1.15, 1.6) y SOLAPA con la reserva")
    else:
        lo, hi = lh["range"]
        if not (hi < RESERVA_INTERLINEADO[0] or lo > RESERVA_INTERLINEADO[1]):
            malos.append(f"receta.json usa interlineado [{lo}, {hi}], que solapa con "
                         f"la reserva {list(RESERVA_INTERLINEADO)} de `banco-k` (§3.7)")
    for f in RECETA["blocks"][0]["typography"].get("font_family") or []:
        if f == RESERVA_FUENTE:
            malos.append(f"el bloque fija font_family={RESERVA_FUENTE} (§3.7)")
    return malos


# ---------------------------------------------------------------- muestreo
def _palabras(alto: float, cuerpo: float, lh: float, ancho: float, fuente: str) -> int:
    lineas = max(1.0, alto / (cuerpo * lh))
    wpl = max(1.5, CONST_PALABRA[fuente] * ancho / cuerpo)
    return int(min(PALABRAS_RECETA[1], max(PALABRAS_RECETA[0], round(lineas * wpl))))


def plan(n_parrafos: int, rng: random.Random) -> list[int]:
    """Cuantos parrafos lleva cada pagina, sumando EXACTAMENTE n_parrafos."""
    out: list[int] = []
    quedan = n_parrafos
    while quedan > 0:
        posibles = [k for k in PARRAFOS_POR_PAGINA if k <= quedan]
        if not posibles:                    # el resto es 1: se suma a la ultima
            out[-1] += quedan
            break
        out.append(rng.choice(posibles))
        quedan -= out[-1]
    return out


def cuerpos_emparejados(ranuras, rng: random.Random) -> list[float]:
    """Un cuerpo por ranura: se SORTEAN los n cuerpos y el MAYOR va a la ranura MAS ANCHA.

    ⚠ Por que, medido el 2026-10-01 con el primer render de este dataset (sin emparejar):
    91 descartes y 6 paginas perdidas enteras -- 977 parrafos en vez de 1000 --, y el
    cuarto de cuerpo mas grande quedo un 11 % por debajo del uniforme ([270, 251, 237,
    217] por cuartos de [11, 30], ~244 cada uno). La causa: con la geometria x4 hay
    ranuras de 120 px de ancho, y ahi un cuerpo de 30 no mete ni las 6 palabras minimas
    sin pasarse de alto. Descartar esas paginas es SORTEAR-Y-RECHAZAR sobre el cuerpo,
    que es justo lo que las reglas de `bor-p` prohiben.

    Emparejar no cambia NINGUNA distribucion marginal: los cuerpos se sortean igual
    (uniformes en [11, 30]) y las ranuras salen del mismo reparto. Lo unico que cambia es
    QUE cuerpo cae en QUE ranura dentro de una pagina."""
    cuerpos = sorted((round(rng.uniform(*CUERPO), 1) for _ in ranuras), reverse=True)
    por_ancho = sorted(range(len(ranuras)), key=lambda i: ranuras[i][2] - ranuras[i][0],
                       reverse=True)
    salida = [0.0] * len(ranuras)
    for c, i in zip(cuerpos, por_ancho):
        salida[i] = c
    return salida


def factores(ranura, rng: random.Random, cuerpo: float) -> dict:
    """Los factores de UN parrafo, derivados de su ranura (§ variar tamanos). El cuerpo
    viene ya sorteado y emparejado (`cuerpos_emparejados`)."""
    rw, rh = ranura[2] - ranura[0], ranura[3] - ranura[1]
    fuente = rng.choice(FUENTES)
    lh = round(rng.uniform(*INTERLINEADO), 3)
    # el tope es el MENOR de: lo que da la ranura y lo que declara la receta
    ancho = max(ANCHO_RECETA[0], min(rw, ANCHO_RECETA[1], rw * rng.uniform(*FRAC_ANCHO)))
    alto = max(G.ALTO_MIN, min(rh * FRAC_ALTO[1], rh * rng.uniform(*FRAC_ALTO)))
    return {"fuente": fuente, "cuerpo": cuerpo, "interlineado": lh,
            "ancho": round(ancho, 1), "alto_objetivo": round(alto, 1),
            "palabras": _palabras(alto, cuerpo, lh, ancho, fuente),
            "ux": rng.random(), "uy": rng.random()}


# ---------------------------------------------------------------- render
def _receta(bloques, posiciones):
    from app.models.recipe import Recipe                        # noqa: PLC0415
    return Recipe.model_validate({
        "canvas": {"width": G.LIENZO, "height": G.LIENZO},
        "background": RECETA["background"],
        "fonts": FUENTES,
        "blocks": [{
            "kind": "paragraph", "count": 1, "width": b["ancho"],
            "content": {"source": "words", "lang": "es", "words": b["palabras"]},
            "typography": {
                "font_family": b["fuente"], "font_size": b["cuerpo"],
                "line_height": b["interlineado"], "color": "#000000",
                "min_contrast": 1.0, "align": "justify", "font_weight": 400,
            },
            "placement": {"x": float(x), "y": float(y), "angle": 0,
                          "avoid_overlap": False},
        } for b, (x, y) in zip(bloques, posiciones)],
    })


def _cajas_tinta(labels, n: int):
    """Union de las cajas de PALABRA, por bloque. None donde el bloque no puso tinta."""
    acc: dict[str, list[float]] = {}
    for w in labels.words:
        x0, y0, ww, hh = w.box
        c = acc.setdefault(w.block_id, [1e9, 1e9, -1e9, -1e9])
        c[0] = min(c[0], x0); c[1] = min(c[1], y0)
        c[2] = max(c[2], x0 + ww); c[3] = max(c[3], y0 + hh)
    return [tuple(acc[f"b{i}"]) if f"b{i}" in acc else None for i in range(n)]


async def _una_pagina(r, resolve, bloques, ranuras, semilla):
    """Los dos pases de UNA pagina. Devuelve (imagen, cajas, bloques) o None."""
    # --- pase A: apilados en (0,0), solo para MEDIR
    medidas = None
    for _ in range(MAX_REPAROS):
        res = await r.render(resolve(_receta(bloques, [(0, 0)] * len(bloques)), semilla))
        cajas = _cajas_tinta(res.labels, len(bloques))
        if any(c is None for c in cajas):
            for i, c in enumerate(cajas):
                if c is None:
                    bloques[i]["palabras"] = max(6, int(bloques[i]["palabras"] * 1.6))
            continue
        medidas = [(c[2] - c[0], c[3] - c[1], c[0], c[1]) for c in cajas]
        malos = False
        for i, (w, h, _dx, _dy) in enumerate(medidas):
            rw = ranuras[i][2] - ranuras[i][0]
            rh = ranuras[i][3] - ranuras[i][1]
            if h > rh or w > rw:                       # no cabe: se ENCOGE, no se tira
                bloques[i]["palabras"] = max(
                    6, int(bloques[i]["palabras"] * min(0.9, (rh / h) * 0.92)))
                malos = True
            elif h < G.ALTO_MIN:                       # demasiado corto para tener borde
                bloques[i]["palabras"] = int(bloques[i]["palabras"] * 1.5) + 2
                malos = True
        if not malos:
            break
    else:
        return None
    if medidas is None:
        return None

    # --- colocacion: cada caja en SU ranura, en la fraccion sorteada
    cajas, posiciones = [], []
    for i, (w, h, dx, dy) in enumerate(medidas):
        caja = G.colocar(ranuras[i], w, h, bloques[i]["ux"], bloques[i]["uy"])
        if caja is None:
            return None
        cajas.append(caja)
        # la tinta arranca desplazada respecto de la esquina del bloque (el ascendente
        # de la fuente): se corrige con lo medido, para que caiga donde se calculo
        posiciones.append((caja[0] - dx, caja[1] - dy))

    # --- pase B: la pagina que se guarda
    res = await r.render(resolve(_receta(bloques, posiciones), semilla))
    reales = _cajas_tinta(res.labels, len(bloques))
    if any(c is None for c in reales):
        return None
    return res.image.convert("L"), [tuple(c) for c in reales], bloques


def reducir(img) -> np.ndarray:
    """1024x1024 L -> 256x256 uint16 con la SUMA de cada bloque 4x4.

    Es el metodo del §3.8 de `banco-k` (`datos.py` de alli hace exactamente esto): la
    suma de 16 pixeles uint8 llega como mucho a 4080, cabe en uint16 y es EXACTA. El
    factor 16 frente al promedio no cambia nada que sea lineal y estandarizado, y por
    eso no se divide: dividir aqui seria cuantizar."""
    a = np.asarray(img.convert("L"), dtype=np.uint16)
    lado = G.GUARDADO
    assert a.shape == (G.LIENZO, G.LIENZO), f"render de {a.shape}, se esperaba {G.LIENZO}"
    return a.reshape(lado, G.REDUCCION, lado, G.REDUCCION).sum(axis=(1, 3)).astype(np.uint16)


TOLERANCIA_TINTA = 1   # bloques; ver tinta_fuera_de_cajas


def tinta_fuera_de_cajas(paginas: np.ndarray, cajas: np.ndarray,
                         tolerancia: int = TOLERANCIA_TINTA) -> list[str]:
    """¿Hay tinta en la pagina GUARDADA fuera de toda caja? Devuelve los fallos.

    Es la comprobacion que el dato reducido necesita y `bor-p` no: la etiqueta sale
    del layout (cajas de palabra) y la imagen de los pixeles, y entre las dos media
    una reduccion /4. Un bloque cuenta como tinta si su suma no es papel blanco puro
    (4080). La caja se lleva a bloques con floor/ceil y se le da `tolerancia` bloques
    por lado: las cajas se guardan redondeadas al px de render, asi que un borde en
    99,6 se guarda como 100 y la tinta del px 99 cae en el bloque de al lado. Lo que
    pase de esa tolerancia es una etiqueta que no describe la imagen."""
    blanco = 255 * G.REDUCCION ** 2
    lado = paginas.shape[-1]
    malos = []
    for p in range(len(paginas)):
        tinta = paginas[p] < blanco
        dentro = np.zeros_like(tinta)
        for r in cajas[cajas[:, 0] == p]:
            x0 = max(0, int(np.floor(r[1] / G.REDUCCION)) - tolerancia)
            x1 = min(lado, int(np.ceil(r[2] / G.REDUCCION)) + tolerancia)
            y0 = max(0, int(np.floor(r[3] / G.REDUCCION)) - tolerancia)
            y1 = min(lado, int(np.ceil(r[4] / G.REDUCCION)) + tolerancia)
            dentro[y0:y1, x0:x1] = True
        n = int((tinta & ~dentro).sum())
        if n:
            malos.append(f"pagina {p}: {n} bloque(s) con tinta fuera de toda caja")
    return malos


async def generar(n_parrafos: int, semilla: int, max_paginas: int | None = None,
                  destino: Path = PAGINAS) -> dict:
    """Rinde el dataset. `max_paginas` corta despues de N paginas SIN cambiar nada:
    el plan se calcula igual y la semilla de cada pagina sale de su indice, asi que
    las N primeras son identicas a las N primeras de la corrida entera. Es lo que
    hace que `--rederivar N` compruebe algo en 20 s en vez de en 11 min.

    ⚠ `destino` existe para que `--rederivar` NO pise la etapa local. La primera
    version rendia sobre `datos/paginas/` y la borraba al empezar: comprobar que la
    receta es honesta destruia el dataset que estabas a punto de publicar."""
    gen = exigir_generador()
    sys.path.insert(0, str(gen))
    from app.core.renderer import Renderer                      # noqa: PLC0415
    from app.core.resolver import resolve                       # noqa: PLC0415

    rng = random.Random(semilla)
    reparto = plan(n_parrafos, rng)
    if max_paginas is not None:
        reparto = reparto[:max_paginas]
    destino.mkdir(parents=True, exist_ok=True)
    for viejo in destino.glob("*.npy"):
        viejo.unlink()

    r = Renderer()
    await r.start()
    filas, meta_pag, descartes = [], [], []
    renders = 0
    t0 = time.time()
    try:
        for p, n in enumerate(reparto):
            for intento in range(MAX_REINTENTOS_PAGINA):
                sem = semilla * 1000003 + p * 97 + intento
                rp = random.Random(sem)
                celdas = G.celdas(n, rp)
                if len(celdas) < n:          # el papel no da: se baja la densidad
                    n = len(celdas)
                ranuras = [G.ranura(c) for c in celdas]
                cuerpos = cuerpos_emparejados(ranuras, rp)
                bloques = [factores(s, rp, c) for s, c in zip(ranuras, cuerpos)]
                out = await _una_pagina(r, resolve, bloques, ranuras, sem)
                renders += 2
                if out is None:
                    descartes.append({"pagina": p, "intento": intento,
                                      "por_que": "no se pudo ajustar el tamano"})
                    continue
                img, cajas, bloques = out
                fallos, med = G.revisar(cajas)
                if fallos:                    # la RED DE SEGURIDAD: se tira entera
                    descartes.append({"pagina": p, "intento": intento,
                                      "por_que": "; ".join(fallos[:3])})
                    continue
                idx = len(meta_pag)
                np.save(destino / f"{idx:06d}.npy", reducir(img))
                # las derivadas, en px GUARDADOS: es donde se recortan las ventanas
                g = [G.a_guardado(c) for c in cajas]
                for i, c in enumerate(cajas):
                    otras = [o for k, o in enumerate(g) if k != i]
                    filas.append({
                        "pagina": idx,
                        "izq": int(round(c[0])), "der": int(round(c[2])),
                        "sup": int(round(c[1])), "inf": int(round(c[3])),
                        "margen_r4": round(G.margen(g[i], G.GUARDADO), 2),
                        "separacion_r4": round(min((G.separacion(g[i], o) for o in otras),
                                                   default=float(G.GUARDADO)), 2),
                        "ventana_limpia_r4": G.ventana_limpia(g[i], otras, G.GUARDADO),
                        **{k: v for k, v in bloques[i].items() if k not in ("ux", "uy")},
                    })
                meta_pag.append({"n": len(cajas), "semilla": sem, **med})
                break
            else:
                descartes.append({"pagina": p, "intento": "todos",
                                  "por_que": f"{MAX_REINTENTOS_PAGINA} intentos sin pagina valida"})
            if (p + 1) % 20 == 0:
                print(f"  pagina {p+1}/{len(reparto)} · {len(filas)} parrafos · "
                      f"{renders} renders · {time.time()-t0:.0f} s", flush=True)
    finally:
        await r.stop()
    return {"filas": filas, "paginas": meta_pag, "descartes": descartes,
            "renders": renders, "segundos": round(time.time() - t0, 1),
            "paginas_planeadas": len(reparto)}


# ---------------------------------------------------------------- empaquetado
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _particion(n_paginas: int, semilla: int) -> np.ndarray:
    """train/val/eval POR PAGINA, no por parrafo, y eso es la decision.

    Las ventanas de entrenamiento se recortan despues; dos ventanas de la MISMA
    pagina comparten fondo, fuente y vecinos, asi que repartirlas entre train y eval
    seria una fuga que no falla por ningun lado y que solo se ve como un resultado
    demasiado bueno. Repartiendo paginas, cualquier recorte posterior hereda el
    reparto y no puede filtrar."""
    orden = np.random.default_rng(semilla).permutation(n_paginas)
    codigos = np.empty(n_paginas, dtype=np.uint8)
    n_tr = int(round(REPARTO["train"] * n_paginas))
    n_va = int(round(REPARTO["val"] * n_paginas))
    codigos[orden[:n_tr]] = 0
    codigos[orden[n_tr:n_tr + n_va]] = 1
    codigos[orden[n_tr + n_va:]] = 2
    return codigos


def empaquetar(res: dict, semilla: int) -> dict:
    ficheros = sorted(PAGINAS.glob("*.npy"))
    if not ficheros:
        raise SystemExit("✗ no hay paginas en datos/paginas/. Genera primero.")
    paginas = np.stack([np.load(f) for f in ficheros]).astype(np.uint16)
    assert paginas.shape[1:] == (G.GUARDADO, G.GUARDADO), paginas.shape
    assert int(paginas.max()) <= 255 * G.REDUCCION ** 2, "la suma del bloque se sale de uint16"

    filas = res["filas"]

    # R15: la receta MANDA, asi que lo generado tiene que caber en lo que declara.
    # Sin esto, un rango de `receta.json` que nadie lee vuelve a ser decoracion.
    fuera = []
    for r in filas:
        if not (ANCHO_RECETA[0] - 0.5 <= r["ancho"] <= ANCHO_RECETA[1] + 0.5):
            fuera.append(f"ancho {r['ancho']} fuera de {list(ANCHO_RECETA)}")
        if not (PALABRAS_RECETA[0] <= r["palabras"] <= PALABRAS_RECETA[1]):
            fuera.append(f"palabras {r['palabras']} fuera de {list(PALABRAS_RECETA)}")
        if not (CUERPO[0] - 0.05 <= r["cuerpo"] <= CUERPO[1] + 0.05):
            fuera.append(f"cuerpo {r['cuerpo']} fuera de {list(CUERPO)}")
        if not (INTERLINEADO[0] - 1e-6 <= r["interlineado"] <= INTERLINEADO[1] + 1e-6):
            fuera.append(f"interlineado {r['interlineado']} fuera de {list(INTERLINEADO)}")
        if r["fuente"] not in FUENTES:
            fuera.append(f"fuente {r['fuente']} no esta en la receta")
    if fuera:
        raise SystemExit("✗ lo generado NO cabe en lo que declara receta.json:\n  "
                         + "\n  ".join(sorted(set(fuera))[:6])
                         + "\nArregla la receta o el muestreo; no se empaqueta.")

    cajas = np.array([[r["pagina"], r["izq"], r["der"], r["sup"], r["inf"]]
                      for r in filas], dtype=np.int32)
    derivadas = np.array([[r["margen_r4"], r["separacion_r4"], r["ventana_limpia_r4"]]
                          for r in filas], dtype=np.float32)
    particion = _particion(len(paginas), semilla)
    tinta_fuera = tinta_fuera_de_cajas(paginas, cajas)
    if tinta_fuera:
        raise SystemExit("✗ hay tinta FUERA de las cajas en la pagina reducida:\n  "
                         + "\n  ".join(tinta_fuera[:6])
                         + "\nLa etiqueta no describe la imagen; no se empaqueta.")

    DATOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATOS / "paginas.npz",
        paginas=paginas,
        particion=particion,
        n_parrafos=np.array([p["n"] for p in res["paginas"]], dtype=np.uint8))
    np.savez_compressed(DATOS / "cajas.npz", cajas=cajas, derivadas=derivadas)
    (DATOS / "meta.json").write_text(json.dumps(
        {"parrafos": filas, "paginas": res["paginas"], "descartes": res["descartes"]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    sep = derivadas[:, 1]
    mar = derivadas[:, 0]
    ven = derivadas[:, 2]
    nombres = np.array(["train", "val", "eval"])
    ancho_r4 = (cajas[:, 2] - cajas[:, 1]) / G.REDUCCION
    alto_r4 = (cajas[:, 4] - cajas[:, 3]) / G.REDUCCION
    manifiesto = {
        "nombre": NOMBRE,
        "experimento": "bor-p4",
        "generado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "semilla": semilla,
        "lienzo": G.LIENZO,
        "reduccion": G.REDUCCION,
        "marco": G.GUARDADO,
        "almacenamiento": ("uint16, SUMA del bloque 4x4 (el §3.8 de `banco-k`): exacto, sin "
                           "cuantizacion. Papel blanco = 4080; tinta negra pura = 0"),
        "fondo": "solid #ffffff (papel limpio; NADA de fondos sucios)",
        "etiqueta": ("4 enteros por parrafo en pixeles del LIENZO DE RENDER (1024): izq, "
                     "der, sup, inf. Es la caja de TINTA (union de cajas de palabra), no "
                     "la de maquetacion"),
        "etiqueta_en_la_pagina_guardada": ("coord / 4, sin redondear: la misma aritmetica "
                                           "que el banco aplica a las suyas (§6.4 de alli)"),
        "alineacion": "justify: sin ella la tinta no llega al borde derecho de la caja",
        "paginas": len(paginas),
        "parrafos": len(filas),
        "parrafos_por_pagina": {str(k): int(v) for k, v in
                                zip(*np.unique([p["n"] for p in res["paginas"]],
                                               return_counts=True))},
        "geometria": {
            "k_max_guardado": G.K_MAX,
            "radio_guardado": G.RADIO,
            "separacion_dura": {"guardado": G.SEPARACION_DURA_R4, "render": G.SEPARACION_DURA},
            "margen_duro": {"guardado": G.MARGEN_DURO_R4, "render": G.MARGEN_DURO},
            "separacion_construida": {"guardado": G.SEPARACION / G.REDUCCION,
                                      "render": G.SEPARACION},
            "margen_construido": {"guardado": G.MARGEN / G.REDUCCION, "render": G.MARGEN},
            "ancho_min_render": G.ANCHO_MIN,
            "alto_min_render": G.ALTO_MIN,
            "de_donde_salen": ("todo de k_max=19 A LA ESCALA GUARDADA, que es donde corre el "
                               "kernel: radio 9, separacion dura 2*9+1=19, margen duro 9 "
                               "(una convolucion valid k=19 sobre 256 solo cubre [9, 246]). "
                               "En render, x4: 76 y 36. La construccion es la de `bor-p` "
                               "x4: separacion 160, margen 128"),
            "metrica": "L-infinito (Chebyshev): es la que corresponde a una ventana CUADRADA",
        },
        "medido": {
            "escala": "px de la pagina GUARDADA (256), que es donde se recortan las ventanas",
            "separacion_min": round(float(sep.min()), 2),
            "separacion_mediana": round(float(np.median(sep)), 2),
            "margen_min": round(float(mar.min()), 2),
            "margen_mediana": round(float(np.median(mar)), 2),
            "ventana_limpia_min": int(ven.min()),
            "ventana_limpia_mediana": int(np.median(ven)),
            "ventana_limpia_p10": int(np.percentile(ven, 10)),
            "que_es_ventana_limpia": ("el lado IMPAR de la ventana mas grande que se "
                                      "puede centrar en CUALQUIER punto del borde de un "
                                      "parrafo sin salirse del papel y sin ver tinta de "
                                      "otro. Es lo que se puede recortar despues"),
            "ancho_tinta": [round(float(ancho_r4.min()), 2), round(float(ancho_r4.max()), 2)],
            "alto_tinta": [round(float(alto_r4.min()), 2), round(float(alto_r4.max()), 2)],
            "tinta_fuera_de_cajas": 0,
            "tinta_fuera_de_cajas_sin_tolerancia": len(
                tinta_fuera_de_cajas(paginas, cajas, tolerancia=0)),
            "tolerancia_tinta_bloques": TOLERANCIA_TINTA,
        },
        "factores": {
            "fuentes": FUENTES,
            "cuerpo_px": list(CUERPO),
            "cuerpo_px_usado": [round(float(min(r["cuerpo"] for r in filas)), 1),
                                round(float(max(r["cuerpo"] for r in filas)), 1)],
            "interlineado": list(INTERLINEADO),
            "interlineado_usado": [round(float(min(r["interlineado"] for r in filas)), 3),
                                   round(float(max(r["interlineado"] for r in filas)), 3)],
            "ancho_px_tope_receta": list(ANCHO_RECETA),
            "ancho_px_usado": [round(float(min(r["ancho"] for r in filas)), 1),
                               round(float(max(r["ancho"] for r in filas)), 1)],
            "palabras_tope_receta": list(PALABRAS_RECETA),
            "palabras_usado": [int(min(r["palabras"] for r in filas)),
                               int(max(r["palabras"] for r in filas))],
            "ancho_y_alto_se_derivan": ("de la celda que el reparto le toco al parrafo; "
                                        "el rango de la receta es el TOPE, comprobado al "
                                        "empaquetar"),
            "color_texto": "#000000 fijo (no se vario: el encargo pide papel limpio)",
            "parrafos_por_pagina": list(PARRAFOS_POR_PAGINA),
        },
        "reserva_banco_k_§3.7": {
            "respetada": True,
            "fuente_prohibida": RESERVA_FUENTE,
            "interlineado_prohibido": list(RESERVA_INTERLINEADO),
            "interlineado_usado": list(INTERLINEADO),
            "regla": ("`banco-k` reserva LiberationMono y el interlineado [1.45, 1.60] "
                      "para uso exclusivo suyo. Un kernel aprendido sobre datos que los "
                      "alcancen tiene fuga de distribucion AUNQUE las muestras sean "
                      "distintas, y sale optimista al evaluarlo alli. Este dataset no "
                      "los usa, y `nn/receta.json` lo declara EXPLICITO porque omitir "
                      "cualquiera de las dos claves equivale a usarlas"),
        },
        "reparto": {
            "criterio": "POR PAGINA, no por parrafo: dos ventanas de la misma pagina "
                        "comparten fuente, fondo y vecinos",
            "fracciones": REPARTO,
            "paginas": {n: int((particion == i).sum()) for i, n in enumerate(nombres)},
            "parrafos": {n: int((particion[cajas[:, 0]] == i).sum())
                         for i, n in enumerate(nombres)},
        },
        "descartadas": len(res["descartes"]),
        "paginas_planeadas": res["paginas_planeadas"],
        "renders": res["renders"],
        "segundos": res["segundos"],
        "ficheros": {
            "paginas.npz": {
                "sha256_16": _sha(paginas.tobytes()),
                "forma": list(paginas.shape),
                "dtype": "uint16",
                "bytes": (DATOS / "paginas.npz").stat().st_size,
                "como_se_lee": "np.load('paginas.npz')['paginas'][i]  -> (256, 256) uint16",
            },
            "cajas.npz": {
                "sha256_16": _sha(cajas.tobytes()),
                "columnas": ["pagina", "izq", "der", "sup", "inf"],
                "escala_columnas": "px del lienzo de RENDER (1024); /4 en la guardada",
                "derivadas": ["margen_r4", "separacion_r4", "ventana_limpia_r4"],
                "escala_derivadas": "px de la pagina GUARDADA (256)",
                "bytes": (DATOS / "cajas.npz").stat().st_size,
            },
        },
    }
    (DATOS / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifiesto


# ---------------------------------------------------------------- publicar
def _publicar() -> int:
    """Publica en el repo de datos. NUNCA pisa uno existente (regla 3 del repo)."""
    if not (DATOS / "manifiesto.json").is_file():
        print("✗ no hay etapa local. Genera primero: --parrafos 1000")
        return 1
    if ruta_dataset(NOMBRE) is not None:
        print(f"✗ '{NOMBRE}' YA esta publicado. Un dataset publicado NO se reescribe:")
        print("  dato nuevo = nombre nuevo (cambia la r<fecha> de NOMBRE).")
        return 1
    destino = exigir_datos() / SUBDIR_DATASETS / NOMBRE
    destino.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATOS.iterdir()):
        if f.suffix in (".npz", ".json"):
            (destino / f.name).write_bytes(f.read_bytes())
            print(f"  {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
    _readme(destino)
    print(f"\npublicado en {destino}")
    print("⚠ El push es parte del encargo: el `origin` del repo de datos es el ALMACEN")
    print("  (desde el 2026-10-01), y lo que no esta empujado no existe.")
    return 0


def _readme(destino: Path) -> None:
    m = json.loads((destino / "manifiesto.json").read_text(encoding="utf-8"))
    g, med, rep = m["geometria"], m["medido"], m["reparto"]
    destino.joinpath("README.md").write_text(f"""# `{NOMBRE}`

Paginas de **papel limpio con varios parrafos**, y la caja que encierra cada uno, **a la
escala del banco `banco-k`**: rendidas a {m['lienzo']} y guardadas reducidas /{m['reduccion']}
({m['marco']} x {m['marco']}). Producido por `bor-p4` (`experimentos-cnn`) con
`nn/generar_paginas.py`.

| | |
|---|---|
| paginas | **{m['paginas']}** de {m['marco']} x {m['marco']}, gris de 1 canal |
| almacenamiento | {m['almacenamiento']} |
| parrafos | **{m['parrafos']}** ({m['parrafos_por_pagina']} por pagina) |
| etiqueta | 4 enteros por parrafo en px de **render** ({m['lienzo']}): `izq, der, sup, inf` — la caja de **TINTA**. En la pagina guardada: `coord / {m['reduccion']}` |
| fondo | blanco solido. **Ningun fondo sucio** |
| descartadas | {m['descartadas']} |
| tinta fuera de las cajas | **{med['tinta_fuera_de_cajas']}** bloques (comprobado sobre la pagina reducida) |

## Por que existe, si ya estaba `parrafos1000-pagina1024-r20260909`

Aquel esta a {m['lienzo']} **sin reducir**, con cuerpo 11-30 px. El banco aplica el
kernel **despues** de reducir /4, o sea sobre tinta de 2,75-7,5 px: un kernel aprendido
sobre aquel veria letras ~4x mas gruesas que las que vera alli. Y aquel no se puede
reducir tal cual: su separacion minima (41,88 px) queda en 10,5 a /4, por debajo de lo que
pide un kernel de 19.

## La garantia geometrica, en px GUARDADOS (donde corre el kernel)

| | | por que |
|---|---|---|
| separacion entre parrafos | **>= {g['separacion_dura']['guardado']}** (real: min **{med['separacion_min']}**, mediana {med['separacion_mediana']}) | `2*{g['radio_guardado']}+1`: las ventanas {g['k_max_guardado']}x{g['k_max_guardado']} de dos bordes enfrentados no se solapan |
| margen al borde del papel | **>= {g['margen_duro']['guardado']}** (real: min **{med['margen_min']}**, mediana {med['margen_mediana']}) | una convolucion `valid` k={g['k_max_guardado']} sobre {m['marco']} solo cubre `[{g['radio_guardado']}, {m['marco']-g['radio_guardado']-1}]` |
| solape | **ninguno**, por construccion | |

⚠ La metrica es **L-infinito**, no euclidea: es la de una ventana **cuadrada**.

## Cuanto se puede recortar

`derivadas` de `cajas.npz` trae, por parrafo y en px **guardados**, `ventana_limpia_r4`:
el lado impar de la ventana mas grande que se puede centrar en **cualquier** punto de su
borde sin salirse del papel y sin ver tinta de otro parrafo.

| | |
|---|---|
| minimo | **{med['ventana_limpia_min']} px** |
| percentil 10 | {med['ventana_limpia_p10']} px |
| mediana | {med['ventana_limpia_mediana']} px |

Una ventana de **{med['ventana_limpia_min']} x {med['ventana_limpia_min']}** centrada en el
borde es limpia para **todos** los parrafos, sin filtrar ninguno.

## Reparto: POR PAGINA, no por parrafo

{rep['criterio']}.

| | paginas | parrafos |
|---|---:|---:|
| train | {rep['paginas']['train']} | {rep['parrafos']['train']} |
| val | {rep['paginas']['val']} | {rep['parrafos']['val']} |
| eval | {rep['paginas']['eval']} | {rep['parrafos']['eval']} |

## ⚠ Reserva de `banco-k` (§3.7): RESPETADA

Este dataset **no usa** `{m['reserva_banco_k_§3.7']['fuente_prohibida']}` ni interlineado
en `{m['reserva_banco_k_§3.7']['interlineado_prohibido']}`, que `banco-k` se reserva. Usa
`{m['reserva_banco_k_§3.7']['interlineado_usado']}` y las otras cuatro familias. Un kernel
aprendido aqui **no tiene fuga de distribucion** contra el banco.

## Como se lee

```python
import numpy as np

d = np.load("paginas.npz")
c = np.load("cajas.npz")

paginas = d["paginas"]        # (P, 256, 256) uint16, SUMA del bloque 4x4: blanco = 4080
cajas = c["cajas"]            # (parrafo, 5) int32: pagina, izq, der, sup, inf  (px de RENDER)
caja_r4 = cajas[:, 1:] / 4    # la misma caja en px de la pagina guardada
margen, separacion, ventana = c["derivadas"].T      # px GUARDADOS
mias = cajas[d["particion"][cajas[:, 0]] == 0]      # 0=train 1=val 2=eval
```

`meta.json` trae los factores de cada parrafo (fuente, cuerpo, interlineado, palabras) y
el porque de cada pagina descartada.
""", encoding="utf-8")


# ---------------------------------------------------------------- comprobar
def _comprobar() -> int:
    p = ruta_dataset(NOMBRE)
    if p is None:
        print(f"✗ '{NOMBRE}' no esta publicado. Publica con --publicar")
        return 1
    m = json.loads((p / "manifiesto.json").read_text(encoding="utf-8"))
    paginas = np.load(p / "paginas.npz")["paginas"]
    c = np.load(p / "cajas.npz")
    malos = []

    if _sha(paginas.tobytes()) != m["ficheros"]["paginas.npz"]["sha256_16"]:
        malos.append("las paginas no casan con el manifiesto")
    if _sha(c["cajas"].tobytes()) != m["ficheros"]["cajas.npz"]["sha256_16"]:
        malos.append("las cajas no casan con el manifiesto")

    # la garantia geometrica y la tinta, re-comprobadas sobre el PUBLICADO
    cajas = c["cajas"]
    for pag in range(len(paginas)):
        cs = [tuple(float(v) for v in (r[1], r[3], r[2], r[4]))
              for r in cajas[cajas[:, 0] == pag]]
        fallos, _ = G.revisar(cs)
        if fallos:
            malos.append(f"pagina {pag}: {fallos[0]}")
            if len(malos) > 5:
                break
    malos += tinta_fuera_de_cajas(paginas, cajas)[:5]
    if m["reserva_banco_k_§3.7"]["fuente_prohibida"] in m["factores"]["fuentes"]:
        malos.append("la reserva §3.7 de banco-k esta en las fuentes usadas")

    for x in malos:
        print(f"  ✗ {x}")
    print(f"\n{NOMBRE}: {len(paginas)} paginas, {len(cajas)} parrafos · "
          f"{'TODO CASA' if not malos else str(len(malos)) + ' fallo(s)'}")
    return 1 if malos else 0


# ---------------------------------------------------------------- figuras
def _muestras(n: int) -> int:
    """Rejilla de paginas GUARDADAS (las de 256, ampliadas sin suavizar para que se vea
    lo que vera el kernel) con la caja /4 encima. Es la comprobacion que ningun numero
    da: que la caja abraza la tinta por los cuatro lados DESPUES de reducir."""
    p = ruta_dataset(NOMBRE) or DATOS
    from PIL import Image, ImageDraw                             # noqa: PLC0415
    paginas = np.load(p / "paginas.npz")["paginas"]
    cajas = np.load(p / "cajas.npz")["cajas"]
    idx = np.random.default_rng(3).choice(len(paginas), size=min(n, len(paginas)),
                                          replace=False)
    escala, cols = 2, min(5, len(idx))
    lado = G.GUARDADO * escala
    filas = (len(idx) + cols - 1) // cols
    hoja = Image.new("RGB", (cols * lado, filas * lado), "white")
    for k, i in enumerate(idx):
        gris = (paginas[i].astype(np.float32) / (255 * G.REDUCCION ** 2) * 255).round()
        im = Image.fromarray(gris.astype(np.uint8)).resize((lado, lado), Image.NEAREST)
        im = im.convert("RGB")
        dr = ImageDraw.Draw(im)
        for r in cajas[cajas[:, 0] == i]:
            f = escala / G.REDUCCION
            dr.rectangle([r[1] * f, r[3] * f, r[2] * f, r[4] * f], outline=(220, 30, 30))
        hoja.paste(im, ((k % cols) * lado, (k // cols) * lado))
    salida = EXP / "muestras"
    salida.mkdir(exist_ok=True)
    hoja.save(salida / f"paginas-r4-{len(idx)}.png")
    print(f"escrita {salida / f'paginas-r4-{len(idx)}.png'}")
    return 0


def _rederivar(n: int) -> int:
    """¿La receta y la semilla vuelven a dar lo mismo? Es una PRUEBA, no una
    alternativa a publicar: publicado es el mismo porque es el MISMO fichero.

    Rehace las N PRIMERAS PAGINAS y compara las CAJAS, que es lo que el generador
    garantiza entre maquinas; los pixeles se informan aparte, porque la rasterizacion
    de Chromium no esta garantizada entre versiones. Va a `datos/rederivar/`."""
    p = ruta_dataset(NOMBRE)
    if p is None:
        print(f"✗ '{NOMBRE}' no esta publicado")
        return 1
    guardadas = np.load(p / "cajas.npz")["cajas"]
    paginas = np.load(p / "paginas.npz")["paginas"]
    destino = DATOS / "rederivar"
    res = asyncio.run(generar(N_PARRAFOS, SEMILLA, max_paginas=n, destino=destino))
    nuevas = np.array([[r["pagina"], r["izq"], r["der"], r["sup"], r["inf"]]
                       for r in res["filas"]], dtype=np.int32)
    esperadas = guardadas[guardadas[:, 0] < n]
    igual = (nuevas.shape == esperadas.shape and np.array_equal(nuevas, esperadas))
    print(f"\n{n} primeras paginas ({len(nuevas)} parrafos): "
          f"cajas {'IDENTICAS' if igual else 'DISTINTAS'}")
    if not igual and nuevas.shape == esperadas.shape:
        d = np.abs(nuevas - esperadas)
        print(f"  mayor diferencia: {d.max()} px en {(d != 0).sum()} de {d.size} valores")
    rehechas = [np.load(f) for f in sorted(destino.glob("*.npy"))]
    iguales = sum(np.array_equal(a, b) for a, b in zip(rehechas, paginas[:n]))
    print(f"  pixeles: {iguales}/{len(rehechas)} paginas identicas bit a bit "
          f"(informativo: Chromium no garantiza la rasterizacion entre versiones)")
    return 0 if igual else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parrafos", type=int)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--paginas", type=int,
                    help="corta tras N paginas (un ensayo: NO se empaqueta para publicar)")
    ap.add_argument("--publicar", action="store_true")
    ap.add_argument("--comprobar", action="store_true")
    ap.add_argument("--rederivar", type=int)
    ap.add_argument("--muestras", type=int)
    ap.add_argument("--reserva", action="store_true",
                    help="solo comprueba el contrato §3.7 de banco-k y sale")
    a = ap.parse_args()

    malos = comprobar_reserva()
    if malos:
        for x in malos:
            print(f"✗ RESERVA §3.7: {x}")
        print("\nUn kernel aprendido aqui saldria optimista en `banco-k`. No se genera.")
        return 1
    if a.reserva:
        print(f"✓ reserva §3.7 respetada: fuentes {FUENTES} · interlineado "
              f"{list(INTERLINEADO)} (prohibido {RESERVA_FUENTE} y "
              f"{list(RESERVA_INTERLINEADO)})")
        return 0
    if a.publicar:
        return _publicar()
    if a.comprobar:
        return _comprobar()
    if a.rederivar:
        return _rederivar(a.rederivar)
    if a.muestras:
        return _muestras(a.muestras)
    if a.parrafos:
        if a.paginas:
            # un ensayo: rinde N paginas a un sitio aparte y mide la tinta, sin tocar
            # la etapa local que se va a publicar
            destino = DATOS / "ensayo"
            res = asyncio.run(generar(a.parrafos, a.semilla, max_paginas=a.paginas,
                                      destino=destino))
            pags = np.stack([np.load(f) for f in sorted(destino.glob("*.npy"))])
            cajas = np.array([[r["pagina"], r["izq"], r["der"], r["sup"], r["inf"]]
                              for r in res["filas"]], dtype=np.int32)
            for tol in (0, 1):
                f = tinta_fuera_de_cajas(pags, cajas, tolerancia=tol)
                print(f"tolerancia {tol} bloque(s): "
                      f"{'ninguna tinta fuera' if not f else '; '.join(f[:3])}")
            for d in res["descartes"]:
                print(f"  descarte: pagina {d['pagina']} intento {d['intento']}: {d['por_que']}")
            ven = [r["ventana_limpia_r4"] for r in res["filas"]]
            print(f"{len(pags)} paginas · {len(res['filas'])} parrafos · "
                  f"{len(res['descartes'])} descarte(s) · {res['segundos']} s · "
                  f"ventana limpia min {min(ven)}")
            return 0
        res = asyncio.run(generar(a.parrafos, a.semilla))
        m = empaquetar(res, a.semilla)
        print(f"\n{m['paginas']} paginas · {m['parrafos']} parrafos · "
              f"{m['descartadas']} descartada(s) · {m['renders']} renders · "
              f"{m['segundos']} s")
        print(f"separacion min {m['medido']['separacion_min']} · "
              f"margen min {m['medido']['margen_min']} · "
              f"ventana limpia min {m['medido']['ventana_limpia_min']} (px guardados)")
        print(f"etapa local en {DATOS}. Publica con --publicar")
        return 0
    ap.error("dime que hacer: --parrafos N [--paginas N] · --publicar · --comprobar · "
             "--rederivar N · --muestras N · --reserva")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
