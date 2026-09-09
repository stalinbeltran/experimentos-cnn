#!/usr/bin/env python3
"""El dataset de `bor-p`: paginas limpias con VARIOS parrafos, y la caja de cada uno.

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

7. LAS PAGINAS SE GUARDAN COMO PNG DENTRO DEL .npz, no como un array. Un array
   (P, 1024, 1024) uint8 son ~300 MB en RAM y esta maquina tiene 3,8 GB; ademas, con
   PNG en disco desde el primer render, una caida a los 18 minutos no pierde nada. PNG
   es SIN PERDIDA y `--comprobar` lo demuestra: guarda la huella del array DECODIFICADO.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
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
NOMBRE = "parrafos1000-pagina1024-r20260909"
SEMILLA = 1
N_PARRAFOS = 1000
PARRAFOS_POR_PAGINA = (2, 3, 4, 5)
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
MAX_REINTENTOS_PAGINA = 3


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


def factores(ranura, rng: random.Random) -> dict:
    """Los factores de UN parrafo, derivados de su ranura (§ variar tamanos)."""
    rw, rh = ranura[2] - ranura[0], ranura[3] - ranura[1]
    fuente = rng.choice(FUENTES)
    cuerpo = round(rng.uniform(*CUERPO), 1)
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
    for viejo in destino.glob("*.png"):
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
                bloques = [factores(s, rp) for s in ranuras]
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
                img.save(destino / f"{idx:06d}.png", format="PNG", optimize=True)
                for i, c in enumerate(cajas):
                    otras = [o for k, o in enumerate(cajas) if k != i]
                    filas.append({
                        "pagina": idx,
                        "izq": int(round(c[0])), "der": int(round(c[2])),
                        "sup": int(round(c[1])), "inf": int(round(c[3])),
                        "margen": round(G.margen(c), 2),
                        "separacion": round(min((G.separacion(c, o) for o in otras),
                                                default=float(G.LIENZO)), 2),
                        "ventana_limpia": G.ventana_limpia(c, otras),
                        **{k: v for k, v in bloques[i].items() if k not in ("ux", "uy")},
                    })
                meta_pag.append({"n": len(cajas), "semilla": sem, **med})
                break
            else:
                descartes.append({"pagina": p, "intento": "todos",
                                  "por_que": "3 intentos sin pagina valida"})
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
    pngs = sorted(PAGINAS.glob("*.png"))
    if not pngs:
        raise SystemExit("✗ no hay paginas en datos/paginas/. Genera primero.")
    from PIL import Image                                        # noqa: PLC0415

    blob, offsets, huellas = bytearray(), [0], hashlib.sha256()
    for f in pngs:
        b = f.read_bytes()
        blob += b
        offsets.append(len(blob))
        huellas.update(np.asarray(Image.open(io.BytesIO(b)).convert("L"),
                                  dtype=np.uint8).tobytes())

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
    derivadas = np.array([[r["margen"], r["separacion"], r["ventana_limpia"]]
                          for r in filas], dtype=np.float32)
    particion = _particion(len(pngs), semilla)

    DATOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATOS / "paginas.npz",
        png=np.frombuffer(bytes(blob), dtype=np.uint8),
        png_offsets=np.array(offsets, dtype=np.int64),
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
    manifiesto = {
        "nombre": NOMBRE,
        "experimento": "bor-p",
        "generado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "semilla": semilla,
        "lienzo": G.LIENZO,
        "fondo": "solid #ffffff (papel limpio; NADA de fondos sucios, es el encargo)",
        "etiqueta": ("4 enteros por parrafo en pixeles del lienzo: izq, der, sup, inf. "
                     "Es la caja de TINTA (union de cajas de palabra), no la de "
                     "maquetacion"),
        "alineacion": "justify: sin ella la tinta no llega al borde derecho de la caja",
        "paginas": len(pngs),
        "parrafos": len(filas),
        "parrafos_por_pagina": {str(k): int(v) for k, v in
                                zip(*np.unique([p["n"] for p in res["paginas"]],
                                               return_counts=True))},
        "geometria": {
            "k_max": G.K_MAX,
            "radio": G.RADIO,
            "separacion_dura": G.SEPARACION_DURA,
            "separacion_construida": G.SEPARACION,
            "margen_duro": G.MARGEN_DURO,
            "margen_construido": G.MARGEN,
            "ancho_min": G.ANCHO_MIN,
            "alto_min": G.ALTO_MIN,
            "de_donde_salen": ("todo de k_max=19: radio=(19-1)/2=9, "
                               "separacion_dura=2*9+1=19 (las dos ventanas 19x19 de dos "
                               "bordes enfrentados no se solapan), margen_duro=9 (una "
                               "convolucion valid k=19 solo cubre [9, 1014], asi que un "
                               "borde mas cerca del papel NO existe en la salida)"),
            "metrica": "L-infinito (Chebyshev): es la que corresponde a una ventana CUADRADA",
        },
        "medido": {
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
            "ancho_tinta": [round(float(x), 1) for x in
                            (np.min(cajas[:, 2] - cajas[:, 1]),
                             np.max(cajas[:, 2] - cajas[:, 1]))],
            "alto_tinta": [round(float(x), 1) for x in
                           (np.min(cajas[:, 4] - cajas[:, 3]),
                            np.max(cajas[:, 4] - cajas[:, 3]))],
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
                "sha256_16_png": _sha(bytes(blob)),
                "sha256_16_decodificado": huellas.hexdigest()[:16],
                "bytes": (DATOS / "paginas.npz").stat().st_size,
                "como_se_lee": ("png/png_offsets son los PNG concatenados: "
                                "Image.open(BytesIO(d['png'][o[i]:o[i+1]].tobytes()))"),
            },
            "cajas.npz": {
                "sha256_16": _sha(cajas.tobytes()),
                "columnas": ["pagina", "izq", "der", "sup", "inf"],
                "derivadas": ["margen", "separacion", "ventana_limpia"],
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
    print("⚠ El push es parte del encargo: el repo de datos es PRIVADO y git no olvida,")
    print("  pero lo que no esta empujado no existe.")
    return 0


def _readme(destino: Path) -> None:
    m = json.loads((destino / "manifiesto.json").read_text(encoding="utf-8"))
    g, med, rep = m["geometria"], m["medido"], m["reparto"]
    destino.joinpath("README.md").write_text(f"""# `{NOMBRE}`

Paginas de **papel limpio con varios parrafos**, y la caja que encierra cada uno.
Producido por [`bor-p`](https://github.com/stalinbeltran/experimentos-cnn/tree/main/2026-09-09-bordes-parrafo)
con `nn/generar_paginas.py`.

| | |
|---|---|
| paginas | **{m['paginas']}** de {m['lienzo']} x {m['lienzo']} px, gris de 1 canal |
| parrafos | **{m['parrafos']}** ({m['parrafos_por_pagina']} por pagina) |
| etiqueta | 4 enteros por parrafo: `izq, der, sup, inf` — la caja de **TINTA** |
| fondo | blanco solido. **Ningun fondo sucio** |
| descartadas | {m['descartadas']} |

## La garantia geometrica, que es para lo que existe este dataset

Todo sale de **`k_max = {g['k_max']}`**, el kernel mas grande que se va a entrenar:

| | | por que |
|---|---|---|
| separacion entre parrafos | **>= {g['separacion_dura']} px** (real: min **{med['separacion_min']}**, mediana {med['separacion_mediana']}) | `2*{g['radio']}+1`: las ventanas {g['k_max']}x{g['k_max']} de dos bordes enfrentados no se solapan, y ninguna ve tinta del otro parrafo |
| margen al borde del papel | **>= {g['margen_duro']} px** (real: min **{med['margen_min']}**, mediana {med['margen_mediana']}) | una convolucion `valid` k={g['k_max']} solo cubre `[{g['radio']}, {m['lienzo']-g['radio']-1}]`: un borde mas cerca **no existe** en la salida |
| solape | **ninguno**, por construccion | |

⚠ La metrica es **L-infinito**, no euclidea: es la que corresponde a una ventana
**cuadrada**. Dos cajas a 14 px en diagonal estan a 19,8 euclideos —pasarian un filtro
de 19— y una ventana de 19x19 **si** las mezcla.

## Cuanto se puede recortar (las ventanas de entrenamiento van despues)

`derivadas` de `cajas.npz` trae, **por parrafo**, `ventana_limpia`: el lado impar de la
ventana mas grande que se puede centrar en **cualquier** punto de su borde sin salirse
del papel y sin ver tinta de otro parrafo.

| | |
|---|---|
| minimo | **{med['ventana_limpia_min']} px** |
| percentil 10 | {med['ventana_limpia_p10']} px |
| mediana | {med['ventana_limpia_mediana']} px |

O sea: una ventana de hasta **{med['ventana_limpia_min']} x {med['ventana_limpia_min']}**
centrada en el borde es limpia para **todos** los parrafos, sin filtrar ninguno.

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
`{m['reserva_banco_k_§3.7']['interlineado_usado']}` y las otras cuatro familias.

Asi, un kernel aprendido sobre estas paginas **no tiene fuga de distribucion** contra el
banco y su veredicto alli no sale optimista.

## Como se lee

```python
import io, json, numpy as np
from PIL import Image

d = np.load("paginas.npz")
c = np.load("cajas.npz")

def pagina(i):
    o = d["png_offsets"]
    return np.asarray(Image.open(io.BytesIO(d["png"][o[i]:o[i+1]].tobytes())))

cajas = c["cajas"]            # (parrafo, 5) int32: pagina, izq, der, sup, inf
margen, separacion, ventana = c["derivadas"].T
mias = cajas[d["particion"][cajas[:, 0]] == 0]      # 0=train 1=val 2=eval
```

Las paginas van como **PNG concatenados** dentro del `.npz`: sin perdida, y sin pedir
300 MB de RAM para abrirlo. `--comprobar` casa la huella del array **decodificado**.

`meta.json` trae los factores de cada parrafo (fuente, cuerpo, interlineado, palabras) y
el porque de cada pagina descartada.
""", encoding="utf-8")


# ---------------------------------------------------------------- comprobar
def _comprobar() -> int:
    p = ruta_dataset(NOMBRE)
    if p is None:
        print(f"✗ '{NOMBRE}' no esta publicado. Publica con --publicar")
        return 1
    from PIL import Image                                        # noqa: PLC0415
    m = json.loads((p / "manifiesto.json").read_text(encoding="utf-8"))
    d = np.load(p / "paginas.npz")
    c = np.load(p / "cajas.npz")
    malos = []

    o = d["png_offsets"]
    h = hashlib.sha256()
    for i in range(len(o) - 1):
        h.update(np.asarray(Image.open(io.BytesIO(d["png"][o[i]:o[i + 1]].tobytes()))
                            .convert("L"), dtype=np.uint8).tobytes())
    esperado = m["ficheros"]["paginas.npz"]["sha256_16_decodificado"]
    if h.hexdigest()[:16] != esperado:
        malos.append(f"las paginas decodificadas no casan: {h.hexdigest()[:16]} != {esperado}")
    if _sha(c["cajas"].tobytes()) != m["ficheros"]["cajas.npz"]["sha256_16"]:
        malos.append("las cajas no casan con el manifiesto")

    # la garantia geometrica, re-comprobada sobre el PUBLICADO
    cajas = c["cajas"]
    for pag in range(len(o) - 1):
        cs = [tuple(float(v) for v in (r[1], r[3], r[2], r[4]))
              for r in cajas[cajas[:, 0] == pag]]
        fallos, _ = G.revisar(cs)
        if fallos:
            malos.append(f"pagina {pag}: {fallos[0]}")
            if len(malos) > 5:
                break
    if m["reserva_banco_k_§3.7"]["fuente_prohibida"] in m["factores"]["fuentes"]:
        malos.append("la reserva §3.7 de banco-k esta en las fuentes usadas")

    for x in malos:
        print(f"  ✗ {x}")
    print(f"\n{NOMBRE}: {len(o)-1} paginas, {len(cajas)} parrafos · "
          f"{'TODO CASA' if not malos else str(len(malos)) + ' fallo(s)'}")
    return 1 if malos else 0


# ---------------------------------------------------------------- figuras
def _muestras(n: int) -> int:
    p = ruta_dataset(NOMBRE) or DATOS
    from PIL import Image, ImageDraw                             # noqa: PLC0415
    d = np.load(p / "paginas.npz")
    cajas = np.load(p / "cajas.npz")["cajas"]
    o = d["png_offsets"]
    idx = np.random.default_rng(3).choice(len(o) - 1, size=min(n, len(o) - 1),
                                          replace=False)
    lado, cols = 320, min(5, len(idx))
    filas = (len(idx) + cols - 1) // cols
    hoja = Image.new("RGB", (cols * lado, filas * lado), "white")
    for k, i in enumerate(idx):
        im = Image.open(io.BytesIO(d["png"][o[i]:o[i + 1]].tobytes())).convert("RGB")
        dr = ImageDraw.Draw(im)
        for r in cajas[cajas[:, 0] == i]:
            dr.rectangle([int(r[1]), int(r[3]), int(r[2]), int(r[4])],
                         outline=(220, 30, 30), width=4)
        hoja.paste(im.resize((lado, lado), Image.LANCZOS),
                   ((k % cols) * lado, (k // cols) * lado))
    salida = EXP / "muestras"
    salida.mkdir(exist_ok=True)
    hoja.save(salida / f"paginas-{len(idx)}.png")
    print(f"escrita {salida / f'paginas-{len(idx)}.png'}")
    return 0


def _rederivar(n: int) -> int:
    """¿La receta y la semilla vuelven a dar lo mismo? Es una PRUEBA, no una
    alternativa a publicar: publicado es el mismo porque es el MISMO fichero.

    Rehace las N PRIMERAS PAGINAS -- no los N primeros parrafos -- porque la pagina
    es la unidad que se rinde. Va a `datos/rederivar/`, nunca sobre la etapa local."""
    p = ruta_dataset(NOMBRE)
    if p is None:
        print(f"✗ '{NOMBRE}' no esta publicado")
        return 1
    guardadas = np.load(p / "cajas.npz")["cajas"]
    res = asyncio.run(generar(N_PARRAFOS, SEMILLA, max_paginas=n,
                              destino=DATOS / "rederivar"))
    nuevas = np.array([[r["pagina"], r["izq"], r["der"], r["sup"], r["inf"]]
                       for r in res["filas"]], dtype=np.int32)
    esperadas = guardadas[guardadas[:, 0] < n]
    igual = (nuevas.shape == esperadas.shape and np.array_equal(nuevas, esperadas))
    print(f"\n{n} primeras paginas ({len(nuevas)} parrafos): "
          f"{'IDENTICAS' if igual else 'DISTINTAS'}")
    if not igual and nuevas.shape == esperadas.shape:
        d = np.abs(nuevas - esperadas)
        print(f"  mayor diferencia: {d.max()} px en {(d != 0).sum()} de {d.size} valores")
    return 0 if igual else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parrafos", type=int)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
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
        res = asyncio.run(generar(a.parrafos, a.semilla))
        m = empaquetar(res, a.semilla)
        print(f"\n{m['paginas']} paginas · {m['parrafos']} parrafos · "
              f"{m['descartadas']} descartada(s) · {m['renders']} renders · "
              f"{m['segundos']} s")
        print(f"separacion min {m['medido']['separacion_min']} · "
              f"margen min {m['medido']['margen_min']} · "
              f"ventana limpia min {m['medido']['ventana_limpia_min']}")
        print(f"etapa local en {DATOS}. Publica con --publicar")
        return 0
    ap.error("dime que hacer: --parrafos N · --publicar · --comprobar · "
             "--rederivar N · --muestras N · --reserva")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
