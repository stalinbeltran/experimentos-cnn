#!/usr/bin/env python3
"""El dataset de `banco-k`: generar, comprobar y PUBLICAR. Especificacion §3-§4.

    python nn/datos.py --imagenes 1000     genera en la etapa local (datos/)
    python nn/datos.py --publicar          lo publica en el repo de datos (una vez)
    python nn/datos.py --comprobar         casa las huellas del publicado con el manifiesto
    python nn/datos.py --rederivar N       rehace las N primeras y comprueba que salen IGUALES
    python nn/datos.py --muestras N        figura PNG con N muestras al azar (--semilla-fig S)

LO QUE HAY QUE ENTENDER ANTES DE TOCAR ESTO
===========================================

1. EL ORDEN DE MUESTREO ES OBLIGATORIO Y NO ES EL OBVIO (§3.4). Se muestrea PRIMERO
   el tamano de la caja y DESPUES la esquina, restringida al rango que garantiza que
   la caja ENTERA cae en [68, 512]. NO se sobre-genera para rechazar: rechazar elimina
   selectivamente las cajas grandes y perifericas, sesga hacia parrafos pequenyos y
   centrados, y eso sube el control de caja media y comprime el banco contra el techo
   (§10.1). Con el orden correcto, 1000 generadas son 1000 validas.

2. PERO EL ALTO DE UN PARRAFO NO SE PUEDE PEDIR: emerge del ancho, la fuente, el
   cuerpo, el interlineado y el numero de palabras. Medido el 2026-09-08: 240 palabras
   a 28 px dan 2195 px de alto sobre un lienzo de 584. Asi que "muestrear el tamano"
   se hace al reves: se sortea el alto OBJETIVO y se DERIVA el numero de palabras.
   El predictor es
        palabras ~= lineas_objetivo * C(fuente) * ancho / cuerpo
   con C medido por fuente (CONST_PALABRA). Es aproximado a proposito: luego se mide
   el render de verdad y se corrige.

3. POR ESO CADA MUESTRA SE RENDERIZA DOS VECES, y no es desperdicio:
     pase A  en la esquina minima (68, 68) -> se MIDE la caja real
     pase B  en la esquina sorteada para ESA caja -> es la muestra que se guarda
   Sin el pase A no se puede saber el rango de esquina valido, y sin ese rango la
   unica salida seria rechazar, que es lo que el §3.4 prohibe.

4. LA ETIQUETA ES LA CAJA DE TINTA, no la caja de maquetacion. Es la union de las
   cajas de PALABRA que devuelve el generador. Decision de este experimento, con su
   motivo: cada uno de los cuatro bordes tiene que tener evidencia visual, o el
   soft-argmax busca algo que no esta ahi -- la misma clase de error irreducible que
   el §3.3 existe para evitar. Por eso tambien `align: justify`: con alineacion a la
   izquierda el borde derecho lo marca el final ragged de cada linea y la tinta NO
   llega al borde de la caja. Medido: con justify, tinta x1 = 328,1 contra caja
   x1 = 328,0.

5. SE GUARDA uint16 CON LA SUMA DEL BLOQUE 4x4 (§3.8), no el promedio. 16 pixeles
   uint8 suman como maximo 4080, que cabe en uint16: la representacion es EXACTA, sin
   cuantizacion. El factor 16 global desaparece con la estandarizacion del §6.5.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))

from expcnn import SUBDIR_DATASETS, exigir_datos, exigir_generador, ruta_dataset  # noqa: E402

# ---------------------------------------------------------------- constantes
NOMBRE = "parrafos1000-584px-r4-r20260908b"
LIENZO = 584
REDUCCION = 4
MARCO = LIENZO // REDUCCION          # 146
DESCARTE = 9                          # §6.2: descarte fijo por lado
FINAL = MARCO - 2 * DESCARTE          # 128
COLOCACION = (68, 512)                # §3.3, en el marco de 584
N_TOTAL = 1000
PARTICIONES = {"train": 100, "monitor": 100, "eval": 800}
SEMILLA = 1
DATOS = EXP / "datos"

# §3.5: los siete factores. Ancho y alto son los CRITICOS y quieren >= 2x.
#
# ⚠⚠ ESTOS RANGOS SON LA SEGUNDA VERSION, Y EL MOTIVO ES UNA MEDIDA. Los primeros
# (ancho 180-400, alto 120-400, uniformes) dieron un control de CAJA MEDIA de
# IoU = 0,5261 sobre `eval`, muy por encima del umbral <= 0,40 del §10.1: un
# predictor constante que ni mira la imagen acertaba la mitad. Con eso el banco no
# discrimina nada, y el §10.1 dice exactamente que hacer -- «se AMPLIA el rango de
# ancho y alto de caja, NO el de posicion, y NO se continua».
#
# Y hay una segunda parte que no es el rango sino la FORMA de sortearlo: LOG-uniforme
# en vez de uniforme. Uniforme en pixeles concentra las cajas en las grandes (una caja
# de 400 y otra de 380 son casi la misma), y las grandes ademas no pueden moverse: en
# una ventana de 444 px, una caja de 400 tiene 44 px de juego. Log-uniforme reparte las
# ESCALAS, que es lo que el §3.5 quiere decir con «>= 2x entre minimo y maximo».
#
# Medido ANTES de renderizar nada, porque la caja media depende SOLO de la distribucion
# de cajas y no de las imagenes -- asi que se puede simular en segundos en vez de pagar
# 21 min de renders por cada intento (buscar_rangos.py; el simulador da 0,513 contra el
# 0,528 real con los rangos viejos, o sea que es fiel):
#
#     rangos viejos, uniforme     -> 0,513 simulado   (0,5261 real)  ✗
#     estos rangos, uniforme      -> 0,366 simulado                  margen escaso
#     estos rangos, LOG-uniforme  -> 0,231 simulado                  ✓
ANCHO = (80.0, 420.0)                 # 5,25x  (log-uniforme)
ALTO_OBJETIVO = (45.0, 420.0)         # 9,33x  (log-uniforme)
CUERPO = (11.0, 30.0)                 # 2,73x. El tope sube a 30 para acercarse al
                                      # §3.2: «altura de linea ~10 px en el marco
                                      # reducido» son ~40 px en el de 584.
INTERLINEADO = (1.15, 1.60)
GRIS = (0.0, 0.40)                    # fraccion de negro->gris: 0 = #000, 0,4 = #666
FUENTES = ["DejaVuSans", "DejaVuSerif", "LiberationMono", "LiberationSans",
           "LiberationSerif"]
# §3.7 RESERVA: de uso exclusivo del banco. Los procedimientos que producen kernels
# NO pueden usar esta familia ni este rango de densidad. Se declara aqui y en REGLAS.md.
RESERVA_FUENTE = "LiberationMono"
RESERVA_DENSIDAD = (1.45, 1.60)       # interlineado alto, reservado

# Palabras por linea normalizadas: wpl * cuerpo / ancho. Medido el 2026-09-08 con una
# render por fuente (ver instrucciones/01-encargo.md). Es un PREDICTOR, no un dato
# exacto: el pase A mide la caja de verdad y `_reparar` corrige si hace falta.
CONST_PALABRA = {"DejaVuSans": 0.293, "DejaVuSerif": 0.289, "LiberationMono": 0.248,
                 "LiberationSans": 0.327, "LiberationSerif": 0.375}
# El ancho minimo en cuerpos: por debajo, `justify` estira lineas de 2 palabras.
ANCHO_MIN_CUERPOS = 6.0
MAX_LADO = COLOCACION[1] - COLOCACION[0]     # 444: lo mas grande que puede caber


# ---------------------------------------------------------------- muestreo
def _hipercubo(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """Hipercubo latino en [0,1)^dims (§3.6).

    Cada dimension se parte en n estratos y se visita cada uno UNA vez, en orden
    aleatorio. Con muestreo independiente, `train` (100 de 1000) puede agruparse por
    azar en una region; con esto cubre el espacio parejo, que es lo que el §3.6 pide."""
    u = np.empty((n, dims))
    for d in range(dims):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return u


def _log_uniforme(lo: float, hi: float, u: float) -> float:
    """Sortea entre `lo` y `hi` repartiendo las ESCALAS, no los pixeles.

    Uniforme en px concentra las cajas en las grandes -- 400 y 380 son casi la misma
    caja -- y las grandes ademas casi no se pueden mover dentro de la ventana de 444.
    El resultado es que todas se parecen y el control de caja media se dispara."""
    if hi <= lo:
        return float(lo)
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def factores(n: int, semilla: int) -> list[dict]:
    """Los factores de cada imagen. Sorteados ANTES de renderizar nada (§3.4)."""
    rng = np.random.default_rng(semilla)
    # 0 cuerpo · 1 ancho · 2 alto objetivo · 3 interlineado · 4 gris · 5 ux · 6 uy
    u = _hipercubo(n, 7, rng)
    # La familia se reparte en bloques iguales y se permuta: balance exacto, no por azar.
    fam = np.array([FUENTES[i % len(FUENTES)] for i in range(n)])
    fam = fam[rng.permutation(n)]
    out = []
    for i in range(n):
        cuerpo = CUERPO[0] + u[i, 0] * (CUERPO[1] - CUERPO[0])
        # El ancho depende del cuerpo: con un cuerpo grande en una caja estrecha,
        # `justify` estira lineas de 2 palabras. Se acota por abajo y se DICE.
        ancho_lo = max(ANCHO[0], ANCHO_MIN_CUERPOS * cuerpo)
        ancho = _log_uniforme(ancho_lo, ANCHO[1], u[i, 1])
        alto = _log_uniforme(ALTO_OBJETIVO[0], ALTO_OBJETIVO[1], u[i, 2])
        lh = INTERLINEADO[0] + u[i, 3] * (INTERLINEADO[1] - INTERLINEADO[0])
        g = int(round(255 * (GRIS[0] + u[i, 4] * (GRIS[1] - GRIS[0]))))
        out.append({
            "i": i, "fuente": str(fam[i]), "cuerpo": round(cuerpo, 2),
            "ancho": round(ancho, 1), "alto_objetivo": round(alto, 1),
            "interlineado": round(lh, 3), "gris": f"#{g:02x}{g:02x}{g:02x}",
            "gris_nivel": g, "ux": float(u[i, 5]), "uy": float(u[i, 6]),
            "palabras": _palabras(alto, lh, cuerpo, ancho, str(fam[i])),
        })
    return out


def _palabras(alto: float, lh: float, cuerpo: float, ancho: float, fuente: str) -> int:
    """Cuantas palabras hacen falta para acercarse a `alto`. Predictor del punto 2."""
    lineas = max(1.0, alto / (cuerpo * lh))
    wpl = max(1.5, CONST_PALABRA[fuente] * ancho / cuerpo)
    return max(4, int(round(lineas * wpl)))


# ---------------------------------------------------------------- render
def _receta(f: dict, x: float, y: float, palabras: int):
    from app.models.recipe import Recipe
    return Recipe.model_validate({
        "canvas": {"width": LIENZO, "height": LIENZO},
        "background": {"kind": "solid", "color": "#ffffff"},
        "fonts": [f["fuente"]],
        "blocks": [{
            "kind": "paragraph", "count": 1, "width": f["ancho"],
            "content": {"source": "words", "lang": "es", "words": palabras},
            "typography": {
                "font_family": f["fuente"], "font_size": f["cuerpo"],
                "line_height": f["interlineado"], "color": f["gris"],
                "min_contrast": 1.0, "align": "justify", "font_weight": 400,
            },
            # §3.4: la esquina se FIJA en px. `area` no sirve: resta un tamano
            # ESTIMADO (`_place`: hi_y = max(y0, y1 - rh)) y para parrafos la
            # estimacion se queda corta, asi que la caja se sale del lienzo.
            "placement": {"x": float(x), "y": float(y), "angle": 0,
                          "avoid_overlap": False},
        }],
    })


def _caja_tinta(labels) -> tuple[float, float, float, float] | None:
    """Union de las cajas de PALABRA: la extension real de la tinta (punto 4)."""
    ws = labels.words
    if not ws:
        return None
    return (min(w.box[0] for w in ws), min(w.box[1] for w in ws),
            max(w.box[0] + w.box[2] for w in ws), max(w.box[1] + w.box[3] for w in ws))


def _reducir(img) -> np.ndarray:
    """584x584 -> 146x146 uint16 con la SUMA de cada bloque 4x4 (§3.8). Exacto."""
    a = np.asarray(img.convert("L"), dtype=np.uint16)
    return a.reshape(MARCO, REDUCCION, MARCO, REDUCCION).sum(axis=(1, 3))


async def generar(n: int, semilla: int) -> dict:
    gen = exigir_generador()
    sys.path.insert(0, str(gen))
    from app.core.renderer import Renderer          # noqa: PLC0415
    from app.core.resolver import resolve           # noqa: PLC0415

    fs = factores(n, semilla)
    r = Renderer()
    await r.start()
    imgs, etiquetas, meta = [], [], []
    reparos, renders = 0, 0
    t0 = time.time()
    try:
        for f in fs:
            palabras = f["palabras"]
            # --- pase A: medir la caja real en la esquina minima ---
            for intento in range(6):
                spec = resolve(_receta(f, COLOCACION[0], COLOCACION[0], palabras),
                               semilla * 100000 + f["i"])
                res = await r.render(spec)
                renders += 1
                caja = _caja_tinta(res.labels)
                if caja is None:
                    palabras = max(4, int(palabras * 1.5))
                    reparos += 1
                    continue
                w, h = caja[2] - caja[0], caja[3] - caja[1]
                if w <= MAX_LADO and h <= MAX_LADO:
                    break
                # REPARO, no rechazo: la muestra se conserva, se le baja la densidad
                # hasta que quepa. Se cuenta y se reporta, porque encoge el alto.
                palabras = max(4, int(palabras * min(0.9, MAX_LADO / h * 0.92)))
                reparos += 1
            else:
                raise RuntimeError(f"imagen {f['i']}: no cabe ni con 6 reparos")

            # --- la esquina, sorteada para ESTA caja (§3.4) ---
            lo, hi = COLOCACION
            x = lo + f["ux"] * max(0.0, (hi - lo) - w)
            y = lo + f["uy"] * max(0.0, (hi - lo) - h)
            # La tinta arranca desplazada respecto de la esquina del bloque: se corrige
            # con el desplazamiento medido en el pase A, para que la TINTA caiga donde
            # se sorteo y no el bloque.
            dx, dy = caja[0] - COLOCACION[0], caja[1] - COLOCACION[0]

            # --- pase B: la muestra que se guarda ---
            spec = resolve(_receta(f, x - dx, y - dy, palabras),
                           semilla * 100000 + f["i"])
            res = await r.render(spec)
            renders += 1
            caja = _caja_tinta(res.labels)
            if caja is None:
                raise RuntimeError(f"imagen {f['i']}: pase B sin palabras")

            izq, sup, der, inf = (int(round(v)) for v in caja)
            imgs.append(_reducir(res.image))
            etiquetas.append((izq, der, sup, inf))
            meta.append({**{k: v for k, v in f.items() if k not in ("ux", "uy")},
                         "palabras_final": palabras, "lineas": len(res.labels.lines),
                         "ancho_real": round(caja[2] - caja[0], 1),
                         "alto_real": round(caja[3] - caja[1], 1)})
            if (f["i"] + 1) % 50 == 0:
                print(f"  {f['i']+1}/{n}  ({time.time()-t0:.0f} s, "
                      f"{renders} renders, {reparos} reparos)", flush=True)
    finally:
        await r.stop()
    return {"imgs": imgs, "etiquetas": etiquetas, "meta": meta,
            "reparos": reparos, "renders": renders, "segundos": time.time() - t0}


# ---------------------------------------------------------------- aserciones
def aseverar(etiquetas: np.ndarray) -> None:
    """§3.3 y §6.4. Aserciones, no supuestos: un fallo aqui contamina la metrica."""
    lo, hi = COLOCACION
    izq, der, sup, inf = (etiquetas[:, k] for k in range(4))

    # §3.3 danyo (1): fuera del rango valido -> borde fuera del marco final
    for nombre, v in (("izq", izq), ("der", der), ("sup", sup), ("inf", inf)):
        mal = np.where((v < lo) | (v > hi))[0]
        assert len(mal) == 0, (
            f"§3.3(1): {len(mal)} caja(s) con {nombre} fuera de [{lo}, {hi}] "
            f"(p.ej. imagen {mal[:5].tolist()} = {v[mal[:5]].tolist()})")
    # §3.3 danyo (2): fuera del lienzo -> parrafo cortado, etiqueta FALSA
    mal = np.where((izq < 0) | (sup < 0) | (der > LIENZO) | (inf > LIENZO))[0]
    assert len(mal) == 0, f"§3.3(2): {len(mal)} caja(s) fuera del lienzo de {LIENZO}"
    # cajas coherentes
    assert (der > izq).all() and (inf > sup).all(), "caja con lado <= 0"

    # §6.4: la transformacion sobre una muestra CONOCIDA, no sobre el rango teorico
    for c584, esperado in ((68, 8.0), (512, 119.0), (256, 55.0)):
        obtenido = c584 / REDUCCION - DESCARTE
        assert abs(obtenido - esperado) < 1e-9, f"§6.4: {c584} -> {obtenido} != {esperado}"
    # y sobre las etiquetas de verdad: todas alcanzables en el marco de 128
    f128 = etiquetas / REDUCCION - DESCARTE
    assert f128.min() >= 0 and f128.max() <= FINAL - 1 + 1e-9, (
        f"§6.4: etiquetas transformadas fuera de [0, {FINAL-1}]: "
        f"[{f128.min():.2f}, {f128.max():.2f}]")


def _restos_mayores(porbin: list[int], cupo: int) -> list[int]:
    """Reparte `cupo` entre bins proporcionalmente, con total EXACTO.

    Suelo + los restos mas grandes primero, y sin pasarse del tamanyo de cada bin."""
    n = sum(porbin)
    if n == 0 or cupo <= 0:
        return [0] * len(porbin)
    exacto = [cupo * b / n for b in porbin]
    dado = [min(int(e), porbin[i]) for i, e in enumerate(exacto)]
    resto = cupo - sum(dado)
    orden = sorted(range(len(porbin)), key=lambda i: exacto[i] - dado[i], reverse=True)
    j = 0
    while resto > 0 and j < len(orden) * 4:
        i = orden[j % len(orden)]
        if dado[i] < porbin[i]:
            dado[i] += 1
            resto -= 1
        j += 1
    return dado


def cupos(n: int) -> dict[str, int]:
    """Cuantas van a cada particion. Con n = N_TOTAL son EXACTAMENTE 100/100/800.

    Escala para poder ensayar con pocas imagenes sin tocar el codigo que luego corre
    de verdad: un ensayo que necesita otro camino no ensaya el camino que importa."""
    if n == N_TOTAL:
        return dict(PARTICIONES)
    c = {k: max(1, round(v * n / N_TOTAL)) for k, v in PARTICIONES.items()}
    c["eval"] = n - c["train"] - c["monitor"]
    return c


def particionar(etiquetas: np.ndarray, semilla: int) -> dict[str, np.ndarray]:
    """§3.6: estratificacion EXPLICITA por area de caja, en 4 bins de cuartil."""
    rng = np.random.default_rng(semilla + 999)
    pedidos = cupos(len(etiquetas))
    area = (etiquetas[:, 1] - etiquetas[:, 0]) * (etiquetas[:, 3] - etiquetas[:, 2])
    bins = np.quantile(area, [0.25, 0.5, 0.75])
    cuartil = np.digitize(area, bins)          # 0..3
    porbin = [int((cuartil == q).sum()) for q in range(4)]
    # El cupo por cuartil se reparte con RESTOS MAYORES, no redondeando cada uno por su
    # cuenta: redondear independientemente pierde o inventa muestras (con n=12 y cupo 1,
    # round(1*3/12) = 0 en los cuatro cuartiles y `train` se quedaba VACIO).
    plan = {k: _restos_mayores(porbin, pedidos[k]) for k in ("train", "monitor")}
    idx = {k: [] for k in PARTICIONES}
    for q in range(4):
        cual = np.where(cuartil == q)[0]
        cual = cual[rng.permutation(len(cual))]
        a, b = plan["train"][q], plan["monitor"][q]
        idx["train"] += cual[:a].tolist()
        idx["monitor"] += cual[a:a + b].tolist()
        idx["eval"] += cual[a + b:].tolist()
    salida = {k: np.array(sorted(v), dtype=np.int32) for k, v in idx.items()}
    total = sum(len(v) for v in salida.values())
    assert total == len(area), f"el reparto pierde muestras: {total} != {len(area)}"
    for k in ("train", "monitor"):
        assert abs(len(salida[k]) - pedidos[k]) <= 4, (
            f"{k} tiene {len(salida[k])}, se pidieron {pedidos[k]}")
    return salida


def balance_marginal(meta: list[dict], part: dict[str, np.ndarray]) -> dict:
    """§3.6: el balance marginal SE VERIFICA, no se fuerza. Devuelve lo medido."""
    out = {}
    for campo in ("cuerpo", "interlineado", "gris_nivel", "ancho_real", "alto_real"):
        out[campo] = {k: (round(float(np.mean([meta[i][campo] for i in v])), 3)
                          if len(v) else None) for k, v in part.items()}
    out["fuente"] = {k: {f: sum(1 for i in v if meta[i]["fuente"] == f) for f in FUENTES}
                     for k, v in part.items()}
    return out


def _huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- guardar
def _guardar(r: dict, semilla: int) -> int:
    imgs = np.stack(r["imgs"]).astype(np.uint16)
    etiquetas = np.array(r["etiquetas"], dtype=np.int32)
    print(f"\nimagenes: {imgs.shape} {imgs.dtype}  ({imgs.nbytes/1e6:.1f} MB en crudo)")
    assert imgs.max() <= 255 * REDUCCION ** 2, "la suma del bloque se sale de uint16"

    aseverar(etiquetas)
    print("aserciones §3.3 y §6.4: OK")

    part = particionar(etiquetas, semilla)
    print("particiones: " + " · ".join(f"{k}={len(v)}" for k, v in part.items()))
    bal = balance_marginal(r["meta"], part)

    DATOS.mkdir(parents=True, exist_ok=True)
    huellas = {}
    for k, idx in part.items():
        f = DATOS / f"{k}.npz"
        np.savez_compressed(f, imagenes=imgs[idx], etiquetas=etiquetas[idx], indices=idx)
        huellas[k] = {"n": int(len(idx)), "sha256_16": _huella(imgs[idx]),
                      "sha256_16_etiquetas": _huella(etiquetas[idx]),
                      "bytes_npz": f.stat().st_size}
        print(f"  {k:8} {len(idx):>4} muestras  {f.stat().st_size/1e6:>5.1f} MB  "
              f"{huellas[k]['sha256_16']}")

    manifiesto = {
        "nombre": NOMBRE,
        "experimento": "banco-k",
        "especificacion": "v1.2",
        "generado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "semilla": semilla,
        "lienzo": LIENZO, "reduccion": REDUCCION, "marco": MARCO,
        "descarte_por_lado": DESCARTE, "marco_final": FINAL,
        "colocacion": list(COLOCACION),
        "almacenamiento": "uint16, SUMA del bloque 4x4 (§3.8): exacto, sin cuantizacion",
        "etiqueta": "4 enteros en el marco de 584: izq, der, sup, inf. Es la caja de "
                    "TINTA (union de cajas de palabra), no la de maquetacion",
        "alineacion": "justify: sin ella la tinta no llega al borde derecho de la caja",
        "factores": {
            "ancho_px": list(ANCHO), "alto_objetivo_px": list(ALTO_OBJETIVO),
            "ancho_y_alto_se_sortean": "LOG-uniforme (§3.5: reparte escalas)",
            "cuerpo_px": list(CUERPO), "interlineado": list(INTERLINEADO),
            "gris_fraccion": list(GRIS), "fuentes": FUENTES,
            "ancho_minimo_en_cuerpos": ANCHO_MIN_CUERPOS,
            "muestreo": "hipercubo latino en 7 dimensiones (§3.6); familia en bloques "
                        "iguales permutados",
        },
        "reserva_§3.7": {"fuente": RESERVA_FUENTE, "interlineado": list(RESERVA_DENSIDAD),
                         "regla": "de uso EXCLUSIVO del banco: los procedimientos que "
                                  "producen kernels no pueden usar esta familia ni este "
                                  "rango de interlineado"},
        "estratificacion": "explicita por AREA de caja en 4 bins de cuartil (§3.6)",
        "balance_marginal_medido": bal,
        "imagenes_pedidas": N_TOTAL, "imagenes_validas": int(len(etiquetas)),
        "rechazadas": 0,
        "reparos_de_densidad": r["reparos"],
        "renders": r["renders"], "segundos": round(r["segundos"], 1),
        "particiones": huellas,
        "rango_etiquetas_584": [int(etiquetas.min()), int(etiquetas.max())],
        "rango_etiquetas_128": [round(float(etiquetas.min() / REDUCCION - DESCARTE), 2),
                                round(float(etiquetas.max() / REDUCCION - DESCARTE), 2)],
    }
    (DATOS / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(DATOS / "meta.npz", meta=json.dumps(r["meta"]))
    print(f"\nreparos de densidad: {r['reparos']} · rechazadas: 0 (§3.4)")
    print(f"rango de etiquetas: {manifiesto['rango_etiquetas_584']} en 584  ->  "
          f"{manifiesto['rango_etiquetas_128']} en 128")
    print(f"manifiesto: {DATOS / 'manifiesto.json'}")
    return 0


def _publicar() -> int:
    """Publica en el repo de datos. NUNCA pisa uno existente (regla 3 del repo)."""
    if not (DATOS / "manifiesto.json").is_file():
        print("✗ no hay etapa local. Genera primero: --imagenes 1000")
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
            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    _readme(destino)
    print(f"\npublicado en {destino}")
    print("⚠ El push es parte del encargo: el repo de datos es PRIVADO y git no olvida,")
    print("  pero lo que no esta empujado no existe.")
    return 0


def _readme(destino: Path) -> None:
    m = json.loads((destino / "manifiesto.json").read_text(encoding="utf-8"))
    p = m["particiones"]
    destino.joinpath("README.md").write_text(f"""# `{NOMBRE}`

Dataset de **párrafos** para el banco de evaluación de kernels (`banco-k` de
`experimentos-cnn`), especificación **{m['especificacion']}** §3-§4.

| | |
|---|---|
| lienzo | {m['lienzo']} × {m['lienzo']} px, fondo blanco, 1 canal |
| reducción | /{m['reduccion']} por promedio de área → **{m['marco']} × {m['marco']}** |
| almacenamiento | **{m['almacenamiento']}** |
| etiqueta | {m['etiqueta']} |
| muestras | {m['imagenes_validas']} · train {p['train']['n']} / monitor {p['monitor']['n']} / eval {p['eval']['n']} |
| estratificación | {m['estratificacion']} |
| generado | {m['generado']} · semilla {m['semilla']} |

## Por qué se guarda a 146 y no a 128

El recorte a 128 **depende del kernel** (§6.2: convolución `valid` con `k` y luego
recorte para que el descarte sea siempre 9 px por lado). Un dataset ya recortado no
dejaría aplicar ningún kernel. Este dato es **pre-kernel** a propósito.

Y las etiquetas van en el marco de **584 sin transformar**, porque
`(coord/4) − 9` es parte del **pipeline** (§6.4), no del dato.

## ⚠ Reserva contra fuga de distribución (§3.7)

**`{m['reserva_§3.7']['fuente']}`** e interlineado en
`{m['reserva_§3.7']['interlineado']}` son de **uso exclusivo del banco**. Un
procedimiento que produzca kernels usando el mismo generador tiene **fuga de
distribución aunque las muestras sean distintas**; esta reserva es lo que la acota.

## Huellas

| partición | n | sha256-16 imágenes | sha256-16 etiquetas |
|---|---|---|---|
""" + "\n".join(f"| `{k}` | {v['n']} | `{v['sha256_16']}` | `{v['sha256_16_etiquetas']}` |"
                for k, v in p.items()) + f"""

Se comprueban con `python nn/datos.py --comprobar` desde la carpeta del experimento.

## Quién lo usa

`experimentos-cnn/<carpeta de banco-k>` — y **compartir el dataset no es compartir
condiciones**: otro experimento puede leer otras columnas y medir otra cosa.
""", encoding="utf-8")


def _comprobar() -> int:
    p = ruta_dataset(NOMBRE)
    if p is None:
        print(f"✗ '{NOMBRE}' no esta publicado. Publica con --publicar")
        return 1
    m = json.loads((p / "manifiesto.json").read_text(encoding="utf-8"))
    malos = 0
    for k, esperado in m["particiones"].items():
        d = np.load(p / f"{k}.npz")
        h = _huella(d["imagenes"])
        he = _huella(d["etiquetas"])
        ok = h == esperado["sha256_16"] and he == esperado["sha256_16_etiquetas"]
        print(f"  {'ok ' if ok else 'MAL'} {k:8} n={len(d['etiquetas']):>4} "
              f"imagenes {h} etiquetas {he}")
        if not ok:
            malos += 1
        aseverar(d["etiquetas"])
    print(f"\n{'✓ el publicado casa con su manifiesto' if not malos else '✗ NO casa'}")
    return 1 if malos else 0


def _rederivar(n: int, semilla: int) -> int:
    """¿La receta y la semilla vuelven a dar lo mismo? Prueba de honestidad.

    No es una alternativa a publicar: contesta si la RECETA es honesta. Se hacen las
    N primeras y no las 1000 porque cuesta ~2 renders por muestra y la propiedad que
    se comprueba (determinismo) no necesita las mil."""
    ruta = ruta_dataset(NOMBRE) or DATOS
    if not (ruta / "train.npz").is_file():
        print("✗ no hay nada con que comparar")
        return 1
    todo = {}
    for k in PARTICIONES:
        d = np.load(ruta / f"{k}.npz")
        for j, i in enumerate(d["indices"]):
            todo[int(i)] = (d["imagenes"][j], d["etiquetas"][j])
    r = asyncio.run(generar(n, semilla))
    malos = 0
    for i in range(n):
        if i not in todo:
            continue
        img_v, et_v = todo[i]
        if not np.array_equal(img_v, r["imgs"][i]) or not np.array_equal(
                et_v, np.array(r["etiquetas"][i], dtype=np.int32)):
            print(f"  MAL imagen {i}: no coincide")
            malos += 1
    print(f"\n{n - malos}/{n} identicas" + ("" if malos else " → la receta es honesta"))
    return 1 if malos else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenes", type=int, default=0)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--publicar", action="store_true")
    ap.add_argument("--comprobar", action="store_true")
    ap.add_argument("--rederivar", type=int, default=0)
    ap.add_argument("--muestras", type=int, default=0)
    ap.add_argument("--semilla-fig", type=int, default=7)
    a = ap.parse_args()
    if a.imagenes:
        return _guardar(asyncio.run(generar(a.imagenes, a.semilla)), a.semilla)
    if a.publicar:
        return _publicar()
    if a.comprobar:
        return _comprobar()
    if a.rederivar:
        return _rederivar(a.rederivar, a.semilla)
    if a.muestras:
        from muestras import figura                 # noqa: PLC0415
        return figura(a.muestras, a.semilla_fig)
    ap.error("dime que hacer: --imagenes N · --publicar · --comprobar · "
             "--rederivar N · --muestras N")


if __name__ == "__main__":
    raise SystemExit(main())
