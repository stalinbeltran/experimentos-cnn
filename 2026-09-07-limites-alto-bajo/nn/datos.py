#!/usr/bin/env python3
"""El dataset: parrafos limpios -> ventanas de 32x32 con SUS DOS esquinas.

    python nn/datos.py --imagenes 300      genera en la etapa local
    python nn/datos.py --publicar          lo copia al repo de DATOS (no pisa)
    python nn/datos.py --comprobar         el publicado, ¿coincide con el manifiesto?
    python nn/datos.py --rederivar         ¿la receta y la semilla lo vuelven a dar?

QUE CAMBIA RESPECTO DE `esq-k`, Y QUE NO
    NO cambia nada de lo que decide la comparabilidad: misma receta, misma
    semilla, misma reduccion por promedio de area, mismas condiciones de caja,
    mismo reparto (12 imagenes para las muestras, 15 % de val). O sea que las
    IMAGENES son las mismas; lo que cambia son las ventanas que se sacan de
    ellas y sus etiquetas.

    SI cambia la etiqueta: ahora son SEIS numeros por ventana --
    (existe_tl, x_tl, y_tl, existe_br, x_br, y_br) -- y las clases negativas
    incluyen las dos esquinas de la OTRA diagonal (tr, bl), que son el negativo
    duro de este experimento: tienen exactamente la misma forma local que las
    buscadas, girada 90 grados.

⚠ UNA VENTANA NUNCA CONTIENE LAS DOS ESQUINAS, y hay que decirlo porque es un
    limite del experimento y no un detalle: el parrafo mide >= 64 px reducidos
    (condicion 2) y la ventana 32, asi que tl y br estan siempre a mas de una
    ventana de distancia. Se comprueba y se cuenta al empaquetar, en vez de
    darlo por supuesto.

⚠ POR QUE SE COPIA ESTE FICHERO EN VEZ DE IMPORTARLO DE `esq-k`
    Regla del repo: ningun experimento importa de otro, y `comun/` se crea
    cuando DOS experimentos tengan que medir con la MISMA regla -- que no es el
    caso: aqui la etiqueta tiene seis numeros y alli tres. La copia es
    deliberada y su precio (dos ficheros que pueden divergir) es menor que el de
    un `comun/` inventado con el segundo experimento.

LAS CONDICIONES, IGUALES QUE EN `esq-k` (alli esta el porque de cada numero)
    1. parrafo a >= 32 px (reducidos) del borde de la imagen.
    2. parrafo >= 64 px de lado = 2 ventanas.
    3. la esquina cae en [8, 23] de la ventana: >= 5 px de margen ciego (k=11) y
       >= 8 px de contexto a cada lado. Para br el contexto es el mismo numero
       con los papeles cambiados: >= 8 px de parrafo arriba-izquierda y >= 8 px
       de fondo abajo-derecha.
    4. `has_overlap` falso.

DONDE VIVE EL DATO, Y POR QUE ALLI
    En el repo de DATOS (`foveal-vision-data/experimentos-cnn/<nombre>/`), por
    orden del dueno del 2026-09-07: «Guarda los datasets en el repo de data, de
    modo que sean siempre los mismos, por consistencia».

    NO en este repo, y eso no es una preferencia: este es PUBLICO y aquel es
    PRIVADO (comprobado el 2026-09-06 contra la API de GitHub), y git no olvida.
    Y no en `window-datasets/`, que es de `foveal-vision` y lo resuelve su propio
    `settings`: meter ahi un dataset de otra forma seria una colision silenciosa.

    ⚠ Y ESO ARREGLA UN AGUJERO REAL, no solo ordena. Las 10 muestras congeladas
    de `esq-k` se declaraban commiteadas y NO lo estaban: el `.gitignore` de este
    repo tiene `*.npz` para que no se cuele el dato de entrada, y se llevaba
    tambien la verificacion. Medido el 2026-09-07 en un clon limpio: su
    `nn/muestras.npz` no existe (`git ls-files | grep npz` da cero), o sea que la
    figura de verificacion de aquel experimento no se podia regenerar sin volver
    a rendir 300 imagenes. Publicadas con el dataset, dejan de perderse.

    "Se re-deriva" seguia siendo verdad y aun asi no bastaba: un dato re-derivado
    es el mismo *mientras nada cambie*, y "nada cambia" no es comprobable hacia
    el futuro -- basta que el generador cambie una fuente. Publicado, es el mismo
    porque es EL MISMO FICHERO. La re-derivabilidad se conserva como PRUEBA
    (`--rederivar`), que es para lo que sirve.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
from expcnn import SUBDIR_DATASETS, exigir_datos, exigir_generador, ruta_dataset

VENTANA = 32
REDUCCION = 4
MARGEN_BORDE = 32
LADO_MIN = 64
K_MAX = 11
CIEGO = (K_MAX - 1) // 2
CONTEXTO = 8
OFF_MIN, OFF_MAX = max(CIEGO, CONTEXTO), VENTANA - 1 - CONTEXTO
SEMILLA = 1
N_MUESTRA_IMG = 12
FRAC_VAL = 0.15

# EL NOMBRE DEL DATASET PUBLICADO. Por orden del dueno (2026-09-07), el dato de
# entrada vive en el repo de DATOS y es siempre el mismo fichero, no uno
# equivalente re-derivado. `dato nuevo = nombre nuevo`, que es la regla que ese
# repo ya tiene escrita: nunca se reescribe uno publicado.
DATASET = "limites300-32px-r4-r20260907"
PARTES = ("train", "val", "muestra")
CONGELADAS = "muestras-congeladas.npz"

# La etapa LOCAL, donde se genera antes de publicar. Ignorada por git.
DATOS = EXP / "datos"

# ⚠ SE ETIQUETAN LAS CUATRO ESQUINAS, no solo la diagonal que mide `esq-2d`.
# No cuesta nada (las cuatro coordenadas ya se conocen al recortar) y es lo que
# hace que "si pueden usar el mismo dataset no hay problema" sea cierto tambien
# para el experimento siguiente: el que mire la OTRA diagonal, o las cuatro, no
# tiene que re-rendir nada. `esq-2d` lee `tl` y `br` e ignora el resto; que un
# recorte de `tr` traiga su etiqueta no lo convierte en positivo suyo.
ETIQUETADAS = ("tl", "tr", "bl", "br")

# LOS DOS LIMITES HORIZONTALES de un parrafo, que es lo que mide `lim-h`.
LIMITES = ("sup", "inf")

# ⚠ DE DONDE SALE LA ALTURA DE CADA UNO. Orden del dueno (2026-09-07): «La altura
# tomala de la pos y de tl y bl (borde alto y bajo)». Y se EXTIENDE a `tr`/`br`
# porque no es interpretacion, es aritmetica de la caja: con caja (x,y,w,h) se
# tiene tl=(x,y) y tr=(x+w,y) -- LA MISMA `y` --, y bl=(x,y+h) y br=(x+w,y+h),
# tambien la misma. Una ventana de `tr` contiene el borde ALTO exactamente igual
# que una de `tl`; lo unico que cambia es por donde lo corta.
#
# Dejarlas fuera no seria "ser fiel a la orden": seria etiquetar como AUSENTE un
# limite que esta ahi, con tinta a 1 px, en 78 de 389 ventanas. Eso no es ruido
# de etiqueta, es una ANTI-etiqueta, y ademas obligaria a la red a resolver
# `existe` por el canto VERTICAL de la esquina -- o sea a volver a ser `esq-cq`
# en silencio.
DE_ESQUINA = {"tl": "sup", "tr": "sup", "bl": "inf", "br": "inf"}


def objetivo(d):
    """La etiqueta de `lim-h`: (existe, y). UNA salida, y SOLO la altura.

    `existe = existe_sup | existe_inf`, y la coordenada es la fila del que exista.

    ⚠ NO HAY `x`, y no es una simplificacion: un limite horizontal es una LINEA,
    asi que su `x` no esta definida. La `x` que guardaba `esq-cq` era el punto de
    anclaje SORTEADO al recortar, o sea ruido de muestreo; meterla en la perdida
    seria entrenar contra ruido, y en la metrica haria que la mitad de una
    distancia euclidea no significase nada.

    ⚠ VIVE AQUI Y SE IMPORTA, no se copia. La usan TRES consumidores
    (`entrenar_local`, `muestras`, `informe`) y tres copias de una derivacion
    divergen sin que nadie se entere -- es la misma razon por la que el
    coordinador tiene `workspaces-locales.mjs` en vez de la comparacion repetida
    en dos scripts.

    ⚠ Y el `assert` no es decorativo: toda la pregunta de `esq-cq` descansa en que
    una ventana no pueda tener `tl` Y `br` a la vez, porque hay UNA sola salida de
    posicion y dos esquinas no caben en ella. Si algun dia el dataset cambia
    (parrafo mas corto, ventana mas grande), esto tiene que fallar A GRITOS y no
    quedarse con una de las dos en silencio.

    La garantia de fondo es geometrica y es la misma de siempre: LADO_MIN = 64 px
    de alto minimo y el offset va de OFF_MIN a OFF_MAX, asi que desde el ancla la
    ventana alcanza como mucho VENTANA - 1 - OFF_MIN px, muy por debajo de 64. El
    borde alto y el bajo no caben nunca en la misma ventana."""
    n = len(d["ventanas"])
    existe = np.zeros(n, np.int64)
    y = np.full(n, -1.0, np.float32)
    juntas = sum(np.asarray(d[f"existe_{c}"]) for c in LIMITES)
    if (juntas > 1).any():
        raise AssertionError(
            f"{int((juntas > 1).sum())} ventana(s) tienen {' Y '.join(LIMITES)} a la vez: "
            f"con UNA sola salida de altura, `lim-h` no puede representarlas. "
            f"El dataset o los parametros de recorte han cambiado.")
    for c in LIMITES:
        m = np.asarray(d[f"existe_{c}"]) == 1
        existe[m] = 1
        y[m] = np.asarray(d[f"y_{c}"])[m]
    return existe, y


def _reducir(img: Image.Image) -> np.ndarray:
    """Reduce por PROMEDIO DE AREA y devuelve TINTA: fondo 0, tinta 255."""
    g = img.convert("L")
    w, h = g.size
    chico = g.resize((w // REDUCCION, h // REDUCCION), Image.BOX)
    return 255 - np.asarray(chico, dtype=np.uint8)


def _caja_valida(caja, W, H) -> str | None:
    x, y, w, h = caja
    if w < LADO_MIN or h < LADO_MIN:
        return f"parrafo pequeno ({w:.0f}x{h:.0f} < {LADO_MIN})"
    if x < MARGEN_BORDE or y < MARGEN_BORDE:
        return f"pegado al borde sup/izq ({x:.0f},{y:.0f} < {MARGEN_BORDE})"
    if x + w > W - MARGEN_BORDE or y + h > H - MARGEN_BORDE:
        return "pegado al borde inf/der"
    return None


def _recorte(vista, cx, cy, ox, oy):
    """(ventana, x0, y0) cuyo pixel (ox, oy) es el punto (cx, cy) de la imagen.

    Devuelve tambien el ORIGEN porque sin el no se puede contar cuantas esquinas
    de la caja caen dentro del recorte, y ese conteo es el unico aval de que la
    salida unica de `esq-cq` es representable. Ver `_esquinas_dentro`."""
    x0, y0 = int(round(cx - ox)), int(round(cy - oy))
    H, W = vista.shape
    if x0 < 0 or y0 < 0 or x0 + VENTANA > W or y0 + VENTANA > H:
        return None
    return vista[y0:y0 + VENTANA, x0:x0 + VENTANA], x0, y0


def _esquinas_dentro(esquinas: dict, x0: int, y0: int) -> int:
    """Cuantas de las CUATRO esquinas de la caja caen dentro del recorte.

    ⚠ Se cuenta contra la GEOMETRIA, no contra la etiqueta que se acaba de
    escribir. El contador anterior hacia `sum(et[c][0]) > 1` sobre un `et` en el
    que `_ventanas` habia marcado UNA sola esquina: no podia dar otra cosa que 0,
    o sea un assert incapaz de fallar (R14). Y de ese 0 dependia la afirmacion
    "ninguna ventana contiene mas de una esquina", que es lo que hace que una
    salida unica de posicion sea siquiera representable."""
    return sum(1 for cx, cy in esquinas.values()
               if x0 <= cx < x0 + VENTANA and y0 <= cy < y0 + VENTANA)


# etiqueta vacia: ni esquina ni limite
VACIA = {c: (0, -1.0, -1.0) for c in ETIQUETADAS}
VACIA_LIM = {c: (0, -1.0) for c in LIMITES}


def _limites_dentro(caja, x0: int, y0: int) -> int:
    """Cuantos de los DOS limites horizontales cruzan el recorte.

    ⚠ GEOMETRICO, contra la caja, no contra la etiqueta que se acaba de escribir.
    Es la leccion que este mismo fichero ya pago una vez con las esquinas: un
    contador que suma lo que el propio codigo acaba de poner es un assert que no
    puede fallar. Y de este numero depende que una salida UNICA de altura sea
    siquiera representable."""
    x, y, w, h = caja
    return sum(1 for fila in (y, y + h) if y0 <= fila < y0 + VENTANA)


def _ventanas(vista, caja, rng):
    """Las 20 ventanas de una imagen: 8 positivas (40 %) y 12 negativas.

    ⚠ EL REPARTO NO SE HEREDA DE `esq-cq`, SE RECONSTRUYE -- y el 40 % es a
    proposito: es el mismo de alli, asi que el suelo del "siempre si" vuelve a
    ser 0,572 POR CONSTRUCCION y las dos tablas de `f1` se leen juntas.

    Las 6 positivas de esquina son literalmente las de `esq-cq` (2 tl · 2 br ·
    1 tr · 1 bl), asi que las celdas mantienen su n (78/39/39/78 en val) y sus
    SE. Las 2 nuevas son el CASO PURO -- un limite sin ninguna esquina --, que en
    `esq-cq` existia como negativo y cuya posicion se sorteaba y SE TIRABA.

    Los negativos cubren los modos de fallo de ESTE experimento, que no son los
    de aquel: el interior (lineas de texto, que son bordes horizontales falsos a
    montones) y los bordes VERTICALES izquierdo y derecho, que tienen tinta
    alineada y NO son un limite horizontal. La otra diagonal deja de ser negativo
    duro: aqui `tr` y `bl` son positivas."""
    x, y, w, h = caja
    esquinas = {"tl": (x, y), "tr": (x + w, y), "bl": (x, y + h), "br": (x + w, y + h)}
    fuera = []

    def sortear_off():
        return rng.randint(OFF_MIN, OFF_MAX), rng.randint(OFF_MIN, OFF_MAX)

    def anadir(v, x0, y0, clase, esq=None, lim=None, oy=None):
        et = dict(VACIA)
        el = dict(VACIA_LIM)
        if esq is not None:
            et[esq] = (1, float(ox_actual[0]), float(oy))
        if lim is not None:
            el[lim] = (1, float(oy))
        fuera.append((v, et, el, clase, _limites_dentro(caja, x0, y0)))

    ox_actual = [-1.0]

    # --- POSITIVAS de esquina (6): las MISMAS que en `esq-cq` ------------------
    for cual, veces in (("tl", 2), ("br", 2), ("tr", 1), ("bl", 1)):
        for _ in range(veces):
            ox, oy = sortear_off()
            r = _recorte(vista, *esquinas[cual], ox, oy)
            if r is not None:
                v, x0, y0 = r
                ox_actual[0] = ox
                anadir(v, x0, y0, f"esquina-{cual}", esq=cual,
                       lim=DE_ESQUINA[cual], oy=oy)

    # --- POSITIVAS puras (2): el limite SIN esquina ----------------------------
    # ⚠ ESTAS SON LA RAZON DE REGENERAR. En `esq-cq` existian y su `oy` se
    # sorteaba y se tiraba (`dict(VACIA)`), asi que eran positivos sin etiqueta.
    if w > 2 * VENTANA:
        for lado, fila, lim in (("superior", y, "sup"), ("inferior", y + h, "inf")):
            bx = rng.uniform(x + VENTANA, x + w - VENTANA)
            ox, oy = sortear_off()
            r = _recorte(vista, bx, fila, ox, oy)
            if r is not None:
                v, x0, y0 = r
                ox_actual[0] = -1.0
                anadir(v, x0, y0, f"borde-{lado}", lim=lim, oy=oy)

    # --- NEGATIVO DURO (4): los bordes VERTICALES -----------------------------
    # Tinta alineada y un canto claro, pero NO es un limite horizontal. Es el
    # negativo que este experimento necesita y que `esq-cq` no tenia: alli la
    # otra diagonal hacia ese papel, y aqui `tr`/`bl` son POSITIVAS.
    if h > 2 * VENTANA:
        for lado, col in (("izquierdo", x), ("derecho", x + w)):
            for _ in range(2):
                by = rng.uniform(y + VENTANA, y + h - VENTANA)
                r = _recorte(vista, col, by, *sortear_off())
                if r is not None:
                    v, x0, y0 = r
                    fuera.append((v, dict(VACIA), dict(VACIA_LIM),
                                  f"borde-{lado}", _limites_dentro(caja, x0, y0)))

    # --- NEGATIVO (4): interior del parrafo ------------------------------------
    # Lleno de bordes horizontales FALSOS: cada linea de texto tiene arriba y
    # abajo un canto. Es el modo de fallo mas probable de este experimento.
    if w > 2 * VENTANA and h > 2 * VENTANA:
        for _ in range(4):
            r = _recorte(vista, rng.uniform(x + VENTANA, x + w - VENTANA),
                         rng.uniform(y + VENTANA, y + h - VENTANA),
                         VENTANA // 2, VENTANA // 2)
            if r is not None:
                v, x0, y0 = r
                fuera.append((v, dict(VACIA), dict(VACIA_LIM), "interior",
                              _limites_dentro(caja, x0, y0)))

    # --- NEGATIVO (4): fondo vacio --------------------------------------------
    H, W = vista.shape
    puestos = 0
    for _ in range(24):
        if puestos >= 4:
            break
        fx, fy = rng.uniform(0, W - VENTANA), rng.uniform(0, H - VENTANA)
        if fx + VENTANA < x or fx > x + w or fy + VENTANA < y or fy > y + h:
            r = _recorte(vista, fx, fy, 0, 0)
            if r is not None:
                v, x0, y0 = r
                fuera.append((v, dict(VACIA), dict(VACIA_LIM), "fondo",
                              _limites_dentro(caja, x0, y0)))
                puestos += 1
    return fuera


async def generar(n_imagenes: int) -> dict:
    gen = exigir_generador()
    sys.path.insert(0, str(gen))
    from app.core.renderer import Renderer          # noqa: E402
    from app.core.resolver import resolve           # noqa: E402
    from app.models.recipe import Recipe            # noqa: E402

    receta = Recipe.model_validate(json.loads((AQUI / "receta.json").read_text()))
    r = Renderer()
    await r.start()
    imgs, descartes, vistas = [], {}, []
    try:
        for s in range(n_imagenes):
            res = await r.render(resolve(receta, SEMILLA * 100000 + s))
            if res.labels.has_overlap or not res.labels.blocks:
                descartes["solape o sin bloque"] = descartes.get("solape o sin bloque", 0) + 1
                continue
            b = res.labels.blocks[0].box
            caja = tuple(float(v) / REDUCCION for v in b)
            vista = _reducir(res.image)
            vistas.append(caja)
            H, W = vista.shape
            motivo = _caja_valida(caja, W, H)
            if motivo:
                descartes[motivo.split(" (")[0]] = descartes.get(motivo.split(" (")[0], 0) + 1
                continue
            imgs.append((vista, caja, s))
    finally:
        await r.stop()
    return {"imgs": imgs, "descartes": descartes, "pedidas": n_imagenes, "cajas": vistas}


def _empaquetar(imgs, rng):
    V, C, IMG = [], [], []
    E = {c: [] for c in ETIQUETADAS}
    X = {c: [] for c in ETIQUETADAS}
    Y = {c: [] for c in ETIQUETADAS}
    EL = {c: [] for c in LIMITES}
    YL = {c: [] for c in LIMITES}
    dobles = 0
    for vista, caja, s in imgs:
        for v, et, el, clase, n_geom in _ventanas(vista, caja, rng):
            V.append(v); C.append(clase); IMG.append(s)
            # GEOMETRICO, no tautologico: ver `_limites_dentro`.
            dobles += int(n_geom > 1)
            for c in ETIQUETADAS:
                e, ox, oy = et[c]
                E[c].append(e); X[c].append(ox); Y[c].append(oy)
            for c in LIMITES:
                e, oy = el[c]
                EL[c].append(e); YL[c].append(oy)
    salida = {"ventanas": np.stack(V), "clase": np.array(C),
              "imagen": np.array(IMG, np.int64)}
    # ⚠ Las CUATRO esquinas se siguen etiquetando aunque `lim-h` no las lea como
    # tales: no cuesta nada, permite el desglose por celda (tl/tr/bl/br) que es
    # como se compara con `esq-cq`, y libera al experimento siguiente.
    for c in ETIQUETADAS:
        salida[f"existe_{c}"] = np.array(E[c], np.int64)
        salida[f"x_{c}"] = np.array(X[c], np.float32)
        salida[f"y_{c}"] = np.array(Y[c], np.float32)
    for c in LIMITES:
        salida[f"existe_{c}"] = np.array(EL[c], np.int64)
        salida[f"y_{c}"] = np.array(YL[c], np.float32)
    return salida, dobles


def _elegir_muestras(d, rng):
    """Las 10, cubriendo las SEIS celdas positivas y los TRES negativos.

    Con seis celdas y diez huecos no cabe todo por duplicado, asi que se prioriza
    lo que decide el experimento: el caso PURO (el limite sin esquina) va con dos
    -- uno de cada lado --, porque es lo unico que `esq-cq` no podia medir. Y el
    negativo `interior` va si o si: es el modo de fallo mas probable de aqui (una
    linea de texto es un borde horizontal falso, y hay decenas por parrafo)."""
    C = [str(c) for c in d["clase"]]
    idx = list(range(len(C)))
    elegidas = []
    # las cuatro celdas de esquina, una de cada
    for cual in ("tl", "tr", "bl", "br"):
        pos = [i for i in idx if str(C[i]) == f"esquina-{cual}"]
        if pos:
            elegidas += rng.sample(pos, 1)
    # el CASO PURO, uno por lado -- la razon de ser de este dataset
    for lado in ("superior", "inferior"):
        cands = [i for i in idx if C[i] == f"borde-{lado}" and i not in elegidas]
        if cands:
            elegidas += rng.sample(cands, 1)
    # los tres negativos, y el interior duplicado por ser el fallo probable
    for prefijo, cuantas in (("interior", 2), ("borde-izquierdo", 1), ("fondo", 1)):
        cands = [i for i in idx if C[i].startswith(prefijo) and i not in elegidas]
        if cands:
            elegidas += rng.sample(cands, min(cuantas, len(cands)))
    return elegidas[:10]


def _manifiesto() -> dict | None:
    man = AQUI / "manifiesto.json"
    if not man.exists():
        print("✗ no hay manifiesto.json: genera primero con --imagenes N")
        return None
    return json.loads(man.read_text())


def _huella(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "(no existe)"


def publicar() -> int:
    """Copia la etapa local al repo de DATOS. NO pisa lo ya publicado.

    ⚠ "dato nuevo = nombre nuevo, nunca se reescribe uno" es la regla que el
    propio repo de datos tiene escrita en su `.gitignore`, y es la que hace que
    "siempre el mismo dataset" signifique algo: si publicar pudiera sobrescribir,
    un `--publicar` distraido cambiaria el dato bajo los pies de todo lo ya
    medido, sin un solo error."""
    esperado = _manifiesto()
    if esperado is None:
        return 1
    faltan = [p for p in (*PARTES, "") if not (DATOS / (f"{p}.npz" if p else CONGELADAS)).exists()]
    if faltan:
        print(f"✗ la etapa local esta incompleta ({DATOS}). Genera: --imagenes 300")
        return 1
    destino = exigir_datos() / SUBDIR_DATASETS / DATASET
    if destino.exists():
        print(f"✗ '{DATASET}' YA esta publicado en {destino}")
        print("  Un dataset publicado no se reescribe: si el dato cambia, cambia el")
        print("  NOMBRE (la fecha de render). Si no cambia, no hay nada que hacer.")
        return 1
    destino.mkdir(parents=True)
    for nombre in (*[f"{p}.npz" for p in PARTES], CONGELADAS):
        (destino / nombre).write_bytes((DATOS / nombre).read_bytes())
        print(f"  {nombre:>26}  {_huella(destino / nombre)}")
    (destino / "manifiesto.json").write_text(json.dumps(esperado, indent=2), encoding="utf-8")
    (destino / "README.md").write_text(_readme(esperado), encoding="utf-8")
    print(f"\npublicado en {destino}")
    print("⚠ El repo de datos es PRIVADO y esto no esta empujado todavia:")
    print(f"  cd {exigir_datos()} && git add {SUBDIR_DATASETS}/{DATASET} && "
          f"git commit && git push")
    return 0


def _readme(man: dict) -> str:
    p = man["particiones"]
    filas = "\n".join(
        f"| `{n}.npz` | {p[n]['ventanas']} | "
        + " · ".join(f"{c} {p[n]['positivas'][c]}" for c in ETIQUETADAS)
        + f" | `{p[n]['sha256_16']}` |" for n in PARTES)
    return f"""# `{man['nombre']}`

Dataset de entrada de los experimentos de
[`experimentos-cnn`](https://github.com/stalinbeltran/experimentos-cnn) que detectan **esquinas
de parrafo**. Lo produce `nn/datos.py` del experimento `esq-2d`; aqui vive porque el dato de
entrada no puede estar en aquel repo, que es **publico**.

Parrafos limpios rendidos con `image-text-sample-generator` (receta y semilla en el
manifiesto), reducidos **por {man['reduccion']}** con promedio de area, y recortados en ventanas
de **{man['ventana']}x{man['ventana']}** en TINTA (fondo 0, tinta 255).

| particion | ventanas | positivas | sha256 |
|---|--:|---|---|
{filas}
| `muestras-congeladas.npz` | {man['muestras']['n']} | la verificacion a ojo | `{man['muestras']['sha256_16']}` |

**Se etiquetan LAS CUATRO esquinas** (`existe_<c>`, `x_<c>`, `y_<c>` con `c` en
{', '.join('`'+c+'`' for c in ETIQUETADAS)}), no solo la diagonal que mide `esq-2d`: no cuesta
nada y hace que el experimento siguiente no tenga que re-rendir nada.

⚠ **Ninguna ventana contiene mas de una esquina**
({man['ventanas_con_los_dos_limites']} de {sum(p[n]['ventanas'] for n in PARTES)}): el parrafo
mide >= 64 px reducidos y la ventana 32. Se cuenta al generar, no se supone.

⚠ **Admite kernels hasta k = {man['k_max_representable']}** sin regenerarlo: la esquina se sortea
entre los pixeles {OFF_MIN} y {OFF_MAX} de la ventana, y el mapa de un kernel `k` solo representa
de `(k-1)/2` a `31-(k-1)/2`.

⚠ **No se reescribe.** Dato nuevo = nombre nuevo (la `r<fecha>` es la epoca de render), que es la
regla de este repo. Que se re-derive de la receta y la semilla esta comprobado
(`python nn/datos.py --rederivar`), pero eso es una PRUEBA, no un sustituto: un dato re-derivado
es el mismo mientras nada cambie, y "nada cambia" no se puede comprobar hacia el futuro.
"""


def comprobar() -> int:
    """¿El dataset PUBLICADO es el que dice el manifiesto? (barato, sin rendir)"""
    esperado = _manifiesto()
    if esperado is None:
        return 1
    pub = ruta_dataset(DATASET)
    if pub is None:
        print(f"✗ '{DATASET}' no esta publicado. Publicalo: --publicar")
        return 1
    print(f"publicado en {pub}\n")
    ok = True
    for nombre, esp in esperado["particiones"].items():
        h = _huella(pub / f"{nombre}.npz")
        ok &= (h == esp["sha256_16"])
        print(f"  {nombre:>8}: {h} contra {esp['sha256_16']} … "
              f"{'igual' if h == esp['sha256_16'] else 'DISTINTO'}")
    h = _huella(pub / CONGELADAS)
    esp = esperado["muestras"]["sha256_16"]
    ok &= (h == esp)
    print(f"  {'muestras':>8}: {h} contra {esp} … {'igual' if h == esp else 'DISTINTO'}")
    print("\n✓ el dataset publicado es el del manifiesto." if ok else
          "\n✗ el publicado NO coincide con el manifiesto: no entrenes hasta aclararlo.")
    return 0 if ok else 1


def rederivar() -> int:
    """Re-deriva el dataset de cero y contrasta su huella contra `manifiesto.json`.

    ⚠ Esto NO es "comprobar que los ficheros de disco no se han corrompido": es
    comprobar que la receta y la semilla vuelven a producir EL MISMO dato. La
    diferencia importa, porque en este sistema ya se dio por reproducible un
    dataset que no lo era, y costo 20 corridas ya pagadas. Regenera de verdad
    (unos 6 min) en un directorio aparte y compara SHA-256."""
    import shutil, tempfile
    esperado = _manifiesto()
    if esperado is None:
        return 1
    global DATOS
    original, tmp = DATOS, Path(tempfile.mkdtemp(prefix="comprobar-"))
    try:
        DATOS = tmp
        rc = _generar_y_guardar(esperado["imagenes_pedidas"], escribir_manifiesto=False)
        if rc:
            return rc
        print("\ncontraste con el manifiesto:")
        ok = True
        for nombre, esp in esperado["particiones"].items():
            f = tmp / f"{nombre}.npz"
            h = hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "(no existe)"
            igual = h == esp["sha256_16"]
            ok &= igual
            print(f"  {nombre:>8}: {h} contra {esp['sha256_16']} … {'igual' if igual else 'DISTINTO'}")
        print("\n✓ el dataset SE REPRODUCE." if ok else
              "\n✗ NO se reproduce: entonces el publicado es la UNICA copia, y eso"
              "\n  ya esta cubierto -- pero anotalo como defecto abierto.")
        return 0 if ok else 1
    finally:
        DATOS = original
        shutil.rmtree(tmp, ignore_errors=True)


def _generar_y_guardar(n_imagenes: int, escribir_manifiesto: bool = True) -> int:
    DATOS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEMILLA)
    r = asyncio.run(generar(n_imagenes))
    imgs, descartes = r["imgs"], r["descartes"]
    print(f"imagenes: {len(imgs)} validas de {n_imagenes} pedidas")
    for motivo, n in sorted(descartes.items()):
        print(f"  descartadas por {motivo}: {n}")
    if not imgs:
        print("✗ ninguna imagen cumple las condiciones: revisa `area` en receta.json")
        return 1
    if len(imgs) < N_MUESTRA_IMG + 20:
        print(f"✗ hacen falta al menos {N_MUESTRA_IMG + 20} imagenes validas y hay {len(imgs)}")
        return 1

    muestra_img, resto = imgs[:N_MUESTRA_IMG], imgs[N_MUESTRA_IMG:]
    corte = int(len(resto) * (1 - FRAC_VAL))
    partes = {"train": resto[:corte], "val": resto[corte:], "muestra": muestra_img}

    manifiesto = {"semilla": SEMILLA, "ventana": VENTANA, "reduccion": REDUCCION,
                  "imagenes_pedidas": n_imagenes, "imagenes_validas": len(imgs),
                  "descartes": descartes, "particiones": {}}
    total_dobles = 0
    for nombre, trozo in partes.items():
        d, dobles = _empaquetar(trozo, rng)
        total_dobles += dobles
        f = DATOS / f"{nombre}.npz"
        np.savez_compressed(f, **d)
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        pos = {c: int(d[f"existe_{c}"].sum()) for c in ETIQUETADAS}
        manifiesto["particiones"][nombre] = {
            "ventanas": int(len(d["clase"])), "positivas": pos,
            "imagenes": sorted({int(i) for i in d["imagen"]}), "sha256_16": h}
        print(f"  {nombre:>8}: {len(d['clase']):>5} ventanas · "
              + " ".join(f"{c} {pos[c]}" for c in ETIQUETADAS) + f" · {h}")
        if nombre == "muestra":
            sel = _elegir_muestras(d, random.Random(SEMILLA))
            fc = DATOS / CONGELADAS
            np.savez_compressed(fc, **{k: v[sel] for k, v in d.items()})
            manifiesto["muestras"] = {
                "n": len(sel),
                **{c: int(d[f"existe_{c}"][sel].sum()) for c in ETIQUETADAS},
                **{c: int(d[f"existe_{c}"][sel].sum()) for c in LIMITES},
                "clases": [str(c) for c in d["clase"][sel]],
                "sha256_16": hashlib.sha256(fc.read_bytes()).hexdigest()[:16]}
            print(f"  muestras congeladas: {len(sel)} ("
                  + " ".join(f"{c} {manifiesto['muestras'][c]}" for c in ETIQUETADAS)
                  + f") · {manifiesto['muestras']['sha256_16']}")

    # Las dos garantias que no pueden quedar en la confianza.
    ids = {n: set(v["imagenes"]) for n, v in manifiesto["particiones"].items()}
    fuga = (ids["muestra"] & ids["train"]) | (ids["muestra"] & ids["val"])
    if fuga:
        print(f"✗ FUGA: {len(fuga)} imagen(es) de muestra estan tambien en train/val")
        return 1
    print("  ✓ sin fuga: ninguna imagen de muestra aparece en train ni en val")
    manifiesto["nombre"] = DATASET
    manifiesto["esquinas_etiquetadas"] = list(ETIQUETADAS)
    manifiesto["k_max_representable"] = 2 * (VENTANA - 1 - OFF_MAX) + 1
    for n in PARTES:
        d = np.load(DATOS / f"{n}.npz", allow_pickle=True)
        pos = int(((d["existe_sup"] == 1) | (d["existe_inf"] == 1)).sum())
        manifiesto["particiones"][n]["positivas_limite"] = pos
        manifiesto["particiones"][n]["frac_positivas"] = round(pos / len(d["clase"]), 4)
    manifiesto["ventanas_con_los_dos_limites"] = total_dobles
    if total_dobles:
        print(f"  ⚠ {total_dobles} ventana(s) contienen LOS DOS LIMITES")
    else:
        print("  ✓ ninguna ventana contiene LOS DOS limites "
              "(parrafo >= 64 px de alto, ventana 32) — la salida unica es representable")
    manifiesto["limites_etiquetados"] = list(LIMITES)
    print(f"  ✓ admite kernels hasta k = {manifiesto['k_max_representable']} sin regenerarlo "
          f"(la esquina cae entre los px {OFF_MIN} y {OFF_MAX})")
    if escribir_manifiesto:
        (AQUI / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenes", type=int, default=0)
    p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--rederivar", action="store_true")
    a = p.parse_args()
    if a.imagenes:
        rc = _generar_y_guardar(a.imagenes)
        if rc or not a.publicar:
            return rc
    if a.publicar:
        return publicar()
    if a.comprobar:
        return comprobar()
    if a.rederivar:
        return rederivar()
    p.error("dime que hacer: --imagenes N · --publicar · --comprobar · --rederivar")


if __name__ == "__main__":
    raise SystemExit(main())
