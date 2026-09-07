#!/usr/bin/env python3
"""El dataset: parrafos limpios -> ventanas de 32x32 con SUS DOS esquinas.

    python nn/datos.py --imagenes 300      genera
    python nn/datos.py --comprobar         re-deriva y compara la huella

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

QUE SE COMMITEA Y QUE NO
    El `.npz` NO se commitea: se re-deriva de la receta y la semilla, y su huella
    SHA-256 si esta en el manifiesto. Las 10 muestras SI (son la verificacion, y
    tienen que ser las mismas para siempre aunque el generador cambie).
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
from expcnn import exigir_generador

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

DATOS = EXP / "datos"
MUESTRAS_NPZ = AQUI / "muestras.npz"


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
    """Ventana cuyo pixel (ox, oy) es el punto (cx, cy) de la imagen."""
    x0, y0 = int(round(cx - ox)), int(round(cy - oy))
    H, W = vista.shape
    if x0 < 0 or y0 < 0 or x0 + VENTANA > W or y0 + VENTANA > H:
        return None
    return vista[y0:y0 + VENTANA, x0:x0 + VENTANA]


# etiqueta vacia: sin ninguna de las dos esquinas
VACIA = {"tl": (0, -1.0, -1.0), "br": (0, -1.0, -1.0)}


def _ventanas(vista, caja, rng):
    """Las 10 ventanas de una imagen: 2+2 positivas y 6 negativas, con su clase.

    El reparto (4 positivas de 10) es el mismo de `esq-k` a proposito, para que
    el suelo del "siempre si" sea comparable. Los negativos cubren los cuatro
    modos de fallo de ESTE experimento: la otra diagonal (tr, bl), un borde sin
    esquina por arriba y por abajo, el interior y el fondo."""
    x, y, w, h = caja
    esquinas = {"tl": (x, y), "tr": (x + w, y), "bl": (x, y + h), "br": (x + w, y + h)}
    fuera = []

    def sortear_off():
        return rng.randint(OFF_MIN, OFF_MAX), rng.randint(OFF_MIN, OFF_MAX)

    for cual in ("tl", "br"):                             # POSITIVAS: 2 de cada
        for _ in range(2):
            ox, oy = sortear_off()
            v = _recorte(vista, *esquinas[cual], ox, oy)
            if v is not None:
                et = dict(VACIA)
                et[cual] = (1, float(ox), float(oy))
                fuera.append((v, et, f"esquina-{cual}"))

    for nombre in ("tr", "bl"):                           # negativo duro: la OTRA diagonal
        ox, oy = sortear_off()
        v = _recorte(vista, *esquinas[nombre], ox, oy)
        if v is not None:
            fuera.append((v, dict(VACIA), f"otra-esquina-{nombre}"))

    if w > 2 * VENTANA:                                   # negativo: bordes sin esquina
        for lado, cy in (("superior", y), ("inferior", y + h)):
            bx = rng.uniform(x + VENTANA, x + w - VENTANA)
            v = _recorte(vista, bx, cy, *sortear_off())
            if v is not None:
                fuera.append((v, dict(VACIA), f"borde-{lado}"))

    if w > 2 * VENTANA and h > 2 * VENTANA:               # negativo: interior
        v = _recorte(vista, rng.uniform(x + VENTANA, x + w - VENTANA),
                     rng.uniform(y + VENTANA, y + h - VENTANA), VENTANA // 2, VENTANA // 2)
        if v is not None:
            fuera.append((v, dict(VACIA), "interior"))

    H, W = vista.shape                                    # negativo: fondo vacio
    for _ in range(6):
        fx, fy = rng.uniform(0, W - VENTANA), rng.uniform(0, H - VENTANA)
        if fx + VENTANA < x or fx > x + w or fy + VENTANA < y or fy > y + h:
            v = _recorte(vista, fx, fy, 0, 0)
            if v is not None:
                fuera.append((v, dict(VACIA), "fondo"))
            break
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
    E = {c: [] for c in ("tl", "br")}
    X = {c: [] for c in ("tl", "br")}
    Y = {c: [] for c in ("tl", "br")}
    dobles = 0
    for vista, caja, s in imgs:
        for v, et, clase in _ventanas(vista, caja, rng):
            V.append(v); C.append(clase); IMG.append(s)
            dobles += int(et["tl"][0] == 1 and et["br"][0] == 1)
            for c in ("tl", "br"):
                e, ox, oy = et[c]
                E[c].append(e); X[c].append(ox); Y[c].append(oy)
    salida = {"ventanas": np.stack(V), "clase": np.array(C),
              "imagen": np.array(IMG, np.int64)}
    for c in ("tl", "br"):
        salida[f"existe_{c}"] = np.array(E[c], np.int64)
        salida[f"x_{c}"] = np.array(X[c], np.float32)
        salida[f"y_{c}"] = np.array(Y[c], np.float32)
    return salida, dobles


def _elegir_muestras(d, rng):
    """Las 10: 3 con tl, 3 con br y 4 negativas que cubren los modos de fallo.

    Con dos esquinas, una figura de 5 positivas de la MISMA no verificaria la
    mitad del experimento; y los dos negativos duros (tr, bl) tienen que estar
    si o si, porque son la unica forma de ver si el kernel confunde diagonales."""
    C = d["clase"]
    idx = list(range(len(C)))
    elegidas = []
    for cual, cuantas in (("tl", 3), ("br", 3)):
        pos = [i for i in idx if d[f"existe_{cual}"][i] == 1]
        elegidas += rng.sample(pos, cuantas)
    for prefijo in ("otra-esquina-tr", "otra-esquina-bl", "interior", "fondo"):
        cands = [i for i in idx if str(C[i]).startswith(prefijo) and i not in elegidas]
        if cands:
            elegidas += rng.sample(cands, 1)
    return elegidas[:10]


def comprobar() -> int:
    """Re-deriva el dataset y contrasta su huella contra `manifiesto.json`."""
    import shutil, tempfile
    man = AQUI / "manifiesto.json"
    if not man.exists():
        print("✗ no hay manifiesto.json: genera primero con --imagenes N")
        return 1
    esperado = json.loads(man.read_text())
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
              "\n✗ NO se reproduce: hay que guardar el .npz, no enlazarlo.")
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
        n_tl, n_br = int(d["existe_tl"].sum()), int(d["existe_br"].sum())
        manifiesto["particiones"][nombre] = {
            "ventanas": int(len(d["clase"])), "positivas_tl": n_tl, "positivas_br": n_br,
            "imagenes": sorted({int(i) for i in d["imagen"]}), "sha256_16": h}
        print(f"  {nombre:>8}: {len(d['clase']):>5} ventanas ({n_tl} tl · {n_br} br) · {h}")
        if nombre == "muestra" and escribir_manifiesto:
            sel = _elegir_muestras(d, random.Random(SEMILLA))
            np.savez_compressed(MUESTRAS_NPZ, **{k: v[sel] for k, v in d.items()})
            manifiesto["muestras"] = {
                "n": len(sel), "tl": int(d["existe_tl"][sel].sum()),
                "br": int(d["existe_br"][sel].sum()),
                "clases": [str(c) for c in d["clase"][sel]]}
            print(f"  muestras congeladas: {len(sel)} "
                  f"({manifiesto['muestras']['tl']} tl · {manifiesto['muestras']['br']} br)"
                  f" -> {MUESTRAS_NPZ.name}")

    # Las dos garantias que no pueden quedar en la confianza.
    ids = {n: set(v["imagenes"]) for n, v in manifiesto["particiones"].items()}
    fuga = (ids["muestra"] & ids["train"]) | (ids["muestra"] & ids["val"])
    if fuga:
        print(f"✗ FUGA: {len(fuga)} imagen(es) de muestra estan tambien en train/val")
        return 1
    print("  ✓ sin fuga: ninguna imagen de muestra aparece en train ni en val")
    manifiesto["ventanas_con_las_dos_esquinas"] = total_dobles
    if total_dobles:
        print(f"  ⚠ {total_dobles} ventana(s) contienen LAS DOS esquinas: el supuesto "
              f"de `mag` (una sola posicion) deja de valer ahi")
    else:
        print("  ✓ ninguna ventana contiene las dos esquinas (parrafo >= 64 px, ventana 32)")
    if escribir_manifiesto:
        (AQUI / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenes", type=int, default=300)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    return comprobar() if a.comprobar else _generar_y_guardar(a.imagenes)


if __name__ == "__main__":
    raise SystemExit(main())
