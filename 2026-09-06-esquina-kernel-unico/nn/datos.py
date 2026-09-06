#!/usr/bin/env python3
"""El dataset: parrafos limpios -> ventanas de 32x32 con su etiqueta.

    python nn/datos.py --imagenes 300      genera
    python nn/datos.py --comprobar         re-deriva y compara la huella

LA RECETA PIDE, LAS ETIQUETAS PRUEBAN
    `placement.area` mantiene el parrafo lejos del borde, pero lo hace sobre una
    ESTIMACION del tamano (`resolver.estimate_size`): el alto real no se sabe
    hasta que el navegador maqueta. Asi que ninguna imagen entra sin comprobar
    su caja REAL, que sale del DOM y es exacta. Las que no cumplen se descartan
    y se cuentan -- si el descarte es alto, lo que esta mal es el `area`.

LAS CONDICIONES, Y DE DONDE SALE CADA NUMERO
    1. parrafo a >= 32 px (reducidos) del borde de la imagen.
       La esquina ocupa del px 5 al 26 de la ventana, asi que en el caso extremo
       la ventana se extiende 26 px hacia un lado. 32 = una ventana entera, para
       no dejarlo al filo.
    2. parrafo >= 64 px (reducidos) de lado = 2 ventanas: asi cabe una ventana
       ENTERA dentro del cuerpo (negativo de interior) y sobre un borde sin
       tocar ninguna esquina.
    3. la esquina cae en [5, 26] de la ventana: con k = 11 el mapa no puede
       senalar mas cerca de 5 px del borde. Se fija con el k MAS GRANDE del
       barrido, asi que los cinco brazos ven el mismo problema.
    4. `has_overlap` falso.

LA REDUCCION ES PROMEDIO DE AREA (`Image.BOX`), no interpolacion suave: es
    literalmente "menos pixeles, menos detalle", que es la transformacion que el
    dueno ya aplicaba a mano.

QUE SE COMMITEA Y QUE NO
    El `.npz` del entrenamiento NO se commitea: se re-deriva de la receta y la
    semilla. Pero su huella SHA-256 SI, y `--comprobar` la contrasta -- porque
    "es reproducible" es una afirmacion que hay que poder comprobar, no suponer.
    Las 10 muestras SI se commitean (10 KB): son la verificacion, y tienen que
    ser las mismas para siempre aunque el generador cambie.
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
# `expcnn` se importa como paquete instalado (`uv pip install -e .`), no por
# ruta: el acoplamiento se declara, no se deduce del disco.
from expcnn import exigir_generador

VENTANA = 32
REDUCCION = 4
MARGEN_BORDE = 32          # px reducidos, condicion 1
LADO_MIN = 64              # px reducidos, condicion 2
K_MAX = 11
CIEGO = (K_MAX - 1) // 2   # 5 px, condicion 3
# ⚠ El margen ciego (5 px) es solo el limite de ABAJO. Hay otro por arriba que
# no viene del kernel sino de la TAREA: con la esquina en el px 26, la ventana
# contiene 6 px de parrafo y no hay esquina que ver -- se veia en la primera
# figura, muestras 4 y 5. Se exige contexto a los dos lados: >= 8 px de fondo
# arriba/izquierda y >= 9 px de parrafo abajo/derecha.
CONTEXTO = 8
OFF_MIN, OFF_MAX = max(CIEGO, CONTEXTO), VENTANA - 1 - CONTEXTO
SEMILLA = 1
N_MUESTRA_IMG = 12         # imagenes reservadas para las 10 muestras
FRAC_VAL = 0.15

DATOS = EXP / "datos"
MUESTRAS_NPZ = AQUI / "muestras.npz"


def _reducir(img: Image.Image) -> np.ndarray:
    """Reduce por PROMEDIO DE AREA y devuelve TINTA: fondo 0, tinta 255.

    ⚠ La inversion no es cosmetica. La conv no lleva bias, asi que un peso
    positivo significa "espero tinta aqui" y uno negativo "espero fondo" -- que
    es exactamente la forma de un detector de esquina en cuadrante. Con la
    imagen sin invertir, el fondo blanco (valor alto) dominaria el mapa y los
    signos querrian decir lo contrario. Para MIRARLA se vuelve a invertir."""
    g = img.convert("L")
    w, h = g.size
    chico = g.resize((w // REDUCCION, h // REDUCCION), Image.BOX)
    return 255 - np.asarray(chico, dtype=np.uint8)


def _caja_valida(caja, W, H) -> str | None:
    """None si cumple; si no, el motivo del descarte."""
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


def _ventanas(vista, caja, rng):
    """Las ventanas de una imagen: 4 positivas y 4 negativas, con su clase."""
    x, y, w, h = caja
    esquinas = {"tl": (x, y), "tr": (x + w, y), "bl": (x, y + h), "br": (x + w, y + h)}
    fuera = []

    def sortear_off():
        return rng.randint(OFF_MIN, OFF_MAX), rng.randint(OFF_MIN, OFF_MAX)

    for _ in range(4):                                    # POSITIVAS
        ox, oy = sortear_off()
        v = _recorte(vista, *esquinas["tl"], ox, oy)
        if v is not None:
            fuera.append((v, 1, ox, oy, "esquina-tl"))

    for nombre in ("tr", "bl", "br"):                     # negativo duro: otra esquina
        ox, oy = sortear_off()
        v = _recorte(vista, *esquinas[nombre], ox, oy)
        if v is not None:
            fuera.append((v, 0, -1, -1, f"otra-esquina-{nombre}"))

    if w > 2 * VENTANA:                                   # negativo: borde superior sin esquina
        bx = rng.uniform(x + VENTANA, x + w - VENTANA)
        v = _recorte(vista, bx, y, *sortear_off())
        if v is not None:
            fuera.append((v, 0, -1, -1, "borde-superior"))

    if w > 2 * VENTANA and h > 2 * VENTANA:               # negativo: interior del parrafo
        v = _recorte(vista, rng.uniform(x + VENTANA, x + w - VENTANA),
                     rng.uniform(y + VENTANA, y + h - VENTANA), VENTANA // 2, VENTANA // 2)
        if v is not None:
            fuera.append((v, 0, -1, -1, "interior"))

    H, W = vista.shape                                    # negativo: fondo vacio
    for _ in range(6):
        fx, fy = rng.uniform(0, W - VENTANA), rng.uniform(0, H - VENTANA)
        if fx + VENTANA < x or fx > x + w or fy + VENTANA < y or fy > y + h:
            v = _recorte(vista, fx, fy, 0, 0)
            if v is not None:
                fuera.append((v, 0, -1, -1, "fondo"))
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
            caja = tuple(float(v) / REDUCCION for v in b)   # box = [x, y, w, h]
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
    V, E, X, Y, C, IMG = [], [], [], [], [], []
    for vista, caja, s in imgs:
        for v, ex, ox, oy, clase in _ventanas(vista, caja, rng):
            V.append(v); E.append(ex); X.append(ox); Y.append(oy); C.append(clase); IMG.append(s)
    return (np.stack(V), np.array(E, np.int64), np.array(X, np.float32),
            np.array(Y, np.float32), np.array(C), np.array(IMG, np.int64))


def _elegir_muestras(V, E, C, rng):
    """Las 10: 5 positivas y 5 negativas que cubren los cuatro modos de fallo.

    Cumple el minimo del dueno (>= 4 con la esquina) y ademas hace que las 10
    ejerciten algo: una figura con 10 fondos vacios no verifica nada."""
    idx = list(range(len(E)))
    pos = [i for i in idx if E[i] == 1]
    # Variadas a proposito: dos veces la misma clase de esquina verifica la mitad.
    quiero = [("otra-esquina-tr", 1), ("otra-esquina-br", 1),
              ("borde-superior", 1), ("interior", 1), ("fondo", 1)]
    elegidas = rng.sample(pos, 5)
    for prefijo, cuantas in quiero:
        cands = [i for i in idx if str(C[i]).startswith(prefijo) and i not in elegidas]
        elegidas += rng.sample(cands, min(cuantas, len(cands)))
    return elegidas[:10]


def comprobar() -> int:
    """Re-deriva el dataset y contrasta su huella contra `manifiesto.json`.

    ⚠ Esto NO es "comprobar que los ficheros de disco no se han corrompido": es
    comprobar que la receta y la semilla vuelven a producir EL MISMO dato. La
    diferencia importa, porque en este sistema ya se dio por reproducible un
    dataset que no lo era, y costo 20 corridas ya pagadas. Regenera de verdad
    (unos 2,5 min) en un directorio aparte y compara SHA-256."""
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
        if ok:
            print("\n✓ el dataset SE REPRODUCE: la receta y la semilla dan el mismo dato.")
        else:
            print("\n✗ NO se reproduce. El `.npz` deja de ser regenerable y hay que")
            print("  guardarlo, no enlazarlo (y anotar el defecto como abierto).")
        return 0 if ok else 1
    finally:
        DATOS = original
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenes", type=int, default=300)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    return _generar_y_guardar(a.imagenes)


def _generar_y_guardar(n_imagenes: int, escribir_manifiesto: bool = True) -> int:
    a = argparse.Namespace(imagenes=n_imagenes)
    DATOS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEMILLA)
    r = asyncio.run(generar(a.imagenes))
    imgs, descartes = r["imgs"], r["descartes"]
    print(f"imagenes: {len(imgs)} validas de {a.imagenes} pedidas")
    for motivo, n in sorted(descartes.items()):
        print(f"  descartadas por {motivo}: {n}")
    if r["cajas"]:
        ws = [c[2] for c in r["cajas"]]; hs = [c[3] for c in r["cajas"]]
        xs = [c[0] for c in r["cajas"]]; ys = [c[1] for c in r["cajas"]]
        print(f"  cajas vistas (px reducidos): ancho {min(ws):.0f}-{max(ws):.0f} · "
              f"alto {min(hs):.0f}-{max(hs):.0f} · x {min(xs):.0f}-{max(xs):.0f} · "
              f"y {min(ys):.0f}-{max(ys):.0f}   [hace falta lado >= {LADO_MIN}, margen >= {MARGEN_BORDE}]")
    if not imgs:
        print("✗ ninguna imagen cumple las condiciones: revisa `area` en receta.json")
        return 1

    if len(imgs) < N_MUESTRA_IMG + 20:
        print(f"✗ hacen falta al menos {N_MUESTRA_IMG + 20} imagenes validas "
              f"({N_MUESTRA_IMG} se reservan para las muestras) y hay {len(imgs)}. "
              f"Pide mas con --imagenes.")
        return 1
    muestra_img, resto = imgs[:N_MUESTRA_IMG], imgs[N_MUESTRA_IMG:]
    corte = int(len(resto) * (1 - FRAC_VAL))
    partes = {"train": resto[:corte], "val": resto[corte:], "muestra": muestra_img}

    manifiesto = {"semilla": SEMILLA, "ventana": VENTANA, "reduccion": REDUCCION,
                  "imagenes_pedidas": a.imagenes, "imagenes_validas": len(imgs),
                  "descartes": descartes, "particiones": {}}
    for nombre, trozo in partes.items():
        V, E, X, Y, C, IMG = _empaquetar(trozo, rng)
        f = DATOS / f"{nombre}.npz"
        np.savez_compressed(f, ventanas=V, existe=E, x=X, y=Y, clase=C, imagen=IMG)
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        manifiesto["particiones"][nombre] = {
            "ventanas": int(len(E)), "positivas": int(E.sum()),
            "imagenes": sorted({int(i) for i in IMG}), "sha256_16": h}
        print(f"  {nombre:>8}: {len(E):>5} ventanas ({int(E.sum())} positivas) · {h}")
        if nombre == "muestra" and escribir_manifiesto:
            sel = _elegir_muestras(V, E, C, random.Random(SEMILLA))
            np.savez_compressed(MUESTRAS_NPZ, ventanas=V[sel], existe=E[sel], x=X[sel],
                                y=Y[sel], clase=C[sel], imagen=IMG[sel])
            manifiesto["muestras"] = {"n": len(sel), "positivas": int(E[sel].sum()),
                                      "clases": [str(c) for c in C[sel]]}
            print(f"  muestras congeladas: {len(sel)} ({int(E[sel].sum())} positivas) -> {MUESTRAS_NPZ.name}")

    # La garantia que no puede quedar en la confianza: las 10 no estan en train ni en val.
    ids = {n: set(v["imagenes"]) for n, v in manifiesto["particiones"].items()}
    fuga = (ids["muestra"] & ids["train"]) | (ids["muestra"] & ids["val"])
    if fuga:
        print(f"✗ FUGA: {len(fuga)} imagen(es) de muestra estan tambien en train/val")
        return 1
    print("  ✓ sin fuga: ninguna imagen de muestra aparece en train ni en val")
    if escribir_manifiesto:
        (AQUI / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
