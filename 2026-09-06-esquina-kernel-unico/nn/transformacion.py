#!/usr/bin/env python3
"""LA TRANSFORMACION: aplicar el kernel ganador a una entrada cualquiera.

    python nn/transformacion.py --kernel              que aprendio, en numeros y en imagen
    python nn/transformacion.py --entradas 20         20 entradas NO vistas, con su mapa
    python nn/transformacion.py --paginas 10          10 PAGINAS ENTERAS, con su mapa

Es el producto del experimento: no la red, sino **el filtro**. Por eso vive en su
propia funcion y acepta cualquier tamano de entrada, no solo la ventana 32x32 con
la que se entreno -- una transformacion que solo sirve para la forma exacta del
entrenamiento no es una transformacion, es una capa.

CONTRATO DE `aplicar()`
    entra   una imagen en GRIS tal como la rinde el generador (uint8, fondo claro)
    sale    el mapa de respuesta (float32), mas pequeno en k-1 px por lado

    ⚠ La conversion a TINTA y el /255 van DENTRO. Es lo que evita el error que
    no avisa: pasarle la imagen sin invertir da un mapa con el signo cambiado y
    una figura que parece razonable.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from modelo import BRAZOS, VENTANA, construir

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
SALIDA = EXP / "muestras"
GANADOR = "k07"          # el que gana por el criterio congelado (empate -> el mas pequeno)
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def cargar_kernel(brazo: str = GANADOR) -> np.ndarray:
    """Los k x k pesos aprendidos. Sin bias: la conv no lo tiene."""
    est = torch.load(AQUI / "pesos" / brazo / "best.pt", map_location="cpu", weights_only=False)
    return est["modelo"]["conv.weight"].squeeze().numpy()


def aplicar(gris: np.ndarray, brazo: str = GANADOR) -> np.ndarray:
    """El mapa de respuesta del kernel sobre una imagen en gris (uint8)."""
    tinta = (255 - gris.astype(np.float32)) / 255.0        # el contrato, dentro
    k = cargar_kernel(brazo)
    x = torch.from_numpy(tinta)[None, None]
    w = torch.from_numpy(k)[None, None]
    return torch.nn.functional.conv2d(x, w).squeeze().numpy()


# --- dibujo ----------------------------------------------------------------

def _coma(v: float, dec: int = 2) -> str:
    """Decimales con coma, como el resto del proyecto."""
    return f"{v:.{dec}f}".replace(".", ",")


def _fuente(t: int):
    try:
        return ImageFont.truetype(FUENTE, t)
    except OSError:
        return ImageFont.load_default()


def _divergente(m: np.ndarray, escala: int, dos_lados: bool = False) -> Image.Image:
    """⚠ `dos_lados` normaliza el positivo y el negativo POR SEPARADO, y hay que
    decirlo donde se mire: el kernel ganador suma -73,7, o sea que responde muy
    negativo a toda la tinta, y con una escala unica el pico positivo de la
    esquina --que es lo unico que la cabeza lee-- queda aplastado y la figura
    parece un detector de tinta. Con dos escalas se ve lo que hace de verdad; el
    precio es que las magnitudes de un lado y del otro ya no son comparables."""
    if dos_lados:
        pos = float(m.max()) if m.max() > 0 else 1.0
        neg = float(-m.min()) if m.min() < 0 else 1.0
        t = np.where(m > 0, m / pos, m / neg)
    else:
        t = m / (float(np.abs(m).max()) or 1.0)
    t = np.clip(t, -1, 1)
    rgb = np.stack([np.where(t > 0, 255, 255 * (1 + t)),
                    255 * (1 - np.abs(t)),
                    np.where(t < 0, 255, 255 * (1 - t))], -1).astype(np.uint8)
    im = Image.fromarray(rgb)
    return im.resize((im.width * escala, im.height * escala), Image.NEAREST)


def figura_kernel(brazo: str) -> Path:
    k = cargar_kernel(brazo)
    lado, esc = k.shape[0], 46
    im = _divergente(k, esc, dos_lados=True)
    W, H = max(im.width + 24, 610), im.height + 126
    lienzo = Image.new("RGB", (W, H), (250, 250, 250))
    lienzo.paste(im, (12, 76))
    d = ImageDraw.Draw(lienzo)
    d.text((12, 10), f"El kernel aprendido · {brazo} ({lado}x{lado})", fill=(20, 20, 20), font=_fuente(17))
    d.text((12, 34), "rojo: pesa POSITIVO donde espera TINTA · azul: NEGATIVO donde espera FONDO",
           fill=(90, 90, 90), font=_fuente(13))
    d.text((12, 50), "(los dos signos van a escalas independientes: con una sola, el -9,4 de la "
                     "izquierda aplasta el resto)", fill=(140, 140, 140), font=_fuente(11))
    d.text((12, im.height + 88), f"max {k.max():+.3f} · min {k.min():+.3f} · suma {k.sum():+.3f}",
           fill=(70, 70, 70), font=_fuente(13))
    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"kernel-{brazo}.png"
    lienzo.save(destino)
    return destino


def figura_entradas(brazo: str, n: int) -> Path:
    """n entradas de la particion `muestra` -- NUNCA vistas en entrenamiento."""
    z = np.load(EXP / "datos" / "muestra.npz", allow_pickle=True)
    congeladas = set(np.load(AQUI / "muestras.npz", allow_pickle=True)["clase"].tolist())
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(z["existe"]))[:n]

    esc, cols = 6, 5
    filas = (n + cols - 1) // cols
    lado = VENTANA * esc
    pie, sep = 24, 12
    ch = lado * 2 + pie + sep
    W = cols * lado + (cols + 1) * sep
    H = filas * ch + (filas + 1) * sep + 56
    lienzo = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(lienzo)
    f = _fuente(12)
    k = cargar_kernel(brazo)
    red = construir(brazo)
    red.load_state_dict(torch.load(AQUI / "pesos" / brazo / "best.pt",
                                   map_location="cpu", weights_only=False)["modelo"])
    red.eval()
    d.text((sep, 10), f"La transformacion del kernel {brazo} sobre {n} entradas NUNCA VISTAS "
                      f"(particion `muestra`, fuera de train y val)", fill=(20, 20, 20), font=_fuente(17))
    d.text((sep, 34), "arriba la entrada · abajo su mapa de respuesta en las mismas coordenadas "
                      "· verde: la esquina verdadera · ROJO = el maximo, que es lo unico que lee la "
                      "cabeza (azul y rojo van a escalas independientes: ver el docstring)",
           fill=(90, 90, 90), font=f)

    for j, i in enumerate(idx):
        c, fila = j % cols, j // cols
        x0 = sep + c * (lado + sep)
        y0 = 56 + sep + fila * (ch + sep)
        v = z["ventanas"][i]
        vis = Image.fromarray(np.stack([255 - v] * 3, -1)).resize((lado, lado), Image.NEAREST)
        lienzo.paste(vis, (x0, y0))
        d.rectangle([x0, y0, x0 + lado - 1, y0 + lado - 1], outline=(120, 120, 120))

        mapa = aplicar(255 - v, brazo)                     # `aplicar` espera GRIS
        ym = y0 + lado + sep
        base = np.full((VENTANA, VENTANA, 3), 228, np.uint8)
        off = (k.shape[0] - 1) // 2
        col = np.asarray(_divergente(mapa, 1, dos_lados=True))
        base[off:off + mapa.shape[0], off:off + mapa.shape[1]] = col
        lienzo.paste(Image.fromarray(base).resize((lado, lado), Image.NEAREST), (x0, ym))
        d.rectangle([x0, ym, x0 + lado - 1, ym + lado - 1], outline=(120, 120, 120))
        if z["existe"][i] == 1:
            cx, cy = (z["x"][i] + 0.5) * esc, (z["y"][i] + 0.5) * esc
            for oy in (y0, ym):
                dd = ImageDraw.Draw(lienzo)
                dd.line([(x0 + cx - 9, oy + cy), (x0 + cx + 9, oy + cy)], fill=(0, 170, 0), width=2)
                dd.line([(x0 + cx, oy + cy - 9), (x0 + cx, oy + cy + 9)], fill=(0, 170, 0), width=2)
        d.text((x0, ym + lado + 4), f"{j+1}. {z['clase'][i]}", fill=(70, 70, 70), font=f)

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"transformacion-{brazo}-{n}-entradas.png"
    lienzo.save(destino)
    return destino


def paginas(n: int) -> list:
    """Las n primeras paginas VALIDAS, ENTERAS (200x200 reducidas) y sin recortar.

    ⚠ Son las de la particion `muestra` -- las primeras N_MUESTRA_IMG validas,
    que `datos.py` reserva y de las que comprueba que no aparecen ni en train ni
    en val. Por eso esta funcion se NIEGA a dar mas: la pagina 13 ya entreno la
    red, y una figura sobre dato visto no verifica nada.

    Se RE-DERIVAN de la receta y la semilla, no se leen de disco: el dato de
    entrada no se copia a este repo (repo publico) y las paginas enteras tampoco
    se commitean. Cuesta ~1,1 s por pagina mas ~2 s de arrancar el navegador
    (medido el 2026-09-07 en este droplet de 2 vCPU).
    """
    # Perezoso a proposito: `datos` arrastra playwright y el generador, y las
    # otras dos figuras de este fichero no los necesitan.
    from datos import N_MUESTRA_IMG, generar

    if n > N_MUESTRA_IMG:
        raise SystemExit(
            f"✗ solo hay {N_MUESTRA_IMG} paginas fuera de train/val (la particion "
            f"`muestra`), y se piden {n}. Pedir mas seria mirar dato que la red ya "
            f"vio: la figura dejaria de verificar nada.")
    pedidas = n + 3        # ~11% de descartes (267 validas de 300, manifiesto.json)
    while True:
        r = asyncio.run(generar(pedidas))
        if len(r["imgs"]) >= n:
            return r["imgs"][:n]
        if pedidas > 3 * n + 10:
            raise SystemExit(f"✗ {pedidas} renders y solo {len(r['imgs'])} paginas "
                             f"validas: revisa `area` en receta.json")
        pedidas += 5


def figura_paginas(brazo: str, n: int = 10, esc: int = 2) -> Path:
    """La MISMA `aplicar()`, pero sobre la pagina entera en vez de la ventana.

    Ahi esta el punto de que sea una transformacion y no una capa: acepta
    cualquier tamano de entrada. Lo que cambia es el PROBLEMA, y cambia de
    verdad -- entrenando, cada ventana traia una sola esquina y siempre con
    contexto a los dos lados; una pagina entera trae las cuatro esquinas, los
    cuatro bordes, el interior, el fondo y todos los comienzos de linea, y hay
    UN solo maximo para todo eso. Que el maximo caiga en la esquina superior
    izquierda no estaba medido: la cabeza que lo leia solo veia 32x32.

    ⚠ La distancia se mide en PIXELES REDUCIDOS (la imagen es 800x800 y se
    reduce por 4 antes de nada), que es la unidad en la que se midio el
    experimento entero.
    """
    pags = paginas(n)
    k = cargar_kernel(brazo)
    off = (k.shape[0] - 1) // 2

    filas_de = []
    for vista, caja, s in pags:
        gris = 255 - vista                                 # `aplicar` espera GRIS
        mapa = aplicar(gris, brazo)
        r, c = np.unravel_index(int(np.argmax(mapa)), mapa.shape)
        mx, my = c + off, r + off                          # el maximo, en coords de imagen
        d = float(np.hypot(mx - caja[0], my - caja[1]))
        filas_de.append({"s": s, "vista": vista, "caja": caja, "mapa": mapa,
                         "max": (mx, my), "pico": float(mapa.max()), "dist": d})

    aciertos2 = sum(1 for f in filas_de if f["dist"] <= 2)
    aciertos1 = sum(1 for f in filas_de if f["dist"] <= 1)
    peor = max(f["dist"] for f in filas_de)

    H, W = pags[0][0].shape
    lado_w, lado_h = W * esc, H * esc
    cols = 5
    filas = (n + cols - 1) // cols
    pie, sep = 22, 12
    ch = lado_h * 2 + pie + sep
    ancho = cols * lado_w + (cols + 1) * sep
    alto = filas * ch + (filas + 1) * sep + 78
    lienzo = Image.new("RGB", (ancho, alto), (250, 250, 250))
    d = ImageDraw.Draw(lienzo)
    f12 = _fuente(12)
    d.text((sep, 10), f"El kernel {brazo} aplicado a {n} PAGINAS ENTERAS "
                      f"({W}x{H} px reducidos), no a ventanas de {VENTANA}x{VENTANA}",
           fill=(20, 20, 20), font=_fuente(17))
    d.text((sep, 34), "arriba la pagina · abajo su mapa de respuesta en las mismas coordenadas "
                      "· VERDE: la esquina superior-izquierda verdadera · ROJO: el maximo del mapa",
           fill=(90, 90, 90), font=f12)
    d.text((sep, 52), f"el maximo cae a <=2 px de la esquina en {aciertos2}/{n} paginas "
                      f"(<=1 px en {aciertos1}/{n}) · el peor a {_coma(peor)} px   "
                      f"[azul y rojo van a escalas independientes: ver el docstring de _divergente]",
           fill=(90, 90, 90), font=f12)

    for j, fila in enumerate(filas_de):
        col, ren = j % cols, j // cols
        x0 = sep + col * (lado_w + sep)
        y0 = 78 + sep + ren * (ch + sep)
        vis = Image.fromarray(np.stack([255 - fila["vista"]] * 3, -1))
        lienzo.paste(vis.resize((lado_w, lado_h), Image.NEAREST), (x0, y0))
        d.rectangle([x0, y0, x0 + lado_w - 1, y0 + lado_h - 1], outline=(120, 120, 120))

        ym = y0 + lado_h + sep
        base = np.full((H, W, 3), 228, np.uint8)
        mapa = fila["mapa"]
        base[off:off + mapa.shape[0], off:off + mapa.shape[1]] = np.asarray(
            _divergente(mapa, 1, dos_lados=True))
        lienzo.paste(Image.fromarray(base).resize((lado_w, lado_h), Image.NEAREST), (x0, ym))
        d.rectangle([x0, ym, x0 + lado_w - 1, ym + lado_h - 1], outline=(120, 120, 120))

        cx, cy = (fila["caja"][0] + 0.5) * esc, (fila["caja"][1] + 0.5) * esc
        mx, my = (fila["max"][0] + 0.5) * esc, (fila["max"][1] + 0.5) * esc
        for oy in (y0, ym):
            d.line([(x0 + cx - 11, oy + cy), (x0 + cx + 11, oy + cy)], fill=(0, 170, 0), width=2)
            d.line([(x0 + cx, oy + cy - 11), (x0 + cx, oy + cy + 11)], fill=(0, 170, 0), width=2)
            d.ellipse([x0 + mx - 7, oy + my - 7, x0 + mx + 7, oy + my + 7],
                      outline=(220, 0, 0), width=2)
        d.text((x0, ym + lado_h + 3),
               f"{j+1}. pagina s={fila['s']} · maximo a {_coma(fila['dist'])} px "
               f"· pico {_coma(fila['pico'])}", fill=(70, 70, 70), font=f12)

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"transformacion-{brazo}-{n}-paginas.png"
    lienzo.save(destino)

    print(f"kernel {brazo} sobre {n} paginas enteras ({W}x{H} px reducidos), "
          f"particion `muestra` (fuera de train y val):\n")
    print("   #  pagina   esquina real      maximo del mapa   distancia    pico")
    for j, fila in enumerate(filas_de):
        print(f"  {j+1:>2}   s={fila['s']:<3}  ({fila['caja'][0]:6.1f},{fila['caja'][1]:6.1f})   "
              f"({fila['max'][0]:>3},{fila['max'][1]:>3})           "
              f"{fila['dist']:5.2f} px   {fila['pico']:6.2f}")
    print(f"\n  a <=2 px: {aciertos2}/{n} · a <=1 px: {aciertos1}/{n} · el peor: {peor:.2f} px")
    return destino


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", default=GANADOR, choices=sorted(BRAZOS))
    p.add_argument("--kernel", action="store_true")
    p.add_argument("--entradas", type=int, default=0)
    p.add_argument("--paginas", type=int, default=0)
    a = p.parse_args()
    if a.kernel:
        k = cargar_kernel(a.brazo)
        print(f"kernel {a.brazo} ({k.shape[0]}x{k.shape[0]}), sin bias:\n")
        for fila in k:
            print("  " + " ".join(f"{v:+7.3f}" for v in fila))
        print(f"\n  suma {k.sum():+.3f} · max {k.max():+.3f} · min {k.min():+.3f}")
        print(f"  {figura_kernel(a.brazo).relative_to(EXP)}")
    if a.entradas:
        print(f"  {figura_entradas(a.brazo, a.entradas).relative_to(EXP)}")
    if a.paginas:
        print(f"  {figura_paginas(a.brazo, a.paginas).relative_to(EXP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
