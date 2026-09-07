#!/usr/bin/env python3
"""LAS TRANSFORMACIONES: cada kernel aprendido, aplicado a PAGINAS ENTERAS.

    python nn/transformacion.py --paginas 10            los cinco kernels
    python nn/transformacion.py --paginas 10 --brazo k13   solo uno
    python nn/transformacion.py --kernel                que aprendio cada uno

Sale en `muestras/transformacion-<brazo>-<n>-paginas.png`.

QUE ES ESTO Y POR QUE NO ES LA FIGURA DE LAS MUESTRAS
    Las figuras de `muestras.py` ensenan la red sobre las VENTANAS de 32x32 con
    las que entreno. Esta ensena el KERNEL SUELTO sobre la pagina entera
    (200x200 px reducidos), que es un problema distinto y mas dificil: la ventana
    traia una esquina y siempre con contexto a los dos lados; la pagina trae las
    cuatro esquinas, los cuatro bordes, el interior, el fondo y el comienzo de
    cada linea, y hay UN maximo y UN minimo para todo eso.

    El producto de este experimento no es la red: es el filtro. Por eso `aplicar()`
    acepta cualquier tamano de entrada -- una transformacion que solo sirve para
    la forma exacta del entrenamiento no es una transformacion, es una capa.

⚠⚠ AQUI SE LEE EL EXTREMO DEL MAPA, NO LA CABEZA, Y HAY QUE DECIRLO
    La cabeza lee la ESPERANZA bajo softmax(beta*M), no el argmax. Sobre la
    ventana de 32x32 las dos cosas casi coinciden; sobre una pagina de 200x200 no,
    porque el softmax reparte masa entre ~36.000 posiciones en vez de ~700 y la
    beta aprendida es baja (1,1-1,6 al final del barrido, medido 2026-09-07): la
    esperanza se iria al centro de la pagina y no diria nada. Asi que esta figura
    mide el FILTRO --donde cae su maximo y su minimo-- y no la red entera. Es una
    medida distinta, no la misma en otro sitio.

CONTRATO DE `aplicar()`
    entra   una imagen en GRIS tal como la rinde el generador (uint8, fondo claro)
    sale    el mapa de respuesta (float32), mas pequeno en k-1 px por lado

    ⚠ La conversion a TINTA y el /255 van DENTRO. Es lo que evita el error que no
    avisa: pasarle la imagen sin invertir da un mapa con el signo cambiado y una
    figura que parece razonable.

⚠ POR QUE ESTE FICHERO ES UNA COPIA ADAPTADA Y NO UN IMPORT
    Regla del repo: ningun experimento importa de otro. Y ademas no seria el
    mismo fichero: alli hay UNA lectura (el maximo) y aqui DOS (maximo y minimo),
    y la figura tiene que ensenar las dos o no verifica la mitad del experimento.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from modelo import BRAZOS, construir

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
SALIDA = EXP / "muestras"
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def epoca_de(brazo: str) -> int:
    """La epoca del `best.pt` que se va a usar. La figura la lleva en el NOMBRE:
    sin eso, regenerarla despues de entrenar pisa en silencio la del punto de
    partida, y las dos son el mismo fichero con contenidos incomparables."""
    est = torch.load(AQUI / "pesos" / brazo / "best.pt", map_location="cpu",
                     weights_only=False)
    return int(est["epoca"])


def cargar_kernel(brazo: str) -> np.ndarray:
    """Los k x k pesos aprendidos. Sin bias: la conv no lo tiene.

    Se pasa por `construir()` en vez de leer el tensor a pelo porque el kernel
    EFECTIVO no siempre es el guardado: `ant` lo proyecta antisimetrico. Leerlo
    crudo daria otro filtro sin avisar."""
    red = construir(brazo)
    est = torch.load(AQUI / "pesos" / brazo / "best.pt", map_location="cpu",
                     weights_only=False)
    red.load_state_dict(est["modelo"])
    with torch.no_grad():
        return red.kernel().squeeze().numpy().copy()


def aplicar(gris: np.ndarray, brazo: str) -> np.ndarray:
    """El mapa de respuesta del kernel sobre una imagen en gris (uint8)."""
    tinta = (255 - gris.astype(np.float32)) / 255.0        # el contrato, dentro
    k = cargar_kernel(brazo)
    x = torch.from_numpy(tinta)[None, None]
    w = torch.from_numpy(k)[None, None]
    return torch.nn.functional.conv2d(x, w).squeeze().numpy()


def paginas(n: int) -> list:
    """Las n primeras paginas VALIDAS, ENTERAS y sin recortar.

    ⚠ Son las de la particion `muestra` --las primeras N_MUESTRA_IMG validas, que
    `datos.py` reserva y de las que comprueba que no aparecen ni en train ni en
    val--, asi que la red NUNCA las vio. Por eso se niega a dar mas.

    Se RE-DERIVAN de la receta y la semilla: el dataset publicado guarda ventanas
    de 32x32, no paginas. Cuesta ~1,1 s por pagina mas ~2 s de arrancar el
    navegador (medido el 2026-09-07 en este droplet de 2 vCPU)."""
    from datos import N_MUESTRA_IMG, generar    # perezoso: arrastra el generador

    if n > N_MUESTRA_IMG:
        raise SystemExit(f"✗ solo hay {N_MUESTRA_IMG} paginas fuera de train/val y se "
                         f"piden {n}: seria mirar dato que la red ya vio.")
    pedidas = n + 3        # ~11 % de descartes (267 validas de 300, manifiesto.json)
    while True:
        r = asyncio.run(generar(pedidas))
        if len(r["imgs"]) >= n:
            return r["imgs"][:n]
        if pedidas > 3 * n + 10:
            raise SystemExit(f"✗ {pedidas} renders y solo {len(r['imgs'])} validas")
        pedidas += 5


# --- dibujo ----------------------------------------------------------------

def _coma(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def _fuente(t: int):
    try:
        return ImageFont.truetype(FUENTE, t)
    except OSError:
        return ImageFont.load_default()


def _divergente(m: np.ndarray, escala: int, dos_lados: bool = True) -> Image.Image:
    """⚠ `dos_lados` normaliza el positivo y el negativo POR SEPARADO, y hay que
    decirlo donde se mire: si un lado domina en magnitud, con una escala unica el
    otro queda aplastado -- y aqui los DOS importan, porque uno lleva `tl` y el
    otro `br`. El precio es que las magnitudes de un lado y del otro ya no son
    comparables entre si."""
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


def _marca(d, x, y, color, esc, r=8, w=2, aspa=False):
    cx, cy = (x + 0.5) * esc, (y + 0.5) * esc
    if aspa:
        q = r * 0.71
        d.line([(cx - q, cy - q), (cx + q, cy + q)], fill=color, width=w)
        d.line([(cx - q, cy + q), (cx + q, cy - q)], fill=color, width=w)
    else:
        d.line([(cx - r, cy), (cx + r, cy)], fill=color, width=w)
        d.line([(cx, cy - r), (cx, cy + r)], fill=color, width=w)


def figura_paginas(brazo: str, pags: list, esc: int = 2) -> tuple[Path, dict]:
    k = cargar_kernel(brazo)
    off = (k.shape[0] - 1) // 2
    n = len(pags)

    filas = []
    for vista, caja, s in pags:
        mapa = aplicar(255 - vista, brazo)                 # `aplicar` espera GRIS
        r, c = np.unravel_index(int(np.argmax(mapa)), mapa.shape)
        x, y, w, h = caja
        mx, my = c + off, r + off                          # el MAXIMO, y ya
        # ⚠ UNA sola lectura: `esq-cq` no dice CUAL esquina es, asi que el maximo
        # acierta si cae cerca de CUALQUIERA de las dos. El minimo ya no se lee
        # (en `esq-2d` era de donde salia `br`).
        # ⚠ SOLO LA ALTURA. El limite es una linea horizontal: lo que se puntua
        # es |dy| a la fila verdadera mas cercana, no una distancia euclidea.
        d_sup = abs(float(my - y))
        d_inf = abs(float(my - (y + h)))
        filas.append({"s": s, "vista": vista, "caja": caja, "mapa": mapa,
                      "max": (mx, my), "d_tl": d_sup, "d_br": d_inf,
                      "d": min(d_sup, d_inf),
                      "cual": "sup" if d_sup <= d_inf else "inf"})

    ok = sum(1 for f in filas if f["d"] <= 2)
    # A cual se pega el maximo cuando acierta: es la pregunta que sustituye al
    # desglose por salida, porque aqui la red no declara cual esquina cree ver.
    ok_tl = sum(1 for f in filas if f["d"] <= 2 and f["cual"] == "sup")
    ok_br = sum(1 for f in filas if f["d"] <= 2 and f["cual"] == "inf")

    H, W = pags[0][0].shape
    lw, lh = W * esc, H * esc
    cols, pie, sep = 5, 22, 12
    nfilas = (n + cols - 1) // cols
    ch = lh * 2 + pie + sep
    ancho = cols * lw + (cols + 1) * sep
    alto = nfilas * ch + (nfilas + 1) * sep + 78
    lienzo = Image.new("RGB", (ancho, alto), (250, 250, 250))
    d = ImageDraw.Draw(lienzo)
    f12 = _fuente(12)
    ep = epoca_de(brazo)
    d.text((sep, 10), f"El kernel {brazo} ({k.shape[0]}x{k.shape[0]}) aplicado a {n} PAGINAS "
                      f"ENTERAS ({W}x{H} px reducidos), no a ventanas de 32x32"
                      f"  ·  epoca {ep}{'  (SIN ENTRENAR)' if ep == 0 else ''}",
           fill=(20, 20, 20), font=_fuente(17))
    d.text((sep, 34), "arriba la pagina · abajo su mapa de respuesta en las mismas coordenadas "
                      "· VERDE: el limite SUPERIOR · AZUL: el INFERIOR (lineas, no puntos)",
           fill=(90, 90, 90), font=f12)
    d.text((sep, 52), f"ROJO: la FILA del maximo del mapa -> cae a <=2 px de ALGUN limite "
                      f"en {ok}/{n}  (se pega al superior en {ok_tl}, al inferior en {ok_br})"
                      f"   ·  solo cuenta |dy|: la columna del maximo es diagnostico",
           fill=(90, 90, 90), font=f12)

    for j, fila in enumerate(filas):
        col, ren = j % cols, j // cols
        x0 = sep + col * (lw + sep)
        y0 = 78 + sep + ren * (ch + sep)
        vis = Image.fromarray(np.stack([255 - fila["vista"]] * 3, -1))
        lienzo.paste(vis.resize((lw, lh), Image.NEAREST), (x0, y0))
        d.rectangle([x0, y0, x0 + lw - 1, y0 + lh - 1], outline=(120, 120, 120))

        ym = y0 + lh + sep
        base = np.full((H, W, 3), 228, np.uint8)
        mapa = fila["mapa"]
        base[off:off + mapa.shape[0], off:off + mapa.shape[1]] = np.asarray(
            _divergente(mapa, 1))
        lienzo.paste(Image.fromarray(base).resize((lw, lh), Image.NEAREST), (x0, ym))
        d.rectangle([x0, ym, x0 + lw - 1, ym + lh - 1], outline=(120, 120, 120))

        x, y, w, h = fila["caja"]
        for oy in (y0, ym):
            dd = ImageDraw.Draw(lienzo)
            # LINEAS de lado a lado: los limites son horizontales
            for fy, col in ((y, (0, 170, 0)), (y + h, (0, 90, 200))):
                yy = oy + fy * esc
                dd.line([x0, yy, x0 + lw - 1, yy], fill=col, width=2)
            ym_ = oy + fila["max"][1] * esc
            dd.line([x0, ym_, x0 + lw - 1, ym_], fill=(220, 0, 0), width=2)
        d.text((x0, ym + lh + 3),
               f"{j+1}. s={fila['s']} · |dy| {_coma(fila['d'],1)} px al {fila['cual']} "
               f"(sup {_coma(fila['d_tl'],1)} · inf {_coma(fila['d_br'],1)})",
               fill=(70, 70, 70), font=f12)

    SALIDA.mkdir(exist_ok=True)
    ep = epoca_de(brazo)
    et = "ep000-sin-entrenar" if ep == 0 else f"ep{ep:03d}"
    destino = SALIDA / f"transformacion-{brazo}-{n}-paginas-{et}.png"
    lienzo.save(destino)
    return destino, {"ok": ok, "tl": ok_tl, "br": ok_br, "n": n,
                     "med": float(np.median([f["d"] for f in filas])),
                     "med_tl": float(np.median([f["d_tl"] for f in filas])),
                     "med_br": float(np.median([f["d_br"] for f in filas]))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", default=None, choices=sorted(BRAZOS))
    p.add_argument("--paginas", type=int, default=0)
    p.add_argument("--kernel", action="store_true")
    a = p.parse_args()
    brazos = [a.brazo] if a.brazo else list(BRAZOS)

    if a.kernel:
        for b in brazos:
            k = cargar_kernel(b)
            print(f"\nkernel {b} ({k.shape[0]}x{k.shape[0]}), sin bias:")
            print(f"  suma {k.sum():+.3f} · max {k.max():+.3f} · min {k.min():+.3f}")
    if a.paginas:
        # Se rinden UNA vez y se reusan en los cinco: son las mismas paginas a
        # proposito, para que la unica diferencia entre figuras sea el kernel.
        pags = paginas(a.paginas)
        print(f"{len(pags)} paginas de la particion `muestra` (fuera de train y val)\n")
        print(f"{'brazo':>6} {'fila max a <=2px':>18} {'(sup':>6} {'inf)':>6} "
              f"{'mediana':>9} {'med sup':>8} {'med inf':>8}")
        for b in brazos:
            destino, r = figura_paginas(b, pags)
            print(f"{b:>6} {str(r['ok'])+'/'+str(r['n']):>18} {r['tl']:>6} {r['br']:>6} "
                  f"{r['med']:>8.1f}px {r['med_tl']:>7.1f}px {r['med_br']:>7.1f}px   "
                  f"{destino.relative_to(EXP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
