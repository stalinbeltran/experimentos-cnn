#!/usr/bin/env python3
"""Figuras para MIRAR el dataset de `banco-k`. Se llama desde `datos.py --muestras N`.

    python nn/datos.py --muestras 12 [--semilla-fig 7]

Produce dos PNG en `muestras/`:

  parrafos-<N>-marco146.png   lo que se guarda: 146x146, con la caja de TINTA (la
                              etiqueta) y el recorte a 128 marcado
  parrafos-<N>-marco128.png   lo que la RED ve en la condicion identidad: el recorte
                              central a 128x128, con la etiqueta ya transformada

POR QUE DOS Y NO UNA: la etiqueta vive en el marco de 584 y la red trabaja en el de
128 tras `(coord/4) - 9` (§6.4). Un solo dibujo no permite ver si la transformacion
esta bien; con los dos, un desplazamiento de 9 px salta a la vista -- y ese es
justamente el fallo que el §14 describe como «todas las condiciones uniformemente
malas, sin senyal de la causa».

Sin matplotlib a proposito: no esta en el venv del repo y una figura de rejilla con
cajas se hace con PIL, que si esta. Una dependencia nueva para dibujar rectangulos
no se paga.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
DATOS = EXP / "datos"
MUESTRAS = EXP / "muestras"
REDUCCION, DESCARTE, FINAL = 4, 9, 128
ESCALA = 2                # las de 146 px se ven pequenyas en pantalla
MARGEN, PIE = 6, 26


def _a_gris(a: np.ndarray) -> np.ndarray:
    """uint16 (suma del bloque 4x4) -> uint8 para mirar. Solo para la figura."""
    return np.clip(a.astype(np.float32) / (REDUCCION ** 2), 0, 255).astype(np.uint8)


def _celda(img: np.ndarray, caja, pie: str, recorte: bool) -> Image.Image:
    """Una muestra: la imagen, su caja en rojo, y el recorte a 128 en azul."""
    base = Image.fromarray(_a_gris(img)).convert("RGB")
    base = base.resize((base.width * ESCALA, base.height * ESCALA), Image.NEAREST)
    d = ImageDraw.Draw(base)
    izq, der, sup, inf = caja
    d.rectangle([izq * ESCALA, sup * ESCALA, der * ESCALA, inf * ESCALA],
                outline=(220, 30, 30), width=1)
    if recorte:
        d.rectangle([DESCARTE * ESCALA, DESCARTE * ESCALA,
                     (DESCARTE + FINAL) * ESCALA, (DESCARTE + FINAL) * ESCALA],
                    outline=(40, 90, 220), width=1)
    lienzo = Image.new("RGB", (base.width, base.height + PIE), (255, 255, 255))
    lienzo.paste(base, (0, 0))
    ImageDraw.Draw(lienzo).text((2, base.height + 3), pie, fill=(20, 20, 20))
    return lienzo


def _rejilla(celdas: list[Image.Image], cols: int) -> Image.Image:
    w, h = celdas[0].size
    filas = (len(celdas) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + MARGEN) + MARGEN, filas * (h + MARGEN) + MARGEN),
                    (245, 245, 245))
    for i, c in enumerate(celdas):
        out.paste(c, (MARGEN + (i % cols) * (w + MARGEN),
                      MARGEN + (i // cols) * (h + MARGEN)))
    return out


def figura(n: int, semilla: int) -> int:
    if not (DATOS / "manifiesto.json").is_file():
        print("✗ no hay etapa local. Genera primero: --imagenes 1000")
        return 1
    m = json.loads((DATOS / "manifiesto.json").read_text(encoding="utf-8"))
    meta = json.loads(np.load(DATOS / "meta.npz", allow_pickle=False)["meta"].item()) \
        if (DATOS / "meta.npz").is_file() else None

    # Se muestrean de las TRES particiones, proporcionalmente: mirar solo `train` (100
    # de 1000) daria una idea sesgada de un dataset cuyo 80 % es `eval`.
    imgs, cajas, pies, ind = [], [], [], []
    rng = np.random.default_rng(semilla)
    for k in ("train", "monitor", "eval"):
        d = np.load(DATOS / f"{k}.npz")
        cuantas = max(1, round(n * len(d["etiquetas"]) / m["imagenes_validas"]))
        for j in rng.choice(len(d["etiquetas"]), size=min(cuantas, len(d["etiquetas"])),
                            replace=False):
            imgs.append(d["imagenes"][j])
            cajas.append(d["etiquetas"][j])
            ind.append(int(d["indices"][j]))
            pies.append(k)
    print(f"{len(imgs)} muestras de {len(set(pies))} particiones")

    c146, c128 = [], []
    for img, caja, part, i in zip(imgs, cajas, pies, ind):
        e = meta[i] if meta else {}
        # el marco de 146: la etiqueta de 584 dividida por 4
        caja146 = tuple(v / REDUCCION for v in (caja[0], caja[1], caja[2], caja[3]))
        etq = f"{part[:3]} #{i}"
        if e:
            etq += f" {e['fuente'][:7]} {e['cuerpo']:.0f}px lh{e['interlineado']:.2f}"
        c146.append(_celda(img, (caja146[0], caja146[1], caja146[2], caja146[3]),
                           etq, recorte=True))
        # el marco de 128: recorte central y etiqueta transformada (§6.4)
        rec = img[DESCARTE:DESCARTE + FINAL, DESCARTE:DESCARTE + FINAL]
        caja128 = tuple(v / REDUCCION - DESCARTE for v in (caja[0], caja[1], caja[2], caja[3]))
        c128.append(_celda(rec, caja128, f"{etq}  [{caja128[0]:.0f},{caja128[1]:.0f}]"
                           f"x[{caja128[2]:.0f},{caja128[3]:.0f}]", recorte=False))

    MUESTRAS.mkdir(parents=True, exist_ok=True)
    cols = 4 if len(c146) >= 8 else 3
    for nombre, celdas in (("marco146", c146), ("marco128", c128)):
        f = MUESTRAS / f"parrafos-{len(celdas)}-{nombre}.png"
        _rejilla(celdas, cols).save(f, optimize=True)
        print(f"  {f.relative_to(EXP)}  ({f.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestras", type=int, default=12)
    ap.add_argument("--semilla-fig", type=int, default=7)
    a = ap.parse_args()
    raise SystemExit(figura(a.muestras, a.semilla_fig))
