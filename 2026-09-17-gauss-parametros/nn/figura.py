#!/usr/bin/env python3
"""El mismo barrido que la app, pero en UNA imagen. numpy + pillow.

    python nn/figura.py                          sigmas por defecto, k ajustado
    python nn/figura.py --sigmas 0.4,0.8,1.5,3   los que quieras
    python nn/figura.py --n 6 --salida /tmp/x.png

POR QUE EXISTE SI YA HAY UNA APP
================================
Porque la app depende de un puerto, de `ufw` y de la red de en medio, y eso ha
fallado **cuatro veces** en esta maquina (ver `telegram-coordinator/CLAUDE.md` §
«LO PRIMERO SI ACABAS DE NACER», punto 1). Una figura es un fichero: se mira desde
Telegram, desde `scp`, o desde donde sea, y no se puede quedar inalcanzable.

La app sirve para BUSCAR (mover el mando y mirar); la figura, para FIJAR lo que se
vio y poder enseyarlo despues. No se sustituyen.

⚠ QUE SE PUEDE COMMITEAR DE AQUI, Y QUE NO. Una figura ELEGIDA es un resultado y
va a `muestras/` (es lo que ya hace `banco-k` con `condiciones-4x6.png`). Un
VOLCADO de todas las combinaciones NO: seria republicar el dataset con otro nombre,
y el dataset vive en un repo PRIVADO mientras este es PUBLICO. Por eso el destino
por defecto es `/tmp`, y dejarla en el repo es una decision que hay que teclear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import banco                                          # noqa: E402
import forma                                          # noqa: E402
from app import Visor, a_png                          # noqa: E402

SIGMAS = (0.4, 0.8, 1.5, 2.5)
MARGEN_SUP = 34
MARGEN_IZQ = 4
ESCALA = 2


def figura(sigmas, n: int, semilla: int, salida: Path) -> int:
    v = Visor(n, semilla)
    cols = [("identidad", None, None)] + [
        (f"sigma {s:g}", forma.k_minimo(s) or banco.K_MAX, s) for s in sigmas]

    lado = banco.FINAL * ESCALA
    an = MARGEN_IZQ + len(cols) * (lado + 6)
    al = MARGEN_SUP + len(v.sel) * (lado + 6)
    out = Image.new("RGB", (an, al), "#1a1a1a")
    d = ImageDraw.Draw(out)

    for c, (titulo, k, s) in enumerate(cols):
        x = MARGEN_IZQ + c * (lado + 6)
        sub = "recorte puro, sin kernel" if k is None else f"k={k}"
        if k is not None:
            t = forma.truncamiento(k, s)
            if t > forma.TRUNCAMIENTO_AVISO:
                sub += f"  CORTA {t:.0%}"
        d.text((x + 3, 4), titulo, fill="#eee")
        d.text((x + 3, 18), sub, fill="#888")
        z = v.z(k, s if s is not None else 0.0)
        for f in range(len(v.sel)):
            y = MARGEN_SUP + f * (lado + 6)
            png = a_png(z[f], v.ventana, v.cajas[f], escala=ESCALA)
            import io
            out.paste(Image.open(io.BytesIO(png)), (x, y))

    salida.parent.mkdir(parents=True, exist_ok=True)
    out.save(salida, optimize=True)
    r = v.m.resumen()
    print(f"\n{salida}  ({salida.stat().st_size/1000:.0f} KB, {out.size[0]}x{out.size[1]})")
    print(f"  {len(v.sel)} muestras x {len(cols)} condiciones · `train` · "
          f"{r['n_reservadas']} descartadas por §3.7")
    print(f"  escala de gris FIJA [{v.ventana[0]:.2f}, {v.ventana[1]:.2f}], de la "
          f"identidad: las columnas se pueden comparar entre si")
    print(f"  la caja roja es la ETIQUETA, que es lo unico que el banco mide\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default=",".join(f"{s:g}" for s in SIGMAS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--semilla", type=int, default=17)
    ap.add_argument("--salida", default="/tmp/gauss-p.png")
    a = ap.parse_args()
    try:
        sig = [float(x) for x in a.sigmas.split(",") if x.strip()]
    except ValueError:
        ap.error(f"--sigmas tiene que ser una lista de numeros: «{a.sigmas}»")
    if not sig:
        ap.error("--sigmas vacio")
    for s in sig:
        if s <= 0:
            ap.error(f"sigma tiene que ser > 0 y hay un {s}")
    return figura(sig, a.n, a.semilla, Path(a.salida))


if __name__ == "__main__":
    raise SystemExit(main())
