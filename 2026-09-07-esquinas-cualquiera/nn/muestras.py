#!/usr/bin/env python3
"""La figura de verificacion: las 10 muestras, todas juntas, en UNA imagen por red.

    python nn/muestras.py                     los cinco brazos
    python nn/muestras.py --brazo k07         solo uno
    python nn/muestras.py --etiqueta ep000    sufijo del fichero

Sale en `muestras/<brazo>-<etiqueta>.png`.

QUE ENSENA CADA CELDA
    arriba   la ventana 32x32 AMPLIADA x8 con vecino mas proximo. Ampliando con
             interpolacion se verian pixeles que no existen, y esta figura existe
             para poder creerse lo que se ve.
    abajo    el MAPA DE RESPUESTA en falso color divergente centrado en cero
             (azul negativo, rojo positivo), COLOCADO EN SU SITIO: la posicion
             (i,j) del mapa mira el pixel i+(k-1)/2 de la ventana. El gris de
             alrededor es el margen ciego de no usar padding.
    marcas   verde = la esquina VERDADERA (la que sea) · rojo = la PREDICHA.

⚠ EL PIE DICE DE QUE CLASE ERA, aunque la tarea ya no distinga. Es la unica
    forma de ver a ojo si "detecta cualquiera" se resolvio de verdad o se
    resolvio detectando siempre la misma -- que es el fallo de `esq-2d` con otro
    nombre, y no se veria en un numero global.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datos import congeladas
from modelo import BRAZOS, VENTANA, construir, simetria

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
SALIDA = EXP / "muestras"
ESCALA = 8
LADO = VENTANA * ESCALA
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _fuente(t: int):
    try:
        return ImageFont.truetype(FUENTE, t)
    except OSError:
        return ImageFont.load_default()


def _tinta_a_rgb(v: np.ndarray) -> Image.Image:
    """Lo guardado es TINTA (fondo 0). Se invierte para mirarlo, porque un
    parrafo en negro sobre blanco es lo que el ojo sabe leer."""
    return Image.fromarray(np.stack([255 - v] * 3, -1)).resize((LADO, LADO), Image.NEAREST)


def _mapa_a_rgb(m: np.ndarray, k: int) -> Image.Image:
    esc = float(np.abs(m).max()) or 1.0
    t = np.clip(m / esc, -1, 1)
    rgb = np.stack([np.where(t > 0, 255, 255 * (1 + t)),
                    255 * (1 - np.abs(t)),
                    np.where(t < 0, 255, 255 * (1 - t))], -1).astype(np.uint8)
    lienzo = np.full((VENTANA, VENTANA, 3), 228, np.uint8)     # gris = no representable
    off = (k - 1) // 2
    lienzo[off:off + m.shape[0], off:off + m.shape[1]] = rgb
    return Image.fromarray(lienzo).resize((LADO, LADO), Image.NEAREST)


def _cruz(d, x, y, color, r=14, w=3):
    cx, cy = (x + 0.5) * ESCALA, (y + 0.5) * ESCALA
    d.line([(cx - r, cy), (cx + r, cy)], fill=color, width=w)
    d.line([(cx, cy - r), (cx, cy + r)], fill=color, width=w)


def figura(brazo: str, d10: dict, etiqueta: str) -> Path:
    red = construir(brazo)
    pesos = AQUI / "pesos" / brazo / "last.pt"
    epoca = 0
    if pesos.exists():
        est = torch.load(pesos, map_location="cpu", weights_only=False)
        red.load_state_dict(est["modelo"])
        epoca = est["epoca"]
    red.eval()

    V = d10["ventanas"].astype(np.float32) / 255.0
    with torch.no_grad():
        logit, px, py, mapa = red(torch.from_numpy(V).unsqueeze(1))
        prob = torch.sigmoid(logit)

    cols, filas = 5, 2
    pie, sep = 46, 14
    ch = LADO * 2 + pie + sep
    W = cols * LADO + (cols + 1) * sep
    H = filas * ch + (filas + 1) * sep + 58
    lienzo = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(lienzo)
    f, fb = _fuente(14), _fuente(17)

    k = red.k
    err = [float(np.hypot(float(px[i]) - d10["x"][i], float(py[i]) - d10["y"][i]))
           if d10["existe"][i] == 1 else None for i in range(len(V))]
    aciertos = sum(1 for e in err if e is not None and e <= 2)
    n_pos = int(d10["existe"].sum())
    sim, _ = simetria(red.kernel())
    d.text((sep, 10), f"{brazo} · estructura `{red.estructura}` · kernel {k}x{k} · "
                      f"{red.n_parametros()} parametros · epoca {epoca}"
                      f"{'  (SIN ENTRENAR)' if epoca == 0 else ''} · acierta a <=2 px en "
                      f"{aciertos}/{n_pos} · kernel {100*sim:.0f}% simetrico",
           fill=(20, 20, 20), font=fb)
    d.text((sep, 33), f"arriba: ventana 32x32 (x8) · abajo: mapa del kernel en las MISMAS "
                      f"coordenadas (azul<0 rojo>0; gris = margen ciego {(k-1)//2} px) "
                      f"· verde: la esquina verdadera (la que sea) · rojo: la predicha",
           fill=(90, 90, 90), font=f)

    for i in range(len(V)):
        col, fila = i % cols, i // cols
        x0 = sep + col * (LADO + sep)
        y0 = 58 + sep + fila * (ch + sep)
        lienzo.paste(_tinta_a_rgb(d10["ventanas"][i]), (x0, y0))
        ym = y0 + LADO + sep
        lienzo.paste(_mapa_a_rgb(mapa[i].numpy(), k), (x0, ym))
        for oy, r, w in ((y0, 14, 3), (ym, 10, 2)):
            cap = ImageDraw.Draw(lienzo)
            if d10["existe"][i] == 1:
                _cruz(cap, d10["x"][i] + x0 / ESCALA, d10["y"][i] + oy / ESCALA,
                      (0, 190, 0), r=r + 2, w=w)
            _cruz(cap, float(px[i]) + x0 / ESCALA, float(py[i]) + oy / ESCALA,
                  (230, 0, 0), r=r, w=w)
            d.rectangle([x0, oy, x0 + LADO - 1, oy + LADO - 1], outline=(120, 120, 120))

        real = "SI" if d10["existe"][i] == 1 else "no"
        txt, color = "", (70, 70, 70)
        if err[i] is not None:
            # El umbral de 2 px es el de la metrica principal del criterio, para
            # que la figura y el criterio digan lo mismo y no haya que traducir.
            txt = f" · {err[i]:.1f} px {'ACIERTA' if err[i] <= 2 else 'falla'}"
            color = (0, 130, 0) if err[i] <= 2 else (150, 60, 60)
        d.text((x0, ym + LADO + 5), f"{i+1}. {d10['clase'][i]}", fill=(20, 20, 20), font=f)
        d.text((x0, ym + LADO + 23), f"existe {real} · pred {float(prob[i]):.2f}{txt}",
               fill=color, font=f)

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"{brazo}-{etiqueta}.png"
    lienzo.save(destino)
    return destino


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", default=None, choices=sorted(BRAZOS))
    p.add_argument("--etiqueta", default=None)
    a = p.parse_args()

    d10 = congeladas()
    print(f"{len(d10['existe'])} muestras · {int(d10['existe'].sum())} con esquina")
    for b in ([a.brazo] if a.brazo else list(BRAZOS)):
        pesos = AQUI / "pesos" / b / "last.pt"
        et = a.etiqueta or ("ep000-sin-entrenar" if not pesos.exists() else "actual")
        print(f"  {b}: {figura(b, d10, et).relative_to(EXP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
