#!/usr/bin/env python3
"""La figura de verificacion: las 10 muestras, todas juntas, en UNA imagen por red.

    python nn/muestras.py                     todas las redes, en el estado que tengan
    python nn/muestras.py --brazo k03         solo una
    python nn/muestras.py --etiqueta ep000    sufijo del fichero

Sale en `muestras/<brazo>-<etiqueta>.png`.

QUE ENSENA CADA CELDA, Y POR QUE ESO Y NO OTRA COSA
    arriba   la ventana 32x32 AMPLIADA x8 con vecino mas proximo. Sin ampliar no
             se ve nada; ampliando con interpolacion se verian pixeles que no
             existen, y esta figura existe para poder creerse lo que se ve.
    abajo    el MAPA DE RESPUESTA del kernel, en falso color divergente centrado
             en cero (azul negativo, rojo positivo), normalizado por su mayor
             magnitud. Es donde se ve de un vistazo si el kernel pica en la
             esquina o se ha vuelto un difuminador.
    marcas   verde = esquina VERDADERA (solo si existe) · rojo = PREDICHA.

    El mapa es mas pequeno que la ventana (m = 32 - k + 1) y se amplia al mismo
    tamano, asi que las dos vistas se pueden mirar juntas -- pero el mapa NO
    cubre el borde: es el margen ciego de (k-1)/2 px que trae el no usar padding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from modelo import BRAZOS, VENTANA, construir

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
SALIDA = EXP / "muestras"
ESCALA = 8
LADO = VENTANA * ESCALA
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _fuente(tam: int):
    try:
        return ImageFont.truetype(FUENTE, tam)
    except OSError:
        return ImageFont.load_default()


def _tinta_a_rgb(v: np.ndarray) -> Image.Image:
    """Lo guardado es TINTA (fondo 0). Se invierte para mirarlo, porque un
    parrafo en negro sobre blanco es lo que el ojo sabe leer."""
    g = 255 - v
    return Image.fromarray(np.stack([g] * 3, -1)).resize((LADO, LADO), Image.NEAREST)


def _mapa_a_rgb(m: np.ndarray, k: int) -> Image.Image:
    """Divergente centrado en 0 (azul<0, blanco 0, rojo>0), COLOCADO EN SU SITIO.

    ⚠ El mapa mide m x m con m = 32 - k + 1, y su posicion (i, j) mira el centro
    de su campo receptivo, o sea el pixel i + (k-1)/2 de la ventana. Estirarlo a
    32x32 lo pondria en otro sistema de coordenadas que la ventana de arriba, y
    esta figura existe para comparar las dos: donde pica el mapa contra donde
    esta la esquina. Asi que se pega en su offset real y el resto se pinta gris
    -- que ademas ENSENA el margen ciego que trae no usar padding."""
    esc = float(np.abs(m).max()) or 1.0
    t = np.clip(m / esc, -1, 1)
    r = np.where(t > 0, 255, 255 * (1 + t))
    g = 255 * (1 - np.abs(t))
    b = np.where(t < 0, 255, 255 * (1 - t))
    rgb = np.stack([r, g, b], -1).astype(np.uint8)
    lienzo = np.full((VENTANA, VENTANA, 3), 228, np.uint8)      # gris = no representable
    off = (k - 1) // 2
    lienzo[off:off + m.shape[0], off:off + m.shape[1]] = rgb
    return Image.fromarray(lienzo).resize((LADO, LADO), Image.NEAREST)


def _cruz(d: ImageDraw.ImageDraw, x: float, y: float, color, r: int = 14, w: int = 3):
    cx, cy = (x + 0.5) * ESCALA, (y + 0.5) * ESCALA
    d.line([(cx - r, cy), (cx + r, cy)], fill=color, width=w)
    d.line([(cx, cy - r), (cx, cy + r)], fill=color, width=w)


def figura(brazo: str, datos: dict, etiqueta: str) -> Path:
    red = construir(brazo)
    pesos = AQUI / "pesos" / brazo / "last.pt"
    epoca = 0
    if pesos.exists():
        est = torch.load(pesos, map_location="cpu", weights_only=False)
        red.load_state_dict(est["modelo"])
        epoca = est["epoca"]
    red.eval()

    V = datos["ventanas"].astype(np.float32) / 255.0
    with torch.no_grad():
        logit, px, py, mapa = red(torch.from_numpy(V).unsqueeze(1))
        prob = torch.sigmoid(logit)

    cols, filas = 5, 2
    pie, sep = 46, 14
    cw, ch = LADO, LADO * 2 + pie + sep
    W = cols * cw + (cols + 1) * sep
    H = filas * ch + (filas + 1) * sep + 58
    lienzo = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(lienzo)
    f, fb = _fuente(14), _fuente(17)

    k = BRAZOS[brazo]
    d.text((sep, 10), f"{brazo} · kernel {k}x{k} · {k*k + 3} parametros · "
                      f"epoca {epoca}{'  (SIN ENTRENAR)' if epoca == 0 else ''}",
           fill=(20, 20, 20), font=fb)
    d.text((sep, 33), f"arriba: ventana 32x32 (x8) · abajo: mapa del kernel en las MISMAS "
                      f"coordenadas (azul<0 rojo>0; gris = margen ciego {(k-1)//2} px) "
                      f"· verde: verdadera · rojo: predicha",
           fill=(90, 90, 90), font=f)

    for i in range(len(V)):
        c, fila = i % cols, i // cols
        x0 = sep + c * (cw + sep)
        y0 = 58 + sep + fila * (ch + sep)

        lienzo.paste(_tinta_a_rgb(datos["ventanas"][i]), (x0, y0))
        cap = ImageDraw.Draw(lienzo)
        if datos["existe"][i] == 1:
            _cruz(cap, datos["x"][i] + x0 / ESCALA, datos["y"][i] + y0 / ESCALA, (0, 200, 0))
        _cruz(cap, float(px[i]) + x0 / ESCALA, float(py[i]) + y0 / ESCALA, (230, 0, 0))
        d.rectangle([x0, y0, x0 + LADO - 1, y0 + LADO - 1], outline=(120, 120, 120))

        ym = y0 + LADO + sep
        lienzo.paste(_mapa_a_rgb(mapa[i].numpy(), k), (x0, ym))
        mp = ImageDraw.Draw(lienzo)
        if datos["existe"][i] == 1:
            _cruz(mp, datos["x"][i] + x0 / ESCALA, datos["y"][i] + ym / ESCALA, (0, 160, 0), r=10, w=2)
        _cruz(mp, float(px[i]) + x0 / ESCALA, float(py[i]) + ym / ESCALA, (180, 0, 0), r=10, w=2)
        d.rectangle([x0, ym, x0 + LADO - 1, ym + LADO - 1], outline=(120, 120, 120))

        clase = str(datos["clase"][i])
        real = "SI" if datos["existe"][i] == 1 else "no"
        err = ""
        if datos["existe"][i] == 1:
            e = ((float(px[i]) - datos["x"][i]) ** 2 + (float(py[i]) - datos["y"][i]) ** 2) ** 0.5
            err = f" · error {e:.1f} px"
        d.text((x0, ym + LADO + 5), f"{i+1}. {clase}", fill=(20, 20, 20), font=f)
        d.text((x0, ym + LADO + 23), f"existe {real} · pred {float(prob[i]):.2f}{err}",
               fill=(70, 70, 70), font=f)

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"{brazo}-{etiqueta}.png"
    lienzo.save(destino)
    return destino


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", default=None)
    p.add_argument("--etiqueta", default=None)
    a = p.parse_args()

    f = AQUI / "muestras.npz"
    if not f.exists():
        print(f"✗ no estan las muestras ({f.name}). Generalas: python nn/datos.py")
        return 1
    z = np.load(f, allow_pickle=True)
    datos = {k: z[k] for k in ("ventanas", "existe", "x", "y", "clase")}
    print(f"{len(datos['existe'])} muestras · {int(datos['existe'].sum())} con la esquina")

    brazos = [a.brazo] if a.brazo else list(BRAZOS)
    for b in brazos:
        pesos = AQUI / "pesos" / b / "last.pt"
        et = a.etiqueta or ("ep000-sin-entrenar" if not pesos.exists() else "actual")
        print(f"  {b}: {figura(b, datos, et).relative_to(EXP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
