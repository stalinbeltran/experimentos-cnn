#!/usr/bin/env python3
"""La figura de verificacion: las 10 muestras, todas juntas, en UNA imagen por red.

    python nn/muestras.py                     los cinco brazos
    python nn/muestras.py --brazo rot-k07     solo uno
    python nn/muestras.py --etiqueta ep000    sufijo del fichero

⚠ SALEN LOS CINCO, y eso vale MIENTRAS SEAN CINCO: cada figura pesa ~80 KB, o
    sea ~0,4 MB el barrido entero, dentro del tope de ~5 MB por experimento que
    declara el CLAUDE.md del repo. Si algun dia se arma alguna de las
    alternativas anotadas y los brazos se multiplican, esto vuelve a necesitar un
    filtro: 25 figuras son ~2 MB y el eje se lee en la tabla de metricas, no
    mirando 25 veces las mismas 10 ventanas.

Sale en `muestras/<brazo>-<etiqueta>.png`.

QUE ENSENA CADA CELDA, Y POR QUE ESO Y NO OTRA COSA
    arriba   la ventana 32x32 AMPLIADA x8 con vecino mas proximo. Sin ampliar no
             se ve nada; ampliando con interpolacion se verian pixeles que no
             existen, y esta figura existe para poder creerse lo que se ve.
    abajo    el MAPA DE RESPUESTA del kernel en falso color divergente centrado
             en cero (azul negativo, rojo positivo), COLOCADO EN SU SITIO: el
             mapa mide m = 32 - k + 1 y su posicion (i,j) mira el centro de su
             campo receptivo, o sea el pixel i + (k-1)/2 de la ventana. El gris
             de alrededor es el margen ciego de no usar padding.
    marcas   VERDE = la esquina verdadera (solo hay una por ventana, o ninguna).
             ROJO = la tl predicha · NARANJA = la br predicha. Las dos se pintan
             siempre, porque la red siempre opina de las dos.

⚠ CON DOS ESQUINAS, LA FIGURA TIENE QUE ENSENAR LAS DOS PREDICCIONES aunque
    solo una sea correcta: el modo de fallo que mas importa aqui es "la red pone
    las dos en el mismo sitio", y eso solo se ve si se dibujan las dos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datos import CONGELADAS, DATASET
from expcnn import exigir_dataset
from modelo import BRAZOS, ESQUINAS, VENTANA, construir, simetria

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
SALIDA = EXP / "muestras"
ESCALA = 8
LADO = VENTANA * ESCALA
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COLOR = {"tl": (230, 0, 0), "br": (245, 140, 0)}      # predichas
VERDE = (0, 190, 0)


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
    esc = float(np.abs(m).max()) or 1.0
    t = np.clip(m / esc, -1, 1)
    rgb = np.stack([np.where(t > 0, 255, 255 * (1 + t)),
                    255 * (1 - np.abs(t)),
                    np.where(t < 0, 255, 255 * (1 - t))], -1).astype(np.uint8)
    lienzo = np.full((VENTANA, VENTANA, 3), 228, np.uint8)      # gris = no representable
    off = (k - 1) // 2
    lienzo[off:off + m.shape[0], off:off + m.shape[1]] = rgb
    return Image.fromarray(lienzo).resize((LADO, LADO), Image.NEAREST)


def _cruz(d, x, y, color, r=14, w=3, aspa=False):
    """`aspa` dibuja una X en vez de una +. Hace falta: la tl y la br predichas
    caen EN EL MISMO SITIO mientras la red no ha aprendido nada (y ese es
    justamente el modo de fallo que hay que poder ver), y dos cruces iguales
    superpuestas se leen como una sola."""
    cx, cy = (x + 0.5) * ESCALA, (y + 0.5) * ESCALA
    if aspa:
        q = r * 0.71
        d.line([(cx - q, cy - q), (cx + q, cy + q)], fill=color, width=w)
        d.line([(cx - q, cy + q), (cx + q, cy - q)], fill=color, width=w)
    else:
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
        salida, mapa = red(torch.from_numpy(V).unsqueeze(1))
    pred = {c: (torch.sigmoid(salida[c][0]).numpy(), salida[c][1].numpy(), salida[c][2].numpy())
            for c in red.esquinas}

    cols, filas = 5, 2
    pie, sep = 62, 14
    cw, ch = LADO, LADO * 2 + pie + sep
    W = cols * cw + (cols + 1) * sep
    H = filas * ch + (filas + 1) * sep + 76
    lienzo = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(lienzo)
    f, fb = _fuente(14), _fuente(17)

    k = red.k
    aciertos = {c: [0, 0] for c in ESQUINAS}       # [aciertos, total] por esquina
    for i in range(len(V)):
        for c in red.esquinas:
            if d10[f"existe_{c}"][i] != 1:
                continue
            e = float(np.hypot(pred[c][1][i] - d10[f"x_{c}"][i],
                               pred[c][2][i] - d10[f"y_{c}"][i]))
            aciertos[c][0] += int(e <= 2); aciertos[c][1] += 1
    resumen = " · ".join(f"{c} {aciertos[c][0]}/{aciertos[c][1]}"
                         for c in red.esquinas if aciertos[c][1])
    sim, anti = simetria(red.kernel())
    dr.text((sep, 10), f"{brazo} · estructura `{red.estructura}` · kernel {k}x{k} · "
                       f"{red.n_parametros()} parametros ({red.n_cabeza()} de cabeza) · "
                       f"epoca {epoca}{'  (SIN ENTRENAR)' if epoca == 0 else ''} · "
                       f"acierta a <=2 px: {resumen}",
            fill=(20, 20, 20), font=fb)
    dr.text((sep, 32), f"arriba: ventana 32x32 (x8) · abajo: mapa del kernel en las MISMAS "
                       f"coordenadas (azul<0 rojo>0; gris = margen ciego {(k-1)//2} px) "
                       f"· kernel {100*anti:.0f}% antisimetrico bajo giro de 180",
            fill=(90, 90, 90), font=f)
    dr.text((sep, 50), "VERDE: la esquina verdadera (hay una por ventana, o ninguna) · "
                       "cruz ROJA: la tl predicha · ASPA NARANJA: la br predicha "
                       "— las dos se pintan siempre, y con forma distinta porque al empezar "
                       "caen en el mismo sitio",
            fill=(90, 90, 90), font=f)

    for i in range(len(V)):
        col, fila = i % cols, i // cols
        x0 = sep + col * (cw + sep)
        y0 = 76 + sep + fila * (ch + sep)
        lienzo.paste(_tinta_a_rgb(d10["ventanas"][i]), (x0, y0))
        ym = y0 + LADO + sep
        lienzo.paste(_mapa_a_rgb(mapa[i].numpy(), k), (x0, ym))

        for oy, r, w in ((y0, 14, 3), (ym, 10, 2)):
            cap = ImageDraw.Draw(lienzo)
            for c in ESQUINAS:                          # la verdadera, si existe
                if d10[f"existe_{c}"][i] == 1:
                    _cruz(cap, d10[f"x_{c}"][i] + x0 / ESCALA,
                          d10[f"y_{c}"][i] + oy / ESCALA, VERDE, r=r + 2, w=w)
            for c in red.esquinas:                      # las predichas, siempre
                _cruz(cap, float(pred[c][1][i]) + x0 / ESCALA,
                      float(pred[c][2][i]) + oy / ESCALA, COLOR[c], r=r, w=w,
                      aspa=(c == "br"))
            dr.rectangle([x0, oy, x0 + LADO - 1, oy + LADO - 1], outline=(120, 120, 120))

        dr.text((x0, ym + LADO + 5), f"{i+1}. {d10['clase'][i]}", fill=(20, 20, 20), font=f)
        lin = ym + LADO + 23
        for c in red.esquinas:
            real = d10[f"existe_{c}"][i] == 1
            txt = f"{c}: existe {'SI' if real else 'no'} · pred {float(pred[c][0][i]):.2f}"
            color = (70, 70, 70)
            if real:
                e = float(np.hypot(pred[c][1][i] - d10[f"x_{c}"][i],
                                   pred[c][2][i] - d10[f"y_{c}"][i]))
                # El umbral de 2 px es el de la metrica principal del criterio,
                # para que la figura y el criterio digan lo mismo.
                txt += f" · {e:.1f} px {'ACIERTA' if e <= 2 else 'falla'}"
                color = (0, 130, 0) if e <= 2 else (150, 60, 60)
            dr.text((x0, lin), txt, fill=color, font=f)
            lin += 18

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / f"{brazo}-{etiqueta}.png"
    lienzo.save(destino)
    return destino


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", default=None, choices=sorted(BRAZOS))
    p.add_argument("--etiqueta", default=None)
    a = p.parse_args()

    f = exigir_dataset(DATASET) / CONGELADAS
    if not f.exists():
        print(f"✗ el dataset publicado no trae {CONGELADAS}")
        return 1
    z = np.load(f, allow_pickle=True)
    d10 = {k: z[k] for k in z.files}
    print(f"{len(d10['clase'])} muestras · {int(d10['existe_tl'].sum())} con tl · "
          f"{int(d10['existe_br'].sum())} con br")

    for b in ([a.brazo] if a.brazo else list(BRAZOS)):
        pesos = AQUI / "pesos" / b / "last.pt"
        et = a.etiqueta or ("ep000-sin-entrenar" if not pesos.exists() else "actual")
        print(f"  {b}: {figura(b, d10, et).relative_to(EXP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
