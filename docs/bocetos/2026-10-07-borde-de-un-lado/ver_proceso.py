#!/usr/bin/env python3
"""Boceto (NO es un experimento, no entrena nada): cómo vería un dígito un banco de detectores de borde DE UN SOLO LADO.

Cada detector es UN kernel 3×3 seguido de ReLU, aplicado a la tinta TAL CUAL (sin pre-proceso: es la primera capa de la
red, no un script delante de ella). El kernel del detector θ es la derivada en la dirección θ:

    k_θ = cos θ · Sx + sin θ · Sy        (Sx, Sy: Sobel, normalizados)
    salida_θ = ReLU(k_θ ⋆ tinta)

ReLU deja pasar sólo donde la tinta AUMENTA al avanzar en θ: «blanco → negro yendo hacia θ». El de θ + 180° es
«negro → blanco yendo hacia θ». Un trazo, recorrido en θ, tiene un borde de entrada y uno de salida con signos opuestos,
así que cada detector ve UNO de los dos, por grueso que sea el trazo.

    /tmp/vizenv/bin/python ver_proceso.py --datos <dir con datos.npz de uci-optdigits-orig-32px-r20261005>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import correlate

AQUI = Path(__file__).resolve().parent
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4      # tinta crece hacia la DERECHA
SY = SX.T                                                            # tinta crece hacia ABAJO
DIRS = [("→", 0), ("↘", 45), ("↓", 90), ("↙", 135), ("←", 180), ("↖", 225), ("↑", 270), ("↗", 315)]


def kernel(grados):
    t = np.deg2rad(grados)
    return np.cos(t) * SX + np.sin(t) * SY


def detectar(img, grados):
    return np.maximum(0, correlate(img.astype(float), kernel(grados), mode="constant"))


def dos_bordes(img):
    return np.hypot(correlate(img.astype(float), SX, mode="constant"), correlate(img.astype(float), SY, mode="constant"))


def fig_concepto(dest):
    img = np.zeros((32, 32)); img[4:28, 10:20] = 1                    # barra vertical de 10 px
    fila = 16
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    for a, (t, m) in zip(ax[:4], [("tinta (trazo de 10 px)", img), ("|gradiente|: DOS bordes", dos_bordes(img)),
                                  ("detector → (blanco→negro)\nsólo el borde IZQUIERDO", detectar(img, 0)),
                                  ("detector ← (= negro→blanco hacia →)\nsólo el borde DERECHO", detectar(img, 180))]):
        a.imshow(m, cmap="gray_r" if m is img else "magma"); a.set_title(t, fontsize=9); a.axis("off")
        a.axhline(fila, color="c", lw=.6)
    x = np.arange(32)
    ax[4].plot(x, img[fila], "k", label="tinta"); ax[4].plot(x, detectar(img, 0)[fila], "r", label="detector →")
    ax[4].plot(x, detectar(img, 180)[fila], "b", label="detector ←"); ax[4].legend(fontsize=8)
    ax[4].set_title("corte por la fila cian", fontsize=9)
    fig.suptitle("Un detector de un solo lado ve UN borde del trazo, sea cual sea su grosor", fontsize=11)
    fig.tight_layout(); fig.savefig(dest, dpi=110); plt.close(fig)


def fig_kernels(dest):
    fig, ax = plt.subplots(1, 8, figsize=(14, 2.4))
    for a, (f, g) in zip(ax, DIRS):
        k = kernel(g); a.imshow(k, cmap="bwr", vmin=-.8, vmax=.8)
        for (i, j), v in np.ndenumerate(k):
            a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
        a.set_title(f"{f}  ({g}°)"); a.axis("off")
    fig.suptitle("Los 8 kernels 3×3 (rojo = tinta esperada, azul = fondo esperado). Fijos aquí; en la red, aprendibles",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(dest, dpi=110); plt.close(fig)


def superponer(img):
    """4 direcciones en color sobre la tinta en gris: rojo →, verde ↓, azul ←, amarillo ↑."""
    rgb = np.dstack([1 - .25 * img] * 3)
    for g, c in [(0, (1, 0, 0)), (90, (0, .7, 0)), (180, (0, 0, 1)), (270, (.9, .7, 0))]:
        r = detectar(img, g); r = r / (r.max() or 1)
        rgb = rgb * (1 - r[..., None]) + np.array(c) * r[..., None]
    return np.clip(rgb, 0, 1)


def fig_digitos(img_por_fila, titulos, dest, sup):
    n = len(img_por_fila)
    fig, ax = plt.subplots(n, 11, figsize=(19, 1.95 * n))
    ax = np.atleast_2d(ax)
    for r, (img, tit) in enumerate(zip(img_por_fila, titulos)):
        celdas = [("tinta", img, "gray_r"), ("|grad| (2 bordes)", dos_bordes(img), "magma")]
        celdas += [(f"det {f}", detectar(img, g), "magma") for f, g in DIRS]
        for c, (t, m, cm) in enumerate(celdas):
            ax[r, c].imshow(m, cmap=cm); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(t, fontsize=9)
        ax[r, 10].imshow(superponer(img)); ax[r, 10].axis("off")
        if r == 0: ax[r, 10].set_title("→ rojo ↓ verde\n← azul ↑ amarillo", fontsize=8)
        ax[r, 0].text(-4, 16, tit, ha="right", va="center", fontsize=9)
    fig.suptitle(sup, fontsize=11); fig.tight_layout(); fig.savefig(dest, dpi=100); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--datos", required=True)
    a = p.parse_args()
    d = np.load(Path(a.datos) / "datos.npz")
    X, y = d["imagenes"], d["etiquetas"]
    out = AQUI / "imagenes"; out.mkdir(exist_ok=True)
    fig_concepto(out / "1-concepto-barra.png")
    fig_kernels(out / "2-kernels.png")
    tinta = X.reshape(len(X), -1).sum(1)
    ochos = np.where(y == 8)[0]; ochos = ochos[np.argsort(tinta[ochos])]
    sel = [ochos[int(q * (len(ochos) - 1))] for q in (0.02, 0.5, 0.98, 1.0)]
    fig_digitos([X[i] for i in sel], [f"8 #{i}\n{tinta[i]} px" for i in sel], out / "3-ochos-fino-a-grueso.png",
                "Ochos de UCI, del más fino al más grueso (sin pre-proceso). Cada detector dibuja UN lado del trazo")
    otros = [np.where(y == k)[0][np.argmax(tinta[y == k])] for k in range(10)]
    fig_digitos([X[i] for i in otros], [f"{y[i]} #{i}" for i in otros], out / "4-los-diez-mas-gruesos.png",
                "El dígito MÁS grueso de cada clase")
    print("\n".join(str(f) for f in sorted(out.iterdir())))


if __name__ == "__main__":
    main()
