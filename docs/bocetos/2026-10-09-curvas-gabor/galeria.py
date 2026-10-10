#!/usr/bin/env python3
"""Galería del detector de curvas (pedido del dueño, 2026-10-10): trazos de entrada característicos, variando UNA cosa
cada vez —grosor, radio, posición/orientación—, y lo que el detector hace con cada uno.

Cada columna es un trazo y tiene tres filas:
    1  ENTRADA          la imagen 32×32 tal cual
    2  FILTRO DE RECTAS la energía del banco de 12 Gabor (max_i r_i) y, encima, un palito por píxel con la orientación
                        de la recta que mejor responde ahí: la «cadena de rectas cortas» que lee el detector
    3  DETECTOR         |κ| (°/px) en los píxeles donde se pudo medir el giro; gris = trazo sin medir; y el veredicto

El detector es el de `curvas.py` sin tocar una línea (se importa). Sólo dibuja: no ajusta nada, no entrena nada.

    python galeria.py            → imagenes/10-galeria-*.png y resultados-galeria.json  (λ = 6, el de curvas.py)
    python galeria.py --lam 2    → lo mismo con el Gabor ESTRECHADO: imagenes/10-galeria-*-lam2.png y
                                   resultados-galeria-lam2.json. Sólo cambia λ (ancho de la franja central = λ/2 y
                                   envolvente transversal σ = λ/2); el kernel sigue 9×9 y TAU y demás umbrales, iguales
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import curvas as cv                                                 # noqa: E402
import torch                                                        # noqa: E402
from curvas import arco, recta, detectar, plt                       # noqa: E402

OK, MAL = "#1b7f3b", "#c2410c"


def _casar(vs, esperado: str, r_real: float | None) -> tuple[str, bool]:
    """Texto del veredicto y si coincide con lo esperado (para una curva, además el radio dentro de ±25 %)."""
    if not vs:
        return "NADA", esperado == "nada"
    v = vs[0]
    txt = v["que"].upper()
    bien = v["que"] == esperado
    if v["que"] == "curva":
        txt += f"\nR≈{v['radio']:.1f}" + (f" (real {r_real:g})" if r_real else "")
        if r_real and bien:
            bien = abs(np.log(v["radio"] / r_real)) < np.log(1.25)
    if len(vs) > 1:
        txt += f"\n+{len(vs) - 1} comp."
    return txt, bool(bien)


def figura(nombre_png: str, titulo: str, items: list[tuple[str, np.ndarray, str, float | None]]) -> list[dict]:
    cv._estilo()
    n = len(items)
    fig, axs = plt.subplots(3, n, figsize=(1.55 * n + 1.2, 6.3))
    res = []
    for j, (etq, x, esperado, r_real) in enumerate(items):
        c, kappa, mask, vs = detectar(x)
        a0, a1, a2 = axs[0, j], axs[1, j], axs[2, j]
        a0.imshow(x, cmap="gray_r", vmin=0, vmax=1)
        a0.set_title(etq, fontsize=8.5)
        a1.imshow(c["Emax"], cmap="Greys", vmin=0, vmax=max(3.3, c["Emax"].max()))
        cv._ticks(a1, c, mask)
        a2.imshow(x, cmap="gray_r", vmin=0, vmax=1, alpha=0.12)
        a2.imshow(np.where(mask & ~np.isfinite(kappa), 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.5)
        a2.imshow(np.where(np.isfinite(kappa), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=10)
        for v in vs:
            a2.contour(v["nucleo"].astype(float), levels=[0.5], colors=[cv.AZU], linewidths=0.6)
        txt, bien = _casar(vs, esperado, r_real)
        a2.set_xlabel(("✓ " if bien else "✗ ") + txt, fontsize=8.5, color=OK if bien else MAL, linespacing=1.1)
        for a in (a0, a1, a2):                                     # los palitos se salen del borde: fijar límites
            a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, x.shape[1] - 0.5); a.set_ylim(x.shape[0] - 0.5, -0.5)
        res.append({"trazo": etq, "esperado": esperado, "radio_real": r_real, "acierta": bien,
                    "componentes": [{k: v for k, v in z.items() if k not in ("mascara", "nucleo")} for z in vs]})
    for i, rot in enumerate(["1 · ENTRADA", "2 · FILTRO DE RECTAS\n(energía + orientación)", "3 · DETECTOR\n|κ| °/px · veredicto"]):
        axs[i, 0].set_ylabel(rot, fontsize=8.5, color=cv.T1)
    acier = sum(r["acierta"] for r in res)
    fig.suptitle(f"{titulo}   —   {acier}/{n} como se esperaba", color=cv.T1, fontsize=11)
    fig.text(0.5, 0.005, "fila 3: naranja = giro medido (más oscuro = gira más, radio ≈ 57,3/κ) · gris = trazo donde no se "
             "puede medir · contorno azul = lomo usado para el veredicto · ✓ curva: radio dentro de ±25 %",
             ha="center", fontsize=7.5, color=cv.T2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(cv.IMG / nombre_png, dpi=110); plt.close(fig)
    print("→", cv.IMG / nombre_png, f"{acier}/{n}")
    return res


def trazos() -> dict[str, tuple[str, list[tuple[str, np.ndarray, str, float | None, int]]]]:
    """Los 32 trazos de la galería, por grupo: (nombre del png, título, [(etiqueta, imagen, esperado, R real, grosor)]).
    Los comparte `bordes.py`, para que las dos galerías enseñen exactamente los mismos trazos."""
    G = [1, 2, 3, 4, 6, 8, 10, 12]
    grosor_arco = [(f"{g} px", arco(16, 10, 12, 0, 22, g), "curva", 12.0, g) for g in G]
    grosor_recta = [(f"{g} px", recta(16, 16, 22, 30, g), "recta", None, g) for g in G]
    radios = [(f"R = {r}", arco(16, 12 if r < 20 else 14, r, 0, 20, 3), "curva", float(r), 3) for r in (4, 6, 9, 12, 18, 27, 40)]
    radios.append(("recta (R = ∞)", recta(16, 16, 20, 0, 3), "recta", None, 3))
    pos = [("centro", arco(16, 12, 9, 0, 18, 3), "curva", 9.0, 3),
           ("arriba-izq.", arco(9, 5, 9, 0, 18, 3), "curva", 9.0, 3),
           ("abajo-der.", arco(23, 18, 9, 0, 18, 3), "curva", 9.0, 3),
           ("pegada al borde", arco(16, 1.5, 9, 0, 18, 3), "curva", 9.0, 3),
           ("cortada por el borde", arco(3, 12, 9, 0, 18, 3), "curva", 9.0, 3),
           ("girada 45°", arco(16, 14, 9, 45, 18, 3), "curva", 9.0, 3),
           ("girada 90°", arco(14, 16, 9, 90, 18, 3), "curva", 9.0, 3),
           ("girada 225°", arco(18, 18, 9, 225, 18, 3), "curva", 9.0, 3)]
    return {"grosor-arco": ("1-grosor-arco", "GROSOR · arco R = 12, 22 px de largo", grosor_arco),
            "grosor-recta": ("2-grosor-recta", "GROSOR · recta a 30°, 22 px de largo", grosor_recta),
            "radio": ("3-radio", "RADIO · arcos de 3 px de grosor y 20 px de largo", radios),
            "posicion": ("4-posicion", "POSICIÓN y ORIENTACIÓN · arco R = 9, 3 px, 18 px de largo", pos)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=cv.LAM, help="λ del Gabor (px); 2 = el mínimo de la rejilla (Nyquist)")
    lam = ap.parse_args().lam
    suf, tit = "", ""
    if lam != cv.LAM:                    # el detector lee el banco global en cada llamada: basta con sustituirlo
        cv.LAM = lam
        cv.GABOR = torch.from_numpy(np.stack([cv.kernel_gabor(cv.K, t, lam) for t in cv.THETAS])[:, None])
        suf, tit = f"-lam{lam:g}", f"  ·  Gabor λ = {lam:g} (franja central {lam / 2:g} px)"
    out = {k: figura(f"10-galeria-{png}{suf}.png", t + tit, [it[:4] for it in items])
           for k, (png, t, items) in trazos().items()}
    (AQUI / f"resultados-galeria{suf}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
