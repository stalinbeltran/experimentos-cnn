#!/usr/bin/env python3
"""Un EJEMPLO, paso a paso, de por qué una curva gruesa sale como recta (pedido por el dueño, 2026-10-08).

Toma del banco una curva de radio 9 y grosor 8 (y, al lado, una recta del mismo grosor) y la pasa por el brazo
Sobel · 9×9 · N = 1000 · semilla 0 de la rejilla, enseñando cada paso del detector:

    1 la imagen  →  2 Sobel  →  3 los 4 kernels (K0, K45 y sus rot90)  →  4 el mapa de respuesta K⋆x de cada uno
    →  5 el MAX de cada mapa (y dónde está)  →  6 logit = a·max + c; detecta si > 0; orientación = el mayor

    python nn/ejemplo_curva.py   → resultados/ejemplo-curva.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp                                    # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402
from matplotlib.colors import TwoSlopeNorm                         # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos as D                                                  # noqa: E402
import modelo as M                                                 # noqa: E402
import prepro                                                      # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, NAR = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834"
ORI = ("0° —", "45° \\", "90° |", "135° /")


def brazo(pre="sobel", k=9, n=1000, s=0) -> M.Lineal:
    for f in sorted((AQUI.parent / "resultados" / "trozos").glob("rejilla-*.jsonl")):
        for r in map(json.loads, f.open()):
            if (r["pre"], r["k"], r["n"], r["semilla"]) == (pre, k, n, s):
                m = M.Lineal(k, 1)
                with torch.no_grad():
                    m.K[0, 0] = torch.tensor(r["K0"]); m.K[1, 0] = torch.tensor(r["K45"]); m.a.fill_(r["a"]); m.c.fill_(r["c"])
                return m.eval()
    raise SystemExit("✗ no encuentro ese brazo en resultados/trozos/")


def main() -> int:
    b = D.cargar(D.BANCO)
    t, g, rad, ang = b["tipo"], b["grosor"], b["radio"], b["angulo"]
    i_cur = int(np.flatnonzero((t == D.CURVA) & (rad == 9) & (g == 8) & (ang == 0))[0])
    i_rec = int(np.flatnonzero((t == D.CONTINUA) & (g == 8) & (b["largo"] == 22) & (ang == 0))[0])
    m = brazo()
    with torch.no_grad():
        Ks = m.kernels()[:, 0].numpy()
    a, c = float(m.a.detach()), float(m.c.detach())

    filas = []
    for i, nombre in ((i_cur, "CURVA · radio 9 · grosor 8"), (i_rec, "RECTA · 0° · grosor 8 · largo 22")):
        x = b["imagenes"][i].astype(np.float32)[None, None]
        xs = prepro.aplicar(x, "sobel")
        with torch.no_grad():
            mapas = F.conv2d(torch.from_numpy(xs), m.kernels(), padding=4)[0].numpy()     # (4, 32, 32)
            logit = m(torch.from_numpy(xs))[0].numpy()
        filas.append((nombre, x[0, 0], xs[0, 0], mapas, logit))

    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(5, 8, height_ratios=[0.25, 1, 1, 0.25, 1], hspace=0.55, wspace=0.35)
    # fila de los kernels (son los mismos para las dos imágenes)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0, 0.5, "Paso 3 — los 4 kernels 9×9 aprendidos (Sobel, N = 1000, semilla 0). K90 y K135 son rot90 EXACTOS de K0 y K45. "
            "Rojo +, azul −.", color=T1, fontsize=10, va="center")
    sub = gs[1, :].subgridspec(1, 8, wspace=0.3)
    for j in range(4):
        axk = fig.add_subplot(sub[0, 2 + j]); K = Ks[j]; mx = abs(K).max()
        axk.imshow(K, cmap="RdBu_r", norm=TwoSlopeNorm(0, -mx, mx)); axk.set_xticks([]); axk.set_yticks([])
        axk.set_title(f"kernel {ORI[j]}")
    for r, (nombre, x, xs, mapas, logit) in enumerate(filas):
        fila = gs[2 if r == 0 else 4, :].subgridspec(1, 9, wspace=0.35)
        if r == 1:
            axh = fig.add_subplot(gs[3, :]); axh.axis("off")
        a1 = fig.add_subplot(fila[0, 0]); a1.imshow(x, cmap="gray_r", vmin=0, vmax=1); a1.set_title("1 · imagen"); a1.set_ylabel(nombre.replace(" · ", "\n", 1), fontsize=9, color=T1)
        a2 = fig.add_subplot(fila[0, 1]); a2.imshow(xs, cmap="gray_r", vmin=0, vmax=1); a2.set_title("2 · tras Sobel")
        mx = max(mapas.max(), 1e-6)
        for j in range(4):
            aj = fig.add_subplot(fila[0, 2 + j]); M_ = mapas[j]
            # sólo la parte POSITIVA (la que puede ganar el max), en una escala común a los 4; el borde de Sobel en gris
            aj.imshow(np.clip(M_, 0, None), cmap="Oranges", vmin=0, vmax=mx)
            aj.contour(xs, levels=[0.3], colors=[T2], linewidths=0.6)
            aj.set_xlim(-0.5, 31.5); aj.set_ylim(31.5, -0.5)
            yx = np.unravel_index(M_.argmax(), M_.shape)
            aj.add_patch(mp.Rectangle((yx[1] - 4.5, yx[0] - 4.5), 9, 9, fill=False, ec="#2a78d6", lw=1.5))
            aj.plot(yx[1], yx[0], "o", color="#2a78d6", ms=3)
            aj.set_title(f"4 · respuesta {ORI[j]}\n5 · max = {M_.max():.2f}")
            aj.set_xticks([]); aj.set_yticks([])
        for axx in (a1, a2):
            axx.set_xticks([]); axx.set_yticks([])
        # superponer la ventana ganadora sobre la imagen de Sobel
        jg = int(logit.argmax()); yx = np.unravel_index(mapas[jg].argmax(), mapas[jg].shape)
        a2.add_patch(mp.Rectangle((yx[1] - 4.5, yx[0] - 4.5), 9, 9, fill=False, ec=NAR, lw=1.5))
        a2.set_xlim(-0.5, 31.5); a2.set_ylim(31.5, -0.5)
        a6 = fig.add_subplot(fila[0, 7:9])
        cols = [NAR if j == jg and logit[j] > 0 else "#2a78d6" for j in range(4)]
        a6.barh(range(4), logit, color=cols, height=0.6); a6.axvline(0, color=T2, lw=1)
        a6.set_yticks(range(4)); a6.set_yticklabels(ORI); a6.invert_yaxis(); a6.set_xlim(-7, 6)
        for j in range(4):
            a6.text(logit[j] + (0.15 if logit[j] >= 0 else -0.15), j, f"{logit[j]:+.1f}", va="center", ha="left" if logit[j] >= 0 else "right", fontsize=8, color=T1)
        veredicto = (f"DETECTA recta {ORI[jg]}" if logit.max() > 0 else "no detecta nada")
        a6.set_title(f"6 · logit = {a:.2f}·max {c:+.2f}\n→ {veredicto}", color=NAR if logit.max() > 0 else T1, loc="left")
        a6.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paso 4: sólo la parte POSITIVA de cada mapa (naranja; escala común por fila), con el borde de Sobel en gris. "
                 "Paso 5: el MAX (recuadro azul = su ventana 9×9).\nFuera de ±4 px de un borde el mapa vale exactamente 0. "
                 "El fallo real: el kernel de 0° responde MÁS a la esquina (1,67) y a la curva (2,21) que al borde recto (1,49).",
                 color=T1, fontsize=11)
    RES.mkdir(exist_ok=True); fig.savefig(RES / "ejemplo-curva.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("→", RES / "ejemplo-curva.png")
    for nombre, *_, logit in filas:
        print(f"  {nombre}: logits {np.round(logit, 2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
