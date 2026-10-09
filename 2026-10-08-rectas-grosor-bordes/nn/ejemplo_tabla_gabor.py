#!/usr/bin/env python3
"""Una imagen de cada fila de la tabla «de dónde sale el umbral del Gabor» (pedido por el dueño, 2026-10-09), con sus
4 mapas de respuesta y su max contra el umbral (1,99 = 5 % de FP sobre los negativos de entrenamiento).

Para los grupos (ruido, puntos sueltos, manchas, rectas finas) se toma el ejemplo cuyo max está más cerca de la MEDIANA
del grupo: el representativo, ni el peor ni el mejor. Rectángulo y curva son los de `ejemplo_curva.py`.

    python nn/ejemplo_tabla_gabor.py   → resultados/ejemplo-tabla-gabor.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp                                    # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos as D                                                  # noqa: E402
import ejemplo_fijos as EF                                         # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, NAR, AZU = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"


def representativo(imgs: np.ndarray) -> tuple[np.ndarray, float, float]:
    mx = EF.mapas(imgs.astype(np.float32)[:, None], EF.GABOR, False).max(axis=(1, 2, 3))
    i = int(np.abs(mx - np.median(mx)).argmin())
    return imgs[i], float(np.median(mx)), float(mx[i])


def main() -> int:
    e, b = D.cargar(D.ENTRENO), D.cargar(D.BANCO)
    u = EF.umbral(EF.GABOR, False)
    casos = []
    for t in (D.NEG_RUIDO, D.NEG_PUNTOS, D.NEG_MANCHA):
        img, med, _ = representativo(e["imagenes"][e["tipo"] == t])
        casos.append((f"{D.NOMBRE_TIPO[t]} (negativo)", img, med))
    for g in (2, 3, 4):
        img, med, _ = representativo(b["imagenes"][(b["tipo"] == D.CONTINUA) & (b["grosor"] == g) & (b["largo"] == 22)])
        casos.append((f"recta {g} px · largo 22", img, med))
    t, g = b["tipo"], b["grosor"]
    casos.append(("rectángulo: recta 0° · 8 px", b["imagenes"][int(np.flatnonzero((t == D.CONTINUA) & (g == 8) & (b["largo"] == 22) & (b["angulo"] == 0))[0])], None))
    casos.append(("curva · radio 9 · 8 px", b["imagenes"][int(np.flatnonzero((t == D.CURVA) & (b["radio"] == 9) & (g == 8) & (b["angulo"] == 0))[0])], None))

    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})
    n = len(casos)
    fig = plt.figure(figsize=(13, 2.0 * n + 1.6))
    gs = fig.add_gridspec(n + 1, 7, width_ratios=[1.6, 1, 1, 1, 1, 1, 2.4], height_ratios=[0.9] + [1] * n, hspace=0.45, wspace=0.25)
    for j in range(4):
        ak = fig.add_subplot(gs[0, 2 + j]); K = EF.GABOR[j]; mx = abs(K).max()
        ak.imshow(K, cmap="RdBu_r", vmin=-mx, vmax=mx); ak.set_xticks([]); ak.set_yticks([]); ak.set_title(f"kernel {EF.ORI[j]}")
    a0 = fig.add_subplot(gs[0, 0:2]); a0.axis("off")
    a0.text(0, 0.5, f"Gabor 9×9 sin entrenar\numbral = {u:.2f}\n(5 % de FP en negativos)", fontsize=10, va="center", color=T1)
    todos = [EF.mapas(img.astype(np.float32)[None, None], EF.GABOR, False)[0] for _, img, _ in casos]
    top = max(m.max() for m in todos)
    for r, ((nombre, img, med), Mp) in enumerate(zip(casos, todos), start=1):
        a1 = fig.add_subplot(gs[r, 1]); a1.imshow(img, cmap="gray_r", vmin=0, vmax=1); a1.set_xticks([]); a1.set_yticks([])
        a0 = fig.add_subplot(gs[r, 0]); a0.axis("off"); a0.text(1, 0.5, nombre, ha="right", va="center", fontsize=9, color=T1)
        mxs = Mp.max(axis=(1, 2))
        for j in range(4):
            aj = fig.add_subplot(gs[r, 2 + j]); m_ = Mp[j]
            aj.imshow(np.clip(m_, 0, None), cmap="Oranges", vmin=0, vmax=top)      # MISMA escala en toda la figura
            aj.contour(img.astype(float), levels=[0.5], colors=[T2], linewidths=0.5)
            yx = np.unravel_index(m_.argmax(), m_.shape)
            aj.add_patch(mp.Rectangle((yx[1] - 4.5, yx[0] - 4.5), 9, 9, fill=False, ec=AZU, lw=1.2))
            aj.set_xlim(-0.5, 31.5); aj.set_ylim(31.5, -0.5); aj.set_xticks([]); aj.set_yticks([])
            aj.set_title(f"{mxs[j]:.2f}", fontsize=8, color=NAR if mxs[j] >= u else T2)
        ab = fig.add_subplot(gs[r, 6]); M = mxs.max()
        ab.barh([0], [M], color=NAR if M >= u else AZU, height=0.5); ab.axvline(u, color=T1, lw=1.2, ls="--")
        ab.set_xlim(0, top * 1.1); ab.set_yticks([]); ab.spines[["top", "right", "left"]].set_visible(False)
        txt = f"max {M:.2f} " + ("≥" if M >= u else "<") + f" {u:.2f} → " + ("DETECTA" if M >= u else "no detecta")
        if med is not None:
            txt += f"\n(mediana del grupo: {med:.2f})"
        ab.set_title(txt, loc="left", fontsize=8, color=NAR if M >= u else T1)
    fig.suptitle("El Gabor fijo sobre un ejemplo de cada fila de la tabla. Naranja: parte positiva del mapa (MISMA escala en "
                 "toda la figura); recuadro azul: la ventana del max;\nnúmero sobre cada mapa: su max (naranja si pasa el "
                 "umbral). A la derecha: el mayor de los 4 contra el umbral (línea discontinua).", color=T1, fontsize=10)
    RES.mkdir(exist_ok=True); fig.savefig(RES / "ejemplo-tabla-gabor.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("→", RES / "ejemplo-tabla-gabor.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
