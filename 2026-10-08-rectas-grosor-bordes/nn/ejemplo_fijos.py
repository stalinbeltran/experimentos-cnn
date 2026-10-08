#!/usr/bin/env python3
"""Las MISMAS dos imágenes de `ejemplo_curva.py` (curva R=9 · 8 px y recta 0° · 8 px · 22 px), pero con detectores FIJOS
sin entrenar, aplicados a la imagen ORIGINAL (pedido por el dueño, 2026-10-08):

    Gabor  los 4 kernels 9×9 de `modelo.kernel_gabor` (par, λ = 6, alargado a lo largo de la recta)
    Sobel  los 4 kernels 3×3 de borde ORIENTADO (0°, 45°, 90°, 135°), en valor absoluto: un borde vale igual
           si va de tinta a fondo que al revés

Mismos pasos: kernels → mapa (parte positiva) → max → max − umbral. El umbral de cada uno, calibrado como en los
experimentos: 5 % de FP sobre los 1000 negativos de entrenamiento.

    python nn/ejemplo_fijos.py   → resultados/ejemplo-fijos.png
"""
from __future__ import annotations

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

RES = AQUI.parent / "resultados"
SUP, T1, T2, NAR, AZU = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"
ORI = ("0° —", "45° \\", "90° |", "135° /")

# Sobel orientado: responde a un borde CON esa orientación (0° = borde horizontal = salto en vertical). y hacia abajo.
S0 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
S45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32)     # borde «\»: salto perpendicular a la diagonal
SOBEL = np.stack([S0, S45, np.rot90(S0).copy(), np.rot90(S45).copy()])
GABOR = np.stack([M.kernel_gabor(9, a) for a in (0, 45, 90, 135)])
DETECTORES = {"Gabor 9×9": (GABOR, False), "Sobel 3×3 (|·|)": (SOBEL, True)}


def mapas(x: np.ndarray, Ks: np.ndarray, absoluto: bool) -> np.ndarray:
    """(N,1,32,32) → (N,4,32,32)."""
    r = F.conv2d(torch.from_numpy(x), torch.from_numpy(Ks[:, None].copy()), padding=Ks.shape[-1] // 2).numpy()
    return np.abs(r) if absoluto else r


def umbral(Ks, absoluto) -> float:
    e = D.cargar(D.ENTRENO)
    neg = e["imagenes"][e["tipo"] >= D.NEG_RUIDO].astype(np.float32)[:, None]
    return float(np.quantile(mapas(neg, Ks, absoluto).max(axis=(1, 2, 3)), 0.95))


def main() -> int:
    b = D.cargar(D.BANCO)
    t, g, rad, ang = b["tipo"], b["grosor"], b["radio"], b["angulo"]
    i_cur = int(np.flatnonzero((t == D.CURVA) & (rad == 9) & (g == 8) & (ang == 0))[0])
    i_rec = int(np.flatnonzero((t == D.CONTINUA) & (g == 8) & (b["largo"] == 22) & (ang == 0))[0])
    imgs = [(b["imagenes"][i_cur], "CURVA · radio 9 · grosor 8"), (b["imagenes"][i_rec], "RECTA · 0° · grosor 8 · largo 22")]

    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(6, 1, height_ratios=[0.8, 1, 1, 0.8, 1, 1], hspace=0.7)
    fila_g = 0
    for nombre_det, (Ks, absoluto) in DETECTORES.items():
        u = umbral(Ks, absoluto)
        sub = gs[fila_g].subgridspec(1, 9, wspace=0.35)
        axt = fig.add_subplot(sub[0, 0:2]); axt.axis("off")
        axt.text(0, 0.5, f"{nombre_det}\nsin entrenar\numbral = {u:.2f}\n(FP 5 % en negativos)", fontsize=11, color=T1, va="center")
        for j in range(4):
            ak = fig.add_subplot(sub[0, 2 + j]); K = Ks[j]; mx = abs(K).max()
            ak.imshow(K, cmap="RdBu_r", norm=TwoSlopeNorm(0, -mx, mx)); ak.set_xticks([]); ak.set_yticks([])
            ak.set_title(f"kernel {ORI[j]}")
        for r, (img, nombre) in enumerate(imgs):
            x = img.astype(np.float32)[None, None]
            Mp = mapas(x, Ks, absoluto)[0]; mxs = Mp.max(axis=(1, 2)); s = mxs - u
            fila = gs[fila_g + 1 + r].subgridspec(1, 9, wspace=0.35)
            a1 = fig.add_subplot(fila[0, 0]); a1.imshow(img, cmap="gray_r", vmin=0, vmax=1); a1.set_xticks([]); a1.set_yticks([])
            a1.set_ylabel(nombre.replace(" · ", "\n", 1), fontsize=9, color=T1); a1.set_title("imagen")
            top = max(Mp.max(), 1e-6)
            for j in range(4):
                aj = fig.add_subplot(fila[0, 2 + j]); m_ = Mp[j]
                aj.imshow(np.clip(m_, 0, None), cmap="Oranges", vmin=0, vmax=top)
                aj.contour(img.astype(float), levels=[0.5], colors=[T2], linewidths=0.6)
                yx = np.unravel_index(m_.argmax(), m_.shape); h = Ks.shape[-1] / 2
                aj.add_patch(mp.Rectangle((yx[1] - h, yx[0] - h), 2 * h, 2 * h, fill=False, ec=AZU, lw=1.5))
                aj.plot(yx[1], yx[0], "o", color=AZU, ms=3)
                aj.set_xlim(-0.5, 31.5); aj.set_ylim(31.5, -0.5); aj.set_xticks([]); aj.set_yticks([])
                aj.set_title(f"respuesta {ORI[j]}\nmax = {mxs[j]:.2f}")
            a6 = fig.add_subplot(fila[0, 7:9]); jg = int(s.argmax())
            a6.barh(range(4), s, color=[NAR if (j == jg and s[j] > 0) else AZU for j in range(4)], height=0.6)
            a6.axvline(0, color=T2, lw=1); a6.set_yticks(range(4)); a6.set_yticklabels(ORI); a6.invert_yaxis()
            lim = max(abs(s).max() * 1.35, 0.5); a6.set_xlim(-lim, lim)
            for j in range(4):
                a6.text(s[j] + (0.03 * lim if s[j] >= 0 else -0.03 * lim), j, f"{s[j]:+.2f}", va="center",
                        ha="left" if s[j] >= 0 else "right", fontsize=8)
            ver = f"DETECTA recta {ORI[jg]}" if s.max() > 0 else "no detecta nada"
            a6.set_title(f"max − umbral\n→ {ver}", loc="left", color=NAR if s.max() > 0 else T1)
            a6.spines[["top", "right"]].set_visible(False)
            print(f"{nombre_det:16s} {nombre:34s} max {np.round(mxs, 2).tolist()}  − umbral {u:.2f} → {ver}")
        fila_g += 3
    fig.suptitle("Las mismas dos imágenes con detectores FIJOS (sin entrenar), sobre la imagen original.\n"
                 "Naranja: parte positiva de cada mapa (escala común por fila); gris: el borde de la tinta; recuadro azul: "
                 "la ventana del max.", color=T1, fontsize=11)
    RES.mkdir(exist_ok=True); fig.savefig(RES / "ejemplo-fijos.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("→", RES / "ejemplo-fijos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
