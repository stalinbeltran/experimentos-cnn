#!/usr/bin/env python3
"""rizo-2 — las RECTAS de un dígito dibujadas como rectas (pedido por el dueño el 2026-10-12): cada celda 8×8 de la feature
«rectas» (4 orientaciones × 2 escalas, valor = máximo en la celda del Gabor integrado a lo largo de 15 px) se pinta como un
SEGMENTO en la posición del máximo dentro de la celda, con la orientación de su Gabor, de LONGITUD y TONO proporcionales
al valor (relativo al máximo del dígito). La dirección de cada cresta se lee del propio kernel (eje principal de su parte
positiva), no de la etiqueta.

    python nn/rectas_dibujo.py --idx 784              → resultados/22-rectas-784.png
    python nn/rectas_dibujo.py --idx 784 --celdas 4   → resultados/22-rectas-784-4x4.png  (la opción 3)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import evaluar as EV                                                # noqa: E402
import digitos as D                                                 # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

ANG = (0, 45, 90, 135)


def direccion(k):
    """Eje principal de la parte positiva del kernel: (dx, dy) unitario en coordenadas de imagen (y hacia abajo)."""
    k = np.asarray(k); yy, xx = np.mgrid[:k.shape[0], :k.shape[1]]; w = np.clip(k, 0, None); w /= w.sum()
    cx, cy = (w * xx).sum(), (w * yy).sum(); c = np.cov(np.stack([xx.ravel() - cx, yy.ravel() - cy]), aweights=w.ravel())
    v = np.linalg.eigh(c)[1][:, -1]; return v / np.linalg.norm(v)


@torch.no_grad()
def mapas(x):
    xt = torch.from_numpy(x.astype(np.float32))[None, None]; out = []
    for e in range(2):
        xe = xt if e == 0 else F.avg_pool2d(xt, 2 ** e)
        m = F.relu(F.conv2d(xe, D.GABOR4, padding=4)); m = F.conv2d(m, D.INTEG, padding=D.L_INT // 2, groups=4); out.append(m[0].numpy())
    return out


def celdas(m, n=8):
    """(4, H, W) → por orientación y celda: valor máximo y posición (x, y) del máximo, en coordenadas 32×32."""
    H = m.shape[1]; t = H // n; esc = 32 / H; out = []
    for o in range(4):
        for i in range(n):
            for j in range(n):
                bloque = m[o, i * t:(i + 1) * t, j * t:(j + 1) * t]; k = int(bloque.argmax()); v = float(bloque.flat[k])
                out.append((o, i, j, v, (j * t + k % t + 0.5) * esc - 0.5, (i * t + k // t + 0.5) * esc - 0.5))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, required=True); ap.add_argument("--largo", type=float, default=D.L_INT, help="longitud del segmento de valor máximo, px"); ap.add_argument("--celdas", type=int, default=8, help="rejilla de celdas (8 = referencia; 4 = opción 3)")
    args = ap.parse_args()
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); x = (d["imagenes"][args.idx] > 0).astype(np.uint8); real = int(d["etiquetas"][args.idx])
    dirs = [direccion(D.GABOR4[o, 0].numpy()) for o in range(4)]
    print("dirección de cada cresta (dx, dy; y hacia abajo):", {a: tuple(np.round(v, 2)) for a, v in zip(ANG, dirs)})
    M = mapas(x); C = [celdas(M[e], args.celdas) for e in range(2)]; vmax = max(v for e in range(2) for (_, _, _, v, _, _) in C[e])
    colores = ["#c2410c", "#1b7f3b", "#2a78d6", "#7c3aed"]
    fig, axs = plt.subplots(2, 5, figsize=(17, 9.4))
    for e in range(2):
        for col in range(5):
            ax = axs[e, col]; ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#d9d8d4"]), vmin=0, vmax=1)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
            for g in range(1, args.celdas):
                ax.axhline(g * 32 / args.celdas - 0.5, color="#eeeeee", lw=0.5, zorder=0); ax.axvline(g * 32 / args.celdas - 0.5, color="#eeeeee", lw=0.5, zorder=0)
            orients = range(4) if col == 4 else [col]; suma = 0.0
            for (o, i, j, v, px, py) in C[e]:
                if o not in orients or v <= 0:
                    continue
                suma += v; r = v / vmax; L = args.largo * r / 2; dx, dy = dirs[o]
                ax.plot([px - dx * L, px + dx * L], [py - dy * L, py + dy * L], color=colores[o], alpha=0.15 + 0.85 * r, lw=0.8 + 2.4 * r, solid_capstyle="round")
            t = f"{ANG[col]}°" if col < 4 else "las cuatro"
            ax.set_title(f"rectas {t} · escala ×{2 ** e} · celdas {args.celdas}×{args.celdas}\nΣ celdas {suma:.1f}", fontsize=9)
    fig.suptitle(f"rizo-2 · 22 · las RECTAS del idx {args.idx} (un {real}) dibujadas como rectas: cada celda {args.celdas}×{args.celdas} → un segmento en el máximo de la celda, "
                 f"con la orientación de su Gabor;\nlongitud y tono ∝ valor de la feature (máx {vmax:.2f} = {args.largo:.0f} px); la cresta de cada ángulo se lee del kernel", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=5); nombre = f"22-rectas-{args.idx}" + ("" if args.celdas == 8 else f"-{args.celdas}x{args.celdas}") + ".png"; fig.savefig(AQUI / "resultados" / nombre, dpi=100); print(f"→ resultados/{nombre}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
