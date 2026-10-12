#!/usr/bin/env python3
"""rizo-2 — CÓMO funciona el detector GRIS, etapa a etapa sobre un dígito (pedido por el dueño el 2026-10-12, al elegir
las rectas-objeto ×1 y quedar el gris como grupo dominante). Es la cadena del boceto curvas-gabor
(`bordes_gabor.BordeGabor` → `curvas.campo` → `curvas.giro` → `digitos_bordes.mapas_bordes`), con la calibración
«gris re-calibrado» (τ 0,1 · σ_E 0,5 · coh 0,25 · λ 3):

  1 dígito · 2 BORDE: banco de Gabor IMPAR (seno; λ_b 10, 9×9, N orientaciones), |respuesta| máxima normalizada por un
  escalón ideal → borde GRIS en la frontera del trazo · 3 CAMPO: Gabor PAR (coseno; 12 orientaciones, λ 3) sobre el
  borde → energía E_max (dónde hay borde, umbral τ), orientación θ (media en ángulo doble) y coherencia (cuánto se cree θ)
  · 4 GIRO: en cada píxel del borde se lee θ a ±4 px a lo largo de la tangente; κ = Δθ / 8 px (°/px); sólo si los dos
  extremos siguen en el borde y son coherentes · 5 el VECTOR de curvatura k = κ·(−sin θ, cos θ), que apunta al centro
  de la curva · 6 los SEIS mapas: recto (|κ| < 2 °/px) · k→ · k← · k↓ · k↑ · golpe (coherencia < 0,25: esquina o cruce)
  · 7 máximo en celdas 8×8 → 6 × 64 = 384 números.

    python nn/gris_dibujo.py --idx 784   → resultados/28-gris-784.png
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
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

NOMBRES = ["recto", "curva k→", "curva k←", "curva k↓", "curva k↑", "golpe"]


def panel(ax, x, titulo):
    RA.pintar_digito(ax, x); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo, fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, default=784); a = ap.parse_args()
    GR.configurar(**R.CAL_GRIS); d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); x = (d["imagenes"][a.idx] > 0).astype(np.float32); real = int(d["etiquetas"][a.idx])
    e = R.GRIS(x); c = C.campo(e); kappa, mask, _ = C.giro(c); M = DB.mapas_bordes(x, R.GRIS); P = F.adaptive_max_pool2d(torch.from_numpy(M)[None], 8)[0].numpy()
    medible = mask & np.isfinite(kappa); kk = np.where(medible, np.clip(np.nan_to_num(kappa), -D.KAPPA_TOPE, D.KAPPA_TOPE), np.nan)
    fig, axs = plt.subplots(3, 6, figsize=(17, 10.4))
    for ax in axs.flat:
        ax.axis("off")
    # fila 0
    panel(axs[0, 0], x, f"1 · idx {a.idx} (un {real})")
    ax = axs[0, 1]; im = ax.imshow(e, cmap="gray_r", vmin=0, vmax=1); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"2 · BORDE: Gabor IMPAR λ_b {R.GRIS.lam:g}, {R.GRIS.k}×{R.GRIS.k}, {R.GRIS.n} orient.\n|resp.| máx / escalón ideal → gris [0, 1]", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ins = ax.inset_axes([0.72, 0.02, 0.26, 0.26]); ins.imshow(R.GRIS.banco[0, 0].numpy(), cmap="RdBu_r"); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("un kernel impar", fontsize=6)
    ax = axs[0, 2]; im = ax.imshow(c["Emax"], cmap="magma"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.contour(mask, levels=[0.5], colors="#2a78d6", linewidths=0.7); ax.set_title(f"3a · CAMPO: Gabor PAR λ {C.LAM:g} sobre el borde → E_max\nazul: E_max > τ = {C.TAU} → «aquí hay borde»", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ins = ax.inset_axes([0.72, 0.02, 0.26, 0.26]); ins.imshow(C.GABOR[0, 0].numpy(), cmap="RdBu_r"); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("un kernel par", fontsize=6)
    ax = axs[0, 3]; th = np.where(mask, c["theta"], np.nan); im = ax.imshow(th, cmap="hsv", vmin=0, vmax=180); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("3b · orientación θ del borde (0–180°)\nmedia en ángulo doble, donde E_max > τ", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ys, xs = np.nonzero(mask)
    for yy, xx in zip(ys[::3], xs[::3]):
        t = np.deg2rad(c["theta"][yy, xx]); ax.plot([xx - 1.2 * np.cos(t), xx + 1.2 * np.cos(t)], [yy - 1.2 * np.sin(t), yy + 1.2 * np.sin(t)], color="k", lw=0.6)
    ax = axs[0, 4]; im = ax.imshow(np.where(mask, c["coh"], np.nan), cmap="viridis", vmin=0, vmax=1); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.contour(mask & (c["coh"] < C.COH_MIN), levels=[0.5], colors="#c2410c", linewidths=0.8); ax.set_title(f"3c · coherencia (cuánto se cree θ)\nnaranja: < {C.COH_MIN} → GOLPE, sin giro", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax = axs[0, 5]; im = ax.imshow(kk, cmap="RdBu_r", vmin=-D.KAPPA_TOPE, vmax=D.KAPPA_TOPE); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"4 · GIRO κ = (θ a +4 px − θ a −4 px) / 8, en °/px\nmedible en {int(medible.sum())} de {int(mask.sum())} px del borde", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    yy0, xx0 = ys[0], xs[0]
    # fila 1: vector de curvatura + los 6 mapas (recto y golpe aparte)
    ax = axs[1, 0]; panel(ax, x, f"5 · VECTOR k = κ·(−sin θ, cos θ), apunta al CENTRO\n|κ| ≥ {C.KAPPA_MIN:g} °/px curva · menos, recto")
    th_r = np.deg2rad(c["theta"]); kx, ky = -np.nan_to_num(kk) * np.sin(th_r), np.nan_to_num(kk) * np.cos(th_r); curvo = medible & (np.abs(np.nan_to_num(kk)) >= C.KAPPA_MIN)
    cy_, cx_ = np.nonzero(curvo)
    ax.quiver(cx_[::2], cy_[::2], kx[cy_, cx_][::2], -ky[cy_, cx_][::2], color="#c2410c", scale=120, width=0.004, zorder=5)
    ry, rx = np.nonzero(medible & ~curvo); ax.plot(rx, ry, ".", color="#2a78d6", ms=2.5, zorder=4); ax.text(0.02, 0.02, "naranja: curva (flecha hacia el centro) · azul: recto", transform=ax.transAxes, fontsize=6.5)
    for q in range(5):
        k = [0, 1, 2, 3, 4][q]; ax = axs[1, 1 + q]; panel(ax, x, f"6 · mapa «{NOMBRES[k]}»" + ("\n|κ| < 2 °/px y medible" if k == 0 else f"\n= max(0, {'+' if k in (1, 3) else '−'}{'kx' if k < 3 else 'ky'})"))
        ax.imshow(np.where(M[k] > 0, M[k], np.nan), cmap="Oranges", vmin=0, vmax=max(1.0, float(M[k].max())), zorder=3)
    # fila 2: golpe + los 6 pooled
    ax = axs[2, 0]; panel(ax, x, "6 · mapa «golpe»\nborde con coherencia < 0,25"); ax.imshow(np.where(M[5] > 0, M[5], np.nan), cmap="Oranges", vmin=0, vmax=1, zorder=3)
    for q in range(5):
        k = q; ax = axs[2, 1 + q]; ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(P[k], cmap="Oranges", vmin=0, vmax=max(1.0, float(P[k].max())))
        ax.set_title(f"7 · «{NOMBRES[k]}» en celdas 8×8 (máx) → 64\nΣ {P[k].sum():.1f} · no nulas {int((P[k] > 0).sum())}", fontsize=8)
    fig.suptitle(f"rizo-2 · 28 · cómo funciona el detector GRIS, etapa a etapa (idx {a.idx}): borde Gabor impar → campo (E, θ, coherencia) → giro κ → vector de curvatura → 6 mapas → celdas 8×8 (384)\n"
                 f"(el mapa «golpe» en celdas y los 6×64 completos no caben: la fila 7 enseña los 5 primeros)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / f"28-gris-{a.idx}.png", dpi=100); print(f"→ resultados/28-gris-{a.idx}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
