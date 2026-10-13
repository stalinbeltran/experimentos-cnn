#!/usr/bin/env python3
"""delg-cnn — dibuja la ESTRUCTURA de la red de sólo kernels que hay en nn/pesos/kernels: el flujo de capas, los
kernels aprendidos de la primera capa y la activación de cada capa sobre un dígito real.

    python nn/estructura.py [--idx 1327] [--pesos k3x3]   → resultados/3-estructura-<pesos>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parents[0])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from expcnn import exigir_dataset  # noqa: E402
from entrenar_local import SoloKernels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, default=1327)
    ap.add_argument("--pesos", default="k3x3", help="carpeta nn/pesos/<pesos>"); a = ap.parse_args()
    pesos = AQUI / "nn" / "pesos" / a.pesos
    cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    convs = [m for m in red.red if isinstance(m, torch.nn.Conv2d)]
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    x = torch.from_numpy((d["imagenes"][a.idx] > 0).astype(np.float32))[None, None]
    acts, h = [], x
    with torch.no_grad():
        for m in red.red:
            h = m(h)
            if isinstance(m, torch.nn.Conv2d):
                acts.append(h[0].numpy())
        M = torch.sigmoid(h)[0, 0].numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    C = cfg["canales"]; L = len(convs); K = cfg.get("kernel", 3)
    fig = plt.figure(figsize=(15, 9.5))
    NC = max(L + 2, C + 1)
    gs = fig.add_gridspec(3, NC, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.25)

    # fila 1: el flujo de capas
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, L + 2); ax.set_ylim(0, 1); ax.axis("off")
    cajas = [("entrada\ntinta 32×32\n1 canal", "#dddddd")]
    rf = 1
    for i, m in enumerate(convs):
        rf += K - 1
        npar = sum(p.numel() for p in m.parameters())
        ultima = i == L - 1
        cajas.append((f"conv {i + 1}\n{K}×{K}, {m.in_channels}→{m.out_channels} can.\n{npar} par.\n"
                      f"{'sigmoide' if ultima else 'ReLU'}\ncampo {rf}×{rf}", "#f6c89f" if not ultima else "#f08a5d"))
    cajas.append(("salida M\n32×32 ∈ [0,1]\nlínea media", "#cfe3f7"))
    for i, (txt, col) in enumerate(cajas):
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.2), 0.84, 0.6, boxstyle="round,pad=0.02", fc=col, ec="#444"))
        ax.text(i + 0.5, 0.5, txt, ha="center", va="center", fontsize=8.5)
        if i < len(cajas) - 1:
            ax.annotate("", xy=(i + 1.08, 0.5), xytext=(i + 0.92, 0.5), arrowprops=dict(arrowstyle="->", lw=1.4))
    ntot = sum(p.numel() for p in red.parameters())
    ax.set_title(f"delg-cnn · SOLO KERNELS [{a.pesos}]: {L} convoluciones {K}×{K} con relleno (sin pooling, sin capa densa) · "
                 f"{ntot} parámetros · campo receptivo final {rf}×{rf} px · entrenada con {cfg['n']} trazos "
                 f"sintéticos de 3–6 px, {cfg['epocas']} épocas, {cfg['segundos']} s", fontsize=10)

    # fila 2: los kernels aprendidos de la primera capa (entrada de 1 canal: son 3×3 legibles)
    w1 = convs[0].weight.detach().numpy()[:, 0]; v = np.abs(w1).max()
    for c in range(C):
        axk = fig.add_subplot(gs[1, c]); axk.imshow(w1[c], cmap="RdBu_r", vmin=-v, vmax=v)
        for (i, j), val in np.ndenumerate(w1[c]):
            axk.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6.5)
        axk.set_xticks([]); axk.set_yticks([]); axk.set_title(f"kernel 1.{c + 1}", fontsize=8)
    axt = fig.add_subplot(gs[1, C:]); axt.axis("off")
    axt.text(0, 0.5, f"los {C} kernels {K}×{K} de la\nprimera capa (rojo +, azul −).\nLas capas 2–{L} tienen {C}×{C}\n"
             f"kernels cada una y la última\n{C}×1: no caben legibles.", fontsize=8.5, va="center")

    # fila 3: activación de cada capa (canal de mayor energía) sobre el dígito
    axd = fig.add_subplot(gs[2, 0]); axd.imshow(x[0, 0], cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    axd.set_title(f"entrada · val {a.idx} ({d['etiquetas'][a.idx]})", fontsize=8); axd.set_xticks([]); axd.set_yticks([])
    for i, A in enumerate(acts[:-1]):
        A = np.maximum(A, 0); c = int(np.argmax(A.reshape(C, -1).sum(1)))
        axa = fig.add_subplot(gs[2, i + 1]); axa.imshow(A[c], cmap="magma", interpolation="nearest")
        axa.set_title(f"tras conv {i + 1} · canal {c + 1}\n(de {C}, el más activo)", fontsize=7.5); axa.set_xticks([]); axa.set_yticks([])
    axm = fig.add_subplot(gs[2, L]); axm.imshow(M, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axm.set_title(f"salida M (conv {L} + sigmoide)", fontsize=8); axm.set_xticks([]); axm.set_yticks([])
    [fig.add_subplot(gs[2, k]).axis("off") for k in range(L + 1, NC)]
    out = AQUI / "resultados" / f"3-estructura-{a.pesos}.png"; fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
