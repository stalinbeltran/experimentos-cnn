#!/usr/bin/env python3
"""delg-cnn — pasa dígitos REALES de UCI (val) por la red de sólo kernels y enseña tinta y salida M, sin binarizar.

    python nn/aplicar_digitos.py [--n 30] [--semilla 1]   → resultados/2-digitos-<n>.png
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

DATASET = "uci-optdigits-orig-32px-r20261005"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=1)
    a = ap.parse_args()
    d = np.load(exigir_dataset(DATASET) / "datos.npz")
    idx_val = np.flatnonzero(d["particion"] == "val")
    rng = np.random.default_rng(a.semilla)
    idx = np.sort(rng.choice(idx_val, a.n, replace=False))
    X = (d["imagenes"][idx] > 0).astype(np.float32)
    pesos = AQUI / "nn" / "pesos" / "kernels"
    cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"]); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    with torch.no_grad():
        M = red(torch.from_numpy(X)[:, None])[:, 0].numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 10; nfil = -(-a.n // ncol)
    fig, axs = plt.subplots(2 * nfil, ncol, figsize=(1.3 * ncol, 2.7 * nfil))
    for j in range(nfil * ncol):
        f, c = divmod(j, ncol)
        for k in range(2):
            ax = axs[2 * f + k, c]; ax.set_xticks([]); ax.set_yticks([])
            if j >= a.n:
                ax.axis("off"); continue
            ax.imshow((X, M)[k][j], cmap=("Greys", "magma")[k], vmin=0, vmax=(1.6, 1)[k], interpolation="nearest")
            if k == 0:
                ax.set_title(f"{d['etiquetas'][idx[j]]}  (val {idx[j]})", fontsize=7)
    fig.suptitle(f"delg-cnn · {a.n} dígitos de val al azar (semilla {a.semilla}): tinta y salida M de la red de sólo kernels "
                 f"({cfg['capas']} conv × {cfg['canales']}, entrenada con {cfg['n']} trazos sintéticos)", fontsize=8)
    fig.tight_layout(); out = AQUI / "resultados" / f"2-digitos-{a.n}.png"; fig.savefig(out, dpi=130)
    print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
