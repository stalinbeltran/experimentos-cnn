#!/usr/bin/env python3
"""delg-cnn — VISTA de los trazos CASI CERRADOS (`sinteticos.casi_cerrado`; el dueño, 2026-10-19) y de lo que hace con ellos el
adelgazador vigente (`k3x3-g0.7x-64-rt`, rizo-2 § 125), que NUNCA ha visto uno. Arriba de cada pareja la tinta (gris σ 0,7, como al
entrenar) con su línea media de verdad; abajo el mapa M y la cresta (M > 0,4) del adelgazador. Sin entrenar nada.

    ../.venv/bin/python nn/casi_cerrado_vista.py   → resultados/4-casi-cerrado.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_local import SoloKernels, ampliar  # noqa: E402
from sinteticos import casi_cerrado, engrosar  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PESOS = "k3x3-g0.7x-64-rt"
REALES = [304, 357, 980, 1187, 1229]                                # 0 de val con la puntita (rizo-2 § 125)


def main():
    rng = np.random.default_rng(19); cs = [casi_cerrado(rng) for _ in range(16)]
    X = [ndimage.gaussian_filter(engrosar(c["puntos"], c["grosor"]).astype(np.float32), 0.7, mode="constant") for c in cs]
    cfg = json.loads((AQUI / "nn" / "pesos" / PESOS / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(AQUI / "nn" / "pesos" / PESOS / "best.pt")); red.eval()
    with torch.no_grad():
        M = red(torch.from_numpy(np.stack([ampliar(x, 2) for x in X]).astype(np.float32))[:, None])[:, 0].numpy()
    nc = 8; fig, axs = plt.subplots(5, nc, figsize=(2.0 * nc, 11.2))
    sys.path.insert(0, str(AQUI.parents[0])); from expcnn import exigir_dataset  # noqa: E402
    real = np.load(exigir_dataset("uci-optdigits-orig-32px-trazo4-r20261018") / "datos.npz")["imagenes"].astype(np.float32)
    for j in range(nc):
        a = axs[0, j]; a.axis("off")
        if j < len(REALES):
            a.axis("on"); a.imshow(real[REALES[j]], cmap="Greys", vmin=0, vmax=1.6); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"REAL #{REALES[j]} (0, val)", fontsize=8, fontweight="bold")
    for k, (c, x, m) in enumerate(zip(cs, X, M)):
        f, j = divmod(k, nc); a1, a2 = axs[2 * f + 1, j], axs[2 * f + 2, j]; p = c["puntos"]
        a1.imshow(x, cmap="Greys", vmin=0, vmax=1.6, extent=(0, 32, 32, 0))
        a1.plot(p[:, 0], p[:, 1], "-", color="#e8432b", lw=1.3)
        a1.plot(*p[0], "o", color="#2a78d6", ms=5); a1.plot(*p[-1], "s", color="#f59e0b", ms=5)
        a1.set_title(c["tipo"], fontsize=8)
        a2.imshow(x, cmap="Greys", vmin=0, vmax=2.5, extent=(0, 32, 32, 0))
        a2.imshow(np.ma.masked_less(m, 0.15), cmap="magma_r", vmin=0, vmax=1, alpha=0.6, extent=(0, 32, 32, 0))
        yy, xx = np.nonzero(skeletonize(m > 0.4)); a2.plot((xx + 0.5) / 2, (yy + 0.5) / 2, ".", color="k", ms=1.6)
        for a in (a1, a2):
            a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a1.set_ylabel("tinta + verdad", fontsize=8, fontweight="bold"); a2.set_ylabel(f"{PESOS}\nM + cresta", fontsize=8, fontweight="bold")
    fig.suptitle("Trazos CASI CERRADOS (sintéticos): se empieza arriba (punto AZUL), se da casi una vuelta o algo más, y se termina DESVIADO "
                 "(cuadrado NARANJA) · fila de arriba: 0 REALES con esa tinta\narriba: tinta y línea media de verdad (rojo) · abajo: lo que hace hoy el adelgazador, que nunca vio uno "
                 "(morado = M, negro = cresta) · título: grados de más/menos sobre la vuelta y desvío final", fontsize=10)
    fig.tight_layout(); out = AQUI / "resultados" / "4-casi-cerrado.png"; fig.savefig(out, dpi=90); print(out)


if __name__ == "__main__":
    main()
