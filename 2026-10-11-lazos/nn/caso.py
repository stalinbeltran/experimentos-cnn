#!/usr/bin/env python3
"""lazos — un caso, de cerca: lo que se dibujó, los píxeles curvos que votan y dónde caen sus votos.

    python nn/caso.py 50        → resultados/caso-50.png  (índice en el .npz del dataset)

El 50 es el 2 que el dueño señaló el 2026-10-11 (4.º de la primera fila de 4-rizos-doses.png): un lazo de radio 12
(el máximo permitido) con 5 votos de 85, cuya corona ancha junta la cabeza y la base del 2, que son curvas distintas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import detector as DT                                               # noqa: E402
import lazos as LZ                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def main() -> int:
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    d = dict(np.load(exigir_dataset(LZ.DATASET) / "datos.npz"))
    x = (d["imagenes"][i] > 0).astype(np.uint8)
    cal = json.loads((AQUI / "resultados" / "metricas.json").read_text())["elegida"]
    L = DT.lazos(x, cal["sigma_v"], cal["v_min"])
    acc = DT.votos(x); c = DT.campo(x); k = DT.giro(c)
    cur = np.isfinite(k) & (np.abs(np.nan_to_num(k)) >= DT.KAPPA_MIN)
    plt.rcParams.update({"font.size": 8, "figure.facecolor": LZ.SUP, "axes.facecolor": LZ.SUP})
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
    LZ.dibujar(axs[0], x, L, "lo que se dibujó (punteado: la corona 0,6–1,4 R)")
    axs[1].imshow(x, cmap="Greys", vmin=0, vmax=2.5)
    axs[1].imshow(np.where(cur, np.abs(k), np.nan), cmap="Oranges", vmin=0, vmax=12)
    axs[1].set_title(f"los {int(cur.sum())} píxeles curvos que votan (|κ|)")
    axs[2].imshow(x, cmap="Greys", vmin=0, vmax=2.5); axs[2].imshow(np.where(acc > 0, acc, np.nan), cmap="Blues")
    axs[2].set_title(f"dónde caen sus {int(acc.sum())} votos")
    t = np.linspace(0, 2 * np.pi, 100)
    for l in L:
        for a in axs[1:]:
            a.plot(l["centro"][0] - 0.5, l["centro"][1] - 0.5, "+", color=LZ.AZU, ms=12, mew=2)
        for f in (0.6, 1.4):
            axs[0].plot(l["centro"][0] - 0.5 + f * l["radio"] * np.cos(t), l["centro"][1] - 0.5 + f * l["radio"] * np.sin(t),
                        ":", color=LZ.T2, lw=0.8)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    txt = " · ".join(f"lazo: centro ({l['centro'][0]:.1f}, {l['centro'][1]:.1f}), R {l['radio']:.0f}, cobertura "
                     f"{l['cobertura']:.2f}, cierre {l['cierre']:.2f}, {l['votos']:.1f} votos" for l in L)
    fig.suptitle(f"lazos · caso {i} (dígito {int(d['etiquetas'][i])}) · {txt}", fontsize=9.5, color=LZ.T1)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / f"caso-{i}.png", dpi=100); plt.close(fig)
    print(f"→ resultados/caso-{i}.png · {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
