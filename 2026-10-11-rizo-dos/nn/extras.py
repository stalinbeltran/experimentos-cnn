#!/usr/bin/env python3
"""rizo-2 — los rizos ADICIONALES que ponen las plantillas (S 0,8) y los rayos (F 0,5) en los 48 doses: cada candidato
con su cruz; el que cae a ≤ 2,5 px de la etiqueta en verde, el resto (los de más) en naranja. Pedido por el dueño el
2026-10-11 antes de componer los detectores.

    python nn/extras.py   → resultados/8-extras-plantillas-rayos.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
import rayos as RY                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

UMBRAL = {"P": 0.8, "R": 0.5}


def main() -> int:
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8)
    E = EV.etiquetas(); det = {"P": lambda x: PL.rizos(x, UMBRAL["P"]), "R": lambda x: RY.rizos(x, UMBRAL["R"], 8.0, 99.0)}
    k = 2; por_fila = 6; filas = int(np.ceil(len(E) / por_fila))
    fig, axs = plt.subplots(filas, por_fila * k, figsize=(1.35 * por_fila * k, 1.45 * filas + 0.8))
    tot = {"P": [0, 0], "R": [0, 0]}
    for j, e in enumerate(E):
        x = IM[e["i"]]
        for c, n in enumerate(det):
            ax = axs[j // por_fila, (j % por_fila) * k + c]; rr = det[n](x)
            ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            buenos = [r for r in rr if EV._dist(r["centro"], e["centro"]) <= EV.TOL]; malos = [r for r in rr if r not in buenos]
            for r in malos:
                ax.plot(r["centro"][0] - 0.5, r["centro"][1] - 0.5, "+", color=EV.NAR, ms=7, mew=1.3)
            for r in buenos:
                ax.plot(r["centro"][0] - 0.5, r["centro"][1] - 0.5, "+", color=EV.VERDE, ms=9, mew=2)
            ax.plot(e["centro"][0] - 0.5, e["centro"][1] - 0.5, "o", mfc="none", mec=EV.VERDE, ms=9, mew=1)
            tot[n][0] += bool(buenos); tot[n][1] += len(malos)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{n} {e['i']} · {len(malos)} de más", fontsize=6.5, pad=2, color=EV.VERDE if buenos else EV.NAR)
    for ax in axs.flat[len(E) * k:]:
        ax.axis("off")
    fig.suptitle(f"rizo-2 · 8 · los rizos ADICIONALES de plantillas (P, S {UMBRAL['P']}) y rayos (R, F {UMBRAL['R']}) en los 48 doses: "
                 f"círculo verde = etiqueta · cruz verde = el candidato que la acierta · cruces naranja = los de más\n"
                 f"P: {tot['P'][0]}/48 aciertan, {tot['P'][1] / 48:.1f} de más por dígito · R: {tot['R'][0]}/48, {tot['R'][1] / 48:.1f} de más", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(AQUI / "resultados" / "8-extras-plantillas-rayos.png", dpi=90)
    print(f"P {tot['P']} · R {tot['R']} → resultados/8-extras-plantillas-rayos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
