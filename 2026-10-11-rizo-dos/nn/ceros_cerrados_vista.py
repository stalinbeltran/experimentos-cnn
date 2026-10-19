#!/usr/bin/env python3
"""rizo-2 § 125 — los 0 de val que el adelgazador de 64 px (§ 122) dejaba ABIERTOS (sin lazo) y el afinado con re-trazo (§ 125) CIERRA
(el dueño, 2026-10-19: «muéstrame los 0 cerrados que antes estaban abiertos»). Fondo: el mapa M de cada adelgazador sobre la tinta
gris; negro = la cresta (tramos); verde = el lazo que encuentra `objetos()`.

    ../.venv/bin/python nn/ceros_cerrados_vista.py   → resultados/125-ceros-cerrados.{png,json}
"""
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"; T4 = "uci-optdigits-orig-32px-trazo4-r20261018"
VAR = [("64 px (§ 122)", f"delgados-k3x3-g0.7x-64-{T4}-x2.npz"), ("+ re-trazo (§ 125)", f"delgados-k3x3-g0.7x-64-rt-{T4}-x2.npz")]


def main():
    d = np.load(exigir_dataset(T4) / "datos.npz"); y, p, img = d["etiquetas"], d["particion"], d["imagenes"].astype(np.float32)
    Ms = [np.load(DELG / f)["M"] for _, f in VAR]
    ceros = np.flatnonzero((p == "val") & (y == 0))
    lz = {int(i): [CC.objetos(M[i].astype(np.float32))["lazos"] for M in Ms] for i in ceros}
    cerrados = [i for i in ceros if not lz[i][0] and lz[i][1]]
    abiertos = [i for i in ceros if lz[i][0] and not lz[i][1]]
    siguen = [i for i in ceros if not lz[i][0] and not lz[i][1]]
    print(f"0 de val: {len(ceros)} · abiertos antes {len(cerrados) + len(siguen)} → cierra {len(cerrados)}, siguen abiertos {len(siguen)}, "
          f"se ABREN {len(abiertos)}")
    nc = 8; filas = int(np.ceil(len(cerrados) / nc)); fig, axs = plt.subplots(2 * filas, nc, figsize=(1.9 * nc, 2.1 * 2 * filas + 0.8))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for k, i in enumerate(cerrados):
        f, c = divmod(k, nc)
        for v, M in enumerate(Ms):
            ax = axs[2 * f + v, c]; m = M[i].astype(np.float32)
            ax.imshow(img[i], cmap="Greys", vmin=0, vmax=2.5, extent=(0, 32, 32, 0))
            ax.imshow(np.ma.masked_less(m, 0.15), cmap="magma_r", vmin=0, vmax=1, alpha=0.55, extent=(0, 32, 32, 0))
            for t in CC.tramos(m)[0]:
                q = np.asarray(t["pts"]) / 2; ax.plot(q[:, 0], q[:, 1], "-", lw=1.4, color="k")
            for L in lz[int(i)][v]:
                q = np.asarray(L["pts"]) / 2; q = np.vstack([q, q[:1]]); ax.plot(q[:, 0], q[:, 1], "-", lw=2.2, color="#1b7f3b")
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i} · {'ABIERTO' if v == 0 else 'CERRADO'}", fontsize=8, color="#b91c1c" if v == 0 else "#1b7f3b")
            if c == 0:
                ax.set_ylabel(VAR[v][0], fontsize=8, fontweight="bold")
    fig.suptitle(f"Los {len(cerrados)} ceros de val que el adelgazador de 64 px dejaba ABIERTOS y el afinado con re-trazo CIERRA "
                 f"(de {len(ceros)} ceros; siguen abiertos {len(siguen)}, se abren {len(abiertos)})\n"
                 "fondo: tinta gris + mapa M · negro = cresta · VERDE = lazo encontrado", fontsize=10)
    fig.tight_layout(); out = AQUI / "resultados" / "125-ceros-cerrados.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({"ceros_val": len(ceros), "cerrados": [int(i) for i in cerrados],
                                                    "siguen_abiertos": [int(i) for i in siguen], "se_abren": [int(i) for i in abiertos]}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
