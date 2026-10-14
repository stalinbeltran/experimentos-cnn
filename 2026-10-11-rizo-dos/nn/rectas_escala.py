#!/usr/bin/env python3
"""rizo-2 — «sólo para ver» (el dueño, 2026-10-14): los mismos dígitos reducidos a 16×16 y 8×8, y las RECTAS de la
cresta sobre cada resolución. Se reduce el MAPA adelgazado por el máximo de cada bloque (2×2 / 4×4), para que un trazo
de 1–2 px no se desvanezca al promediar; los umbrales del detector se escalan con la resolución (longitudes ÷ f,
curvaturas × f, con f = 32 / lado). Las rectas-objeto (Gabor par) sólo existen a 32 px y no se aplican.

    python nn/rectas_escala.py --nombre suave-cresta-todo-conc --entrada /ruta/delgados-k3x3.npz   → resultados/51-rectas-escala-<nombre>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
import rectas_objetos as RO                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

BASE = {k: getattr(CC, k) for k in ("SIGMA", "K_MIN", "K_GOLPE", "L_MIN", "L_RECTA")}


def reducir(m, lado):
    f = m.shape[0] // lado
    return m.reshape(lado, f, lado, f).max(axis=(1, 3))


def escalar(lado):
    f = 32 / lado
    CC.SIGMA = max(BASE["SIGMA"] / f, 0.6); CC.K_MIN = BASE["K_MIN"] * f; CC.K_GOLPE = BASE["K_GOLPE"] * f
    CC.L_MIN = max(int(round(BASE["L_MIN"] / f)), 2); CC.L_RECTA = BASE["L_RECTA"] / f


def rectas_en(m):
    escalar(m.shape[0])
    try:
        return CC.objetos(m)["rectas"]
    finally:
        escalar(32)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--nombre", required=True); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M"); a = ap.parse_args()
    res = json.loads((AQUI / "resultados" / f"40-entrada-{a.nombre}.json").read_text()); fallos = sorted(res["fallos"], key=lambda f: (f["real"], f["predicho"]))
    Mz = np.load(a.entrada)[a.clave].astype(np.float32)
    lados = (32, 16, 8); cols = 7; filas = int(np.ceil(len(fallos) / cols))
    fig, axs = plt.subplots(filas, 3 * cols, figsize=(3 * cols * 1.25, filas * 1.5 + 1.0)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    cuentas = {l: [] for l in lados}
    for k, f in enumerate(fallos):
        i = f["idx"]; r, c = divmod(k, cols)
        for q, lado in enumerate(lados):
            ax = axs[r, 3 * c + q]; m = reducir(Mz[i], lado); rs = rectas_en(m); cuentas[lado].append(len(rs))
            ax.imshow(np.where(m > 0.05, m, np.nan), cmap="Greys", vmin=0, vmax=3.0, interpolation="nearest")
            for z in rs:
                ax.plot(z["pts"][:, 0] - 0.5, z["pts"][:, 1] - 0.5, "-", color=RO.COLORES[z["o"]], lw=1.4 + 2.2 * min(z["largo"] / lado, 1.0), solid_capstyle="round")
            ax.set_xlim(-0.5, lado - 0.5); ax.set_ylim(lado - 0.5, -0.5)
            ax.set_title((f"{f['real']}→{f['predicho']} idx {i} · " if q == 0 else "") + f"{lado}px: {len(rs)}", fontsize=6.5, color=EV.NAR if f["semillas_que_fallan"] == 3 else "#b45309", pad=1.5)
    med = " · ".join(f"{l} px: {np.mean(cuentas[l]):.2f} rectas/dígito" for l in lados)
    fig.suptitle(f"rizo-2 · 51 · las RECTAS de la cresta a 32, 16 y 8 px (mapa reducido por el máximo de cada bloque; umbrales escalados) en los {len(fallos)} fallos de «{a.nombre}»\n"
                 f"color = orientación: 0° naranja · 45° verde · 90° azul · 135° morado · grosor = longitud relativa · {med}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = AQUI / "resultados" / f"51-rectas-escala-{a.nombre}.png"; fig.savefig(out, dpi=100); print(f"→ {out.relative_to(AQUI)} · {med}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
