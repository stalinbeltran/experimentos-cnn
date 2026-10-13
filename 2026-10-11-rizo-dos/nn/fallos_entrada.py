#!/usr/bin/env python3
"""rizo-2 — pinta los FALLOS de val de una corrida de todo_objetos_entrada.py: para cada fallo, la tinta original y la
entrada alternativa con los objetos que vieron los detectores encima (como en todo_objetos: segmentos = rectas, círculos
= curvas del gris por sentido, × = golpe, + = agujero).

    python nn/fallos_entrada.py --nombre delgado-k3x3 --entrada /ruta/delgados-k3x3.npz [--umbral 0.5] [--engrosar 0]
        → resultados/40-entrada-<nombre>-fallos.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import evaluar as EV                                                # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import rizo_huecos as RH                                            # noqa: E402
import todo_objetos as TO                                           # noqa: E402
import todo_objetos_entrada as TE                                   # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", required=True); ap.add_argument("--entrada", default=None); ap.add_argument("--clave", default="M")
    ap.add_argument("--umbral", type=float, default=0.5); ap.add_argument("--engrosar", type=int, default=0)
    a = ap.parse_args(); GR.configurar(**R.CAL_GRIS)
    res = json.loads((AQUI / "resultados" / f"40-entrada-{a.nombre}.json").read_text()); RH.VAR = res.get("var", RH.VAR)
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); tinta = (d["imagenes"] > 0).astype(np.uint8)
    img = TE.engrosar((np.load(a.entrada)[a.clave].astype(np.float32) > a.umbral).astype(np.uint8), a.engrosar) if a.entrada else tinta
    todos = pickle.loads((AQUI / "resultados" / f"40-candidatos-{a.nombre}.pkl").read_bytes())["todos"]
    fallos = sorted(res["fallos"], key=lambda f: (f["real"], f["predicho"]))
    cols = 8; filas = int(np.ceil(len(fallos) / cols))
    fig, axs = plt.subplots(filas, 2 * cols, figsize=(2 * cols * 1.45, filas * 1.75 + 1.0)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for k, f in enumerate(fallos):
        i = f["idx"]; r, c = divmod(k, cols); a0, a1 = axs[r, 2 * c], axs[r, 2 * c + 1]
        RA.pintar_digito(a0, tinta[i]); a0.set_xlim(-0.5, 31.5); a0.set_ylim(31.5, -0.5)
        TO.dibujar_objetos(a1, img[i], todos[i]); a1.set_xlim(-0.5, 31.5); a1.set_ylim(31.5, -0.5)
        col = EV.NAR if f["semillas_que_fallan"] == 3 else "#b45309"
        a0.set_title(f"{f['real']}→{f['predicho']} · idx {i}", color=col, fontsize=7.5, pad=2)
        a1.set_title(f"{f['semillas_que_fallan']}/3 semillas", color=col, fontsize=7, pad=2)
    fig.suptitle(f"rizo-2 · 40 · los {len(fallos)} fallos de val (semilla 0) de todo objetos sobre «{a.nombre}» (val {res['val']:.4f}) · "
                 "por pareja: tinta original | entrada con sus objetos · «real→predicho» (naranja = falla en las 3 semillas)\n"
                 "segmentos = rectas (tono = fuerza) · círculos = curvas del gris por sentido (→ naranja · ← azul · ↓ verde · ↑ morado) · × = golpe · + = agujero", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); out = AQUI / "resultados" / f"40-entrada-{a.nombre}-fallos.png"; fig.savefig(out, dpi=100)
    print(f"→ {out.relative_to(AQUI)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
