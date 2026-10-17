#!/usr/bin/env python3
"""rizo-2 § 109 — VISTA PREVIA: AMPLIAR el ángulo máximo del vértice libre (el dueño, 2026-10-17: «sugiero mantener el nombre,
pero ampliar el ángulo, pues también tenemos un parámetro ángulo»). Nada cambiado en `vertice_libre.py` ni entrenado.

Filas = límites (a 6 px / a 3 px): el de hoy (< 90° / < 120°) y tres más anchos. Columnas = el 3 #409 de § 108 y un dígito de
val de cada clase que tiene CURVAS (0, 8, 6, 2): donde ampliar puede convertir una curva en un rosario de vértices. En cada
panel, los vértices hallados con su dirección y su ángulo; a la izquierda, cuántos vértices da por dígito cada clase en val.

    ../.venv/bin/python nn/vertice_ampliar_vista.py   → resultados/109-vertice-ampliar.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import vertice_libre as V                                                # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"
LIMITES = [(90, 120), (120, 150), (135, 160), (150, 160)]
COL = {"N": "#c2410c", "E": "#2a78d6", "S": "#1b7f3b", "O": "#7c3aed"}
FLECHA = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "O": (-1, 0)}


def main():
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    M = np.load(DELGADOS)["M"].astype(float); y = d["etiquetas"]; va = np.flatnonzero(d["particion"] == "val")
    ids = [409] + [int(va[y[va] == c][0]) for c in (0, 8, 6, 2)]
    cnt = np.bincount(y[va], minlength=10); res = {}
    fig, axs = plt.subplots(len(LIMITES), len(ids) + 1, figsize=(3.1 * (len(ids) + 1), 3.2 * len(LIMITES)),
                            gridspec_kw={"width_ratios": [1.25] + [1] * len(ids)})
    for f, (a6, a3) in enumerate(LIMITES):
        V.A_MAX, V.A_MAX_CORTO = float(a6), float(a3)
        n = np.zeros(10); tres = 0
        for i in va:
            vs = V.vertices_libres(M[i]); n[y[i]] += len(vs)
            if y[i] == 3:
                tres += any(V.celda(v) in (3, 4, 5) and v["dir"] == "O" for v in vs)
        media = n / cnt; res[f"<{a6}/<{a3}"] = {"vertices_por_digito": np.round(media, 2).tolist(), "tres_con_O_en_el_medio": int(tres)}
        a = axs[f, 0]; a.barh(range(10), media, color=["#7c3aed" if c == 3 else "#bbb" for c in range(10)])
        a.set_yticks(range(10)); a.set_yticklabels([str(c) for c in range(10)]); a.invert_yaxis(); a.set_xlim(0, 6)
        for c in range(10):
            a.text(media[c] + 0.1, c, f"{media[c]:.1f}", va="center", fontsize=8)
        a.set_title(f"< {a6}° a 6 px · < {a3}° a 3 px" + ("  (HOY)" if f == 0 else "") +
                    f"\nvértices por dígito en val\nel 3 con O en el medio: {tres}/165", fontsize=10, fontweight="bold" if f == 0 else None)
        for k, i in enumerate(ids):
            a = axs[f, k + 1]; a.imshow(M[i], cmap="Greys", vmin=0, vmax=1, alpha=0.5, extent=(0, 32, 32, 0))
            vs = V.vertices_libres(M[i])
            for v in vs:
                dx, dy = FLECHA[v["dir"]]
                a.annotate("", (v["cx"] + 2.2 * dx, v["cy"] + 2.2 * dy), (v["cx"], v["cy"]),
                           arrowprops=dict(arrowstyle="-|>", color=COL[v["dir"]], lw=1.8))
                a.plot(v["cx"], v["cy"], "o", ms=6, color=COL[v["dir"]], mec="k", mew=0.6)
                a.text(v["cx"] + 2.6 * dx + (0.6 if dx >= 0 else -0.6), v["cy"] + 2.6 * dy, f"{v['ang']:.0f}°", fontsize=8,
                       color=COL[v["dir"]], fontweight="bold", ha="left" if dx >= 0 else "right", va="center",
                       bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
            a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"#{i} (un {y[i]}) · {len(vs)} vértice(s)", fontsize=10)
    fig.suptitle("Ampliar el ángulo del vértice libre: qué gana el 3 (#409, su centro apunta al O con 144°) y qué pasa con las curvas\n"
                 "flecha = hacia dónde apunta: N naranja · E azul · S verde · O morado · número = su ángulo (la pieza «amplitud»)", fontsize=12)
    fig.tight_layout(); out = AQUI / "resultados" / "109-vertice-ampliar.png"; fig.savefig(out, dpi=100)
    out.with_suffix(".json").write_text(json.dumps({"ids": ids, "limites": res}, ensure_ascii=False, indent=1)); print(out)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
