#!/usr/bin/env python3
"""La figura del factor de generalización: lee resultados/factor.json de AQUÍ y el de `feat-ind` (por su id, en el
registro: no por su carpeta) y dibuja resultados/factor-generalizacion.png.

    python nn/figura_factor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn.registro import por_id              # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categóricos 1-4 de la paleta de referencia, validados (validate_palette.js, light: ALL PASS); con marcador distinto
# y etiqueta directa en cada línea, así que la identidad no depende del color
SERIES = [("detectores 8×8 (13)", "#2a78d6", "o", "-"), ("detectores 32×32 (13)", "#eb6834", "s", "-"),
          ("píxeles 8×8", "#1baf7a", "^", "--"), ("píxeles 32×32", "#eda100", "D", "--")]


def main() -> int:
    datos = json.loads((RES / "factor.json").read_text())
    otro = json.loads((por_id("feat-ind").carpeta / "resultados" / "factor.json").read_text())
    casos = {**otro["casos"], **datos["casos"]}
    pool = datos["no_vistos"]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SUP)
    for ax in (a1, a2):
        ax.set_facecolor(SUP); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.set_xscale("log"); ax.set_xlabel("dígitos de entrenamiento del compositor (N, escala log)")
    ns = [int(n) for n in next(iter(casos.values()))]
    a1.plot(ns, [pool / n for n in ns], color=T2, lw=1, ls=":", zorder=1)
    a1.annotate("techo: 100 % de acierto (4540 / N)", (ns[0], pool / ns[0]), xytext=(10, 2), textcoords="offset points",
                ha="left", color=T2, fontsize=10)
    for nombre, col, mk, ls in SERIES:
        f = casos[nombre]; x = [int(n) for n in f]
        a1.plot(x, [f[k]["F"] for k in f], color=col, marker=mk, ms=8, lw=2, ls=ls, mec=SUP, mew=1.5, label=nombre, zorder=3)
        a2.plot(x, [100 * f[k]["acc"] for k in f], color=col, marker=mk, ms=8, lw=2, ls=ls, mec=SUP, mew=1.5, label=nombre, zorder=3)
        a2.annotate(f"{nombre}  {100 * f[str(x[-1])]['acc']:.1f} %", (x[-1], 100 * f[str(x[-1])]["acc"]),
                    xytext=(10, {"detectores 8×8 (13)": 6, "detectores 32×32 (13)": 1, "píxeles 32×32": -8, "píxeles 8×8": -12}.get(nombre, 0)), textcoords="offset points",
                    va="center", color=T1, fontsize=9.5)
    a1.set_yscale("log"); a1.set_ylabel("F = dígitos nuevos acertados ÷ dígitos de train")
    a1.set_title("Factor de generalización F(N)", loc="left", fontsize=13, color=T1)
    f8, f32 = casos["detectores 8×8 (13)"], casos["píxeles 32×32"]
    a1.text(0.98, 0.97, "F a N = 36 / 180 / 1080\n" + "\n".join(
        f"{n:<22}{'  '.join(f'{casos[n][k]["F"]:>5.1f}' for k in ('36', '180', '1080'))}" for n, *_ in SERIES)
        + "\nF cae como 1/N: cada dígito\nnuevo de train aporta menos",
            transform=a1.transAxes, ha="right", va="top", fontsize=9, color=T2, family="monospace")
    a1.legend(frameon=False, loc="lower left", fontsize=9.5)
    a2.set_ylabel("acierto sobre los no vistos (%)"); a2.set_xlim(right=4200)
    a2.set_title("…y el acierto que hay detrás (aquí se ven las diferencias)", loc="left", fontsize=13, color=T1)
    fig.suptitle(f"Dígitos manuscritos: {pool} no vistos (717 de test + 3823 de otros 30 escritores) · compositor lineal posicional · 3 semillas",
                 x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destino = RES / "factor-generalizacion.png"
    fig.savefig(destino, dpi=140, facecolor=SUP)
    print(f"→ {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
