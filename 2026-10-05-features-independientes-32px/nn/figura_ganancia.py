#!/usr/bin/env python3
"""La figura de la ganancia G = (% aciertos) / (% train): lee resultados/ganancia.json de AQUÍ y el de `feat-ind` (por su id,
en el registro) y dibuja resultados/ganancia-generalizacion.png. Se niega si los dos no evaluaron los MISMOS dígitos.

  1. G según el % de train, las 4 representaciones, dataset de 4000.
  2. ¿Es independiente del tamaño del dataset? G de los detectores 8×8 con datasets de 500, 1000, 2000 y 4000.
  3. El control: el acierto según N (muestras, no %), con los mismos cuatro datasets.

    python nn/figura_ganancia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn.registro import por_id              # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# las mismas 4 representaciones con los mismos colores que factor-generalizacion.png (validados, light: ALL PASS)
SERIES = [("detectores 8×8 (13)", "#2a78d6", "o", "-"), ("detectores 32×32 (13)", "#eb6834", "s", "-"),
          ("píxeles 8×8", "#1baf7a", "^", "--"), ("píxeles 32×32", "#eda100", "D", "--")]
# tamaño del dataset = ordinal → una rampa de un solo tono (azul, el de los detectores 8×8), más oscuro = más grande
# (validate_palette.js --ordinal, light: ALL PASS)
RAMPA = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def medias(filas: list, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(pct_train, G, N, acc) medios por p, para un T."""
    ps = sorted({f["p"] for f in filas})
    sel = [[f for f in filas if f["T"] == T and f["p"] == p] for p in ps]
    return (np.array([np.mean([f["pct_train"] for f in s]) for s in sel]), np.array([np.mean([f["G"] for f in s]) for s in sel]),
            np.array([np.mean([f["N"] for f in s]) for s in sel]), np.array([np.mean([f["acc"] for f in s]) for s in sel]))


def main() -> int:
    aqui = json.loads((RES / "ganancia.json").read_text())
    alli = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    for k in ("huella_y", "huella_particiones", "T", "p", "semillas"):
        if aqui[k] != alli[k]:
            raise SystemExit(f"✗ '{k}' no coincide entre feat-ind y feat-ind32: no evaluaron los mismos dígitos. Me niego.")
    casos = {**alli["casos"], **aqui["casos"]}
    Ts = aqui["T"]; Tmax = max(Ts)
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.8), facecolor=SUP)
    for ax in (a1, a2, a3):
        ax.set_facecolor(SUP); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.set_xscale("log")
    pt = np.geomspace(0.02, 0.5, 50)
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.plot(pt * 100, (1 - pt) / pt, color=T2, lw=1, ls=":", zorder=1)
        ax.set_xlabel("% de train (N / T)")
        ax.set_xticks([2, 4, 10, 20, 50]); ax.set_xticklabels(["2", "4", "10", "20", "50"])
    a1.annotate("techo: 100 % de acierto", (3, (1 - 0.03) / 0.03), xytext=(8, 4), textcoords="offset points", color=T2, fontsize=9.5)
    a1.set_ylabel("G = % aciertos ÷ % train  (= aciertos ÷ N)")

    # 1. las cuatro representaciones, dataset de 4000
    for nombre, col, mk, ls in SERIES:
        p, g, _, _ = medias(casos[nombre], Tmax)
        a1.plot(p * 100, g, color=col, marker=mk, ms=8, lw=2, ls=ls, mec=SUP, mew=1.5, label=nombre, zorder=3)
    a1.legend(frameon=False, loc="lower left", fontsize=9.5)
    a1.set_title(f"1 · G con un dataset de {Tmax}", loc="left", fontsize=12.5, color=T1)
    g2 = {n: medias(casos[n], Tmax)[1][0] for n, *_ in SERIES}
    a1.text(0.98, 0.97, "con 2 % de train:\n" + "\n".join(f"{n:<22}G {g2[n]:>4.1f}" for n, *_ in SERIES),
            transform=a1.transAxes, ha="right", va="top", fontsize=9, color=T2, family="monospace")

    # 2 y 3. detectores 8×8 con cuatro tamaños de dataset
    ref = "detectores 8×8 (13)"
    for T, col in zip(Ts, RAMPA):
        p, g, n, acc = medias(casos[ref], T)
        a2.plot(p * 100, g, color=col, marker="o", ms=7, lw=2, mec=SUP, mew=1.2, zorder=3, label=f"T = {T}")
        a3.plot(n, acc * 100, color=col, marker="o", ms=7, lw=2, mec=SUP, mew=1.2, zorder=3, label=f"T = {T}")
    gmin, gmax = medias(casos[ref], min(Ts))[1][0], medias(casos[ref], Tmax)[1][0]
    a2.set_title("2 · ¿Independiente del tamaño del dataset?", loc="left", fontsize=12.5, color=T1)
    a2.text(0.98, 0.97, f"mismo 2 % de train:\nT = {Tmax} → G {gmax:.1f}\nT = {min(Ts)}  → G {gmin:.1f}  ({gmax / gmin:.2f}×)\n"
            "NO: el % de train esconde N,\ny el acierto depende de N", transform=a2.transAxes, ha="right", va="top",
            fontsize=9, color=T2, family="monospace")
    a2.legend(frameon=False, loc="lower left", fontsize=9.5, title="detectores 8×8 (13)", title_fontsize=9.5)
    a3.set_xlabel("muestras de train N (escala log)"); a3.set_ylabel("acierto sobre los no vistos (%)")
    a3.set_title("3 · Control: contra N, los cuatro tamaños casi coinciden", loc="left", fontsize=12.5, color=T1)
    a3.legend(frameon=False, loc="lower right", fontsize=9.5, title="detectores 8×8 (13)", title_fontsize=9.5)
    fig.suptitle("Ganancia G = (% aciertos sobre los no vistos) ÷ (% de train) · datasets balanceados de T dígitos del pool de "
                 "5620 (43 escritores) · compositor lineal posicional · media de 3 semillas", x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destino = RES / "ganancia-generalizacion.png"
    fig.savefig(destino, dpi=130, facecolor=SUP)
    # los pares con el MISMO N desde datasets distintos: es la comprobación numérica del panel 3
    filas = casos[ref]
    pares = {}
    for T in Ts:
        _, _, n, acc = medias(filas, T)
        for ni, ai in zip(n, acc):
            pares.setdefault(int(ni), []).append((T, round(float(ai) * 100, 1)))
    print(f"→ {destino}")
    print("mismo N desde datasets distintos (detectores 8×8):", {k: v for k, v in sorted(pares.items()) if len(v) > 1})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
