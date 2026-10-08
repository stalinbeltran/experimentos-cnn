#!/usr/bin/env python3
"""La gráfica de «entrenado SÓLO con s px» del boceto (pedida por el dueño el 2026-10-08 para entender el resultado). Lee
resultados-desplazamiento.json (prueba_desplazamiento.py) y no calcula nada.

    /tmp/vizenv/bin/python graficas_desplazamiento.py   → imagenes/6-por-separado-y-campo-de-vision.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
SUP, T1, T2, MUT, REJ, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def main():
    r = json.loads((AQUI / "resultados-desplazamiento.json").read_text(encoding="utf-8"))
    D = r["D"]; A = r["acierto"]
    series = [("sin desplazar", "8 vistas · base", MUT, None), ("sólo 4 px", "8 vistas · solo 4", "#86b6ef", 4),
              ("sólo 8 px", "8 vistas · solo 8", "#3987e5", 8), ("sólo 12 px", "8 vistas · solo 12", "#1c5cab", 12),
              ("sólo 16 px", "8 vistas · solo 16", "#0d366b", 16)]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=130, facecolor=SUP); ax.set_facecolor(SUP)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EJE); ax.tick_params(colors=T2, labelsize=9, length=0); ax.grid(axis="y", color=REJ, lw=0.8, zorder=0)
    for nombre, k, c, pico in series:
        ys = [A[k]["media"][str(d)] for d in D]
        ax.plot(D, ys, color=c, lw=2, marker="o", ms=5, zorder=3, markeredgecolor=SUP, markeredgewidth=1.5, label=f"entrenado {nombre}")
        if pico is not None:
            ax.annotate(f"{nombre}\n{A[k]['media'][str(pico)] * 100:.0f}", (pico, A[k]["media"][str(pico)]), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom", color=T1, fontsize=8.5)
    ax.axhline(0.1, color=EJE, lw=1, ls=":", zorder=1)
    ax.text(0, 0.035, "punteada: azar (10 de cada 100)", color=T2, fontsize=8.5)
    ax.axvspan(6, 16.6, color=REJ, alpha=0.35, lw=0, zorder=0)
    ax.set_xticks(D); ax.set_xlim(-0.5, 16.8); ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, labelcolor=T1)
    ax.set_xlabel("cuánto se movió el dígito al probar (px; media de las 8 direcciones)", color=T2, fontsize=9)
    ax.set_ylabel("dígitos acertados de cada 100", color=T2, fontsize=9)
    ax.set_title("Entrenado con UN solo desplazamiento: aprende esa posición", color=T1, fontsize=12, loc="left",
                 pad=30, fontweight="bold")
    ax.text(0, 1.03, "Boceto, 3823 dígitos. Zona gris: el dígito (~20 px) empieza a salirse.", transform=ax.transAxes,
            color=T2, fontsize=9, va="bottom")
    fig.subplots_adjust(top=0.85, bottom=0.24)
    fig.savefig(AQUI / "imagenes" / "6-por-separado-y-campo-de-vision.png", facecolor=SUP); plt.close(fig)
    print("→ imagenes/6-por-separado-y-campo-de-vision.png")


if __name__ == "__main__":
    main()
