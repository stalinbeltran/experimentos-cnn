#!/usr/bin/env python3
"""Las GRÁFICAS de `feat-1lado`, pedidas por el dueño el 2026-10-08 para entender el resultado sin la tabla. Lee lo ya
medido (resultados/evaluacion.json y componer.json; `feat-bor` por su id, sólo como referencia) y no calcula nada nuevo.

    python nn/graficas.py   → resultados/g1-recall-por-grosor.png, g1b-falsas-alarmas.png, g2-digitos.png, g3-curva-gradual.png

Colores: la paleta categórica de referencia (azul, naranja, aqua) validada con el validador de la skill `dataviz`; el aqua
queda por debajo de 3:1 sobre el fondo, así que toda serie lleva su nombre escrito junto a la línea o la barra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
from expcnn import por_id                                         # noqa: E402

RES = EXP / "resultados"
SUP, T1, T2, MUT, REJ, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"lineas": "#2a78d6", "control": "#eb6834", "compartido": "#1baf7a"}
NOMBRE = {"lineas": "tinta (feat-ind32)", "control": "control: 8 bordes a la vez", "compartido": "compartido: un borde cada vez",
          "signo": "feat-bor signo (ayer)"}
GROSORES = (2, 3, 4, 6, 8, 10, 12)


def lienzo(w=7.2, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130, facecolor=SUP)
    ax.set_facecolor(SUP)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EJE)
    ax.tick_params(colors=T2, labelsize=9, length=0)
    ax.grid(axis="y", color=REJ, lw=0.8, zorder=0)
    return fig, ax


def titulo(ax, t, sub):
    ax.set_title(t, color=T1, fontsize=12, loc="left", pad=30, fontweight="bold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=T2, fontsize=9, va="bottom")


def recall_por_grosor(B: dict) -> list[float]:
    """Media de los detectores que tienen positivos de ese grosor (a 12 px sólo hay esquinas)."""
    out = []
    for g in GROSORES:
        v = [d["por_grosor"][str(g)]["recall"] for d in B["detectores"].values() if str(g) in d["por_grosor"]]
        out.append(sum(v) / len(v) if v else None)
    return out


def g1(ev: dict, signo: dict) -> None:
    fig, ax = lienzo()
    G = GROSORES[:-1]                                         # 12 px fuera: a ese grosor sólo hay esquinas
    ax.axvspan(5, 10.6, color=REJ, alpha=0.45, lw=0, zorder=0)
    ax.text(7.8, 0.94, "grosores que NO vio al entrenar", ha="center", color=T2, fontsize=9)
    ax.text(3, 0.94, "los que vio", ha="center", color=T2, fontsize=9)
    series = [(NOMBRE[r], recall_por_grosor(ev["B"][r])[:-1], COL[r], "-") for r in ("lineas", "control", "compartido")] + \
             [(NOMBRE["signo"], recall_por_grosor(signo)[:-1], MUT, "--")]
    for nombre, ys, c, ls in series:
        ax.plot(G, ys, color=c, lw=2, ls=ls, marker="o", ms=5, zorder=3, label=nombre,
                markeredgecolor=SUP, markeredgewidth=1.5, solid_capstyle="round")
    ax.set_xticks(G); ax.set_xlim(1.6, 10.4); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=T1)
    ax.set_xlabel("grosor del trazo (px, en una imagen de 32 × 32)", color=T2, fontsize=9)
    ax.set_ylabel("de cada 100 trazos de su tipo, cuántos encuentra", color=T2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    titulo(ax, "¿Sigue encontrando el trazo cuando es más grueso?",
           "Media de los 13 detectores, entrenados con 2–4 px. Sin 12 px: allí sólo hay esquinas.")
    fig.subplots_adjust(top=0.84, bottom=0.13)
    fig.savefig(RES / "g1-recall-por-grosor.png", facecolor=SUP); plt.close(fig)


def g1b(ev: dict, signo: dict) -> None:
    """Las FALSAS ALARMAS: la otra mitad de g1 (la tinta encuentra el trazo grueso, pero enciende de más)."""
    fig, ax = lienzo(7.2, 4.0)
    bancos = [("lineas", ev["B"]["lineas"]), ("control", ev["B"]["control"]), ("compartido", ev["B"]["compartido"]),
              ("signo", signo)]
    fino = [b["fp_tasa_vistos"] for _, b in bancos]; grueso = [b["fp_tasa_no_vistos"] for _, b in bancos]
    x = range(len(bancos)); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], fino, w, color="#f3b29a", zorder=3, label="trazos finos (2–4 px, los que vio)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], grueso, w, color="#eb6834", zorder=3, label="trazos gruesos (6–12 px, nuevos)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.002, f"{r.get_height() * 100:.0f}",
                    ha="center", va="bottom", color=T1, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([NOMBRE[b].replace(": ", ":\n") for b, _ in bancos], color=T1, fontsize=8.5)
    ax.set_ylim(0, 0.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_ylabel("de cada 100 trazos de OTRO tipo,\ncuántos confunde con el suyo", color=T2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=T1)
    titulo(ax, "Falsas alarmas: la tinta ve trazos que no están", "Media de los 13 detectores. Menos es mejor.")
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.13)
    fig.savefig(RES / "g1b-falsas-alarmas.png", facecolor=SUP); plt.close(fig)


def g2(comp: dict) -> None:
    fig, ax = lienzo(7.2, 4.0)
    bancos = ("lineas", "control", "compartido")
    sin = [comp["bancos"][b]["C1"]["normal"] for b in bancos]
    con = [comp["bancos"][b]["C2_180"]["normal"] for b in bancos]
    x = range(len(bancos)); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], sin, w, color="#86b6ef", zorder=3, label="sin desplazar")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], con, w, color="#2a78d6", zorder=3, label="entrenado con desplazamientos de 0–2 px")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height() * 100:.1f}",
                    ha="center", va="bottom", color=T1, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([NOMBRE[b].replace(": ", ":\n") for b in bancos], color=T1, fontsize=9)
    ax.set_ylim(0.7, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_ylabel("dígitos acertados de cada 100", color=T2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=T1)
    titulo(ax, "Leer dígitos: la tinta sigue ganando", "1617 dígitos de prueba; el compositor aprendió con 180. El eje empieza en 70.")
    fig.subplots_adjust(top=0.84, bottom=0.16)
    fig.savefig(RES / "g2-digitos.png", facecolor=SUP); plt.close(fig)


def g3(comp: dict) -> None:
    fig, ax = lienzo()
    D = comp["D"]; C3 = comp["bancos"]["lineas"]["C3"]
    series = [("sin desplazar", "base", MUT), ("0–2 px", "≤2", "#86b6ef"), ("0–4 px", "≤4", "#2a78d6"), ("0–8 px", "≤8", "#104281")]
    for nombre, k, c in series:
        ys = [C3[k]["media"][str(d)] for d in D]
        ax.plot(D, ys, color=c, lw=2, marker="o", ms=5, zorder=3, markeredgecolor=SUP, markeredgewidth=1.5,
                label=nombre if k == "base" else f"entrenado moviéndolo {nombre}")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=T1)
    ax.axhline(0.1, color=EJE, lw=1, ls=":", zorder=1)
    ax.text(0, 0.035, "punteada: azar (10 de cada 100)", color=T2, fontsize=8.5, ha="left")
    ax.set_xticks(D); ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, 16.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("cuánto se movió el dígito al probar (px; media de las 8 direcciones)", color=T2, fontsize=9)
    ax.set_ylabel("dígitos acertados de cada 100", color=T2, fontsize=9)
    titulo(ax, "Entrenar con el dígito movido: aguanta hasta donde se entrenó",
           "Compositor sobre los detectores de tinta; aprendió con 180 dígitos movidos de 0 a N px.")
    fig.subplots_adjust(top=0.84, bottom=0.13)
    fig.savefig(RES / "g3-curva-gradual.png", facecolor=SUP); plt.close(fig)


def main() -> int:
    ev = json.loads((RES / "evaluacion.json").read_text(encoding="utf-8"))
    comp = json.loads((RES / "componer.json").read_text(encoding="utf-8"))
    signo = json.loads((por_id("feat-bor").carpeta / "resultados" / "evaluacion.json").read_text(encoding="utf-8"))["B"]["signo"]
    g1(ev, signo); g1b(ev, signo); g2(comp); g3(comp)
    print("→ resultados/g1-recall-por-grosor.png, g1b-falsas-alarmas.png, g2-digitos.png, g3-curva-gradual.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
