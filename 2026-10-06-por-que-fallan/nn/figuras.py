#!/usr/bin/env python3
"""Las FIGURAS del estudio, leídas de lo que ya está en `resultados/` (no recalcula nada salvo el histograma de grosor):

  causa-grosor.png     el grosor del trazo: lo que vieron los detectores contra los dígitos (y los 1)
  curvas-rectas.png    ¿se enciende un arco en una recta? fino, grueso y en los 1, por banco
  soluciones.png       cada solución probada: compositor (180), la curva con 36 de train y κ al engrosar/adelgazar
  errores.png          dónde se concentra el error: por tercil de grosor, relleno, inclinación y descentrado

    python nn/figuras.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset                 # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
AZUL, NARANJA, VERDE, MORADO, GRIS, ROJO = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#9a9893", "#d63c3c"


def _plt():
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    return plt


def _ejes(ax):
    ax.set_facecolor(SUP)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.tick_params(axis="y", colors=T2, labelsize=8); ax.tick_params(axis="x", length=0, colors=T1, labelsize=8)
    ax.grid(axis="y", color=REJ, lw=1, zorder=0)


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ------------------------------------------------------------------------------------------------ 1. la causa
def causa_grosor() -> None:
    import medir_grosor as G                                         # noqa: PLC0415
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    s = np.load(exigir_dataset("feat-ind32-sinteticas-32px-r20261005") / "datos.npz")
    m = (s["principal"] != 13) & (s["secundaria"] < 0)
    gd, gs, y = G.grosor(d["imagenes"]), G.grosor(s["imagenes"][m]), d["etiquetas"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.1), dpi=100, facecolor=SUP); _ejes(ax)
    bins = np.arange(0.5, 16.01, 0.5)
    for v, col, lab in ((gs, AZUL, f"features sintéticas con las que se entrenó (mediana {np.median(gs):.1f} px)"),
                        (gd[y != 1], NARANJA, f"dígitos de NIST salvo el 1 (mediana {np.median(gd[y != 1]):.1f} px)"),
                        (gd[y == 1], ROJO, f"los 1 (mediana {np.median(gd[y == 1]):.1f} px)")):
        ax.hist(v, bins, density=True, color=col, alpha=0.55, label=lab, zorder=3)
    ax.set_xlabel("grosor del trazo (px, en el lienzo de 32×32): tinta ÷ esqueleto", color=T2, fontsize=9)
    ax.set_ylabel("densidad", color=T2, fontsize=9); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=T1)
    ax.set_title("La causa principal: el trazo de los dígitos es ~3 veces más grueso\nque el de las features con las que se entrenaron los detectores",
                 color=T1, fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(RES / "causa-grosor.png", facecolor=SUP); plt.close(fig)
    print("  causa-grosor.png")


# ------------------------------------------------------------------------------------------------ 2. curvas y rectas
NOMBRE_BANCO = {"lineas": "líneas (entrenadas finas)", "contorno": "contorno (feat-bor)", "signo": "borde con signo (feat-bor)",
                "lineas-grueso": "líneas entrenadas con 2–12 px (S2)"}
def curvas_rectas() -> None:
    bancos = [b for b in ("lineas", "contorno", "signo", "lineas-grueso") if (RES / "curvas-rectas" / f"{b}-nada.json").is_file()]
    if not bancos:
        return
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=100, facecolor=SUP, gridspec_kw={"width_ratios": [3, 1.3]})
    grupos = (("fino_val", "todo", "fina (2–4 px, val)"), ("grueso", "2–4 px", "prueba gruesa: 2–4 px"),
              ("grueso", "6–12 px", "prueba gruesa: 6–12 px"))
    cols = (AZUL, NARANJA, VERDE, MORADO)
    ancho = 0.8 / len(bancos)
    for k, (que, ax, tit) in enumerate((("arco_se_enciende_en_rectas", axs[0], "en una RECTA se enciende algún arco"),)):
        _ejes(ax)
        for j, b in enumerate(bancos):
            d = _json(RES / "curvas-rectas" / f"{b}-nada.json")
            v = [d[g][t][que] for g, t, _ in grupos]
            xs = np.arange(len(grupos)) + (j - (len(bancos) - 1) / 2) * ancho
            ax.bar(xs, v, ancho * 0.92, color=cols[j], label=NOMBRE_BANCO.get(b, b), zorder=3)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 0.01, f"{vv:.0%}", ha="center", va="bottom", fontsize=7, color=T2)
        ax.set_xticks(range(len(grupos))); ax.set_xticklabels([g[2] for g in grupos])
        ax.set_ylim(0, 1); ax.set_title(tit, color=T1, fontsize=9.5, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=T1)
    ax = axs[1]; _ejes(ax)
    v = [_json(RES / "curvas-rectas" / f"{b}-nada.json")["digitos"]["algun_arco_en_los_1"] for b in bancos]
    ax.bar(range(len(bancos)), v, 0.7, color=cols[:len(bancos)], zorder=3)
    for i, vv in enumerate(v):
        ax.text(i, vv + 0.01, f"{vv:.0%}", ha="center", va="bottom", fontsize=7, color=T2)
    corto = {"lineas": "líneas\nfinas", "contorno": "contorno", "signo": "borde\ncon signo", "lineas-grueso": "líneas\n2–12 px"}
    ax.set_xticks(range(len(bancos))); ax.set_xticklabels([corto.get(b, b) for b in bancos], fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("algún arco en los 1 (dígitos)", color=T1, fontsize=9.5, loc="left")
    fig.suptitle("¿Se distingue una curva de una recta? Fina, sí; gruesa, sólo si el detector vio trazos gruesos",
                 color=T1, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(RES / "curvas-rectas.png", facecolor=SUP); plt.close(fig)
    print("  curvas-rectas.png")


# ------------------------------------------------------------------------------------------------ 3. las soluciones
ORDEN = [  # (combo, etiqueta, iteración)
    ("lineas-nada", "referencia:\nlíneas, crudo", 0),
    ("lineas-norm3", "S3 esqueleto\n+ 3 px", 1),
    ("lineas-adapt", "S7 erosión\nsegún grosor", 6),
    ("contorno-nada", "bordes:\ncontorno", 2),
    ("signo-nada", "bordes:\ncon signo", 2),
    ("lineas-grueso-nada", "S2 entrenados\ncon 2–12 px", 3),
    ("lineas-desinc", "S4\ndesinclinar", 4),
    ("lineas-nada+norm3", "S3' crudo\n+ 3 px", 2),
    ("lineas+lineas-grueso-nada", "S2e finos\n+ gruesos", 3),
    ("lineas+cortas-nada+norm3", "S6a largas\n+ cortas,\ncrudo + 3 px", 5),
    ("lineas+lineas-grueso-nada+norm3", "finos\n+ gruesos,\ncrudo + 3 px", 5),
    ("lineas+lineas-grueso+cortas-nada+norm3", "S6b finos\n+ gruesos\n+ cortas,\ncrudo + 3 px", 5),
]


def soluciones(extra: list | None = None) -> None:
    filas = [(c, e, it) for c, e, it in ORDEN + (extra or []) if (RES / "combos" / f"{c}.json").is_file()]
    if not filas:
        return
    d = {c: _json(RES / "combos" / f"{c}.json") for c, _, _ in filas}
    filas = filas[:1] + sorted(filas[1:], key=lambda r: d[r[0]]["compositor_180"]["media"])     # la referencia, y de peor a mejor
    plt = _plt()
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.8), dpi=100, facecolor=SUP, sharex=True)
    x = np.arange(len(filas))
    ax = axs[0]; _ejes(ax)
    c180 = [d[c]["compositor_180"]["media"] for c, _, _ in filas]
    c36 = [d[c]["curva_717"]["36"] for c, _, _ in filas]
    ref = c180[0]
    ax.bar(x - 0.2, c180, 0.38, color=[AZUL if v >= ref else GRIS for v in c180], zorder=3, label="180 dígitos de train (1617 de val)")
    ax.bar(x + 0.2, c36, 0.38, color=NARANJA, alpha=0.8, zorder=3, label="sólo 36 de train (717 de test)")
    for xx, v in zip(x, c180):
        ax.text(xx - 0.2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=T1)
    for xx, v in zip(x, c36):
        ax.text(xx + 0.2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=T2)
    ax.axhline(ref, color=AZUL, lw=0.8, ls="--", zorder=2)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("acierto leyendo dígitos", color=T2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=T1)
    ax.set_title("Cada solución probada: ¿lee mejor los dígitos? (la referencia primero; las demás, de peor a mejor; gris = peor que ella)",
                 color=T1, fontsize=10, loc="left", pad=24)
    ax = axs[1]; _ejes(ax)
    ke = [d[c]["kappa"]["engrosar"]["kappa"] for c, _, _ in filas]
    ka = [d[c]["kappa"]["adelgazar"]["kappa"] for c, _, _ in filas]
    ax.bar(x - 0.2, ke, 0.38, color=NARANJA, zorder=3, label="al engrosar el dígito 2 px")
    ax.bar(x + 0.2, ka, 0.38, color=VERDE, zorder=3, label="al adelgazarlo 1 px")
    ax.set_ylim(0, 1); ax.set_ylabel("κ: el mismo dígito,\n¿cae en su grupo?", color=T2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=T1)
    ax.set_title("¿Y es robusto al grosor? (κ = 1: el grupo no cambia; 0: como al azar)", color=T1, fontsize=10, loc="left")
    ax.set_xticks(x); ax.set_xticklabels([e for _, e, _ in filas], fontsize=7.5, linespacing=1.1)
    fig.tight_layout(); fig.savefig(RES / "soluciones.png", facecolor=SUP); plt.close(fig)
    print("  soluciones.png")


# ------------------------------------------------------------------------------------------------ 4. dónde está el error
ETIQ = {"lineas-nada": "referencia (líneas, crudo)", "lineas+lineas-grueso+cortas-nada+norm3": "S6b (la mejor: todo, crudo + 3 px)"}


def errores(combos: tuple = ("lineas-nada", "lineas+lineas-grueso+cortas-nada+norm3")) -> None:
    g = _json(RES / "diagnostico.json")
    if not g:
        return
    combos = [c for c in combos if c in g["combos"] and all(f"error_por_tercil_de_{k}" in g["combos"][c] for k in ("grosor", "relleno"))]
    props = [("grosor", "grosor del trazo", g, "diag"), ("relleno", "relleno (zonas macizas)", g, "diag"),
             ("inclinacion", "inclinación", g, "diag"), ("descentrado", "descentrado", g, "diag")]
    plt = _plt()
    fig, axs = plt.subplots(1, len(props), figsize=(11, 3.8), dpi=100, facecolor=SUP, sharey=True)
    cols = (AZUL, VERDE, NARANJA, MORADO)
    for ax, (k, tit, src, _) in zip(axs, props):
        _ejes(ax)
        for j, c in enumerate(combos):
            v = src["combos"][c][f"error_por_tercil_de_{k}"]
            ax.plot(range(3), v, "o-", color=cols[j], lw=2, ms=5, label=ETIQ.get(c, c), zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(["bajo", "medio", "alto"])
        ax.set_title(f"por tercil de {tit}", color=T1, fontsize=9, loc="left")
    axs[0].set_ylabel("error leyendo dígitos (val)", color=T2, fontsize=9)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.91), ncol=2, labelcolor=T1)
    fig.suptitle("¿Dónde se concentran los fallos? (error leyendo los 1617 dígitos de val, por tercil de cada propiedad)",
                 color=T1, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(RES / "errores.png", facecolor=SUP); plt.close(fig)
    print("  errores.png")


def main(argv: list[str]) -> int:
    quiero = set(argv) or {"causa", "curvas", "soluciones", "errores"}
    if "causa" in quiero:
        causa_grosor()
    if "curvas" in quiero:
        curvas_rectas()
    if "soluciones" in quiero:
        soluciones()
    if "errores" in quiero:
        errores()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
