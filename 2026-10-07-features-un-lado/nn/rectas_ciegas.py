#!/usr/bin/env python3
"""¿Qué RECTAS no ve ningún detector de recta? Pedido por el dueño el 2026-10-08, después de ver un «2» con lados rectos que
nadie detectaba: «busca más casos de estos, porque si los detectores no hacen bien su trabajo, su resultado no es confiable».
No entrena nada (0 $, minutos en el dev). Dos bancos: `lineas` (los de tinta de feat-ind32, la referencia) y `control`.

  1. MAPA DE ZONAS CIEGAS (sintético): una recta de 18 px dibujada en el centro, en cada ángulo (0–177°, cada 3°) y
     grosor (2–12 px). ¿La ve alguno de los 4 detectores de recta (pasa su umbral)?
  2. RECTAS REALES en los 1617 dígitos de val: el esqueleto de la tinta, sus tramos rectos de ≥ 12 px (transformada de
     Hough probabilística, skimage), su ángulo y el grosor del trazo ahí (transformada de distancia). Un tramo cuenta como
     VISTO si algún detector de recta pasa su umbral en la celda 8×8 de su punto medio o en una vecina.

    python nn/rectas_ciegas.py   → resultados/rectas-zonas-ciegas.png, rectas-reales.png, rectas-galeria.png, rectas.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from scipy.ndimage import distance_transform_edt                  # noqa: E402
from skimage.morphology import skeletonize                        # noqa: E402
from skimage.transform import probabilistic_hough_line            # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos                                                      # noqa: E402
import evaluar as E                                               # noqa: E402
import features as F                                              # noqa: E402

RES = AQUI.parent / "resultados"
RECTAS = ("recta-H", "recta-B", "recta-V", "recta-S")             # 0°, 45° (\), 90°, 135° (/) — y crece hacia ABAJO
CENTRO = {"recta-H": 0, "recta-B": 45, "recta-V": 90, "recta-S": 135}
SIMBOLO = {"recta-H": "—", "recta-B": "\\", "recta-V": "|", "recta-S": "/"}
MARGEN = F.RECTA_JITTER                                           # ±12°: lo que el vocabulario llama «recta X»
ANGULOS = np.arange(0, 180, 3)
GROSORES = (2, 3, 4, 6, 8, 10, 12)
BANCOS = {"lineas": "tinta (feat-ind32)", "control": "control: 8 bordes a la vez"}
SUP, T1, T2, MUT, REJ, EJE, NAR = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#eb6834"


def recta(cx, cy, largo, ang, grosor):
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    t = np.deg2rad(ang); u = np.array([np.cos(t), np.sin(t)])
    px, py = xx - cx, yy - cy
    return ((np.abs(px * u[0] + py * u[1]) <= largo / 2) & (np.abs(-px * u[1] + py * u[0]) <= grosor / 2)).astype(np.uint8)


def dist_ang(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def en_vocabulario(ang):
    """La recta del vocabulario más cercana, y si el ángulo está dentro de su ±12°."""
    f = min(RECTAS, key=lambda r: dist_ang(ang, CENTRO[r]))
    return f, dist_ang(ang, CENTRO[f]) <= MARGEN


@torch.no_grad()
def mapas_rectas(reds, idx, x):
    """(N, 4, 8, 8) σ de los 4 detectores de recta."""
    xt = torch.from_numpy(x[:, None].astype(np.float32))
    return np.stack([torch.sigmoid(reds[idx[r]](xt))[:, 0].numpy() for r in RECTAS], 1)


def tramos(d):
    """Tramos rectos del esqueleto: [(x0, y0, x1, y1, ángulo, largo, grosor)], los más largos primero, sin repetir."""
    esq = skeletonize(d > 0)
    segs = probabilistic_hough_line(esq, threshold=6, line_length=12, line_gap=1, rng=0)
    dt = distance_transform_edt(d > 0)
    out = []
    for (x0, y0), (x1, y1) in sorted(segs, key=lambda s: -np.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1])):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if any(np.hypot(mx - (a[0] + a[2]) / 2, my - (a[1] + a[3]) / 2) < 5 for a in out):
            continue
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int); ys = np.linspace(y0, y1, n).round().astype(int)
        grosor = float(2 * dt[ys, xs].mean() - 1)
        ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180)
        out.append((x0, y0, x1, y1, ang, float(np.hypot(x1 - x0, y1 - y0)), grosor))
        if len(out) == 3:
            break
    return out


def main() -> int:
    torch.set_num_threads(2)
    idx = {f: i for i, f in enumerate(F.CON_TRAZO)}
    bancos = {b: E.banco(b) for b in BANCOS}
    um = {b: np.array([bancos[b][1][idx[r]] for r in RECTAS]) for b in BANCOS}
    out = {"margen_vocabulario": MARGEN}

    # ---------------- 1. zonas ciegas
    sint = np.stack([recta(16, 16, 18, a, g) for g in GROSORES for a in ANGULOS])
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 5.4), dpi=130, facecolor=SUP, sharex=True)
    out["zonas_ciegas"] = {}
    for ax, b in zip(axs, BANCOS):
        m = mapas_rectas(bancos[b][0], idx, sint).reshape(len(sint), 4, -1).max(2)        # (N, 4)
        visto = (m >= um[b][None]).any(1).reshape(len(GROSORES), len(ANGULOS))
        out["zonas_ciegas"][b] = {f"{g}px": [int(a) for a, v in zip(ANGULOS, visto[i]) if not v] for i, g in enumerate(GROSORES)}
        ax.set_facecolor(SUP)
        ax.imshow(visto, aspect="auto", cmap=matplotlib.colors.ListedColormap([SUP, "#2a78d6"]), vmin=0, vmax=1,
                  extent=(-1.5, 178.5, len(GROSORES) - 0.5, -0.5), interpolation="nearest")
        for c in (0, 45, 90, 135, 180):
            ax.plot([max(-1.5, c - MARGEN), min(178.5, c + MARGEN)], [len(GROSORES) - 0.45] * 2, color=NAR, lw=3,
                    solid_capstyle="butt", clip_on=False)
        ax.set_xlim(-1.5, 178.5)
        ax.set_yticks(range(len(GROSORES))); ax.set_yticklabels([f"{g}" for g in GROSORES], fontsize=8, color=T2)
        ax.set_ylabel("grosor (px)", fontsize=8.5, color=T2)
        ax.set_title(BANCOS[b], fontsize=9.5, color=T1, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[1].set_xticks([0, 12, 33, 45, 57, 78, 90, 102, 123, 135, 147, 168])
    axs[1].set_xticklabels(["0°\n—", "12", "33", "45°\n\\", "57", "78", "90°\n|", "102", "123", "135°\n/", "147", "168"],
                           fontsize=7.5, color=T2)
    axs[1].set_xlabel("ángulo de la recta (azul = algún detector de recta la ve; blanco = ninguno)", fontsize=8.5, color=T2)
    fig.suptitle("¿Qué rectas ve algún detector de recta? (recta de 18 px en el centro)", fontsize=11.5, color=T1, x=0.02,
                 ha="left", fontweight="bold")
    fig.text(0.02, 0.905, "Raya naranja: los ±12° que el vocabulario de entrenamiento llama «recta». Fuera de ella, el "
             "detector nunca vio un ejemplo.", fontsize=8, color=T2)
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.1, right=0.98, hspace=0.35)
    fig.savefig(RES / "rectas-zonas-ciegas.png", facecolor=SUP); plt.close(fig)

    # ---------------- 2. rectas reales en los dígitos de val
    dg = datos.digitos(); va = np.flatnonzero(dg["val"]); X = dg["x"][va, 0]; Y = dg["y"][va]
    M = {b: mapas_rectas(bancos[b][0], idx, X) for b in BANCOS}
    filas = []
    for i, d in enumerate(X):
        for (x0, y0, x1, y1, ang, largo, grosor) in tramos(d):
            cf, cc = int(min(7, ((y0 + y1) / 2) // 4)), int(min(7, ((x0 + x1) / 2) // 4))
            fila = {"i": int(i), "digito": int(Y[i]), "seg": [int(x0), int(y0), int(x1), int(y1)], "angulo": round(ang, 1),
                    "largo": round(largo, 1), "grosor": round(grosor, 1)}
            fila["clase"], fila["en_vocabulario"] = en_vocabulario(ang)
            for b in BANCOS:
                vec = M[b][i][:, max(0, cf - 1):cf + 2, max(0, cc - 1):cc + 2].reshape(4, -1).max(1)   # (4,)
                on = vec >= um[b]
                fila[b] = {"visto": bool(on.any()), "por": [RECTAS[k] for k in np.flatnonzero(on)],
                           "max": {RECTAS[k]: round(float(vec[k]), 3) for k in range(4)}}
            filas.append(fila)
    out["tramos"] = filas

    def resumen(sel):
        return {b: {"n": len(sel), "no_vistos": round(1 - np.mean([f[b]["visto"] for f in sel]), 3) if sel else None} for b in BANCOS}
    out["resumen"] = {"todos": resumen(filas),
                      "angulo_en_vocabulario": resumen([f for f in filas if f["en_vocabulario"]]),
                      "angulo_en_el_hueco": resumen([f for f in filas if not f["en_vocabulario"]]),
                      "en_vocabulario_y_fino_(≤4px)": resumen([f for f in filas if f["en_vocabulario"] and f["grosor"] <= 4]),
                      "en_vocabulario_y_grueso_(>4px)": resumen([f for f in filas if f["en_vocabulario"] and f["grosor"] > 4])}
    out["por_digito"] = {str(c): resumen([f for f in filas if f["digito"] == c]) for c in range(10)}

    # gráfica: % de tramos rectos que NO ve nadie, por ángulo y por grosor
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=130, facecolor=SUP)
    bins_a = np.arange(0, 181, 15); bins_g = [0, 3, 4.5, 6, 8, 20]
    for ax, (clave, bins, etiq) in zip(axs, (("angulo", bins_a, "ángulo del tramo (°)"), ("grosor", bins_g, "grosor del trazo (px)"))):
        ax.set_facecolor(SUP)
        for k, (b, c) in enumerate((("lineas", "#2a78d6"), ("control", NAR))):
            v = np.array([f[clave] for f in filas]); nv = np.array([not f[b]["visto"] for f in filas])
            ys, xs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (v >= lo) & (v < hi)
                if m.sum() >= 10:
                    ys.append(nv[m].mean()); xs.append((lo + hi) / 2)
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=5, markeredgecolor=SUP, markeredgewidth=1.5, label=BANCOS[b])
        if clave == "angulo":
            for cen in (0, 45, 90, 135, 180):
                ax.axvspan(cen - MARGEN, cen + MARGEN, color=REJ, alpha=0.6, lw=0, zorder=0)
            ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xticklabels(["0 —", "45 \\", "90 |", "135 /", "180"], fontsize=8)
        ax.set_ylim(0, 1); ax.set_xlabel(etiq, fontsize=8.5, color=T2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(EJE); ax.tick_params(colors=T2, labelsize=8, length=0); ax.grid(axis="y", color=REJ, lw=0.8)
    axs[0].set_ylabel("de cada 100 tramos rectos,\ncuántos NO ve ningún detector", fontsize=8.5, color=T2)
    axs[0].legend(frameon=False, fontsize=8, loc="upper center", labelcolor=T1)
    fig.suptitle(f"Rectas reales en los dígitos ({len(filas)} tramos de ≥ 12 px en 1617 dígitos): ¿cuántas se escapan?",
                 fontsize=10.5, color=T1, x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.885, "Zona gris: ángulos que el vocabulario llama «recta» (±12°).", fontsize=8, color=T2)
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.11, right=0.98, wspace=0.18)
    fig.savefig(RES / "rectas-reales.png", facecolor=SUP); plt.close(fig)

    # galería: 8 en el hueco del vocabulario + 8 con ángulo «bueno» que tampoco ve la tinta, dígitos distintos
    def elegir(cond, n):
        vistos, sel = set(), []
        for f in sorted(filas, key=lambda f: -f["largo"]):
            if cond(f) and not f["lineas"]["visto"] and (f["i"] not in vistos) and sum(s["digito"] == f["digito"] for s in sel) < 2:
                sel.append(f); vistos.add(f["i"])
            if len(sel) == n:
                break
        return sel
    gal = elegir(lambda f: not f["en_vocabulario"], 8) + elegir(lambda f: f["en_vocabulario"], 8)
    fig, axs = plt.subplots(4, 4, figsize=(8.4, 9.6), dpi=120, facecolor=SUP)
    for ax, f in zip(axs.ravel(), gal):
        ax.imshow(X[f["i"]], cmap="Greys", vmin=0, vmax=1.6); ax.set_xticks([]); ax.set_yticks([])
        x0, y0, x1, y1 = f["seg"]; ax.plot([x0, x1], [y0, y1], color=NAR, lw=3, solid_capstyle="round")
        mx = max(f["lineas"]["max"].values())
        clase = SIMBOLO[f["clase"]]
        motivo = "ángulo sin clase" if not f["en_vocabulario"] else f"ángulo de «{clase}»"
        ax.set_title(f"«{f['digito']}» · {f['angulo']:.0f}° · {f['grosor']:.0f} px\n{motivo} · máx {mx * 100:.0f}",
                     fontsize=8, color=T1, pad=3)
        for s in ax.spines.values():
            s.set_color(REJ)
    for ax in axs.ravel()[len(gal):]:
        ax.axis("off")
    fig.suptitle("Tramos rectos (naranja) que NO ve ningún detector de recta de la tinta", fontsize=11.5, color=T1, x=0.02,
                 ha="left", fontweight="bold")
    fig.text(0.02, 0.945, "Filas 1–2: ángulo fuera de las 4 clases del vocabulario. Filas 3–4: ángulo de una clase, y aun "
             "así no la ve. «máx»: lo más alto de los 4 detectores de recta ahí (de 100).", fontsize=8, color=T2, wrap=True)
    fig.subplots_adjust(top=0.9, bottom=0.02, left=0.02, right=0.98, hspace=0.35, wspace=0.08)
    fig.savefig(RES / "rectas-galeria.png", facecolor=SUP); plt.close(fig)

    (RES / "rectas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"resumen": out["resumen"], "zonas_ciegas": {b: {g: len(v) for g, v in z.items()} for b, z in out["zonas_ciegas"].items()}},
                     ensure_ascii=False, indent=1))
    print("por dígito (no vistos, tinta):", {c: v["lineas"]["no_vistos"] for c, v in out["por_digito"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
