#!/usr/bin/env python3
"""¿Qué ARCOS no ve su detector de arco? Pedido por el dueño el 2026-10-08, tras la auditoría de rectas (nn/rectas_ciegas.py):
«haz lo mismo para arcos; concentrémonos en arcos cortos, de varios radios». Sin entrenar nada (0 $, minutos en el dev).

El vocabulario de entrenamiento (features.py) sólo tiene arcos de APERTURA 100–200°, RADIO 4–14 px, y orientación ±40° de
las cuatro familias:  arco-E ⊂ (centro al Este) · arco-W ⊃ · arco-N ∪ · arco-S ∩.

Se dibuja un arco con el MISMO rasterizador (PIL, `features.rasterizar`), con su punto medio en el centro de la imagen, para
cada apertura (30–180°), radio (3–14 px), grosor (3 y 6 px) y orientación (cada 15°, sin las de los huecos entre familias:
45°, 135°, 225°, 315°). Cuenta como VISTO si el detector de SU familia pasa su umbral cerca del punto medio (su celda 8×8 o
una vecina). Si no, se apunta qué otro detector se enciende ahí.

⚠ A 32×32 muchos arcos cortos NO tienen curva visible: un arco de 45° y radio 12 se separa menos de 1 px de su cuerda
(flecha = r·(1 − cos(apertura/2))), y uno de radio 4 mide 3 px. Ahí que se encienda «recta» no es un error. Por eso cada
caso lleva su FLECHA y su LARGO, y el resumen separa los arcos con curva VISIBLE (flecha ≥ 2 px y largo ≥ 8 px).

    python nn/arcos_ciegos.py   → resultados/arcos-mapa.png, arcos-galeria.png, arcos-diagonales-galeria.png, arcos.json
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

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import evaluar as E                                               # noqa: E402
import features as F                                              # noqa: E402

RES = AQUI.parent / "resultados"
ARCOS = ("arco-E", "arco-S", "arco-W", "arco-N")                  # centro al E, S, W, N = 0°, 90°, 180°, 270°
FORMA = {"arco-E": "⊂", "arco-W": "⊃", "arco-N": "∪", "arco-S": "∩"}
NOMBRE = {f: f for f in F.CON_TRAZO} | {"arco-E": "arco ⊂", "arco-W": "arco ⊃", "arco-N": "arco ∪", "arco-S": "arco ∩",
                                       "recta-V": "recta |", "recta-H": "recta —", "recta-S": "recta /", "recta-B": "recta \\",
                                       "lazo": "lazo", "esquina-NE": "esq └", "esquina-NW": "esq ┘", "esquina-SE": "esq ┌",
                                       "esquina-SW": "esq ┐"}
APERTURAS = (30, 45, 60, 90, 120, 180)
RADIOS = (3, 4, 6, 8, 10, 12, 14)
GROSORES = (3, 6)
ORIENT = [t for t in range(0, 360, 15) if t % 90 != 45]           # sin los huecos de 10° entre familias
BANCOS = {"lineas": "tinta (feat-ind32)", "control": "control: 8 bordes a la vez"}
SUP, T1, T2, REJ, NAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#eb6834"


def flecha(ap, r):
    return r * (1 - np.cos(np.radians(ap / 2)))


def visible(ap, r):
    """¿Se ve la curva a esta resolución? Flecha ≥ 2 px y largo del arco ≥ 8 px."""
    return flecha(ap, r) >= 2 and r * np.radians(ap) >= 8


def familia(theta):
    return ARCOS[int(((theta + 45) % 360) // 90)]


def arco(ap, r, w, theta):
    """El arco del vocabulario (features._arco) con su punto medio en (16, 16)."""
    ancla = np.array([16.0, 16.0])
    c = ancla + r * np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    a = np.radians(theta + 180.0 + np.linspace(-ap / 2, ap / 2, 64))
    pts = c + r * np.stack([np.cos(a), np.sin(a)], 1)
    return F.rasterizar({"pts": [pts], "cerrado": False, "grosor": int(w)})


@torch.no_grad()
def cerca(reds, x):
    """(N, 13): lo más alto de cada detector en las celdas 3..4 (el centro y sus vecinas)."""
    xt = torch.from_numpy(x[:, None].astype(np.float32))
    return np.stack([torch.sigmoid(r(xt))[:, 0, 3:5, 3:5].reshape(len(x), -1).max(1).values.numpy() for r in reds], 1)


def main() -> int:
    torch.set_num_threads(2)
    idx = {f: i for i, f in enumerate(F.CON_TRAZO)}
    casos = [(ap, r, w, t) for ap in APERTURAS for r in RADIOS for w in GROSORES for t in ORIENT]
    X = np.stack([arco(*c) for c in casos])
    out = {"casos": []}
    vis = {}
    for b in BANCOS:
        reds, um, _ = E.banco(b)
        C = cerca(reds, X); on = C >= um[None]
        vis[b] = []
        for k, (ap, r, w, t) in enumerate(casos):
            fam = familia(t); ok = bool(on[k, idx[fam]])
            otros = [F.CON_TRAZO[j] for j in np.argsort(-C[k]) if on[k, j] and F.CON_TRAZO[j] != fam]
            vis[b].append(ok)
            if b == "lineas":
                out["casos"].append({"apertura": ap, "radio": r, "grosor": w, "orientacion": t, "familia": fam,
                                     "flecha": round(float(flecha(ap, r)), 2), "largo": round(float(r * np.radians(ap)), 1),
                                     "curva_visible": bool(visible(ap, r))})
            out["casos"][k][b] = {"visto": ok, "conf": round(float(C[k, idx[fam]]), 3), "en_su_lugar": otros[:3]}
        vis[b] = np.array(vis[b])
    A = np.array([c[0] for c in casos]); R = np.array([c[1] for c in casos]); W = np.array([c[2] for c in casos])

    # mapa: % de orientaciones en que lo ve SU detector, por apertura × radio, para cada banco y grosor
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.2), dpi=130, facecolor=SUP)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("a", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    out["mapa"] = {}
    for fila, b in enumerate(BANCOS):
        for col, w in enumerate(GROSORES):
            ax = axs[fila, col]
            M = np.array([[vis[b][(A == ap) & (R == r) & (W == w)].mean() for ap in APERTURAS] for r in RADIOS])
            out["mapa"][f"{b} {w}px"] = M.round(3).tolist()
            ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(len(RADIOS)):
                for j in range(len(APERTURAS)):
                    vi = visible(APERTURAS[j], RADIOS[i])
                    ax.text(j, i, f"{M[i, j] * 100:.0f}" + ("" if vi else "\n≈recta"), ha="center", va="center",
                            fontsize=8 if vi else 6.5, color="#fcfcfb" if M[i, j] > 0.55 else (T1 if vi else "#898781"))
            # lo que el entrenamiento vio: apertura 100–200 (columnas 120 y 180), radio 4–14 (filas 4..14)
            ax.add_patch(plt.Rectangle((3.5, 0.5), 2, len(RADIOS) - 1, fill=False, ec=NAR, lw=2))
            ax.set_xticks(range(len(APERTURAS))); ax.set_xticklabels([f"{a}°" for a in APERTURAS], fontsize=8, color=T2)
            ax.set_yticks(range(len(RADIOS))); ax.set_yticklabels([f"{r}" for r in RADIOS], fontsize=8, color=T2)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{BANCOS[b]} · trazo de {w} px", fontsize=9, color=T1, loc="left")
            if col == 0:
                ax.set_ylabel("radio (px)", fontsize=8.5, color=T2)
            if fila == 1:
                ax.set_xlabel("apertura del arco (corto ← → largo)", fontsize=8.5, color=T2)
    fig.suptitle("¿Ve el detector de arco su arco? (de cada 100 orientaciones)", fontsize=11.5, color=T1, x=0.02, ha="left",
                 fontweight="bold")
    fig.text(0.02, 0.915, "Recuadro naranja: lo que el entrenamiento vio (apertura 100–200°, radio 4–14 px).\n«≈recta»: a 32 × 32 "
             "esa curva casi no se ve (se separa < 2 px de la recta, o mide < 8 px).", fontsize=8, color=T2)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.98, hspace=0.3, wspace=0.12)
    fig.savefig(RES / "arcos-mapa.png", facecolor=SUP); plt.close(fig)

    # galería: arcos CORTOS (≤ 90°) de varios radios que la tinta no ve, y qué se enciende en su lugar
    sel, usados = [], {}
    for k, c in enumerate(casos):                     # sólo arcos CORTOS con curva VISIBLE, varias formas y radios
        ap, r, w, t = c
        if ap <= 90 and visible(ap, r) and not vis["lineas"][k] and usados.get((ap, r), 0) < 1 and \
                sum(familia(casos[q][3]) == familia(t) for q in sel) < 3:
            sel.append(k); usados[(ap, r)] = 1
        if len(sel) == 12:
            break
    filas_g = max(1, -(-len(sel) // 4))
    fig, axs = plt.subplots(filas_g, 4, figsize=(8.4, 1.0 + 2.6 * filas_g), dpi=120, facecolor=SUP, squeeze=False)
    for ax, k in zip(axs.ravel(), sel):
        ap, r, w, t = casos[k]; c = out["casos"][k]["lineas"]
        ax.imshow(X[k], cmap="Greys", vmin=0, vmax=1.6); ax.set_xticks([]); ax.set_yticks([])
        en = ", ".join(NOMBRE[f] for f in c["en_su_lugar"]) or "nada"
        ax.set_title(f"{FORMA[familia(t)]} {ap}° · radio {r} · {w} px\nsu detector: {c['conf'] * 100:.0f} · ve: {en}",
                     fontsize=8, color=T1, pad=3)
        for s in ax.spines.values():
            s.set_color(REJ)
    for ax in axs.ravel()[len(sel):]:
        ax.axis("off")
    H = 1.0 + 2.6 * filas_g
    fig.suptitle("Arcos cortos CON CURVA VISIBLE que NO ve su detector de arco (tinta)", fontsize=11.5, color=T1, x=0.02,
                 ha="left", fontweight="bold", y=1 - 0.08 / H, va="top")
    fig.text(0.02, 1 - 0.42 / H, "Primera línea: forma, apertura y radio. «su detector»: lo más alto del arco que tocaba (de 100). "
             "«ve»: lo que se enciende en su lugar.", fontsize=8, color=T2)
    fig.subplots_adjust(top=1 - 0.95 / H, bottom=0.02, left=0.02, right=0.98, hspace=0.4, wspace=0.08)
    fig.savefig(RES / "arcos-galeria.png", facecolor=SUP); plt.close(fig)

    # arcos en DIAGONAL (centro al NE, SE, SW, NW): el vocabulario no tiene clase; cuenta como visto si se enciende
    # cualquiera de las dos familias vecinas
    diag = [(ap, r, w, t) for ap in APERTURAS for r in RADIOS for w in GROSORES for t in (45, 135, 225, 315)]
    Xd = np.stack([arco(*c) for c in diag]); Ad = np.array([c[0] for c in diag]); Rd = np.array([c[1] for c in diag])
    Vd = np.array([visible(c[0], c[1]) for c in diag])
    out["diagonales"] = {}
    for b in BANCOS:
        reds, um, _ = E.banco(b)
        on = cerca(reds, Xd) >= um[None]
        vd = np.array([on[k, idx[familia(t - 45)]] or on[k, idx[familia(t + 45)]] for k, (_, _, _, t) in enumerate(diag)])
        out["diagonales"][b] = {"como_el_entrenamiento": round(float(vd[(Ad >= 100) & (Rd >= 4)].mean()), 3),
                                "cortos_≤90_con_curva_visible": round(float(vd[(Ad <= 90) & Vd].mean()), 3)}
        if b == "lineas":                           # galería: arcos en diagonal, con curva visible, que nadie ve
            C = cerca(reds, Xd)
            sel = [k for k, (ap, r, w, t) in enumerate(diag) if Vd[k] and not vd[k] and w == 3][::3][:8]
            fig, axs = plt.subplots(2, 4, figsize=(8.4, 5.8), dpi=120, facecolor=SUP)
            for ax, k in zip(axs.ravel(), sel):
                ap, r, w, t = diag[k]
                otros = [NOMBRE[F.CON_TRAZO[j]] for j in np.argsort(-C[k]) if on[k, j]][:2]
                v1, v2 = familia(t - 45), familia(t + 45)
                ax.imshow(Xd[k], cmap="Greys", vmin=0, vmax=1.6); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{ap}° · radio {r} · entre {FORMA[v1]} y {FORMA[v2]}\nellos: {C[k, idx[v1]] * 100:.0f} y "
                             f"{C[k, idx[v2]] * 100:.0f} · ve: {', '.join(otros) or 'nada'}", fontsize=8, color=T1, pad=3)
                for sp in ax.spines.values():
                    sp.set_color(REJ)
            for ax in axs.ravel()[len(sel):]:
                ax.axis("off")
            fig.suptitle("Arcos en DIAGONAL, con curva visible, que no ve ningún detector de arco (tinta)", fontsize=11.5,
                         color=T1, x=0.02, ha="left", fontweight="bold")
            fig.text(0.02, 0.905, "El vocabulario sólo tiene arcos ⊂ ⊃ ∪ ∩ (±40°); éstos quedan entre dos. «ellos»: lo más alto de "
                     "los dos vecinos (de 100).", fontsize=8, color=T2)
            fig.subplots_adjust(top=0.82, bottom=0.02, left=0.02, right=0.98, hspace=0.45, wspace=0.08)
            fig.savefig(RES / "arcos-diagonales-galeria.png", facecolor=SUP); plt.close(fig)

    def pct(b, m):
        return round(float(vis[b][m].mean()), 3)
    entreno = (A >= 100) & (R >= 4)
    V = np.array([visible(c[0], c[1]) for c in casos])
    out["resumen"] = {b: {"todos": pct(b, A > 0), "como_el_entrenamiento": pct(b, entreno),
                          "cortos_≤90": pct(b, A <= 90), "cortos_≤90_fino": pct(b, (A <= 90) & (W == 3)),
                          "cortos_≤90_con_curva_visible": pct(b, (A <= 90) & V), "n_cortos_visibles": int(((A <= 90) & V).sum()),
                          "cortos_≤90_sin_curva_visible": pct(b, (A <= 90) & ~V),
                          "radio_3": pct(b, R == 3), "grueso_6px_como_el_entrenamiento": pct(b, entreno & (W == 6))}
                      for b in BANCOS}
    (RES / "arcos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"resumen": out["resumen"], "diagonales": out["diagonales"]}, ensure_ascii=False, indent=1))
    from collections import Counter                                # noqa: PLC0415
    for b in BANCOS:
        cnt = Counter(f for c in out["casos"] if c["apertura"] <= 90 and not c[b]["visto"] for f in c[b]["en_su_lugar"][:1])
        sin = sum(1 for c in out["casos"] if c["apertura"] <= 90 and not c[b]["visto"] and not c[b]["en_su_lugar"])
        print(b, "arcos cortos no vistos → lo que se enciende en su lugar:", dict(cnt.most_common(6)), "· nada:", sin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
