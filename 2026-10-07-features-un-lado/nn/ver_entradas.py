#!/usr/bin/env python3
"""Lo que VE el compositor para UN dígito (pedido por el dueño el 2026-10-08: «muéstrame un dígito cualquiera y las
entradas del compositor, para ver dónde encuentra las features»). No mide nada: dibuja.

  entradas-<i>-detectores.png   los 13 mapas 8×8 de los detectores de TINTA (feat-ind32) y de los de CONTROL, con el
                                contorno del dígito encima; el título dice el máximo y si pasa el umbral del detector
  entradas-<i>-lados.png        la entrada de los compositores POR LADO (nn/lados.py): fila A, el borde de cada lado
                                reducido a 8×8; filas B, los 13 detectores del brazo compartido mirando ESE lado solo

    python nn/ver_entradas.py [--indice N]   N = posición del dígito entre los 1617 de val (por defecto, el primer 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import componer as C                                              # noqa: E402
import datos                                                      # noqa: E402
import features as F                                              # noqa: E402
import modelo                                                     # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, BORDE = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834"
CMAP = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
# ⚠ arco-E tiene el CENTRO al Este, o sea forma de ⊂ (corregido el 2026-10-08: la primera versión los tenía al revés)
NOMBRES = {"arco-E": "arco ⊂", "arco-W": "arco ⊃", "arco-N": "arco ∪", "arco-S": "arco ∩", "recta-V": "recta |",
           "recta-H": "recta —", "recta-S": "recta /", "recta-B": "recta \\", "lazo": "lazo ○", "esquina-NE": "esq. └",
           "esquina-NW": "esq. ┘", "esquina-SE": "esq. ┌", "esquina-SW": "esq. ┐"}


def mapa(ax, m8, digito, titulo, vmax=1.0, encendido=None):
    ax.imshow(m8, cmap=CMAP, vmin=0, vmax=vmax, extent=(0, 32, 32, 0), interpolation="nearest")
    ax.contour(np.arange(32) + 0.5, np.arange(32) + 0.5, digito, levels=[0.5], colors=[BORDE], linewidths=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e1e0d9")
    ax.set_title(titulo, fontsize=7.5, color=T1 if encendido in (None, True) else T2, pad=2,
                 fontweight="bold" if encendido else "normal")


def leyenda(fig, texto):
    fig.text(0.01, 0.005, texto, fontsize=8, color=T2, va="bottom")


def main() -> int:
    a = argparse.ArgumentParser(); a.add_argument("--indice", type=int); a = a.parse_args()
    torch.set_num_threads(2)
    dg = datos.digitos(); y = dg["y"]; va = np.flatnonzero(dg["val"]); tr = dg["train"]
    i = a.indice if a.indice is not None else int(np.flatnonzero(y[va] == 2)[0])
    n = va[i]; x = dg["x"][n:n + 1].astype(np.float32); dig = x[0, 0]

    bancos = {"tinta (feat-ind32)": C.banco("lineas")[0], "control: 8 bordes a la vez": C.banco("control")[0]}
    umbral = {}
    import evaluar as E                                           # noqa: PLC0415
    for nombre, clave in (("tinta (feat-ind32)", "lineas"), ("control: 8 bordes a la vez", "control")):
        umbral[nombre] = E.banco(clave)[1]
    # qué dígito dice cada compositor (el de C1: 180 de train, sin desplazar, semilla 1)
    pred = {}
    for nombre, reds in bancos.items():
        Xtr = C.mapas(reds, dg["x"][tr]).astype(np.float32); W = C.ajustar(Xtr, y[tr], 1)
        with torch.no_grad():
            p = torch.softmax(W(torch.from_numpy(C.mapas(reds, x).astype(np.float32))), 1)[0].numpy()
        pred[nombre] = (int(p.argmax()), float(p.max()))

    fig, axs = plt.subplots(2, 14, figsize=(16, 3.4), dpi=130, facecolor=SUP,
                            gridspec_kw={"width_ratios": [1.25] + [1] * 13})
    for fila, (nombre, reds) in enumerate(bancos.items()):
        with torch.no_grad():
            m = [torch.sigmoid(r(torch.from_numpy(x)))[0, 0].numpy() for r in reds]
        ax = axs[fila, 0]
        ax.imshow(dig, cmap="Greys", vmin=0, vmax=1.3); ax.set_xticks([]); ax.set_yticks([])
        d, conf = pred[nombre]
        ax.set_title(f"{nombre}\ncompositor: «{d}» ({conf * 100:.0f} %)", fontsize=7.5, color=T1, pad=2)
        for j, f in enumerate(F.CON_TRAZO):
            mx = float(m[j].max()); on = mx >= umbral[nombre][j]
            mapa(axs[fila, j + 1], m[j], dig, f"{NOMBRES[f]}  {mx * 100:.0f}{' ✓' if on else ''}", encendido=on)
    fig.suptitle(f"Un «{int(y[n])}» de prueba y lo que recibe el compositor: 13 mapas de 8 × 8 (azul = el detector cree ver "
                 f"su trazo ahí)", fontsize=11, color=T1, x=0.01, ha="left")
    leyenda(fig, "Naranja: contorno del dígito. Número: lo más alto del mapa (de 100). ✓ y negrita: pasa el umbral de ese "
                 "detector, o sea «lo encontró». Las 832 casillas azules (13 × 64) son TODO lo que ve el compositor.")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.78, bottom=0.08, wspace=0.08, hspace=0.45)
    fig.savefig(RES / f"entradas-{i}-detectores.png", facecolor=SUP); plt.close(fig)

    # --- por lado: A (borde reducido) y B (13 detectores del compartido mirando ese lado)
    comp = [modelo.cargar(AQUI / "pesos-compartido" / f / "best.pt")[0] for f in F.CON_TRAZO]
    with torch.no_grad():
        xt = torch.from_numpy(x)
        bor = comp[0].canales_borde(xt)[0].numpy()                                  # (8, 32, 32)
        B = np.stack([torch.sigmoid(r.por_canal(xt))[0].numpy() for r in comp])      # (13, 8, 8, 8)
    A8 = bor.reshape(8, 8, 4, 8, 4).mean((2, 4))
    fig, axs = plt.subplots(15, 8, figsize=(8.6, 16.5), dpi=120, facecolor=SUP)
    for k in range(8):
        ax = axs[0, k]; ax.imshow(bor[k], cmap=CMAP, vmin=0, vmax=bor.max()); ax.set_xticks([]); ax.set_yticks([])
        ax.contour(np.arange(32), np.arange(32), dig, levels=[0.5], colors=[BORDE], linewidths=0.6)
        ax.set_title(f"lado {modelo.FLECHAS[k]}", fontsize=9, color=T1, fontweight="bold")
        mapa(axs[1, k], A8[k], dig, "", vmax=A8.max())
        for j in range(13):
            mapa(axs[j + 2, k], B[j, k], dig, "")
    axs[0, 0].set_ylabel("el borde\n(32 × 32)", fontsize=8, color=T1)
    axs[1, 0].set_ylabel("A: lo que ve\nsu compositor\n(8 × 8)", fontsize=8, color=T1)
    for j, f in enumerate(F.CON_TRAZO):
        axs[j + 2, 0].set_ylabel(f"B: {NOMBRES[f]}", fontsize=8, color=T1)
    fig.suptitle(f"El mismo «{int(y[n])}» visto lado por lado: lo que recibe cada compositor por lado", fontsize=11,
                 color=T1, x=0.01, ha="left")
    leyenda(fig, "Fila A: 64 números por lado. Filas B: los 13 detectores (brazo compartido) mirando sólo ese lado, "
                 "832 números por lado. Azul = alto.")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.025, wspace=0.06, hspace=0.12)
    fig.savefig(RES / f"entradas-{i}-lados.png", facecolor=SUP); plt.close(fig)
    print(f"→ resultados/entradas-{i}-detectores.png y entradas-{i}-lados.png (dígito {int(y[n])}, val #{i}); "
          f"compositores: {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
