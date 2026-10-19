#!/usr/bin/env python3
"""rizo-2 § 127 — VISTA PREVIA: el LAZO con la posición de su centro Y SU ALTURA (el dueño, 2026-10-19: «crear un nuevo feature,
que sea como el lazo con posición de su centro, y que indique cuál es su altura»). Sólo imagen; el revisor no cambia.

«Altura» = lo que MIDE el lazo de alto: alto = máx(y) − mín(y) de su polígono (`cresta_curvas.objetos()["lazos"][k]["pts"]`), en px
de 32. La posición vertical ya la da el centro. Sobre el DELGADO del revisor (fuente «nist», `delgados-k3x3.npz`), como todo aquí.

ALTURA RELATIVA (el dueño, 2026-10-19: «haz que la altura sea relativa a 1, así es independiente del tamaño en px»): alto del lazo /
alto del DÍGITO, los dos sobre la LÍNEA MEDIA del delgado (el del dígito: de la primera a la última fila de la cresta). Un 0 ≈ 1; no depende de la escala.

Codificación propuesta, la del bloque `lazos` (presencia) con la altura dentro, igual que `lazos_area` lleva el área: una gaussiana
σ 2 px en el CENTRO de cada lazo con pico = altura relativa, máximo por celda de la rejilla + la cuenta de lazos.

    ../.venv/bin/python nn/lazo_altura_vista.py   → resultados/127-lazo-altura.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import servidor as S                                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

CELDAS = 3


def alto(z) -> float:
    p = np.asarray(z["pts"], float); return float(p[:, 1].max() - p[:, 1].min())


def alto_digito(M) -> float:
    """Sobre la LÍNEA MEDIA (esqueleto de M > 0,4), como el lazo: sobre la banda M > 0,4 un 0 daba 0,85 en vez de ~1."""
    filas = np.flatnonzero(skeletonize(M > 0.4).any(1)); return float(filas[-1] - filas[0]) if len(filas) else float("nan")


def bloque(lazos, ref: float, celdas: int = CELDAS) -> list[float]:
    """ref = el alto del DÍGITO: el pico de cada lazo es su altura relativa."""
    m = CC._mapa(lazos, lambda z: alto(z) / ref, 2.0)
    return F.adaptive_max_pool2d(torch.from_numpy(m)[None], celdas)[0].numpy().ravel().tolist() + [float(len(lazos))]


def main():
    d = S.Datos(); y = d.y; val = np.asarray(d.va)
    lz = {int(i): d.objetos(int(i))["lazos"] for i in val}
    ad = {int(i): alto_digito(d.M[i]) for i in val}
    altos = {c: [alto(z) / ad[int(i)] for i in val if y[i] == c for z in lz[int(i)]] for c in range(10)}
    ref = 1.0; res = {"por_clase": {}}
    for c in range(10):
        a = np.asarray(altos[c]); n_dig = int(np.sum(y[val] == c))
        res["por_clase"][str(c)] = {"digitos": n_dig, "lazos": int(len(a)), **({"altura_rel_p10_p50_p90": [round(float(v), 2) for v in np.percentile(a, [10, 50, 90])]} if len(a) else {})}
        print(c, res["por_clase"][str(c)])

    # muestra: 2 dígitos por clase con lazo (0 4 6 8 9) + un 2 con lazo si lo hay; el más típico y el más raro por alto
    muestra = []
    for c in (0, 6, 9, 8, 4, 2):
        ids = [int(i) for i in val if y[i] == c and lz[int(i)]]
        if not ids:
            continue
        h = np.array([max(alto(z) / ad[i] for z in lz[i]) for i in ids]); o = np.argsort(np.abs(h - np.median(h)))
        muestra += [ids[o[0]], ids[o[-1]]]
    fig = plt.figure(figsize=(18, 10.5), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[3.3, 7])
    a = arriba.subplots(); cls = [c for c in range(10) if altos[c]]
    a.boxplot([altos[c] for c in cls], positions=range(len(cls)), widths=0.55, showfliers=True, flierprops={"markersize": 2})
    for k, c in enumerate(cls):
        p = res["por_clase"][str(c)]
        a.text(k, max(altos[c]) + 0.03, f"{p['lazos']} lazos\nen {p['digitos']} díg.\nmed. {p['altura_rel_p10_p50_p90'][1]:.2f}", ha="center", fontsize=8)
    a.axhline(1, color="#1b7f3b", ls="--", lw=1); a.text(len(cls) - 0.5, 1.02, "= alto del dígito", color="#1b7f3b", fontsize=9, ha="right")
    a.set_xticks(range(len(cls))); a.set_xticklabels([str(c) for c in cls]); a.set_ylabel("alto del lazo / alto del dígito"); a.set_ylim(0, 1.45)
    a.set_title("ALTURA RELATIVA de cada lazo en val, por clase (los cuadros: 25–75 %; la raya: mediana)", fontsize=10)
    axs = abajo.subplots(2, len(muestra) // 2 + len(muestra) % 2, squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for k, i in enumerate(muestra):
        ax = axs.ravel()[k]; ax.axis("on"); ax.imshow(d.M[i], cmap="Greys", vmin=0, vmax=1.3, extent=(0, 32, 32, 0))
        for g in (32 / 3, 64 / 3):
            ax.axhline(g, color="#bbb", lw=0.6); ax.axvline(g, color="#bbb", lw=0.6)
        b = bloque(lz[i], ad[i])
        f0 = np.flatnonzero(skeletonize(d.M[i] > 0.4).any(1)); ax.annotate("", (0.8, f0[0] + 0.5), (0.8, f0[-1] + 0.5), arrowprops={"arrowstyle": "<->", "color": "#666", "lw": 1})
        for cc in range(9):                                               # el valor de la rejilla, en cada celda
            if b[cc] > 0.05:
                ax.text((cc % 3 + 0.5) * 32 / 3, (cc // 3 + 0.5) * 32 / 3, f"{b[cc]:.2f}", ha="center", va="center", fontsize=10,
                        fontweight="bold", color="#1b7f3b", bbox={"fc": "white", "alpha": 0.7, "lw": 0, "pad": 0.5})
        for z in lz[i]:
            p = np.asarray(z["pts"]); q = np.vstack([p, p[:1]]); ax.plot(q[:, 0], q[:, 1], "-", color="#2a78d6", lw=1.6)
            y0, y1 = p[:, 1].min(), p[:, 1].max(); x = p[:, 0].max() + 1.2
            ax.annotate("", (x, y0), (x, y1), arrowprops={"arrowstyle": "<->", "color": "#c2410c", "lw": 1.4})
            ax.text(x + 0.5, (y0 + y1) / 2, f"{(y1 - y0) / ad[i]:.2f}", color="#c2410c", fontsize=10, fontweight="bold", va="center")
            ax.plot(z["cx"], z["cy"], "o", color="#c2410c", ms=4)
        ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{i} ({y[i]}) · dígito {ad[i]:.0f} px · cuenta {int(b[-1])}", fontsize=9)
    abajo.suptitle("sobre el DELGADO · AZUL = lazo · punto = su centro · flecha GRIS = alto del dígito · NARANJA = altura RELATIVA del lazo · "
                   "VERDE = lo que vale cada celda 3×3: gaussiana en el centro con pico = altura relativa", fontsize=10)
    fig.suptitle("VISTA PREVIA · el LAZO con posición de su centro y su ALTURA RELATIVA (alto del lazo / alto del dígito; sólo imagen: el revisor no cambia)", fontsize=12)
    out = AQUI / "resultados" / "127-lazo-altura.png"; fig.savefig(out, dpi=85)
    out.with_suffix(".json").write_text(json.dumps({**res, "muestra": muestra}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
