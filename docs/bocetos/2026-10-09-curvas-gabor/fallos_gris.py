#!/usr/bin/env python3
"""Los fallos del compositor con el borde de Gabor impar GRIS re-calibrado (pedido del dueño, 2026-10-10): λ_b 10 ·
kernel 9 con su calibración de la galería (TAU 0,1 · σE 0,5 · coherencia 0,25 · λ par 3), el mejor a ciegas de
`gris_recalibrado.py` (val 0,960 / a ciegas 0,957) y el que conserva el λ 3 acordado para leer líneas. Semilla 0, val.

Cada fallo va en DOS celdas: el borde gris que entra al detector, y lo que el detector ve sobre él (verde recto ·
naranja curvo · flecha al centro de cada curva · gris claro: borde sin giro medible). «antes también» = ya fallaba con
el borde morfológico (`resultados-fallos-bordes.json`).

    python fallos_gris.py   → imagenes/19-fallos-gris.png · resultados-fallos-gris.json  (~1 min)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

LAM_B, K_B = 10.0, 9
CAL = {"tau": 0.1, "sigma_e": 0.5, "coh_min": 0.25, "lam": 3.0}     # la elegida por la galería (resultados-gris-recalibrado.json)
PARES_POR_FILA = 8


def main() -> int:
    torch.set_num_threads(2)
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    cal = json.loads((AQUI / "resultados-gris-recalibrado.json").read_text())["por_borde"][f"λ_b {LAM_B:g} · K {K_B}"]
    assert all(cal["calibracion"][k] == v for k, v in CAL.items()), "la calibración no es la del JSON"
    GR.configurar(**CAL)
    bg = BG.BordeGabor(LAM_B, k=K_B, binarizar=False)
    X = DB.caract(img, borde=bg)["bordes"]
    pred = DB.entrenar(X[tr], y[tr], 0)(X[va])
    acc = float((pred == y[va]).mean())
    f = np.flatnonzero(pred != y[va]); orden = f[np.lexsort((pred[f], y[va][f]))]
    antes = set(json.loads((AQUI / "resultados-fallos-bordes.json").read_text())["indices_val_fallados"])
    n = len(orden); filas = int(np.ceil(n / PARES_POR_FILA))
    print(f"semilla 0: acierto {acc:.4f} · {n} fallos · {sum(int(va[j]) in antes for j in orden)} ya fallaban antes")

    C._estilo()
    fig, axs = plt.subplots(filas, 2 * PARES_POR_FILA, figsize=(2 * PARES_POR_FILA * 0.98, filas * 1.25 + 0.6))
    axs = np.atleast_2d(axs)
    for k in range(filas * PARES_POR_FILA):
        a_b, a_d = axs[k // PARES_POR_FILA, 2 * (k % PARES_POR_FILA)], axs[k // PARES_POR_FILA, 2 * (k % PARES_POR_FILA) + 1]
        if k >= n:
            a_b.axis("off"); a_d.axis("off"); continue
        j = orden[k]; i = va[j]
        e = bg(img[i].astype(np.float32))
        a_b.imshow(e, cmap="gray_r", vmin=0, vmax=1); a_b.set_xticks([]); a_b.set_yticks([])
        ya = int(i) in antes
        a_b.set_title(f"{y[i]}→{pred[j]}", fontsize=10, pad=2, color=DB.NAR, fontweight="bold")
        BG.dibujar_detector(a_d, e, img[i], "antes también" if ya else "nuevo", C.T2 if ya else DB.AZU)
        for s in a_b.spines.values():
            s.set_color(DB.NAR); s.set_linewidth(1.2)
    por_real = {int(r): int((y[va][f] == r).sum()) for r in range(10)}
    conf = {}
    for j in f:
        kk = f"{y[va][j]}→{pred[j]}"; conf[kk] = conf.get(kk, 0) + 1
    top = dict(sorted(conf.items(), key=lambda z: -z[1])[:8])
    fig.suptitle(f"19 · Los {n} dígitos de val (1617) que FALLA el compositor con el borde de Gabor impar GRIS re-calibrado "
                 f"(λ_b {LAM_B:g} · kernel {K_B}, semilla 0, acierto {acc:.3f}) · «real→leído», por real\n"
                 "cada fallo en dos celdas: el borde gris que entra · lo que ve el detector (verde recto · naranja curvo · "
                 "flecha al centro · gris: sin giro medible) · «antes también» = ya fallaba con el borde morfológico\n"
                 "fallos por real: " + " · ".join(f"{r}: {v}" for r, v in por_real.items()) +
                 "   |   más frecuentes: " + " · ".join(f"{k} ({v})" for k, v in top.items()),
                 fontsize=10, color=C.T1, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.5, w_pad=0.2)
    fig.savefig(DB.IMG / "19-fallos-gris.png", dpi=100); plt.close(fig)
    print("→ 19-fallos-gris.png")
    (AQUI / "resultados-fallos-gris.json").write_text(json.dumps(
        {"borde": f"Gabor impar gris λ_b {LAM_B:g} · K {K_B}", "calibracion": CAL, "semilla": 0, "acierto_val": round(acc, 4),
         "fallos": n, "ya_fallaban_con_morfologico": sum(int(va[j]) in antes for j in orden), "fallos_por_real": por_real,
         "confusiones": conf, "indices_val_fallados": [int(va[j]) for j in orden]}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
