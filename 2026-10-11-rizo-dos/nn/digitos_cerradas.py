#!/usr/bin/env python3
"""rizo-2 — TODOS los dígitos con el compuesto como detector de CURVAS CERRADAS, más los otros detectores para las otras
features (pedido del dueño, 2026-10-11: «evalúa sobre dígitos (todos), usando este como detector de curvas cerradas, y
usando otros detectores para otros features. Muéstrame los dígitos donde falla»).

Cadena, la de siempre en el boceto curvas-gabor (`digitos_delgadas.py`): dígito binario → detector DELGADO (recto ·
vector de curvatura → ← ↓ ↑ · golpe, máximo en celdas 8×8 = 384 características) → compositor LINEAL (180 train,
1617 val, 3823 a ciegas, 3 semillas). Aquí se le añaden las CURVAS CERRADAS del compuesto (`compuesto2.py`, en TODO el
dígito, no sólo arriba): dos mapas 32×32 —una campana (σ 1,5) de altura = encierro en cada candidato, y otra sólo con
el mejor— en celdas 8×8 = 128 características más. Se comparan: otros solos · cerradas solas · otros + cerradas.
⚠ Importa los módulos del boceto `docs/bocetos/2026-10-09-curvas-gabor` para «los otros detectores» tal cual están
(es la referencia publicada; copiarlos aquí sería 6 ficheros); el detector de cerradas es el de este experimento.

    python nn/digitos_cerradas.py [--rmin 0.85 --rmax 12 --sufijo b]   → resultados/13-digitos.json · 13-digitos-fallos.png · 13-digitos-cerradas.png  (~25 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
BOCETO = AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(BOCETO))
import compuesto2 as C2                                             # noqa: E402
import evaluar as EV                                                # noqa: E402
import delgadas as DL                                               # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

_YY, _XX = np.mgrid[0:32, 0:32] + 0.5
R_MIN = 0.5
SUFIJO = ""


def mapas_cerradas(x: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    cs = [c for c in C2.compuesto2(x) if c["r"] >= R_MIN]
    todos = np.zeros((32, 32), np.float32); mejor = np.zeros((32, 32), np.float32)
    for k, c in enumerate(cs):                                      # compuesto2 ya los da ordenados por encierro
        g = np.exp(-((_XX - c["centro"][0]) ** 2 + (_YY - c["centro"][1]) ** 2) / (2 * 1.5 ** 2)) * c["r"]
        todos = np.maximum(todos, g)
        if k == 0:
            mejor = g.astype(np.float32)
    return np.stack([todos, mejor]), cs


@torch.no_grad()
def caract_cerradas(img):
    M, C = [], []
    for x in img:
        m, cs = mapas_cerradas(x); M.append(m); C.append(cs)
    p = F.adaptive_max_pool2d(torch.from_numpy(np.stack(M)), 8).flatten(1).numpy()
    return p, C


def main() -> int:
    global R_MIN, SUFIJO
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--rmin", type=float, default=0.5); ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--sufijo", default=""); a = ap.parse_args(); R_MIN, C2.RMAX, SUFIJO = a.rmin, a.rmax, a.sufijo
    torch.set_num_threads(2); t0 = time.time()
    DL.configurar(**json.loads((BOCETO / "resultados-delgadas.json").read_text())["elegida"])
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    XA = DB.caract(img, borde=lambda z: z.astype(np.float32))["bordes"]
    print(f"otros detectores: {XA.shape} en {time.time() - t0:.0f} s", flush=True)
    XB, cands = caract_cerradas(img)
    print(f"curvas cerradas: {XB.shape} en {time.time() - t0:.0f} s", flush=True)
    conjuntos = {"otros (delgado)": XA, "cerradas solas": XB, "otros + cerradas": np.concatenate([XA, XB], 1)}
    out = {"rmin": R_MIN, "rmax": C2.RMAX, "n": {"train": len(tr), "val": len(va), "ciega": len(ex)}, "caracteristicas": {k: int(v.shape[1]) for k, v in conjuntos.items()}, "resultados": {}}
    preds = {}
    for n, X in conjuntos.items():
        v, c = [], []
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem); pv, pe = pr(X[va]), pr(X[ex])
            v.append(float((pv == y[va]).mean())); c.append(float((pe == y[ex]).mean()))
            if sem == 0:
                preds[n] = (pv, pe)
        out["resultados"][n] = {"val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
                                "ciega": round(float(np.mean(c)), 4), "ciega_rango": [round(min(c), 4), round(max(c), 4)],
                                "por_digito_val_s0": [round(float((preds[n][0][y[va] == k] == k).mean()), 3) for k in range(10)]}
        print(n, out["resultados"][n], flush=True)
    pv0 = preds["otros + cerradas"][0]; pa0 = preds["otros (delgado)"][0]
    f = np.flatnonzero(pv0 != y[va]); out["fallos_val_s0"] = {"otros+cerradas": int(len(f)), "otros": int((pa0 != y[va]).sum()),
                                                            "arregla": int(((pa0 != y[va]) & (pv0 == y[va])).sum()), "rompe": int(((pa0 == y[va]) & (pv0 != y[va])).sum())}
    conf = {}
    for j in f:
        k = f"{y[va][j]}→{pv0[j]}"; conf[k] = conf.get(k, 0) + 1
    out["confusiones_val_s0"] = dict(sorted(conf.items(), key=lambda z: -z[1])[:12])
    out["indices_fallos_val_s0"] = [int(va[j]) for j in f]
    out["cerradas_por_clase_val"] = {str(k): round(float(np.mean([len(cands[i]) for i in va[y[va] == k]])), 2) for k in range(10)}
    (AQUI / "resultados" / f"13{SUFIJO}-digitos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("resultados", "fallos_val_s0", "confusiones_val_s0")}, indent=1, ensure_ascii=False), flush=True)

    # figura: los fallos de «otros + cerradas» en val, con las curvas cerradas que vio (cruz azul = la mejor, claras = las demás)
    orden = f[np.lexsort((pv0[f], y[va][f]))]; n_max = 72; cols = 12; filas = int(np.ceil(min(len(orden), n_max) / cols))
    fig, axs = plt.subplots(max(filas, 1), cols, figsize=(cols * 1.3, max(filas, 1) * 1.5 + 0.8)); axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k < min(len(orden), n_max):
            j = orden[k]; i = va[j]; cs = cands[i]
            ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            for q, c in enumerate(cs):
                ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=8 if q == 0 else 5, mew=1.6 if q == 0 else 1)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
            ax.set_title(f"{y[i]}→{pv0[j]} · idx {i}" + ("" if pa0[j] != y[va][j] else " (otros acertaba)"), color=EV.NAR, fontsize=6.5, pad=2)
    fig.suptitle(f"rizo-2 · 13{SUFIJO} · los {len(f)} dígitos de val (1617) que FALLA el compositor «otros + curvas cerradas» (semilla 0, acierto "
                 f"{out['resultados']['otros + cerradas']['val']:.4f}) · «real→predicho» · cruz azul = la curva cerrada mejor, claras = las demás"
                 + (f" · se enseñan {n_max}" if len(f) > n_max else ""), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(AQUI / "resultados" / f"13{SUFIJO}-digitos-fallos.png", dpi=95); plt.close(fig)

    # figura: un dígito de cada clase con sus curvas cerradas, para ver qué ve el compositor
    fig, axs = plt.subplots(3, 10, figsize=(14, 4.6))
    for k in range(10):
        for r, i in enumerate(va[np.flatnonzero(y[va] == k)[:3]]):
            ax = axs[r, k]; cs = cands[i]
            ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            for q, c in enumerate(cs):
                ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=8 if q == 0 else 5, mew=1.6 if q == 0 else 1)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{k} · {len(cs)} cerradas", fontsize=7, pad=2)
    fig.suptitle(f"rizo-2 · 13{SUFIJO} · lo que el detector de curvas cerradas (compuesto2, encierro ≥ {R_MIN}, rayos a {C2.RMAX:g} px) ve en tres dígitos de cada clase: cruz azul = la mejor", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / f"13{SUFIJO}-digitos-cerradas.png", dpi=95)
    print(f"→ resultados/13-digitos.json · 13-digitos-fallos.png · 13-digitos-cerradas.png · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
