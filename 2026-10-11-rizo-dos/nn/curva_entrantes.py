#!/usr/bin/env python3
"""rizo-2 — CURVA DE DESPLAZAMIENTOS del compositor sólo-cresta (regla del dueño, 2026-10-09): acierto en val con el mapa
delgado M movido d = −4…+4 px, horizontal y vertical por separado (sólo los dígitos que no pierden tinta, como en
resistencia.py), y el % que cambia de lectura respecto de d = 0. Tres compositores de una sola pasada de caract():
  conc 45°  = rectas + arcos + golpes + lazos + concavidades de arco (§ 55, 152)
  entr 45°  = ídem con entrantes a 4 direcciones de 45° (§ 58, 152)
  conos 60° = ídem con entrantes en 4 conos de ±60°, presencia 3×3 (§ 66, 144)
Mismo protocolo: 180 de train × 13 copias movidas, MLP 256, 3 semillas.

    ../.venv/bin/python nn/curva_entrantes.py [--entrada ../2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz]
        → resultados/68-curva-entrantes.json · 68-curva-entrantes.png   (~10 min: 1617 × 17 copias de val)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import digitos as D                                                 # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

DS = R.DS
B = CC.bloques()
COMPOSITORES = {
    "conc 45° (§ 55, 152)": ["rectas", "arcos", "golpes", "lazos", "concavidades"],
    "entrantes 45° (§ 58, 152)": ["rectas", "arcos", "golpes", "lazos", "entrantes"],
    "conos 60° presencia (§ 66, 144)": ["rectas", "arcos", "golpes", "lazos", "entrantes60"],
}


def cortar(F, bloques):
    return np.concatenate([F[:, B[k][0]:B[k][1]] for k in bloques], 1)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", default=str(AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz")); a = ap.parse_args()
    torch.set_num_threads(2); t0 = time.time()
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); y, part = d["etiquetas"].astype(int), d["particion"]
    M = np.load(a.entrada)["M"].astype(np.float32); tinta = (d["imagenes"] > 0).astype(np.uint8)
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"); yva = y[va]
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    Ftr = np.concatenate([CC.caract(R.mover(M[tr], *m)) for m in aum]); Ytr = np.concatenate([y[tr]] * len(aum))
    print(f"train listo: {Ftr.shape} ({time.time() - t0:.0f} s)", flush=True)
    Fva, rec = {}, {}
    for eje in ("x", "y"):
        for dd in DS:
            m = (dd, 0) if eje == "x" else (0, dd)
            if (eje, dd) == ("y", 0):
                Fva[(eje, dd)], rec[(eje, dd)] = Fva[("x", 0)], rec[("x", 0)]; continue
            Fva[(eje, dd)] = CC.caract(R.mover(M[va], *m))
            Im = R.mover(tinta[va], *m); rec[(eje, dd)] = np.array([p.sum() != q.sum() for p, q in zip(Im, tinta[va])])
            print(f"val movida {eje} {dd:+d} ({time.time() - t0:.0f} s)", flush=True)
    out = {"d": DS, "recortados_pc": {eje: [round(100 * float(rec[(eje, dd)].mean()), 1) for dd in DS] for eje in ("x", "y")}, "compositores": {}}
    for n, bl in COMPOSITORES.items():
        X = cortar(Ftr, bl); res = {"x": {"acierto": [], "cambian": []}, "y": {"acierto": [], "cambian": []}}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Ytr, sem); pred = lambda eje, dd: RA.logits(red, mu, sd, cortar(Fva[(eje, dd)], bl)).argmax(1)
            p0 = pred("x", 0)
            for eje in ("x", "y"):
                acc, cam = [], []
                for dd in DS:
                    p = pred(eje, dd); ok = ~rec[(eje, dd)]
                    acc.append(float((p[ok] == yva[ok]).mean()) if ok.any() else np.nan); cam.append(float((p[ok] != p0[ok]).mean()) if ok.any() else np.nan)
                res[eje]["acierto"].append(acc); res[eje]["cambian"].append(cam)
        out["compositores"][n] = {"n": int(X.shape[1])}
        for eje in ("x", "y"):
            out["compositores"][n][eje] = {"acierto": np.nanmean(res[eje]["acierto"], 0).round(4).tolist(), "cambian_pc": (100 * np.nanmean(res[eje]["cambian"], 0)).round(1).tolist()}
        print(f"{n:34s} x {out['compositores'][n]['x']['acierto']} · y {out['compositores'][n]['y']['acierto']} · cambian x {out['compositores'][n]['x']['cambian_pc']} ({time.time() - t0:.0f} s)", flush=True)
    out["segundos"] = round(time.time() - t0)
    (AQUI / "resultados" / "68-curva-entrantes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7)); cols = ["#7a7a7a", "#2a78d6", "#c2410c"]
    for j, eje in enumerate(("x", "y")):
        for (n, m), col in zip(out["compositores"].items(), cols):
            axs[0, j].plot(DS, m[eje]["acierto"], "o-", color=col, label=n, ms=4); axs[1, j].plot(DS, m[eje]["cambian_pc"], "o-", color=col, ms=4)
        ax2 = axs[0, j].twinx(); ax2.bar(DS, out["recortados_pc"][eje], 0.5, color="#e5e3df", zorder=0); ax2.set_ylim(0, 100); ax2.set_ylabel("% recortados (gris)", fontsize=8)
        axs[0, j].set_zorder(ax2.get_zorder() + 1); axs[0, j].patch.set_visible(False)
        axs[0, j].set_title(f"acierto en val contra d ({'horizontal' if eje == 'x' else 'vertical'}), sin recortados", fontsize=9); axs[0, j].set_ylim(0.6, 1.0); axs[0, j].grid(alpha=0.3)
        axs[1, j].set_title("% que cambia de lectura respecto de d = 0", fontsize=9); axs[1, j].set_xlabel("d (px)"); axs[1, j].set_ylim(0, 60); axs[1, j].grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7.5, loc="lower center")
    fig.suptitle("rizo-2 · 68 · resistencia a los DESPLAZAMIENTOS del compositor sólo-cresta (entrada: mapa delgado movido; train con 13 copias ±1–3 px; MLP 256; 3 semillas)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / "68-curva-entrantes.png", dpi=110)
    print("→ resultados/68-curva-entrantes.json · 68-curva-entrantes.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
