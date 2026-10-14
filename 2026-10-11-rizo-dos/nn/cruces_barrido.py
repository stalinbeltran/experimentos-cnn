#!/usr/bin/env python3
"""rizo-2 — BARRIDO de los parámetros del detector de CRUCES (el dueño, 2026-10-15: «los parámetros los vamos a poner a prueba
experimentalmente»). Compositor = el mejor sólo-cresta (§ 66: rectas + arcos + golpes + lazos + entrantes en conos 60°, 144)
+ el bloque de cruces (3×3 de presencia + cuenta, 10) = 154. Mismo protocolo: 180 × 13 copias movidas, MLP 256, 3 semillas,
val (1617) y a ciegas (3823). Los TRAMOS de cada dígito se calculan UNA vez y las cruces se recomputan por combinación.

Rejilla: brazo mínimo {1, 2, 3, 4, 6} px × puente máximo {0, 2, 3, 4, 6} px × colinealidad {30, 40, 50}° = 75 combinaciones,
más la base sin cruces. CRITERIO, escrito antes de mirar: gana la combinación de mejor val medio (3 semillas); se confirma
a ciegas. Si ninguna supera la base en val por más que el rango entre semillas (~0,002), las cruces no aportan y se dice.

    ../.venv/bin/python nn/cruces_barrido.py [--entrada …]   → resultados/70-cruces-barrido.json · 70-cruces-barrido.png  (~15 min)
"""
from __future__ import annotations

import argparse
import itertools
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

BRAZOS = (1, 2, 3, 4, 6); PUENTES = (0, 2, 3, 4, 6); ANGS = (30, 40, 50)
BASE = ["rectas", "arcos", "golpes", "lazos", "entrantes60"]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", default=str(AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz")); a = ap.parse_args()
    torch.set_num_threads(2); t0 = time.time()
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); y, part = d["etiquetas"].astype(int), d["particion"]
    M = np.load(a.entrada)["M"].astype(np.float32)
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    conjuntos = {"tr": np.concatenate([R.mover(M[tr], *m) for m in aum]), "va": M[va], "ex": M[ex]}; Y = {"tr": np.concatenate([y[tr]] * len(aum)), "va": y[va], "ex": y[ex]}
    B = CC.bloques(); base, tramos = {}, {}
    for k, X in conjuntos.items():
        F = CC.caract(X); base[k] = np.concatenate([F[:, B[b][0]:B[b][1]] for b in BASE], 1)
        tramos[k] = [CC.tramos(m)[0] for m in X]
        print(f"{k}: base {base[k].shape}, tramos listos ({time.time() - t0:.0f} s)", flush=True)

    def correr(feats):
        r = {"val": [], "ciega": []}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(feats["tr"], Y["tr"], sem)
            r["val"].append(float((RA.logits(red, mu, sd, feats["va"]).argmax(1) == Y["va"]).mean())); r["ciega"].append(float((RA.logits(red, mu, sd, feats["ex"]).argmax(1) == Y["ex"]).mean()))
        return {"val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4)}
    out = {"base": correr(base), "n_base": int(base["tr"].shape[1]), "rejilla": []}
    print(f"base (sin cruces): {out['base']} ({time.time() - t0:.0f} s)", flush=True)
    for brazo, puente, ang in itertools.product(BRAZOS, PUENTES, ANGS):
        feats = {}
        cuenta_va = []
        for k in conjuntos:
            bl = [CC.bloque_cruces(CC.cruces(t, brazo, puente, ang)) for t in tramos[k]]
            feats[k] = np.concatenate([base[k], np.array(bl, np.float32)], 1)
            if k == "va":
                cuenta_va = [b[-1] for b in bl]
        r = correr(feats); r.update({"brazo": brazo, "puente": puente, "ang": ang, "cruces_por_digito_val": round(float(np.mean(cuenta_va)), 3), "pc_val_con_cruz": round(100 * float(np.mean(np.array(cuenta_va) > 0)), 1)})
        out["rejilla"].append(r); print(f"brazo {brazo} puente {puente} ang {ang}: val {r['val']} ciega {r['ciega']} · {r['pc_val_con_cruz']} % con cruz ({time.time() - t0:.0f} s)", flush=True)
    mejor = max(out["rejilla"], key=lambda r: r["val"]); out["mejor_val"] = mejor; out["segundos"] = round(time.time() - t0)
    (AQUI / "resultados" / "70-cruces-barrido.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    # figura: val y ciega por brazo (eje x), una línea por puente, un panel por ángulo
    fig, axs = plt.subplots(2, len(ANGS), figsize=(4.2 * len(ANGS), 6.5), sharey="row")
    for j, ang in enumerate(ANGS):
        for puente, col in zip(PUENTES, ("#7a7a7a", "#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed")):
            rs = [r for r in out["rejilla"] if r["ang"] == ang and r["puente"] == puente]
            axs[0, j].plot([r["brazo"] for r in rs], [r["val"] for r in rs], "o-", color=col, ms=4, label=f"puente ≤ {puente} px")
            axs[1, j].plot([r["brazo"] for r in rs], [r["ciega"] for r in rs], "o-", color=col, ms=4)
        for i, (k, v) in enumerate((("val", out["base"]["val"]), ("ciega", out["base"]["ciega"]))):
            axs[i, j].axhline(v, color="k", ls="--", lw=1, label="sin cruces" if i == 0 else None); axs[i, j].grid(alpha=0.3); axs[i, j].set_xlabel("brazo mínimo (px)")
        axs[0, j].set_title(f"colinealidad ≤ {ang}° · val", fontsize=9); axs[1, j].set_title(f"colinealidad ≤ {ang}° · a ciegas", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"rizo-2 · 70 · barrido de los parámetros de CRUCES sobre el sólo-cresta de 144 (+10 de cruces): base sin cruces val {out['base']['val']} / ciega {out['base']['ciega']} · "
                 f"mejor val: brazo {mejor['brazo']} puente {mejor['puente']} ang {mejor['ang']} → {mejor['val']} / {mejor['ciega']}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(AQUI / "resultados" / "70-cruces-barrido.png", dpi=110)
    print("→ resultados/70-cruces-barrido.json · 70-cruces-barrido.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
