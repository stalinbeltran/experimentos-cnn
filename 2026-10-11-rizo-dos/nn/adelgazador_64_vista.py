#!/usr/bin/env python3
"""rizo-2 § 122 — el adelgazador a 64 px (camino 1 de § 121; el dueño, 2026-10-18: «prueba 1, para generar otro dataset»).

`delgado-cnn`: los dígitos de MISMO TRAZO (§ 118) se amplían ×2 en gris (`entrenar_local.ampliar`, por áreas de píxel) y la red se
entrena con sintéticos que pasan por la MISMA ampliación (uno y dos trazos, gris σ 0,7) y cuya línea media de verdad se calcula ya a
64 px; 8 capas (alcance 17 px, porque a 64 px los trazos miden ~8 y los cruces ~16). `k3x3-g0.7x-64`.

Mismo CRITERIO que § 121, escrito antes: 1l ≤ 5 % con ramas; 7 con algún nudo ≥ 75 %; y la validación por RE-TRAZO de § 123 (que el
7 deje de tener la barra sin explicar). Las medidas que dependen de umbrales en px (entrantes) no se comparan a 64 px.

    ../.venv/bin/python nn/adelgazador_64_vista.py   → resultados/122-adelgazador-64.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import grupos as GR                                                      # noqa: E402
import servidor as S                                                     # noqa: E402
from adelgazador_cruces_vista import SIETES, nudos                         # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"; T4 = "uci-optdigits-orig-32px-trazo4-r20261018"
VARIANTES = [("hoy (NIST)", "delgados-k3x3.npz", None), ("gris 32 px (§ 119)", f"delgados-k3x3-g0.7-{T4}.npz", T4),
             ("gris 64 px (§ 122)", f"delgados-k3x3-g0.7x-64-{T4}-x2.npz", T4)]


def main():
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); y = d.y
    l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]
    tintas = {None: d.tinta.astype(np.float32), T4: np.load(exigir_dataset(T4) / "datos.npz")["imagenes"].astype(np.float32)}
    Ms = {n: np.load(DELG / f)["M"] for n, f, _ in VARIANTES}; res = {}
    for n, _, _ in VARIANTES:
        M = Ms[n]; r = {"1l_ramas": float(100 * np.mean([len(CC.tramos(M[i].astype(np.float32))[0]) >= 3 for i in l1]))}
        for c in (7, 4):
            r[f"nudo_{c}"] = float(100 * np.mean([nudos(M[i].astype(np.float32)) > 0 for i in np.flatnonzero(y == c)]))
        for c in (0, 6, 8, 9):
            ids = [int(i) for i in d.va if y[i] == c]; r[f"lazos_{c}"] = float(np.mean([len(CC.objetos(M[i].astype(np.float32))["lazos"]) for i in ids]))
        res[n] = r; print(n, {k: round(v, 2) for k, v in r.items()}, flush=True)
    r = res["gris 64 px (§ 122)"]; crit = {"1l_ramas_<=5": r["1l_ramas"] <= 5, "7_nudo_>=75": r["nudo_7"] >= 75}; print("criterio:", crit)
    nv, nm = len(VARIANTES), len(SIETES)
    fig = plt.figure(figsize=(1.6 * nm + 1, 2.0 * nv + 3.6), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[3, 2.0 * nv])
    a = arriba.subplots(); x = np.arange(nv); w = 0.27; cv = ["#999", "#c4b5fd", "#1b7f3b"]
    for j, (clave, et) in enumerate((("nudo_7", "7 con nudo"), ("nudo_4", "4 con nudo"), ("1l_ramas", "1l con ramas"))):
        vals = [res[n][clave] for n, _, _ in VARIANTES]; a.bar(x + (j - 1) * w, vals, w, label=et, color=["#c2410c", "#2a78d6", "#9333ea"][j])
        for k, v in enumerate(vals):
            a.text(k + (j - 1) * w, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    a.axhline(75, color="#c2410c", ls="--", lw=1); a.axhline(5, color="#9333ea", ls=":", lw=1)
    a.set_xticks(x); a.set_xticklabels([n for n, _, _ in VARIANTES]); a.set_ylim(0, 105); a.set_ylabel("%"); a.legend(fontsize=9)
    a.set_title("criterio: 7 con nudo ≥ 75 % (línea naranja) y 1l con ramas ≤ 5 % (línea morada)", fontsize=10)
    axs = abajo.subplots(nv, nm)
    for f_, (n, _, ds) in enumerate(VARIANTES):
        for k, i in enumerate(SIETES):
            ax = axs[f_, k]; m = Ms[n][i].astype(np.float32); s = m.shape[-1] // 32
            ax.imshow(tintas[ds][i], cmap="Greys", vmin=0, vmax=2.4, extent=(0, 32, 32, 0))
            for t in CC.tramos(m)[0]:
                p = np.asarray(t["pts"]) / s; ax.plot(p[:, 0], p[:, 1], "-", lw=1.6)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"#{i}", fontsize=8)
            if k == 0:
                ax.set_ylabel(n, fontsize=9, fontweight="bold")
    fig.suptitle("Adelgazador a 64 px (dígitos de mismo trazo ampliados ×2): ¿vuelve la barra del 7? (un color por tramo)", fontsize=12)
    out = AQUI / "resultados" / "122-adelgazador-64.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({"medidas": res, "criterio": crit}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
