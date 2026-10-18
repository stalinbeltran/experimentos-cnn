#!/usr/bin/env python3
"""rizo-2 § 121 — el adelgazador gris reentrenado con DOS TRAZOS (T y X): ¿recupera la barra del 7 sin devolver las ramas de los 1l?
(el dueño, 2026-10-18: «sí» a reentrenarlo, § 120). `delgado-cnn`: `entrenar_local.py --n 600 --suavizar 0.7 --dos 0.5 --nombre
k3x3-g0.7x` (la mitad de los ejemplos con un segundo trazo anclado en el primero; la mitad de esos, corto como una barra).

CRITERIO, escrito ANTES de medir (README § 121): se acepta si (1) los 1l siguen limpios: ≤ 5 % con ramas y ≤ 10 % con entrantes;
(2) los 7 recuperan la barra: ≥ 75 % con algún nudo (hoy 87 %, el gris de § 119 45 %); (3) a la vista, los 12 sietes que la
perdían la muestran. Sólo entonces se genera y se publica el dataset de 32 px (§ 120).

    ../.venv/bin/python nn/adelgazador_cruces_vista.py   → resultados/121-adelgazador-cruces.{png,json}
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
import suavizado_vista as SV                                             # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DS = "uci-optdigits-orig-32px-trazo4-r20261018"
VARIANTES = [("hoy", "delgados-k3x3.npz", None), ("gris (§ 119)", f"delgados-k3x3-g0.7-{DS}.npz", DS),
             ("gris + DOS TRAZOS", f"delgados-k3x3-g0.7x-{DS}.npz", DS)]
SIETES = [17, 44, 112, 118, 137, 147, 182, 211, 216, 222, 236, 263, 1552]


def nudos(m) -> int:
    tr, _ = CC.tramos(m); c = {}
    for t in tr:
        for r in (t["a"], t["b"]):
            if r > 0:
                c[r] = c.get(r, 0) + 1
    return sum(v >= 3 for v in c.values())


def main():
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); y = d.y
    l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]; va = [int(i) for i in d.va]
    grupos = {"1l": l1} | {str(c): [i for i in va if y[i] == c] for c in range(10) if c != 1}
    tintas = {None: d.tinta.astype(np.float32), DS: np.load(exigir_dataset(DS) / "datos.npz")["imagenes"].astype(np.float32)}
    Ms = {n: np.load(SV.DELG / f)["M"] for n, f, _ in VARIANTES}; res = {}
    for n, _, _ in VARIANTES:
        M = Ms[n]; r = {g: SV.medir(M, ids) for g, ids in grupos.items()}
        for c in (7, 4, 2, 5):
            ids = np.flatnonzero(y == c); r[f"nudo_{c}"] = float(100 * np.mean([nudos(M[i].astype(np.float32)) > 0 for i in ids]))
        res[n] = r
        print(f"{n:20s} 1l: ramas {r['1l']['ramificada']:4.0f} % · entrantes {r['1l']['con_entrantes']:4.0f} % | nudo 7 {r['nudo_7']:.0f} % · 4 {r['nudo_4']:.0f} % · 2 {r['nudo_2']:.0f} % · 5 {r['nudo_5']:.0f} %"
              f" | lazos 0 {r['0']['lazos_por_digito']:.2f} 6 {r['6']['lazos_por_digito']:.2f} 8 {r['8']['lazos_por_digito']:.2f} 9 {r['9']['lazos_por_digito']:.2f}"
              f" | entr/díg 7 {r['7']['entrantes_por_digito']:.1f} 3 {r['3']['entrantes_por_digito']:.1f} 4 {r['4']['entrantes_por_digito']:.1f}")
    n_ = "gris + DOS TRAZOS"; r = res[n_]
    crit = {"1l_ramas_<=5": r["1l"]["ramificada"] <= 5, "1l_entrantes_<=10": r["1l"]["con_entrantes"] <= 10, "7_nudo_>=75": r["nudo_7"] >= 75}
    print("criterio:", crit)

    nv, nm = len(VARIANTES), len(SIETES)
    fig = plt.figure(figsize=(1.6 * nm + 1, 2.0 * nv + 4.2), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[3.2, 2.0 * nv])
    a, b = arriba.subplots(1, 2); x = np.arange(nv); cv = ["#999", "#c4b5fd", "#9333ea"]
    for k, (n, _, _) in enumerate(VARIANTES):
        a.bar(k - 0.2, res[n]["nudo_7"], 0.4, color=cv[k]); a.bar(k + 0.2, res[n]["nudo_4"], 0.4, color=cv[k], hatch="//", ec="w")
        a.text(k - 0.2, res[n]["nudo_7"] + 1.5, f"{res[n]['nudo_7']:.0f}", ha="center", fontweight="bold"); a.text(k + 0.2, res[n]["nudo_4"] + 1.5, f"{res[n]['nudo_4']:.0f}", ha="center", fontsize=9)
    a.axhline(75, color="#c2410c", ls="--", lw=1); a.text(2.45, 76, "criterio 7: ≥ 75 %", color="#c2410c", fontsize=8, ha="right")
    a.set_xticks(x); a.set_xticklabels([n for n, _, _ in VARIANTES]); a.set_ylim(0, 105); a.set_ylabel("% con algún nudo (≥ 3 tramos)")
    a.set_title("¿VUELVE la barra? 7 (lleno) y 4 (rayado) con algún nudo", fontsize=10)
    for k, (n, _, _) in enumerate(VARIANTES):
        b.bar(k - 0.2, res[n]["1l"]["ramificada"], 0.4, color=cv[k]); b.bar(k + 0.2, res[n]["1l"]["con_entrantes"], 0.4, color=cv[k], hatch="//", ec="w")
        b.text(k - 0.2, res[n]["1l"]["ramificada"] + 1.5, f"{res[n]['1l']['ramificada']:.0f}", ha="center", fontweight="bold"); b.text(k + 0.2, res[n]["1l"]["con_entrantes"] + 1.5, f"{res[n]['1l']['con_entrantes']:.0f}", ha="center", fontsize=9)
    b.axhline(5, color="#c2410c", ls="--", lw=1); b.axhline(10, color="#c2410c", ls=":", lw=1)
    b.set_xticks(x); b.set_xticklabels([n for n, _, _ in VARIANTES]); b.set_ylim(0, 65); b.set_ylabel("% de los 264 1l")
    b.set_title("¿y los 1l siguen limpios? con ramas (lleno) y con entrantes (rayado); criterio ≤ 5 % y ≤ 10 %", fontsize=10)
    axs = abajo.subplots(nv, nm)
    for f_, (n, _, ds) in enumerate(VARIANTES):
        for k, i in enumerate(SIETES):
            ax = axs[f_, k]; m = Ms[n][i].astype(np.float32); ax.imshow(tintas[ds][i], cmap="Greys", vmin=0, vmax=2.4, extent=(0, 32, 32, 0))
            for t in CC.tramos(m)[0]:
                p = np.asarray(t["pts"]); ax.plot(p[:, 0], p[:, 1], "-", lw=1.6)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"#{i}", fontsize=8)
            if k == 0:
                ax.set_ylabel(n, fontsize=9, fontweight="bold")
    fig.suptitle("Adelgazador gris reentrenado con DOS TRAZOS (T y X): los 7 que perdían la barra en § 119–120 (un color por tramo)", fontsize=12)
    out = AQUI / "resultados" / "121-adelgazador-cruces.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({"medidas": res, "criterio": crit, "sietes": SIETES}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
