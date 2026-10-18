#!/usr/bin/env python3
"""rizo-2 § 119 — un ADELGAZADOR entrenado con trazos GRISES suavizados, aplicado al dataset de MISMO TRAZO (§ 118) (el dueño,
2026-10-18: «entrena el adelgazador con trazos grises suavizados. ¿Se puede?»).

El adelgazador `k3x3-g0.7` de `delgado-cnn`: el k3x3 de siempre (300 trazos sintéticos, 6 capas, 400 épocas) con la tinta sintética
desenfocada con σ 0,7 en gris, porque ése es el desenfoque del borde medido en el dataset de § 118 (σ ≈ 0,77 px; un sintético con
σ 0,7 mide 0,76, 2026-10-18). Se compara, con los objetos del revisor, en tres variantes:

  · hoy: el dataset de NIST (binario) con el adelgazador de siempre (k3x3, binario);
  · mismo trazo + k3x3: el dataset de § 118 con el adelgazador binario (fuera de lo que vio al entrenar: gris);
  · mismo trazo + k3x3-g0.7: el dataset de § 118 con el adelgazador nuevo.

Los delgados los produce `delgado-cnn/nn/exportar_delgados.py --dataset …` y aquí se leen como dato.

    ../.venv/bin/python nn/adelgazador_trazo_vista.py   → resultados/119-adelgazador-trazo.{png,json}
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
import suavizado_vista as SV                                             # noqa: E402  (las medidas de § 113)
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Polygon                                    # noqa: E402

DS = "uci-optdigits-orig-32px-trazo4-r20261018"
VARIANTES = [("hoy: NIST binario + adelgazador binario", "delgados-k3x3.npz", None),
             ("mismo trazo + adelgazador binario", f"delgados-k3x3-{DS}.npz", DS),
             ("mismo trazo + adelgazador NUEVO (gris σ 0,7)", f"delgados-k3x3-g0.7-{DS}.npz", DS)]
COL = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]
MUESTRA = [736, 2849, 4661, 5538, 3517, 1668, 1552]                      # 1l de § 112–115 y el 7 de § 112


def main():
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor())
    l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]; a1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) != "1l"]
    va = [int(i) for i in d.va]; grupos = {"1l": l1, "1a": a1} | {str(c): [i for i in va if d.y[i] == c] for c in range(10) if c != 1}
    otros = [int(np.array(va)[d.y[va] == c][0]) for c in (0, 3, 4, 8, 9)]; muestra = MUESTRA + otros
    tintas = {None: d.tinta.astype(np.float32), DS: np.load(exigir_dataset(DS) / "datos.npz")["imagenes"].astype(np.float32)}
    Ms = {n: np.load(SV.DELG / f)["M"] for n, f, _ in VARIANTES}
    res = {n: {g: SV.medir(Ms[n], ids) for g, ids in grupos.items()} for n, _, _ in VARIANTES}
    for n, _, _ in VARIANTES:
        r = res[n]; print(f"{n:48s} 1l: entr {r['1l']['con_entrantes']:4.0f} % · ramif {r['1l']['ramificada']:4.0f} % · lazos/1l {r['1l']['lazos_por_digito']:.2f}"
                          f" · rotos 1l {r['1l']['rotos']:.0f} % | lazos 0 {r['0']['lazos_por_digito']:.2f} 6 {r['6']['lazos_por_digito']:.2f} 8 {r['8']['lazos_por_digito']:.2f}"
                          f" 9 {r['9']['lazos_por_digito']:.2f} · entr/díg 7 {r['7']['entrantes_por_digito']:.1f} 3 {r['3']['entrantes_por_digito']:.1f} · rotos val {np.mean([r[g]['rotos'] for g in grupos if g not in ('1l', '1a')]):.1f} %")

    nv, nm = len(VARIANTES), len(muestra)
    fig = plt.figure(figsize=(1.75 * nm + 1, 2.0 * nv + 4.6), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[3.4, 2.0 * nv])
    a, b = arriba.subplots(1, 2, width_ratios=[1, 1.4]); x = np.arange(nv); cv = ["#999", "#c4b5fd", "#9333ea"]
    for k, (n, _, _) in enumerate(VARIANTES):
        r = res[n]["1l"]; a.bar(k - 0.2, r["con_entrantes"], 0.4, color=cv[k]); a.bar(k + 0.2, r["ramificada"], 0.4, color=cv[k], hatch="//", ec="w")
        a.text(k - 0.2, r["con_entrantes"] + 1.5, f"{r['con_entrantes']:.0f}", ha="center", fontsize=10, fontweight="bold")
        a.text(k + 0.2, r["ramificada"] + 1.5, f"{r['ramificada']:.0f}", ha="center", fontsize=9)
    a.set_xticks(x); a.set_xticklabels(["hoy", "mismo trazo\n+ binario", "mismo trazo\n+ NUEVO"]); a.set_ylim(0, 70); a.set_ylabel("% de los 264 1l")
    a.set_title("LA MEDIDA: 1l con entrantes (lleno) y con ramas (rayado)", fontsize=10)
    gs = ["1a", "0", "2", "3", "4", "5", "6", "7", "8", "9"]; w = 0.8 / nv
    for k, (n, _, _) in enumerate(VARIANTES):
        b.bar(np.arange(len(gs)) + (k - 1) * w, [res[n][g]["lazos_por_digito"] for g in gs], w, color=cv[k], label=n)
    b.set_xticks(range(len(gs))); b.set_xticklabels(gs); b.set_ylabel("lazos por dígito"); b.legend(fontsize=8)
    b.set_title("LO QUE CUESTA: lazos por dígito (¿se cierran o se inventan?)", fontsize=10)
    axs = abajo.subplots(nv, nm)
    for f_, (n, _, ds) in enumerate(VARIANTES):
        for k, i in enumerate(muestra):
            ax = axs[f_, k]; m = Ms[n][i].astype(np.float32); ax.imshow(tintas[ds][i], cmap="Greys", vmin=0, vmax=2.2, extent=(0, 32, 32, 0))
            for e in CC.objetos(m)["entrantes"]:
                c = CC.cono(e); ax.add_patch(Polygon(np.asarray(e["pts"]), closed=True, fc=COL[c], ec=COL[c], alpha=0.55, lw=1))
            for t in CC.tramos(m)[0]:
                p = np.asarray(t["pts"]); ax.plot(p[:, 0], p[:, 1], "-", color="#111", lw=1.4)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"#{i} ({d.y[i]})", fontsize=8)
            if k == 0:
                ax.set_ylabel(n.replace(" + ", "\n+ "), fontsize=8, fontweight="bold")
    fig.suptitle("Adelgazador entrenado con trazos GRISES suavizados (σ 0,7), sobre el dataset de MISMO TRAZO (§ 118)\n"
                 "fondo = tinta · negro = cresta · color = entrante: naranja der · azul izq · verde abajo · morado arriba", fontsize=12)
    out = AQUI / "resultados" / "119-adelgazador-trazo.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({"variantes": [[n, f] for n, f, _ in VARIANTES], "muestra": muestra, "medidas": res}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
