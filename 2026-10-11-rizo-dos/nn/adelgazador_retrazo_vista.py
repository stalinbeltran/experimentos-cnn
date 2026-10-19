#!/usr/bin/env python3
"""rizo-2 § 125 — el adelgazador AFINADO con el feedback del re-trazo (§ 123) contra el de 64 px de § 122 (el dueño, 2026-10-19: «procede»).

`delgado-cnn/nn/afinar_retrazo.py` parte de `k3x3-g0.7x-64` y suma a la pérdida de los sintéticos la del RE-TRAZO sobre los dígitos
reales de train + extra. Aquí se miden los dos con las medidas de § 122 (nudos, lazos, 1l con ramas) y de § 123 (sin explicar,
inventado), y el CRITERIO —escrito en el README ANTES de lanzar— se lee SÓLO sobre val (1617), que el afinado no vio:

  dígitos con HUECO (≥ 4 px de núcleo sin re-trazo) ≤ 4 % · inventado ≤ la base · lazos por 0 ≥ 0,90 · 7 con algún nudo ≥ 85 %

⚠ El criterio de partida era «sin explicar < la base», y se cambió ANTES del entrenamiento largo: medido en las pruebas cortas, la
mediana de sin explicar del NÚCLEO (la tinta sin su borde de 1 px) es 0,0 % — el 7 % es todo BORDE (pluma de grosor fijo contra un
trazo de grosor variable) y no se arregla adelgazando. Los fallos de estructura son los HUECOS: el 6 % de val en la base.

    ../.venv/bin/python nn/adelgazador_retrazo_vista.py   → resultados/125-adelgazador-retrazo.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import grupos as GR                                                      # noqa: E402
import servidor as S                                                     # noqa: E402
from adelgazador_cruces_vista import nudos                               # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402
from retrazo import grosor, retrazar                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"; T4 = "uci-optdigits-orig-32px-trazo4-r20261018"
HUECO = 4                                                                 # px de núcleo sin re-trazo que ya son un fallo de estructura
VARIANTES = [("64 px (§ 122)", f"delgados-k3x3-g0.7x-64-{T4}-x2.npz"),
             ("64 px + re-trazo (§ 125)", f"delgados-k3x3-g0.7x-64-rt-{T4}-x2.npz")]


def medir(M, tin, y, ids, l1):
    se, inv, hu = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan)
    for i in ids:
        g = grosor(tin[i])
        if np.isnan(g):
            continue
        r = retrazar(M[i], g, 2) > 0.5
        se[i] = (tin[i] & ~r).sum() / max(tin[i].sum(), 1); inv[i] = (r & ~tin[i]).sum() / max(r.sum(), 1)
        lab, n = ndimage.label(ndimage.binary_erosion(tin[i]) & ~r); hu[i] = max([(lab == k).sum() for k in range(1, n + 1)] or [0])
    en = lambda c: [int(i) for i in ids if y[i] == c]                    # noqa: E731
    hueco = lambda js: float(100 * np.mean(hu[js][~np.isnan(hu[js])] >= HUECO))  # noqa: E731
    m = {"hueco": hueco(ids), "hueco_7": hueco(en(7)), "hueco_0": hueco(en(0)), "sin_explicar": float(100 * np.nanmedian(se[ids])), "inventado": float(100 * np.nanmedian(inv[ids])),
         "sin_explicar_7": float(100 * np.nanmedian(se[en(7)])), "inventado_1": float(100 * np.nanmedian(inv[en(1)]))}
    for c in (7, 4):
        m[f"nudo_{c}"] = float(100 * np.mean([nudos(M[i].astype(np.float32)) > 0 for i in en(c)]))
    for c in (0, 6, 8, 9):
        m[f"lazos_{c}"] = float(np.mean([len(CC.objetos(M[i].astype(np.float32))["lazos"]) for i in en(c)]))
    m["1l_ramas"] = float(100 * np.mean([len(CC.tramos(M[i].astype(np.float32))[0]) >= 3 for i in l1])) if l1 else float("nan")
    return m, hu


def main():
    d = S.Datos(); y = d.y; els = GR.elementos(d, "1", S.almacen_revisor())
    tin = np.load(exigir_dataset(T4) / "datos.npz")["imagenes"].astype(np.float32) > 0.5
    val = np.asarray(d.va); todos = np.arange(len(y))
    l1 = {e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"}
    Ms = {n: np.load(DELG / f)["M"] for n, f in VARIANTES}; res, se_val = {}, {}
    for n, _ in VARIANTES:
        res[n] = {}
        for parte, ids in (("val", val), ("todos", todos)):
            res[n][parte], se = medir(Ms[n], tin, y, ids, sorted(l1 & set(ids.tolist())))
            if parte == "val":
                se_val[n] = se
            print(n, parte, {k: round(v, 2) for k, v in res[n][parte].items()}, flush=True)
    b, r = res[VARIANTES[0][0]]["val"], res[VARIANTES[1][0]]["val"]
    crit = {"hueco <= 4": r["hueco"] <= 4, "inventado <= base": r["inventado"] <= b["inventado"],
            "lazos_0 >= 0,90": r["lazos_0"] >= 0.90, "nudo_7 >= 85": r["nudo_7"] >= 85}
    crit["SE CUMPLE"] = all(crit.values()); print("criterio:", crit, flush=True)

    # figura: arriba las medidas en val (con los números dentro); abajo los 7 y los 0 de val que MÁS cambian, retrazados con cada uno
    claves = [("hueco", "con HUECO %", "≤ 4"), ("hueco_7", "7 con hueco %", ""), ("sin_explicar", "sin explicar % (borde)", ""), ("inventado", "inventado %", "menos es mejor"),
              ("nudo_7", "7 con nudo %", "≥ 85"), ("nudo_4", "4 con nudo %", ""), ("lazos_0", "lazos por 0", "≥ 0,90"),
              ("lazos_8", "lazos por 8", "≈ 2"), ("1l_ramas", "1l con ramas %", "")]
    sev = val[[y[i] == 7 for i in val]]; cer = val[[y[i] == 0 for i in val]]
    d7 = np.nan_to_num(se_val[VARIANTES[0][0]][sev] - se_val[VARIANTES[1][0]][sev]); s7 = sev[np.argsort(-d7)][:8]
    l0 = lambda n, i: len(CC.objetos(Ms[n][i].astype(np.float32))["lazos"])  # noqa: E731
    c0 = [i for i in cer if l0(VARIANTES[0][0], i) != l0(VARIANTES[1][0], i)][:8]
    muestra = [int(i) for i in s7] + [int(i) for i in c0]
    fig = plt.figure(figsize=(max(18, 1.15 * len(muestra) + 1), 9.5), layout="constrained")
    arriba, abajo = fig.subfigures(2, 1, height_ratios=[3.2, 4.4])
    axs = arriba.subplots(1, len(claves)); cv = ["#999", "#1b7f3b"]
    for ax, (k, et, nota) in zip(axs, claves):
        v = [res[n]["val"][k] for n, _ in VARIANTES]; ax.bar([0, 1], v, color=cv, width=0.7)
        for j, x in enumerate(v):
            ax.text(j, x, f"{x:.2f}" if k.startswith("lazos") else f"{x:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["64 px", "+ re-trazo"], fontsize=9); ax.set_title(f"{et}\n{nota}", fontsize=10)
        ax.set_ylim(0, max(v) * 1.25 if max(v) > 0 else 1); ax.set_yticks([])
    arriba.suptitle(f"en VAL (1617, que el afinado no vio) · criterio {'SE CUMPLE' if crit['SE CUMPLE'] else 'NO se cumple'}: "
                    + " · ".join(f"{k} {'✓' if v else '✗'}" for k, v in crit.items() if k != "SE CUMPLE"), fontsize=11)
    if muestra:
        ax2 = abajo.subplots(len(VARIANTES), len(muestra), squeeze=False)
        for f_, (n, _) in enumerate(VARIANTES):
            for k, i in enumerate(muestra):
                rr = retrazar(Ms[n][i], grosor(tin[i]), 2) > 0.5; rgb = np.ones((32, 32, 3))
                rgb[tin[i] & rr] = (0.7, 0.7, 0.7); rgb[tin[i] & ~rr] = (0.85, 0.15, 0.15); rgb[rr & ~tin[i]] = (0.2, 0.45, 0.9)
                ax = ax2[f_, k]; ax.imshow(rgb, interpolation="nearest", extent=(0, 32, 32, 0)); ax.set_xticks([]); ax.set_yticks([])
                for t in CC.tramos(Ms[n][i].astype(np.float32))[0]:
                    p = np.asarray(t["pts"]) / 2; ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, color="k")
                ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_title(f"#{i} ({y[i]})", fontsize=8)
                if k == 0:
                    ax.set_ylabel(n, fontsize=9, fontweight="bold")
        abajo.suptitle("val: los 7 cuyo hueco MÁS baja, y los 0 que cambian de lazos · gris = explicado · ROJO = tinta sin "
                       "línea media · AZUL = trazo inventado · negro = la cresta", fontsize=10)
    out = AQUI / "resultados" / "125-adelgazador-retrazo.png"; fig.savefig(out, dpi=85)
    out.with_suffix(".json").write_text(json.dumps({"medidas": res, "criterio": crit, "muestra": muestra}, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
