#!/usr/bin/env python3
"""rizo-2 § 126 — el adelgazador afinado con re-trazo Y trazos CASI CERRADOS (`delgado-cnn/sinteticos.casi_cerrado`; el dueño,
2026-10-19: «procede») contra el de § 125, que es la misma receta sin ellos.

Dos medidas, y el CRITERIO va escrito en el README antes del entrenamiento:

  1. En 200 trazos casi cerrados NUNCA vistos (semilla 606): ¿conserva la cresta (esqueleto de M > 0,4) la COLA desviada?
     Sólo cuenta la parte de la cola a > 2 px (de 32) del resto del trazo (más cerca, la cresta del lazo la «cubre» aunque se haya
     borrado), y sólo los trazos con ≥ 4 puntos así. Un punto está cubierto si hay cresta a ≤ 1 px (de 32). Por trazo: % de la cola cubierta; una cola
     está CONSERVADA si se cubre ≥ 80 %. Y para que no se gane inventando: % de la cresta a > 1 px de la verdad (sobra).
  2. En los dígitos REALES de val, las medidas de § 125 (`adelgazador_retrazo_vista.medir`).

    ../.venv/bin/python nn/adelgazador_casi_vista.py [--sin-reales]   → resultados/126-adelgazador-casi.{png,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]; DC = AQUI.parent / "2026-10-15-delgado-cnn"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(DC / "nn"))
from entrenar_local import SoloKernels, conjunto_casi                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

T4 = "uci-optdigits-orig-32px-trazo4-r20261018"
VARIANTES = [("64 px (§ 122)", "k3x3-g0.7x-64"), ("+ re-trazo (§ 125)", "k3x3-g0.7x-64-rt"), ("+ re-trazo + casi (§ 126)", "k3x3-g0.7x-64-rtc")]
TOL, CONSERVADA = 2.0, 0.8                                                # 1 px de 32 = 2 de 64 · cola conservada si ≥ 80 % cubierta
SEPARADA = 2 * TOL      # sólo cuenta la cola a > 2 px (de 32) del resto del trazo: más cerca, la cresta del LAZO la «cubre» y una
                        # cola borrada contaba como conservada (1.ª versión: 82 % de colas «conservadas» con el adelgazador de 64 px)


def red(nombre):
    p = DC / "nn" / "pesos" / nombre; cfg = json.loads((p / "config.json").read_text())
    r = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); r.load_state_dict(torch.load(p / "best.pt")); r.eval(); return r


def medir_casi(M, cs):
    cola, sobra = [], []
    for m, c in zip(M, cs):
        cr = skeletonize(m > 0.4); dist = ndimage.distance_transform_edt(~cr)          # distancia (px de 64) a la cresta
        p = c["puntos"] * 2; ij = np.clip(np.floor(p).astype(int), 0, 63)
        d = dist[ij[:, 1], ij[:, 0]]
        resto = np.zeros((64, 64), bool); resto[ij[~c["cola"], 1], ij[~c["cola"], 0]] = True
        sep = ndimage.distance_transform_edt(~resto)[ij[:, 1], ij[:, 0]]            # distancia de cada punto al RESTO del trazo
        vis = c["cola"] & (sep > SEPARADA)                                           # sólo la cola que se DISTINGUE del lazo
        cola.append(float(np.mean(d[vis] <= TOL)) if vis.sum() >= 4 else np.nan)
        verdad = np.zeros((64, 64), bool); verdad[ij[:, 1], ij[:, 0]] = True; dv = ndimage.distance_transform_edt(~verdad)
        sobra.append(float(np.mean(dv[cr] > TOL)) if cr.any() else np.nan)
    cola, sobra = np.array(cola), np.array(sobra)
    ok = ~np.isnan(cola)
    return {"trazos_con_cola_visible": int(ok.sum()), "colas_conservadas": float(100 * np.mean(cola[ok] >= CONSERVADA)), "cola_cubierta_media": float(100 * np.nanmean(cola)),
            "cresta_que_sobra": float(100 * np.nanmedian(sobra))}, cola


def main():
    X, _, cs = conjunto_casi(200, 606, 0.8, 0.7, 2)
    vs = [(n, p) for n, p in VARIANTES if (DC / "nn" / "pesos" / p / "best.pt").is_file()]
    Ms, res, colas = {}, {}, {}
    for n, p in vs:
        with torch.no_grad():
            Ms[n] = red(p)(torch.from_numpy(X)[:, None])[:, 0].numpy()
        res[n] = {}; res[n]["casi"], colas[n] = medir_casi(Ms[n], cs); print(n, "casi", {k: round(v, 1) for k, v in res[n]["casi"].items()}, flush=True)
    if "--sin-reales" not in sys.argv:
        import servidor as S
        import grupos as GR
        from adelgazador_retrazo_vista import medir
        from expcnn import exigir_dataset
        d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); val = np.asarray(d.va)
        tin = np.load(exigir_dataset(T4) / "datos.npz")["imagenes"].astype(np.float32) > 0.5
        l1 = sorted({e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"} & set(val.tolist()))
        for n, p in vs:
            M = np.load(DC / "resultados" / f"delgados-{p}-{T4}-x2.npz")["M"]
            res[n]["val"], _ = medir(M, tin, d.y, val, l1); print(n, "val", {k: round(v, 2) for k, v in res[n]["val"].items()}, flush=True)
    crit = None
    if len(vs) == 3 and "val" in res[vs[2][0]]:
        a, b = res[vs[1][0]], res[vs[2][0]]
        crit = {"colas conservadas >= 90 %": b["casi"]["colas_conservadas"] >= 90,
                "cresta que sobra <= § 125 + 2": b["casi"]["cresta_que_sobra"] <= a["casi"]["cresta_que_sobra"] + 2,
                "val: hueco <= 4,33": b["val"]["hueco"] <= 4.33 + 1e-9, "val: lazos por 0 >= 0,90": b["val"]["lazos_0"] >= 0.90,
                "val: 7 con nudo >= 85": b["val"]["nudo_7"] >= 85, "val: inventado <= 9,25": b["val"]["inventado"] <= 9.25 + 1e-9}
        crit["SE CUMPLE"] = all(crit.values()); print("criterio:", crit, flush=True)

    # figura: arriba las medidas; abajo 12 casi cerrados (los que más cambian entre los dos últimos), verdad + cresta de cada variante
    nv = len(vs); orden = np.argsort(-(np.nan_to_num(colas[vs[-1][0]]) - np.nan_to_num(colas[vs[-2][0]] if nv > 1 else 0)))[:12]
    fig = plt.figure(figsize=(18, 3.2 + 1.9 * nv), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[3.2, 1.9 * nv])
    claves = [("casi", "colas_conservadas", "casi: colas conservadas %"), ("casi", "cresta_que_sobra", "casi: cresta que sobra %")]
    if "val" in res[vs[0][0]]:
        claves += [("val", "hueco", "val: con hueco %"), ("val", "lazos_0", "val: lazos por 0"), ("val", "nudo_7", "val: 7 con nudo %"),
                   ("val", "inventado", "val: inventado %"), ("val", "1l_ramas", "val: 1l con ramas %")]
    axs = arriba.subplots(1, len(claves)); cv = ["#999", "#86c99a", "#1b7f3b"][:nv]
    for ax, (g, k, et) in zip(np.atleast_1d(axs), claves):
        v = [res[n][g][k] for n, _ in vs]; ax.bar(range(nv), v, color=cv)
        for j, x in enumerate(v):
            ax.text(j, x, f"{x:.2f}" if k.startswith("lazos") else f"{x:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(nv)); ax.set_xticklabels(["64", "+rt", "+rt+casi"][:nv], fontsize=8); ax.set_yticks([]); ax.set_title(et, fontsize=9)
        ax.set_ylim(0, max(v) * 1.25 if max(v) > 0 else 1)
    arriba.suptitle("criterio: " + ("—" if crit is None else ("SE CUMPLE · " if crit["SE CUMPLE"] else "NO se cumple · ")
                    + " · ".join(f"{k} {'✓' if v else '✗'}" for k, v in crit.items() if k != "SE CUMPLE")), fontsize=10)
    ax2 = abajo.subplots(nv, len(orden), squeeze=False)
    for f_, (n, _) in enumerate(vs):
        for j, i in enumerate(orden):
            ax = ax2[f_, j]; c = cs[i]; p = c["puntos"]
            ax.imshow(X[i], cmap="Greys", vmin=0, vmax=2.5, extent=(0, 32, 32, 0))
            ax.plot(p[:, 0], p[:, 1], "-", color="#e8432b", lw=0.9, alpha=0.8); ax.plot(p[c["cola"], 0], p[c["cola"], 1], "-", color="#f59e0b", lw=2.2)
            yy, xx = np.nonzero(skeletonize(Ms[n][i] > 0.4)); ax.plot((xx + 0.5) / 2, (yy + 0.5) / 2, ".", color="k", ms=1.5)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"cola {100 * colas[n][i]:.0f} %", fontsize=8, color="#1b7f3b" if colas[n][i] >= CONSERVADA else "#b91c1c")
            if j == 0:
                ax.set_ylabel(n, fontsize=8, fontweight="bold")
    abajo.suptitle("trazos casi cerrados nunca vistos · rojo = línea media de verdad · NARANJA = la cola desviada · negro = cresta del adelgazador", fontsize=10)
    out = AQUI / "resultados" / "126-adelgazador-casi.png"; fig.savefig(out, dpi=85)
    out.with_suffix(".json").write_text(json.dumps({"medidas": res, "criterio": crit}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
