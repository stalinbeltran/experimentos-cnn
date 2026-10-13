#!/usr/bin/env python3
"""rizo-2 — ¿detecta bien los LAZOS el detector de la cresta? (el dueño, 2026-10-13). Se comparan tres detectores de
lazo/hueco sobre val contra la VERDAD = huecos cerrados de la TINTA original (fondo que no toca el borde, área ≥ 3 px;
dilatados 1 px para casar), y aparte contra los huecos del mapa (M > 0,3/0,5/0,7):
  · cresta   — ciclos del grafo de la cresta (cresta_curvas.objetos → lazos), sobre el mapa continuo
  · suaves   — objetos «cerrado» de huecos_suaves (v4: centrar + uno por celda + sin rendijas + ponderar), fuerza ≥ 0,3
  · compuesto2 «lazo» — el detector de rizo-2 (plantillas ∪ rayos, hueco ≥ 2,5 px, rayos 14 px) sobre M > 0,5
Por detector: cobertura (huecos de la tinta con una detección dentro), precisión (detecciones que caen en un hueco de la
tinta o del mapa), duplicados, y la tabla por clase. Y una figura con los fallos de la cresta: huecos de la tinta sin
lazo (¿rotura del mapa?) y lazos sin hueco.

    python nn/lazos_comparar.py --entrada /ruta/delgados-k3x3.npz   → resultados/46-lazos-comparar.{json,png}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                          # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import huecos_suaves as HS                                          # noqa: E402
import huecos_suaves_falsos as HF                                   # noqa: E402
import rizo_huecos as RH                                            # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def huecos_tinta(t):
    lab, n = ndimage.label(t == 0); borde = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1]); out = []
    for k in range(1, n + 1):
        if k not in borde and (lab == k).sum() >= 3:
            out.append(lab == k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M"); a = ap.parse_args()
    HS.configurar(CENTRAR=True, UNO_POR_CELDA=True, SIN_RENDIJAS=True, PONDERAR=True)
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); va = np.flatnonzero(d["particion"] == "val"); y = d["etiquetas"][va]
    M = np.load(a.entrada)[a.clave].astype(np.float32)[va]; T = (d["imagenes"][va] > 0).astype(np.uint8)
    cand = pickle.loads((AQUI / "resultados" / "40-candidatos-delgado-k3x3.pkl").read_bytes())["todos"]
    cer, _ = HS.mapas(M)
    DET = ["cresta", "suaves", "compuesto2-lazo"]
    st = {n: {"det": 0, "en_tinta": 0, "en_mapa_solo": 0, "falsos": 0, "dupl": 0, "cubiertos": 0} for n in DET}
    pc = {c: {n: {"det": 0, "falsos": 0, "cubiertos": 0} for n in DET} | {"huecos_tinta": 0, "digitos": 0} for c in range(10)}
    fallos_cresta = []; total_huecos = 0
    for j, i in enumerate(va):
        c = int(y[j]); m = M[j]; ht = huecos_tinta(T[j]); total_huecos += len(ht); pc[c]["huecos_tinta"] += len(ht); pc[c]["digitos"] += 1
        ht_d = [ndimage.binary_dilation(h, iterations=1) for h in ht]; ref_m = HF.huecos_reales(m); HS.M_ACTUAL = m
        dets = {"cresta": [(z["cx"], z["cy"]) for z in CC.objetos(m)["lazos"]],
                "suaves": [(o["cx"], o["cy"]) for o in HS.objetos(cer[j]) if o["fuerza"] >= 0.3],
                "compuesto2-lazo": [q["centro_hueco"] for q in RH.agujeros((m > 0.5).astype(np.uint8), DV.filtrar(cand[i], *DV.VARIANTES["lazo (h2.5, 14px)"]))[0]]}
        for n, ps in dets.items():
            cubiertos = set(); st[n]["det"] += len(ps); pc[c][n]["det"] += len(ps)
            for (x, yy_) in ps:
                xi, yi = min(int(x), 31), min(int(yy_), 31); k = next((k for k, h in enumerate(ht_d) if h[yi, xi]), None)
                if k is not None:
                    st[n]["en_tinta"] += 1
                    if k in cubiertos:
                        st[n]["dupl"] += 1
                    cubiertos.add(k)
                elif ref_m[yi, xi]:
                    st[n]["en_mapa_solo"] += 1
                else:
                    st[n]["falsos"] += 1; pc[c][n]["falsos"] += 1
            st[n]["cubiertos"] += len(cubiertos); pc[c][n]["cubiertos"] += len(cubiertos)
            if n == "cresta" and (len(cubiertos) < len(ht) or any(not any(h[min(int(yy_), 31), min(int(x), 31)] for h in ht_d) for x, yy_ in ps)):
                fallos_cresta.append((int(i), c, len(ht), len(cubiertos), len(ps)))
    res = {"huecos_tinta_val": total_huecos, "detectores": {}}
    for n in DET:
        s = st[n]; res["detectores"][n] = {**s, "cobertura": round(s["cubiertos"] / total_huecos, 3), "precision_tinta": round(s["en_tinta"] / max(s["det"], 1), 3),
                                          "precision_tinta_o_mapa": round((s["en_tinta"] + s["en_mapa_solo"]) / max(s["det"], 1), 3)}
    res["por_clase"] = {c: {"digitos": pc[c]["digitos"], "huecos_tinta": pc[c]["huecos_tinta"], **{n: pc[c][n] for n in DET}} for c in range(10)}
    res["fallos_cresta"] = fallos_cresta
    (AQUI / "resultados" / "46-lazos-comparar.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "fallos_cresta"}, ensure_ascii=False, indent=1))
    # figura: fallos de la cresta (primero los huecos sin lazo, luego lazos sin hueco), hasta 48
    sel = sorted(fallos_cresta, key=lambda f: (f[3] - f[2], -(f[4] - f[3])))[:48]
    cols = 12; filas = int(np.ceil(len(sel) / cols)); fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.55, filas * 1.8 + 0.9)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (i, c, nh, nc, nd) in zip(axs.flat, sel):
        j = int(np.flatnonzero(va == i)[0]); CC.pintar(ax, M[j])
        for h in huecos_tinta(T[j]):
            yy_, xx_ = np.nonzero(h); ax.plot(xx_, yy_, "s", color="#ffe08a", ms=2.2, mew=0, alpha=0.75, zorder=0)
        ax.set_title(f"{c} · idx {i} · tinta {nh} · lazos {nd} ({nc} ok)", fontsize=7, color="#b45309" if nc < nh else "#1d4ed8")
    fig.suptitle(f"rizo-2 · 46 · fallos del detector de LAZOS de la cresta en val: ámbar = hueco de la tinta original (verdad) · negro + = lazo detectado\n"
                 f"naranja = le falta algún hueco (el mapa tiene una rotura) · azul = lazo de más · cobertura {res['detectores']['cresta']['cobertura']:.3f} · precisión {res['detectores']['cresta']['precision_tinta']:.3f}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / "46-lazos-comparar.png", dpi=100); print("→ resultados/46-lazos-comparar.{json,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
