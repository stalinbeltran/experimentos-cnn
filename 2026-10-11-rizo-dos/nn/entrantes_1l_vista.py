#!/usr/bin/env python3
"""rizo-2 § 112 — los ENTRANTES de los 1l (el dueño, 2026-10-17: «analiza el 1552; sus entrantes conos lo identifican como 1l. Busca
cuáles 1l tienen entrantes»). El 1552 es un 7 (con barra) que LEvertice y Cuspide leen como 1l en las 3 semillas.

Sobre los objetos del revisor (cresta del DELGADO): los 1l son los de `web/grupos.py` + las marcas a mano (`etiquetas-1.json`), igual
que en el entrenamiento (§ 107). Arriba: el 1552 con sus entrantes, cuántos dígitos de cada grupo tienen alguno y el ÁREA de los
entrantes de los 1l contra la de los 7. Abajo: los 1l que tienen LOS DOS entrantes que más empujan al 1552 hacia 1l (→ der en
arriba-centro y ↓ abajo en el centro), y los 1l con los entrantes más grandes. Nada cambiado ni entrenado.

    ../.venv/bin/python nn/entrantes_1l_vista.py   → resultados/112-entrantes-1l.{png,json}
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web"))
import cresta_curvas as CC                                               # noqa: E402
import grupos as GR                                                      # noqa: E402
import servidor as S                                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Polygon                                    # noqa: E402

CONO = ["→ der", "← izq", "↓ abajo", "↑ arriba"]; COL = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]   # los del revisor
CLAVE = (("→ der", "arriba-centro"), ("↓ abajo", "centro"))              # los dos que más empujan al 1552 hacia 1l (Cuspide y LEvertice)


def pinta(a, d, i, titulo):
    a.imshow(d.M[i], cmap="Greys", vmin=0, vmax=2.4, extent=(0, 32, 32, 0))
    for e in d.objetos(i)["entrantes"]:
        c = CC.cono(e); a.add_patch(Polygon(np.asarray(e["pts"]), closed=True, fc=COL[c], ec=COL[c], alpha=0.45, lw=1))
        a.text(e["cx"], e["cy"], f"{e['area']:.0f}", fontsize=7, color="k", ha="center", va="center", fontweight="bold")
    for q in (32 / 3, 64 / 3):
        a.axhline(q, color="#bbb", lw=0.6, ls=":"); a.axvline(q, color="#bbb", lw=0.6, ls=":")
    a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([]); a.set_title(titulo, fontsize=8)


def main():
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); man = {e["idx"]: e for e in els}
    l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]; a1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) != "1l"]
    sietes = [int(i) for i in np.flatnonzero(d.y == 7)]
    parte = {int(i): "train" for i in d.tr}; parte.update({int(i): "val" for i in d.va}); parte.update({int(i): "extra" for i in d.ex})
    X = d.features(3)[0]; a0, z0 = S.rangos(3)["entrantes60"]; E = X[:, a0:z0]; cols = S.columnas("entrantes60", 3)
    clave = [k for k, c in enumerate(cols) if (c["sub"], c["nombre"]) in CLAVE]
    con = lambda ids: [i for i in ids if E[i].max() > 0.5]
    l1c = con(l1); ambos = [i for i in l1 if all(E[i, k] > 0.5 for k in clave)]; ambos7 = sum(all(E[i, k] > 0.5 for k in clave) for i in sietes)
    area = lambda ids: np.array([e["area"] for i in ids for e in d.objetos(i)["entrantes"]])
    a_l1, a_7 = area(l1), area(sietes)
    grandes = sorted(l1c, key=lambda i: -max([e["area"] for e in d.objetos(i)["entrantes"]] or [0]))
    grandes = [i for i in grandes if i not in ambos][:8]

    fig = plt.figure(figsize=(16, 13)); g = fig.add_gridspec(4, 8, height_ratios=[1.5, 1, 1, 0.1])
    pinta(fig.add_subplot(g[0, 0:2]), d, 1552, "#1552 · es un 7 · Cuspide y LEvertice lo leen 1l (3/3 semillas)\nnúmero = área del entrante (px²)")
    b = fig.add_subplot(g[0, 2:4]); n = [len(l1), len(a1), len(sietes)]; c_ = [len(l1c), len(con(a1)), len(con(sietes))]
    b.bar(range(3), [100 * x / y for x, y in zip(c_, n)], color=["#9333ea", "#c4b5fd", "#bbb"])
    for k in range(3):
        b.text(k, 100 * c_[k] / n[k] + 2, f"{c_[k]} de {n[k]}", ha="center", fontsize=9)
    b.set_xticks(range(3)); b.set_xticklabels(["1l", "1a", "7"]); b.set_ylim(0, 112); b.set_ylabel("% con algún entrante")
    b.set_title("¿cuántos tienen algún entrante?\n(un palo recto no debería tener ninguno)", fontsize=10)
    h = fig.add_subplot(g[0, 4:8]); bins = np.arange(0, 62, 2)
    h.hist(np.clip(a_7, 0, 60), bins, density=True, color="#bbb", label=f"entrantes de los 7 (mediana {np.median(a_7):.0f} px²)")
    h.hist(np.clip(a_l1, 0, 60), bins, density=True, color="#9333ea", alpha=0.6, label=f"entrantes de los 1l (mediana {np.median(a_l1):.0f} px²)")
    for e in d.objetos(1552)["entrantes"]:
        h.axvline(e["area"], color="k", lw=1.2, ls="--")
    h.text(17, h.get_ylim()[1] * 0.9, "-- los 3 del 1552:\n10, 11, 16 px²", fontsize=9)
    h.set_xlabel("área del entrante (px², ≥ 60 junto)"); h.legend(fontsize=9); h.set_title("los de los 1l son más pequeños… pero los del 1552 también", fontsize=10)
    for k, i in enumerate(ambos[:8]):
        pinta(fig.add_subplot(g[1, k]), d, i, f"#{i} 1l ({parte[i]}, {'a mano' if man[i]['manual'] else 'regla'})")
    for k, i in enumerate(grandes):
        pinta(fig.add_subplot(g[2, k]), d, i, f"#{i} 1l ({parte[i]}, {'a mano' if man[i]['manual'] else 'regla'})")
    fig.text(0.01, 0.52, f"1l con los DOS del 1552\n(→ der arriba-centro y ↓ abajo centro): {len(ambos)}  ·  7 con los dos: {ambos7}", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.285, "1l con los entrantes más GRANDES", fontsize=11, fontweight="bold")
    ax = fig.add_subplot(g[3, :]); ax.axis("off")
    ax.text(0, 0.5, "color = cono del entrante (hacia dónde abre): " + " · ".join(f"{c}" for c in CONO) + "  (naranja · azul · verde · morado)", fontsize=10)
    fig.suptitle("Los ENTRANTES de los 1l: el 57 % tiene alguno, y por eso un 7 con entrantes pequeños puede leerse 1l (cresta del delgado)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = AQUI / "resultados" / "112-entrantes-1l.png"; fig.savefig(out, dpi=95)
    cnt = collections.Counter((cols[k]["sub"], cols[k]["nombre"]) for i in l1 for k in np.flatnonzero(E[i] > 0.5))
    res = {"n_1l": len(l1), "1l_con_entrantes": len(l1c), "1a_con": c_[1], "n_1a": n[1], "7_con": c_[2], "n_7": n[2],
           "1l_con_por_parte": dict(collections.Counter(parte[i] for i in l1c)),
           "1l_con_decididos": dict(collections.Counter("a mano" if man[i]["manual"] else "regla" for i in l1c)),
           "area_mediana": {"1l": float(np.median(a_l1)), "7": float(np.median(a_7))},
           "celdas_1l": [{"cono": s, "celda": c, "n": m} for (s, c), m in cnt.most_common()],
           "1l_con_los_dos_del_1552": ambos, "7_con_los_dos": int(ambos7),
           "1l_con_entrantes": [{"idx": i, "parte": parte[i], "decidido": "a mano" if man[i]["manual"] else "regla",
                                 "entrantes": [{"cono": CONO[CC.cono(e)], "area": round(e["area"], 1), "tipo": "esquina" if e["s"] == -1 else "arco",
                                                "cx": round(e["cx"], 1), "cy": round(e["cy"], 1)} for e in d.objetos(i)["entrantes"]]} for i in l1c]}
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(out)
    print({k: v for k, v in res.items() if k not in ("1l_con_entrantes", "celdas_1l")})


if __name__ == "__main__":
    main()
