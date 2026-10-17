#!/usr/bin/env python3
"""rizo-2 § 113 — el SUAVIZADO previo al delgado, medido sobre los 1l (el dueño, 2026-10-17: «tenemos ya un preproceso que adelgaza;
vamos a crear otro que los suavice. Nuestra medida van a ser estos 1l. Aplícales un gaussian… lo ruidoso viene de que los dígitos
originales están binarizados, su suavidad original se perdió»).

Los delgados suavizados los produce `delgado-cnn` (`nn/exportar_delgados.py --suavizar σ [--gris]`) y aquí se leen como DATO (regla
de aquel experimento: nunca se importa su código). Para cada variante se pasan los dígitos por los MISMOS objetos del revisor
(`cresta_curvas.objetos`) y se cuenta, por grupo, cuántos tienen algún entrante y cuántos una cresta ramificada (≥ 3 tramos).
Los 1l son los de § 107 (regla + marcas a mano, decididos sobre el delgado de siempre); la medida es que PIERDAN los entrantes
falsos de § 112, y el resto de clases dice lo que cuesta (un 7 o un 4 sin entrantes sería peor).

    ../.venv/bin/python nn/suavizado_vista.py   → resultados/113-suavizado.{png,json}
"""
from __future__ import annotations

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

DELG = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados"
VARIANTES = [("hoy (sin suavizar)", "delgados-k3x3.npz"), ("σ 0,7", "delgados-k3x3-s0.7.npz"), ("σ 1", "delgados-k3x3-s1.npz"),
             ("σ 1,5", "delgados-k3x3-s1.5.npz"), ("σ 1 gris ⚠", "delgados-k3x3-s1g.npz")]
COL = ["#c2410c", "#2a78d6", "#1b7f3b", "#7c3aed"]
MUESTRA = [736, 2849, 4661, 5538, 3517, 4053, 1668, 5266]                 # 4 con las dos celdas del 1552 + 4 con los más grandes (§ 112)


def piezas(m) -> int:
    """Cuántos trozos SUELTOS tiene la cresta (componentes 8-conexas de M > UMBRAL): un trazo que el suavizado rompe sube de 1."""
    from scipy import ndimage
    return int(ndimage.label(m > CC.UMBRAL, structure=np.ones((3, 3)))[1])


def medir(M, ids):
    ent, ram, n_ent, laz, rot = 0, 0, 0, 0, 0
    for i in ids:
        m = M[i].astype(np.float32); ob = CC.objetos(m); tr, _ = CC.tramos(m)
        ent += bool(ob["entrantes"]); n_ent += len(ob["entrantes"]); ram += len(tr) >= 3; laz += len(ob["lazos"]); rot += piezas(m) > 1
    n = len(ids); return {"n": n, "con_entrantes": 100 * ent / n, "ramificada": 100 * ram / n, "entrantes_por_digito": n_ent / n,
                         "lazos_por_digito": laz / n, "rotos": 100 * rot / n}


def main():
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor())
    l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]; a1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) != "1l"]
    va = [int(i) for i in d.va]; grupos = {"1l": l1, "1a": a1} | {str(c): [i for i in va if d.y[i] == c and c != 1] for c in range(10) if c != 1}
    Ms = {}; tintas = {}
    for nombre, f in VARIANTES:
        z = np.load(DELG / f); Ms[nombre] = z["M"]; tintas[nombre] = z["tinta"] if "tinta" in z.files else d.tinta.astype(np.float32)
    res = {nombre: {g: medir(Ms[nombre], ids) for g, ids in grupos.items()} for nombre, _ in VARIANTES}
    for nombre, _ in VARIANTES:
        r = res[nombre]; print(f"{nombre:20s} 1l: entrantes {r['1l']['con_entrantes']:5.1f} %  ramificada {r['1l']['ramificada']:5.1f} %  rotos {r['1l']['rotos']:.1f} % · "
                               + "lazos " + " ".join(f"{g}:{r[g]['lazos_por_digito']:.2f}" for g in ("0", "4", "6", "8", "9"))
                               + " · rotos (val) " + f"{np.mean([r[g]['rotos'] for g in grupos if g not in ('1l', '1a')]):.1f} %")

    nv = len(VARIANTES); nm = len(MUESTRA); fig = plt.figure(figsize=(2.0 * nm + 1, 2.2 * nv + 4.2))
    g = fig.add_gridspec(nv + 1, 2 * nm, height_ratios=[1.9] + [1] * nv)
    a = fig.add_subplot(g[0, 0:nm - 2]); x = np.arange(nv)
    a.bar(x - 0.2, [res[n]["1l"]["con_entrantes"] for n, _ in VARIANTES], 0.4, color="#9333ea", label="con algún entrante")
    a.bar(x + 0.2, [res[n]["1l"]["ramificada"] for n, _ in VARIANTES], 0.4, color="#c4b5fd", label="cresta ramificada (≥ 3 tramos)")
    for k, (n, _) in enumerate(VARIANTES):
        a.text(k - 0.2, res[n]["1l"]["con_entrantes"] + 1.5, f"{res[n]['1l']['con_entrantes']:.0f}", ha="center", fontsize=10, fontweight="bold")
        a.text(k + 0.2, res[n]["1l"]["ramificada"] + 1.5, f"{res[n]['1l']['ramificada']:.0f}", ha="center", fontsize=9)
    a.set_xticks(x); a.set_xticklabels([n for n, _ in VARIANTES], fontsize=9); a.set_ylim(0, 75); a.set_ylabel("% de los 264 1l")
    a.legend(fontsize=8, loc="upper right"); a.set_title("LA MEDIDA: los 1l (un palo recto no debería tener entrantes)", fontsize=10)
    otros = [k for k in grupos if k != "1l"]; w = 0.8 / nv; colv = [plt.cm.viridis(k / (nv - 1)) for k in range(nv)]
    b = fig.add_subplot(g[0, nm - 2:2 * nm - 5])
    for k, (n, _) in enumerate(VARIANTES):
        b.bar(np.arange(len(otros)) + (k - nv / 2 + 0.5) * w, [res[n][o]["entrantes_por_digito"] for o in otros], w, label=n, color=colv[k])
    b.set_xticks(range(len(otros))); b.set_xticklabels(otros); b.set_ylabel("entrantes por dígito"); b.legend(fontsize=7, ncol=2)
    b.set_title("LO QUE CUESTA (1): entrantes por dígito\nen las demás clases (val; 1a: todos)", fontsize=10)
    c = fig.add_subplot(g[0, 2 * nm - 4:]); lz = ["0", "4", "6", "8", "9"]
    for k, (n, _) in enumerate(VARIANTES):
        c.bar(np.arange(len(lz)) + (k - nv / 2 + 0.5) * w, [res[n][o]["lazos_por_digito"] for o in lz], w, color=colv[k])
    c.set_xticks(range(len(lz))); c.set_xticklabels(lz); c.set_ylabel("lazos por dígito")
    c.set_title("(2): lazos por dígito\n(¿cierra o abre huecos?)", fontsize=10)
    filas = []
    for f_, (n, _) in enumerate(VARIANTES):
        for k, i in enumerate(MUESTRA):
            ax = fig.add_subplot(g[f_ + 1, 2 * k:2 * k + 2]); m = Ms[n][i].astype(np.float32)
            ax.imshow(tintas[n][i].astype(np.float32), cmap="Greys", vmin=0, vmax=2.5, extent=(0, 32, 32, 0))
            ob = CC.objetos(m)
            for e in ob["entrantes"]:
                cc = CC.cono(e); ax.add_patch(Polygon(np.asarray(e["pts"]), closed=True, fc=COL[cc], ec=COL[cc], alpha=0.55, lw=1))
            tr = CC.tramos(m)[0]
            for t in tr:
                p = np.asarray(t["pts"]); ax.plot(p[:, 0], p[:, 1], "-", color="#111", lw=1.3)
            ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i} · {len(ob['entrantes'])} entr. · {len(tr)} tramos", fontsize=8)
            if k == 0:
                filas.append((ax, n))
    fig.suptitle("SUAVIZAR la tinta antes de adelgazar (gaussiana + rebinarizar a 0,5): ¿pierden los 1l sus entrantes falsos?\n"
                 "fondo = tinta que ve la red (ya suavizada) · negro = cresta del delgado · color = entrante: naranja der · azul izq · verde abajo · morado arriba", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    for ax, n in filas:
        q = ax.get_position(); fig.text(0.004, q.y0 + q.height / 2, n, rotation=90, fontsize=10, fontweight="bold", va="center")
    out = AQUI / "resultados" / "113-suavizado.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({"variantes": dict(VARIANTES), "muestra": MUESTRA, "medidas": res}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
