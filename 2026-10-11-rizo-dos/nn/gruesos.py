#!/usr/bin/env python3
"""rizo-2 § 116 — el SUBCONJUNTO de los dígitos GRUESOS (el dueño, 2026-10-17: «vamos a crear un subset, que no vamos a usar por
ahora, para los dígitos gruesos. Volvamos al punto de partida donde veíamos las ramas en el 1l, y dime cuántos quedan con ramas si
quitamos los gruesos; muéstrame los que quedan»). NO lo usa ningún compositor ni el revisor: es un dato para más adelante.

GROSOR de un dígito = el ANCHO del trazo medido sobre su cresta: 2 × la distancia al borde de la tinta (transformada de distancia
de la tinta binaria), interpolada en cada punto sub-píxel de la cresta del DELGADO de siempre, y promediada. Es continuo y vale
igual para un 1 vertical que para un 8 (se descartaron: la mediana de la misma distancia, que da saltos de 4 · 4,5 · 5,7 · 6…, y
área de tinta / largo de cresta, que infla los trazos cortos).

GRUESO = el UMBRAL % más grueso de TODO el dataset (5620 dígitos): por defecto el 5 %, ≥ 7,86 px medido ese día.
⚠ Medido ese día: los 1 son, con mucho, los más gruesos (mediana 7,8 px; las demás clases 4,7–5,8), así que el subconjunto es
casi todo 1s (269 de 281) y quitarlo deja 70 de los 264 1l.

Los 1l y sus ramas son los del punto de partida (§ 112): el delgado de siempre, 1l = grupos.py + etiquetas-1.json, rama = cresta
con ≥ 3 tramos.

    ../.venv/bin/python nn/gruesos.py [--pct 5]   → resultados/116-gruesos.{png,json}  (el json ES el subconjunto)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web"))
import cresta_curvas as CC                                               # noqa: E402
import grupos as GR                                                      # noqa: E402
import servidor as S                                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402


def grosor(tinta, M) -> float:
    """2 × la distancia media al borde de la tinta, sobre los puntos de la cresta de M. NaN sin cresta."""
    dt = ndimage.distance_transform_edt(tinta > 0); tr, puros = CC.tramos(M)
    P = np.array([p for t in tr for p in t["pts"]] + [p for c in puros for p in c["pts"]], float)
    if not len(P):
        return float("nan")
    return 2 * float(np.mean(ndimage.map_coordinates(dt, [P[:, 1] - 0.5, P[:, 0] - 0.5], order=1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pct", type=float, default=5.0, help="el PCT %% más grueso del dataset es «grueso»")
    a = ap.parse_args()
    d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]
    G = np.array([grosor(d.tinta[i], d.M[i]) for i in range(len(d.y))]); u = float(np.nanpercentile(G, 100 - a.pct))
    gruesos = [int(i) for i in np.flatnonzero(G >= u)]; es_g = set(gruesos)
    tramos = {i: len(CC.tramos(d.M[i])[0]) for i in l1}; ram = {i: tramos[i] >= 3 for i in l1}
    quedan = [i for i in l1 if i not in es_g]; con = [i for i in quedan if ram[i]]
    parte = {int(i): "train" for i in d.tr}; parte.update({int(i): "val" for i in d.va}); parte.update({int(i): "extra" for i in d.ex})
    man = {e["idx"]: e for e in els}
    print(f"grueso ≥ {u:.2f} px ({a.pct:g} %): {len(gruesos)} dígitos · 1l: {len(l1)} → quedan {len(quedan)} · con ramas {len(con)} (antes {sum(ram.values())})")

    n = len(con); nc = 8; nf = (n + nc - 1) // nc
    fig = plt.figure(figsize=(16, 4.6 + 2.3 * nf), layout="constrained"); arriba, abajo = fig.subfigures(2, 1, height_ratios=[4.2, 2.3 * nf])
    h, b = arriba.subplots(1, 2, width_ratios=[5, 3]); bins = np.linspace(3, 14, 45)
    h.hist(G[(d.y != 1) & ~np.isnan(G)], bins, color="#bbb", label="las demás clases")
    h.hist(G[d.y == 1], bins, color="#9333ea", alpha=0.75, label="los 1 (todos)")
    h.hist([G[i] for i in l1], bins, histtype="step", color="k", lw=1.6, label="los 1l")
    h.axvline(u, color="#c2410c", lw=2, ls="--"); h.text(u + 0.1, h.get_ylim()[1] * 0.85, f"GRUESO ≥ {u:.1f} px\n(el {a.pct:g} % más grueso)", color="#c2410c", fontsize=10, fontweight="bold")
    h.set_xlabel("grosor del trazo (px): 2 × distancia al borde, media sobre la cresta"); h.set_ylabel("dígitos"); h.legend(fontsize=9)
    h.set_title("los 1 son, con mucho, los más gruesos del dataset", fontsize=10)
    filas = [("todos los 1l", l1), ("sin los gruesos", quedan)]
    for k, (nombre, ids) in enumerate(filas):
        r = sum(ram[i] for i in ids); b.barh(k, r, color="#9333ea"); b.barh(k, len(ids) - r, left=r, color="#ddd")
        b.text(len(ids) + 3, k, f"{r} con ramas de {len(ids)} ({100 * r / max(len(ids), 1):.0f} %)", va="center", fontsize=10, fontweight="bold")
    b.set_yticks(range(2)); b.set_yticklabels([f for f, _ in filas]); b.invert_yaxis(); b.set_xlim(0, 400); b.set_xlabel("1l")
    b.set_title("morado = con ramas (cresta con ≥ 3 tramos)", fontsize=10)
    axs = abajo.subplots(nf, nc, squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    for k, i in enumerate(con):
        ax = axs[k // nc, k % nc]; ax.imshow(d.tinta[i], cmap="Greys", vmin=0, vmax=2.5, extent=(0, 32, 32, 0))
        for t in CC.tramos(d.M[i])[0]:
            p = np.asarray(t["pts"]); ax.plot(p[:, 0], p[:, 1], "-", lw=1.8)
        ax.set_xlim(0, 32); ax.set_ylim(32, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"#{i} · {tramos[i]} tramos · {G[i]:.1f} px\n{parte[i]}, {'a mano' if man[i]['manual'] else 'regla'}", fontsize=8)
    fig.suptitle(f"Los 1l con RAMAS que quedan si se quitan los dígitos GRUESOS: {len(con)} de {len(quedan)} "
                 f"(un color por tramo de la cresta, sobre la tinta; delgado de siempre)", fontsize=12)
    out = AQUI / "resultados" / "116-gruesos.png"; fig.savefig(out, dpi=90)
    out.with_suffix(".json").write_text(json.dumps({
        "definicion": "grosor = 2 × media de la distancia al borde de la tinta sobre la cresta del delgado k3x3; grueso = el pct % más grueso",
        "pct": a.pct, "umbral_px": round(u, 3), "gruesos": gruesos, "gruesos_por_clase": np.bincount(d.y[gruesos], minlength=10).tolist(),
        "1l": len(l1), "1l_con_ramas": int(sum(ram.values())), "1l_quedan": len(quedan), "1l_quedan_con_ramas": con,
        "grosor": {int(i): round(float(x), 3) for i, x in enumerate(G)}}, ensure_ascii=False)); print(out)


if __name__ == "__main__":
    main()
