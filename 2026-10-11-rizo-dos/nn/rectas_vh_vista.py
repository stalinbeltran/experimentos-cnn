#!/usr/bin/env python3
"""rizo-2 § 129 — VISTA PREVIA: rectas y curvas de radio amplio, VERTICALES y HORIZONTALES (`nn/rectas_vh.py`; el dueño, 2026-10-19).
Sólo imagen: el revisor no cambia. Sobre el delgado del revisor (fuente «nist»).

Por dígito, tres paneles: el delgado con las líneas (AZUL = vertical, NARANJA = horizontal, gris = cresta que no cuenta), y las dos
rejillas 3×3 (V y H) con su valor en cada celda; entre celdas, una barra VERDE = ENLACE (una misma línea cruza las dos). Debajo,
las RANURAS: las 2 líneas más largas de cada orientación, (posición transversal, inicio–fin) relativos a la caja del dígito.

    ../.venv/bin/python nn/rectas_vh_vista.py   → resultados/129-rectas-vh.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import rectas_vh as RV                                                   # noqa: E402
import servidor as S                                                     # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

COL = {"V": "#2a78d6", "H": "#c2410c"}


def rejilla(ax, g, e, o, titulo):
    ax.imshow(g, cmap="Blues" if o == "V" else "Oranges", vmin=0, vmax=1.3, extent=(0, 3, 3, 0))
    for r in range(3):
        for c in range(3):
            if g[r, c] > 0.02:
                ax.text(c + 0.5, r + 0.5, f"{g[r, c]:.2f}".lstrip("0"), ha="center", va="center", fontsize=8, fontweight="bold")
    for k in (1, 2):
        ax.axhline(k, color="#999", lw=0.6); ax.axvline(k, color="#999", lw=0.6)
    if o == "V":
        for r, c in zip(*np.nonzero(e)):
            ax.plot([c + 0.5, c + 0.5], [r + 0.8, r + 1.2], "-", color="#1b7f3b", lw=4, solid_capstyle="butt")
    else:
        for r, c in zip(*np.nonzero(e)):
            ax.plot([c + 0.8, c + 1.2], [r + 0.5, r + 0.5], "-", color="#1b7f3b", lw=4, solid_capstyle="butt")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo, fontsize=8, color=COL[o])


def main():
    d = S.Datos(); y = d.y; val = np.asarray(d.va); rng = np.random.default_rng(29)
    # medida en val: cuántas líneas V y H por dígito, por clase
    cuenta = {c: [] for c in range(10)}
    for i in val:
        ls = RV.lineas(d.M[i]); cuenta[int(y[i])].append((sum(l["o"] == "V" for l in ls), sum(l["o"] == "H" for l in ls)))
    res = {"TOL": RV.TOL, "K_AMPLIO": RV.K_AMPLIO, "L_LINEA": RV.L_LINEA,
           "por_clase": {str(c): {"V_media": round(float(np.mean([a for a, _ in v])), 2), "H_media": round(float(np.mean([b for _, b in v])), 2)} for c, v in cuenta.items()}}
    for c in range(10):
        print(c, res["por_clase"][str(c)])
    muestra = [int(rng.choice(val[y[val] == c])) for c in range(10) for _ in range(2)]
    nd, porfila = len(muestra), 4; filas = int(np.ceil(nd / porfila))
    fig, axs = plt.subplots(filas, 3 * porfila, figsize=(2.05 * 3 * porfila, 2.75 * filas + 0.9),
                            gridspec_kw={"width_ratios": [1.25, 1, 1] * porfila})
    for k, i in enumerate(muestra):
        f, c = divmod(k, porfila); a0, a1, a2 = axs[f, 3 * c], axs[f, 3 * c + 1], axs[f, 3 * c + 2]
        ls = RV.lineas(d.M[i]); g = RV.rejillas(ls); e = RV.enlaces(ls); rn = RV.ranuras(ls, d.M[i])
        a0.imshow(d.M[i], cmap="Greys", vmin=0, vmax=1.6, extent=(0, 32, 32, 0))
        for t in CC.tramos(d.M[i].astype(np.float32))[0] + CC.tramos(d.M[i].astype(np.float32))[1]:
            p = np.asarray(t["pts"]); a0.plot(p[:, 0], p[:, 1], "-", color="#bbb", lw=1)
        for l in ls:
            a0.plot(l["pts"][:, 0], l["pts"][:, 1], "-", color=COL[l["o"]], lw=2.6)
        for q in (32 / 3, 64 / 3):
            a0.axhline(q, color="#ddd", lw=0.5); a0.axvline(q, color="#ddd", lw=0.5)
        a0.set_xlim(0, 32); a0.set_ylim(32, 0); a0.set_xticks([]); a0.set_yticks([])
        a0.set_title(f"#{i} ({y[i]}) · V {sum(l['o'] == 'V' for l in ls)} · H {sum(l['o'] == 'H' for l in ls)}", fontsize=9, fontweight="bold")
        txt = []
        for o in ("V", "H"):
            ss = [f"{t:.2f} ({a:.2f}–{b:.2f})" for p, t, a, b in rn[o] if p]
            txt.append(f"{o}: " + ("  ".join(ss) if ss else "—"))
        a0.set_xlabel("\n".join(txt), fontsize=6.5)
        rejilla(a1, g["V"], e["V"], "V", "VERTICALES"); rejilla(a2, g["H"], e["H"], "H", "HORIZONTALES")
    fig.suptitle(f"VISTA PREVIA · rectas y curvas de radio amplio (radio ≥ {1 / RV.K_AMPLIO:.0f} px), VERTICALES (azul, a ≤ {RV.TOL:.0f}° de 90°) y "
                 f"HORIZONTALES (naranja, a ≤ {RV.TOL:.0f}° de 0°); las diagonales (de {RV.TOL:.0f}° a {90 - RV.TOL:.0f}°) NO cuentan · ≥ {RV.L_LINEA:.0f} px\n"
                 "rejillas: largo de línea en cada celda / lado de celda · barra VERDE entre celdas = ENLACE (la MISMA línea cruza las dos) · "
                 "debajo, RANURAS: posición transversal (inicio–fin), relativas a la caja del dígito", fontsize=10)
    fig.tight_layout(); out = AQUI / "resultados" / "129-rectas-vh.png"; fig.savefig(out, dpi=80)
    out.with_suffix(".json").write_text(json.dumps({**res, "muestra": muestra}, ensure_ascii=False, indent=1)); print(out)


if __name__ == "__main__":
    main()
