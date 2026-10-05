#!/usr/bin/env python3
"""Las curvas de las dos CNN tradicionales (resultados/curvas-cnn.json, de nn/curvas_cnn.py --resumen) junto a las cuatro de
la ganancia (detectores y píxeles), TODAS sobre las mismas 60 particiones (se comprueba por huella), y la evaluación del
criterio escrito antes de correr (instrucciones/02-criterio.md § «Curvas de CNN»).

  1. la curva de aprendizaje (acierto según N) de las seis, con la banda entre semillas
  2. N(ε): cuántas muestras cuesta cada nivel de acierto, las seis

Mismas funciones que nn/muestras_necesarias.py (media por N, isotónica, interpolación en log N, sin extrapolar).

    python nn/figura_curvas_cnn.py   → resultados/curvas-cnn-comparacion.json y resultados/curvas-cnn.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn.registro import por_id              # noqa: E402
import muestras_necesarias as M                 # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, GRID = M.SUP, M.T1, M.T2, M.GRID
LENET, CNN3 = "LeNet-5 32×32", "CNN 3 capas 8×8 (la del repo)"
# las 4 de siempre + 2 categóricos más (paleta de referencia, slots 5 y 7; validada en orden con las otras 4, light: ALL PASS)
SERIES = M.SERIES + [(CNN3, "#e87ba4", "-."), (LENET, "#4a3aa7", "-.")]
OBJ = (80, 90, 95, 97, 98)


def cruce(ns: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float | None, str]:
    """Primer N (interpolado en log N) a partir del cual b ≥ a y ya no vuelve a quedar debajo."""
    d = b - a
    if (d >= 0).all():
        return None, f"b ≥ a en todo el rango (desde N = {ns[0]})"
    if (d < 0).all():
        return None, f"b < a en todo el rango (hasta N = {ns[-1]})"
    ult = int(np.flatnonzero(d < 0)[-1])                 # el último punto donde b va por debajo
    if ult == len(ns) - 1:
        return None, "b vuelve a quedar por debajo al final"
    t = -d[ult] / (d[ult + 1] - d[ult])
    return float(np.exp(np.log(ns[ult]) + t * (np.log(ns[ult + 1]) - np.log(ns[ult])))), "ok"


def main() -> int:
    g8 = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    g32 = json.loads((RES / "ganancia.json").read_text())
    cc = json.loads((RES / "curvas-cnn.json").read_text())
    for k in ("huella_y", "huella_particiones"):
        if not (g8[k] == g32[k] == cc[k]):
            raise SystemExit(f"✗ '{k}' no coincide entre la ganancia (8×8, 32×32) y las CNN: no son las mismas particiones. Me niego.")
    casos = {**g8["casos"], **g32["casos"], **cc["casos"]}
    faltan = {n: 60 - len(casos[n]) for n, *_ in SERIES if len(casos[n]) != 60}
    curvas = {n: M.curva(casos[n]) for n, *_ in SERIES}
    ns = curvas[LENET][0]
    if any(not np.array_equal(curvas[n][0], ns) for n, *_ in SERIES):
        raise SystemExit("✗ las curvas no tienen los mismos N: no son las mismas particiones. Me niego.")
    tabla = {}
    for n, *_ in SERIES:
        tabla[n] = {f"{o:g}": M.texto_n(*M.n_para(*curvas[n], o), curvas[n][0]) for o in OBJ}
        tabla[n]["max"] = round(float(curvas[n][1][-1]), 2)
    det8 = curvas["detectores 8×8 (13)"][1]
    n_cruce, e_cruce = cruce(ns, det8, curvas[LENET][1])
    n95_lenet = M.n_para(*curvas[LENET], 95)[0]
    n95_det8 = M.n_para(*curvas["detectores 8×8 (13)"], 95)[0]
    pix = np.maximum(curvas["píxeles 8×8"][1], curvas["píxeles 32×32"][1])
    n_cruce_cnn3, e_cnn3 = cruce(ns, curvas["píxeles 8×8"][1], curvas[CNN3][1])
    cnn3_180 = float(np.mean([M.acierto_en(*curvas[CNN3], n) for n in (160, 200)]))
    crit = {
        "1_cruce_det8_lenet": {"N": None if n_cruce is None else round(n_cruce), "estado": e_cruce, "prediccion": "entre 200 y 1000",
                               "cumple": n_cruce is not None and 200 <= n_cruce <= 1000},
        "2_N95_lenet": {"N": None if n95_lenet is None else round(n95_lenet), "N95_det8": round(n95_det8),
                        "veces": None if n95_lenet is None else round(n95_lenet / n95_det8, 2), "prediccion": "250–500",
                        "cumple": n95_lenet is not None and 250 <= n95_lenet <= 500},
        "3_cruce_cnn3_pixeles8": {"N": None if n_cruce_cnn3 is None else round(n_cruce_cnn3), "estado": e_cnn3,
                                  "prediccion": "por debajo hasta ~500, por encima desde ~1000",
                                  "cnn3_menos_mejor_pixeles_por_N": {int(n): round(float(c - p), 2) for n, c, p in zip(ns, curvas[CNN3][1], pix)}},
        "4_coherencia_cnn3_N180": {"curva_N160_200": round(cnn3_180, 2), "ruido_comb_180_1617": 86.9,
                                   "diferencia": round(cnn3_180 - 86.9, 2), "cumple": abs(cnn3_180 - 86.9) <= 5},
    }
    out = {"particiones": {"huella_y": cc["huella_y"], "huella_particiones": cc["huella_particiones"]},
           "faltan_entrenamientos": faltan, "N": ns.tolist(),
           "curvas_pct": {n: np.round(curvas[n][1], 2).tolist() for n, *_ in SERIES}, "N_para": tabla, "criterio": crit}
    (RES / "curvas-cnn-comparacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if faltan:
        print(f"⚠ faltan entrenamientos: {faltan}")
    print(f"{'acierto (%) por N →':<30}" + "".join(f"{int(n):>6}" for n in ns))
    for n, *_ in SERIES:
        print(f"{n:<30}" + "".join(f"{a:>6.1f}" for a in curvas[n][1]))
    print(f"\n{'N para →':<30}" + "".join(f"{f'{o:g} %':>8}" for o in OBJ) + "   máx")
    for n, *_ in SERIES:
        print(f"{n:<30}" + "".join(f"{tabla[n][f'{o:g}']:>8}" for o in OBJ) + f"   {tabla[n]['max']:.1f}")
    print("\ncriterio:")
    for k, v in crit.items():
        print(f"  {k}: " + json.dumps({a: b for a, b in v.items() if a != "cnn3_menos_mejor_pixeles_por_N"}, ensure_ascii=False))
    print("  CNN3 − mejor de los píxeles, por N:", crit["3_cruce_cnn3_pixeles8"]["cnn3_menos_mejor_pixeles_por_N"])
    dibujar(casos, curvas, tabla)
    return 0


def dibujar(casos: dict, curvas: dict, tabla: dict) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=SUP)
    for ax in (a1, a2):
        M._ejes(ax)
    M._eje_n(a1, "x"); a1.set_xlim(8, 2600)
    for n, col, ls in SERIES:
        ns, acc = curvas[n]
        a1.plot(ns, acc, color=col, lw=2.2, ls=ls, label=n, zorder=3)
        por_sem = np.array([M.curva(casos[n], sem=s)[1] for s in M.SEMILLAS])
        a1.fill_between(ns, por_sem.min(0), por_sem.max(0), color=col, alpha=0.12, lw=0, zorder=2)
    a1.set_ylim(a1.get_ylim()[0], 100.5)
    a1.set_xlabel("muestras de train N (escala log)"); a1.set_ylabel("acierto sobre los dígitos nuevos (%)")
    a1.legend(frameon=False, loc="lower right", fontsize=9.5)
    a1.set_title("1 · Curva de aprendizaje: detectores, píxeles y CNN", loc="left", fontsize=12.5, color=T1)
    M._eje_n(a2, "y"); a2.set_xlim(70, 99.5)
    a2.set_xlabel("acierto objetivo sobre los dígitos nuevos (%)"); a2.set_ylabel("muestras de train necesarias N (escala log)")
    for n, col, ls in SERIES:
        ns, acc = curvas[n]
        pts = [(o, M.n_para(ns, acc, o)[0]) for o in M.REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        if pts:
            a2.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2.2, ls=ls, zorder=3)
            a2.plot(pts[-1][0], pts[-1][1], "o", color=col, ms=7, mec=SUP, mew=1.5, zorder=4)
    a2.text(0.02, 0.97, f"{'':<30}{'N(90 %)':>8}{'N(95 %)':>9}{'máx':>7}\n" + "\n".join(
        f"{n:<30}{tabla[n]['90']:>8}{tabla[n]['95']:>9}{tabla[n]['max']:>6.1f}%" for n, *_ in SERIES)
        + "\n● = lo máximo alcanzado con N ≤ 2000", transform=a2.transAxes, ha="left", va="top", fontsize=9, color=T2,
        family="monospace")
    a2.set_title("2 · ¿Cuántas muestras cuesta cada nivel de acierto?", loc="left", fontsize=12.5, color=T1)
    fig.suptitle("Detectores independientes contra CNN entrenadas de punta a punta · las mismas 60 particiones (datasets "
                 "balanceados de 500–4000 del pool de 5620, 43 escritores) · media de 3 semillas, banda = rango · sin extrapolar",
                 x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "curvas-cnn.png", dpi=125, facecolor=SUP)
    print(f"→ {RES / 'curvas-cnn.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
