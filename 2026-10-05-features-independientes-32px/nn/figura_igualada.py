#!/usr/bin/env python3
"""La CNN del repo con el terreno igualado: la escalera (+ padding, + cabeza densa, + capacidad, y el control 3b con el lr de
la del repo) contra C, A y LeNet-5, todas sobre las mismas 60 particiones (se comprueba por huella), y la evaluación del
criterio escrito antes de correr (instrucciones/02-criterio.md § «La CNN del repo con el terreno igualado»).

  1. la curva de aprendizaje de la escalera y las referencias
  2. N(ε): cuántas muestras cuesta cada nivel de acierto

    python nn/figura_igualada.py   → resultados/igualada-comparacion.json y resultados/igualada.png
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
import muestras_necesarias as M                 # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, GRID = M.SUP, M.T1, M.T2, M.GRID
REPO, P1, P2, P3, P3B = ("CNN 3 capas 8×8 (la del repo)", "CNN 3 capas · 1 + padding", "CNN 3 capas · 2 + cabeza densa",
                         "CNN 3 capas · 3 + capacidad (187k), lr 1e-3", "CNN 3 capas · 3b + capacidad, lr 3e-3")
A, B, C, LENET = ("A · CNN 3 capas + compositor", "B · detectores aprendidos", "C · detectores sintéticos + ajuste fino",
                  "LeNet-5 32×32")
# la escalera es ORDINAL: una rampa de un solo tono, más oscuro = más igualada (la azul de la paleta, validada --ordinal);
# 3b, mismo tono que 3 y punteada. Referencias en sus colores de las otras figuras (A amarillo, C verde, LeNet-5 violeta).
COLOR = {REPO: "#86b6ef", P1: "#3987e5", P2: "#1c5cab", P3: "#0d366b", P3B: "#0d366b", A: "#eda100", C: "#008300", LENET: "#4a3aa7"}
ESTILO = {REPO: "-", P1: "-", P2: "-", P3: "-", P3B: ":", A: "--", C: "--", LENET: "-."}


def main() -> int:
    cc = json.loads((RES / "curvas-cnn.json").read_text())
    g = json.loads((RES / "ganancia.json").read_text())
    if not (g["huella_y"] == cc["huella_y"] and g["huella_particiones"] == cc["huella_particiones"]):
        raise SystemExit("✗ las curvas no usan las particiones de la ganancia. Me niego.")
    casos = cc["casos"]
    faltan = {n: 60 - len(casos.get(n, [])) for n in COLOR if len(casos.get(n, [])) != 60}
    if any(n in faltan for n in (P1, P2, P3, P3B)):
        print(f"⚠ faltan entrenamientos: {faltan}")
    curvas = {n: M.curva(casos[n]) for n in list(COLOR) + [B]}
    ns = curvas[REPO][0]
    if any(not np.array_equal(curvas[n][0], ns) for n in curvas):
        raise SystemExit("✗ las curvas no tienen los mismos N. Me niego.")
    a = {n: curvas[n][1] for n in curvas}
    en = lambda n, N: float(np.interp(np.log(N), np.log(ns), a[n]))          # noqa: E731
    m180 = lambda n: float(np.mean([en(n, 160), en(n, 200)]))                 # noqa: E731
    n95 = {n: M.n_para(*curvas[n], 95)[0] for n in curvas}
    tabla = {n: {"90": M.texto_n(*M.n_para(*curvas[n], 90), ns), "95": M.texto_n(*M.n_para(*curvas[n], 95), ns),
                 "N200": round(en(n, 200), 1), "max": round(float(a[n][-1]), 2)} for n in curvas}
    d = lambda x, y: dict(zip(map(int, ns), np.round(a[x] - a[y], 2).tolist()))   # noqa: E731
    grande, peq, muchos = ns >= 400, ns <= 40, ns >= 1000
    rungs95 = [v for v in (n95[P1], n95[P2], n95[P3], n95[P3B]) if v is not None]
    crit = {
        "1_padding": {"P1_menos_repo_N160_200": round(m180(P1) - m180(REPO), 2), "P1_menos_repo_por_N": d(P1, REPO),
                      "prediccion": "dentro de ±3 puntos; si sube > 5, la reducción era el hándicap principal",
                      "cumple": abs(m180(P1) - m180(REPO)) <= 3, "hipotesis_del_dueno": m180(P1) - m180(REPO) > 5},
        "2_cabeza_densa": {"P2_menos_P1_N160_200": round(m180(P2) - m180(P1), 2), "P2_menos_A_por_N": d(P2, A),
                           "prediccion": "≥ +5 sobre el peldaño 1 en N = 160–200; a ±2 de A en todo N ≥ 40",
                           "cumple": bool(m180(P2) - m180(P1) >= 5 and (np.abs(a[P2] - a[A])[ns >= 40] <= 2).all())},
        "3_capacidad": {"P3_menos_P2_por_N": d(P3, P2), "P3_menos_B_por_N": d(P3, B), "C_menos_P3_por_N": d(C, P3),
                        "P3b_menos_P3_por_N": d(P3B, P3),
                        "prediccion": "≥ +1 sobre el 2 desde N = 400; ≈ B con muchos datos; < C en todo N, por ≥ 5 con N ≤ 40",
                        "cumple": {"sobre_P2_desde_400": bool(((a[P3] - a[P2])[grande] >= 1).all()),
                                   "parecido_a_B_desde_1000": bool((np.abs(a[P3] - a[B])[muchos] <= 1).all()),
                                   "bajo_C_en_todo_N": bool((a[C] > a[P3]).all()),
                                   "C_le_saca_5_con_N_le_40": bool(((a[C] - a[P3])[peq] >= 5).all())}},
        "4_N95": {"P1": tabla[P1]["95"], "P2": tabla[P2]["95"], "P3": tabla[P3]["95"], "P3b": tabla[P3B]["95"], "C": tabla[C]["95"],
                  "prediccion": "P2 400–700; P3 200–400; C ≤ la mitad que cualquier peldaño",
                  "cumple": {"P2_400_700": n95[P2] is not None and 400 <= n95[P2] <= 700,
                             "P3_200_400": n95[P3] is not None and 200 <= n95[P3] <= 400,
                             "C_mitad": bool(rungs95) and n95[C] <= 0.5 * min(rungs95)}},
    }
    out = {"faltan_entrenamientos": faltan, "N": ns.tolist(), "curvas_pct": {n: np.round(a[n], 2).tolist() for n in curvas},
           "tabla": tabla, "criterio": crit}
    (RES / "igualada-comparacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'acierto (%) por N →':<46}" + "".join(f"{int(n):>6}" for n in ns))
    for n in list(COLOR) + [B]:
        print(f"{n:<46}" + "".join(f"{v:>6.1f}" for v in a[n]) + f"   N(90) {tabla[n]['90']:>6} · N(95) {tabla[n]['95']:>6}")
    print("\ncriterio:")
    for k, v in crit.items():
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"     {kk}: {vv}")
    dibujar(curvas, casos, tabla)
    return 0


def dibujar(curvas: dict, casos: dict, tabla: dict) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.8), facecolor=SUP)
    for ax in (a1, a2):
        M._ejes(ax)
    M._eje_n(a1, "x"); a1.set_xlim(8, 2600)
    for n in COLOR:
        ns, acc = curvas[n]
        a1.plot(ns, acc, color=COLOR[n], lw=2.4 if n in (P1, P2, P3, P3B, REPO) else 1.8, ls=ESTILO[n],
                label=f"{n}  (N95 {tabla[n]['95']})", zorder=3)
        if n in (P2, P3):
            por_sem = np.array([M.curva(casos[n], sem=s)[1] for s in M.SEMILLAS])
            a1.fill_between(ns, por_sem.min(0), por_sem.max(0), color=COLOR[n], alpha=0.10, lw=0, zorder=2)
    a1.set_ylim(a1.get_ylim()[0], 100.3)
    a1.set_xlabel("muestras de train N (escala log)"); a1.set_ylabel("acierto sobre los dígitos nuevos (%)")
    a1.legend(frameon=False, loc="lower right", fontsize=9, title="N95 = muestras para el 95 %", title_fontsize=9)
    a1.set_title("1 · La CNN del repo, igualando el terreno peldaño a peldaño", loc="left", fontsize=12.5, color=T1)
    M._eje_n(a2, "y"); a2.set_xlim(70, 99.5)
    a2.set_xlabel("acierto objetivo sobre los dígitos nuevos (%)"); a2.set_ylabel("muestras de train necesarias N (escala log)")
    for n in COLOR:
        ns, acc = curvas[n]
        pts = [(o, M.n_para(ns, acc, o)[0]) for o in M.REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        if pts:
            a2.plot([x for x, _ in pts], [v for _, v in pts], color=COLOR[n], lw=2.2, ls=ESTILO[n], zorder=3)
            a2.plot(pts[-1][0], pts[-1][1], "o", color=COLOR[n], ms=6, mec=SUP, mew=1.2, zorder=4)
    a2.text(0.02, 0.97, f"{'':<44}{'N(90)':>6}{'N(95)':>7}{'N=200':>7}\n" + "\n".join(
        f"{n:<44}{tabla[n]['90']:>6}{tabla[n]['95']:>7}{tabla[n]['N200']:>6.1f}%" for n in COLOR),
        transform=a2.transAxes, ha="left", va="top", fontsize=8.5, color=T2, family="monospace")
    a2.set_title("2 · ¿Cuántas muestras cuesta cada nivel de acierto?", loc="left", fontsize=12.5, color=T1)
    fig.suptitle("La CNN de 3 capas del repo con el terreno igualado (+ padding, + cabeza densa, + capacidad) · las mismas 60 "
                 "particiones · 3996 pasos de 20, sin selección ni aumento · media de 3 semillas", x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "igualada.png", dpi=125, facecolor=SUP)
    print(f"→ {RES / 'igualada.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
