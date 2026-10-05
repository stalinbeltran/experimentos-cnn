#!/usr/bin/env python3
"""A, B y C —las CNN con el compositor de los detectores— contra los detectores congelados y las CNN de siempre, TODAS sobre
las mismas 60 particiones (se comprueba por huella), y la evaluación del criterio escrito antes de correr
(instrucciones/02-criterio.md § «CNN con el compositor de los detectores»).

  1. Definir las features o aprenderlas: detectores congelados · C (ajuste fino) · C en su fase congelada · B · LeNet-5
  2. ¿Arregla el compositor a la CNN del repo?: CNN de 3 capas · A · y como referencia detectores y píxeles 8×8

Mismas funciones que nn/muestras_necesarias.py (media por N, isotónica, interpolación en log N, sin extrapolar).

    python nn/figura_compositor.py   → resultados/compositor-comparacion.json y resultados/compositor.png
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
DET8, PIX8 = "detectores 8×8 (13)", "píxeles 8×8"
A, B, C, C0 = "A · CNN 3 capas + compositor", "B · detectores aprendidos", "C · detectores sintéticos + ajuste fino", "C · fase congelada"
CNN3, LENET = "CNN 3 capas 8×8 (la del repo)", "LeNet-5 32×32"
# paleta de referencia, slots 1-7 en orden (validada junta, light: ALL PASS); el color sigue a la entidad en toda la figura
COLOR = {DET8: "#2a78d6", B: "#eb6834", PIX8: "#1baf7a", A: "#eda100", CNN3: "#e87ba4", C: "#008300", C0: "#008300",
         LENET: "#4a3aa7"}
ESTILO = {DET8: "-", B: "-", PIX8: "--", A: "-", CNN3: "-.", C: "-", C0: ":", LENET: "-."}
PANEL1 = [DET8, C, C0, B, LENET]
PANEL2 = [DET8, A, PIX8, CNN3]


def main() -> int:
    g8 = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    cc = json.loads((RES / "curvas-cnn.json").read_text())
    if not (g8["huella_y"] == cc["huella_y"] and g8["huella_particiones"] == cc["huella_particiones"]):
        raise SystemExit("✗ las CNN y la ganancia no usan las mismas particiones. Me niego.")
    casos = {**g8["casos"], **cc["casos"]}
    casos[C0] = [{**f, "acc": f["acc_congelado"]} for f in casos[C]]          # la misma corrida, al acabar la fase congelada
    faltan = {n: 60 - len(casos[n]) for n in COLOR if len(casos[n]) != 60}
    curvas = {n: M.curva(casos[n]) for n in COLOR}
    ns = curvas[DET8][0]
    if any(not np.array_equal(curvas[n][0], ns) for n in COLOR):
        raise SystemExit("✗ las curvas no tienen los mismos N. Me niego.")
    a = {n: curvas[n][1] for n in COLOR}
    en = lambda n, N: float(np.interp(np.log(N), np.log(ns), a[n]))          # noqa: E731
    tabla = {n: {"90": M.texto_n(*M.n_para(*curvas[n], 90), ns), "95": M.texto_n(*M.n_para(*curvas[n], 95), ns),
                 "max": round(float(a[n][-1]), 2)} for n in COLOR}
    pix_mejor = np.maximum(a[PIX8], M.curva(json.loads((RES / "ganancia.json").read_text())["casos"]["píxeles 32×32"])[1])
    # criterio
    d_a = a[A] - a[CNN3]
    a180 = float(np.mean([en(A, 160), en(A, 200)])) - float(np.mean([en(CNN3, 160), en(CNN3, 200)]))
    desde_a = next((int(n) for i, n in enumerate(ns) if (a[A][i:] >= pix_mejor[i:]).all()), None)
    d_b = a[B] - a[DET8]
    cruce_b = next((int(n) for i, n in enumerate(ns) if (d_b[i:] >= 0).all()), None)
    d_c = a[C] - a[DET8]
    crit = {
        "1_A_contra_cnn3": {"A_menos_cnn3_por_N": dict(zip(map(int, ns), np.round(d_a, 2).tolist())),
                            "en_N160_200": round(a180, 2), "prediccion": "+4 a +8 en N = 160–200; A ≥ mejor píxeles desde N ≈ 80",
                            "A_por_encima_en_todo_N": bool((d_a > 0).all()), "A_sobre_mejor_pixeles_desde_N": desde_a,
                            "cumple": bool((d_a > 0).all() and 4 <= a180 <= 8)},
        "2_B_contra_det8": {"B_menos_det8_por_N": dict(zip(map(int, ns), np.round(d_b, 2).tolist())),
                            "B_por_encima_desde_N": cruce_b, "prediccion": "B debajo con N ≤ 200, cruce entre 200 y 1000",
                            "cumple": bool(cruce_b is not None and 200 < cruce_b <= 1000 and (d_b[ns <= 200] < 0).all())},
        "3_C": {"C_menos_det8_por_N": dict(zip(map(int, ns), np.round(d_c, 2).tolist())),
                "C_menos_B_por_N": dict(zip(map(int, ns), np.round(a[C] - a[B], 2).tolist())),
                "C_menos_lenet_por_N": dict(zip(map(int, ns), np.round(a[C] - a[LENET], 2).tolist())),
                "prediccion": "C ≥ det8 − 1 en todo N; C > B en N ≤ 200; C > LeNet-5 en todo N",
                "cumple": {"no_peor_que_det8": bool((d_c >= -1).all()), "sobre_B_en_N_le_200": bool(((a[C] - a[B])[ns <= 200] > 0).all()),
                           "sobre_lenet_en_todo_N": bool((a[C] > a[LENET]).all())}},
        "4_fase_congelada": {"C0_menos_det8_por_N": dict(zip(map(int, ns), np.round(a[C0] - a[DET8], 2).tolist())),
                             "ajuste_fino_C_menos_C0_por_N": dict(zip(map(int, ns), np.round(a[C] - a[C0], 2).tolist())),
                             "prediccion": "C0 por encima de det8 en N ≥ 400 (protocolo del compositor)",
                             "cumple": bool((a[C0] - a[DET8])[ns >= 400].min() > 0)},
    }
    out = {"faltan_entrenamientos": faltan, "N": ns.tolist(), "curvas_pct": {n: np.round(a[n], 2).tolist() for n in COLOR},
           "N_para": tabla, "criterio": crit}
    (RES / "compositor-comparacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if faltan:
        print(f"⚠ faltan entrenamientos: {faltan}")
    print(f"{'acierto (%) por N →':<42}" + "".join(f"{int(n):>6}" for n in ns))
    for n in COLOR:
        print(f"{n:<42}" + "".join(f"{v:>6.1f}" for v in a[n]) + f"   N(90) {tabla[n]['90']:>6} · N(95) {tabla[n]['95']:>6}")
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
    fig, ejes = plt.subplots(1, 2, figsize=(17, 6.8), facecolor=SUP, sharey=True)
    titulos = ["1 · Definir las features o aprenderlas", "2 · ¿Arregla el compositor a la CNN del repo?"]
    for ax, series, titulo in zip(ejes, (PANEL1, PANEL2), titulos):
        M._ejes(ax); M._eje_n(ax, "x"); ax.set_xlim(8, 2600)
        for n in series:
            ns, acc = curvas[n]
            ax.plot(ns, acc, color=COLOR[n], lw=2.2, ls=ESTILO[n], label=f"{n}  (N95 {tabla[n]['95']})", zorder=3)
            if n != C0:
                por_sem = np.array([M.curva(casos[n], sem=s)[1] for s in M.SEMILLAS])
                ax.fill_between(ns, por_sem.min(0), por_sem.max(0), color=COLOR[n], alpha=0.10, lw=0, zorder=2)
        ax.set_xlabel("muestras de train N (escala log)")
        ax.legend(frameon=False, loc="lower right", fontsize=9.5, title="N95 = muestras para el 95 %", title_fontsize=9)
        ax.set_title(titulo, loc="left", fontsize=12.5, color=T1)
    ejes[0].set_ylabel("acierto sobre los dígitos nuevos (%)")
    ejes[0].set_ylim(ejes[0].get_ylim()[0], 100.3)
    fig.suptitle("CNN con el compositor de los detectores (σ de 13 mapas 8×8 → lineal) · las mismas 60 particiones · 3996 pasos "
                 "de 20, sin selección ni aumento · media de 3 semillas, banda = rango", x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "compositor.png", dpi=125, facecolor=SUP)
    print(f"→ {RES / 'compositor.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
