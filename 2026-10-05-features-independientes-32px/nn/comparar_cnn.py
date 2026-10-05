#!/usr/bin/env python3
"""Comparación con CNN «tradicionales» (entrenadas de punta a punta sobre los dígitos), pedida por el dueño el 2026-10-05.
NO entrena ninguna CNN: lee las que ya se entrenaron en el repo, que usan EXACTAMENTE el mismo dato que la corrida 2 de
`feat-ind` y la C4 de aquí — `uci-optdigits-8px-r20261002`, 180 de train y 1617 de validación —, así que en N = 180 la
comparación es directa:

  dim-nist   brazo w8: 2 capas sin padding, kernel 4, 1.258 parámetros, 5 semillas
  ruido-comb limpio: 3 × conv 3×3 (C = 8) + GAP, 1.338 parámetros, 5 semillas (la de ruido-nist, con 2 semillas más)
  ruido-comb recorte@0.6+gaussiano@0.2-linea: la misma CNN con el mejor aumento de datos medido, 5 semillas

Se leen por su id en el registro (nunca por su carpeta) y se comprueba que su dataset es ese. Lo único que se calcula aquí
es la regresión logística sobre los píxeles con ese mismo 180/1617 (segundos), para tener la referencia sin red.

Y como sólo hay un punto (N = 180), la otra comparación es por MUESTRAS EQUIVALENTES: cuántas muestras necesita cada una
de nuestras curvas (resultados/muestras-necesarias, mismas funciones) para el acierto de cada CNN.

    python nn/comparar_cnn.py      → resultados/comparacion-cnn.json y resultados/comparacion-cnn.png
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset              # noqa: E402
from expcnn.registro import por_id              # noqa: E402
import componer as K                            # noqa: E402
import datos                                    # noqa: E402
import muestras_necesarias as M                 # noqa: E402

RES = AQUI.parent / "resultados"
DATO_8PX = "uci-optdigits-8px-r20261002"
CNN = [  # (etiqueta, id del experimento, campo, valor, descripción)
    ("CNN 2 capas (dim-nist)", "dim-nist", "brazo", "w8", "2 capas sin padding, kernel 4"),
    ("CNN 3 capas (ruido-comb, limpio)", "ruido-comb", "escenario", "limpio", "3 × conv 3×3, C = 8, GAP"),
    ("CNN 3 capas + mejor aumento de datos", "ruido-comb", "escenario", "recorte@0.6+gaussiano@0.2-linea",
     "la misma, con recorte + ruido gaussiano en línea"),
]
SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MARCAS = ["v", "^", "P"]


def cnn_existentes() -> list[dict]:
    out = []
    for etiqueta, ident, campo, valor, desc in CNN:
        filas = [json.loads(f.read_text()) for f in sorted((por_id(ident).carpeta / "nn" / "pesos").glob("*/summary.json"))]
        filas = [s for s in filas if s.get(campo) == valor]
        if not filas:
            raise SystemExit(f"✗ {ident}: no hay corridas con {campo} = {valor}")
        malos = {s["dataset"] for s in filas} - {DATO_8PX}
        if malos:
            raise SystemExit(f"✗ {ident}/{valor} usó {malos}, no {DATO_8PX}: no es el mismo dato. Me niego.")
        acc = [s["acc_val"] * 100 for s in filas]
        out.append({"etiqueta": etiqueta, "experimento": ident, campo: valor, "descripcion": desc,
                    "parametros": filas[0]["parametros"], "semillas": len(acc), "acc_val_pct": round(st.mean(acc), 2),
                    "sd_pct": round(st.stdev(acc), 2), "por_semilla_pct": [round(a, 2) for a in acc]})
    return out


def logistica_pixeles() -> dict:
    """Regresión logística (el compositor lineal, mismas épocas/lr/L2) sobre los píxeles, 180 / 1617, 3 semillas."""
    z = np.load(exigir_dataset(DATO_8PX) / "datos.npz")
    x8 = (z["imagenes"].astype(np.float32) / 16).reshape(len(z["etiquetas"]), -1)
    y, tr = z["etiquetas"].astype(np.int64), z["particion"] == "train"
    d32 = datos.digitos()                              # los 1797 windep, mismo orden y reparto (comprobado en datos.py)
    if not (np.array_equal(d32["y"], y) and np.array_equal(d32["train"], tr)):
        raise SystemExit("✗ los dígitos de 32×32 no tienen el mismo orden o reparto que los de 8 px. Me niego.")
    x32 = d32["x"].reshape(len(y), -1).astype(np.float32)
    out = {}
    for nombre, x in (("píxeles 8×8", x8), ("píxeles 32×32", x32)):
        acc = [float((K.predecir(K.ajustar(x[tr], y[tr], s), x[~tr]) == y[~tr]).mean()) * 100 for s in K.SEMILLAS]
        out[nombre] = {"acc_val_pct": round(st.mean(acc), 2), "sd_pct": round(st.stdev(acc), 2), "n_train": int(tr.sum()),
                       "n_val": int((~tr).sum())}
    return out


def main() -> int:
    cnn = cnn_existentes()
    pix = logistica_pixeles()
    comp8 = json.loads((por_id("feat-ind").carpeta / "resultados" / "compositores.json").read_text())
    comp32 = json.loads((RES / "compositores.json").read_text())
    for c in (comp8, comp32):
        if (c["n_train"], c["n_val"]) != (180, 1617):
            raise SystemExit("✗ los compositores no son 180 / 1617. Me niego.")
    puntos = [{"etiqueta": "detectores 8×8 (13)", "acc_val_pct": round(comp8["posicional"]["acc_val_media"] * 100, 2),
               "sd_pct": round(comp8["posicional"]["acc_val_sd"] * 100, 2), "aprende_de_los_180": "lineal 832 → 10 (8.330 parámetros)"},
              {"etiqueta": "detectores 32×32 (13)", "acc_val_pct": round(comp32["posicional"]["acc_val_media"] * 100, 2),
               "sd_pct": round(comp32["posicional"]["acc_val_sd"] * 100, 2), "aprende_de_los_180": "lineal 832 → 10 (8.330 parámetros)"},
              {"etiqueta": "píxeles 32×32", **pix["píxeles 32×32"], "aprende_de_los_180": "lineal 1024 → 10 (10.250 parámetros)"},
              {"etiqueta": "píxeles 8×8", **pix["píxeles 8×8"], "aprende_de_los_180": "lineal 64 → 10 (650 parámetros)"}]
    for c in cnn:
        puntos.append({"etiqueta": c["etiqueta"], "acc_val_pct": c["acc_val_pct"], "sd_pct": c["sd_pct"],
                       "aprende_de_los_180": f"toda la red ({c['parametros']:,} parámetros)".replace(",", ".")})

    # muestras equivalentes: cuántas necesitan nuestras curvas para el acierto de cada CNN
    aqui = json.loads((RES / "ganancia.json").read_text())
    alli = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    casos = {**alli["casos"], **aqui["casos"]}
    curvas = {n: M.curva(casos[n]) for n, *_ in M.SERIES}
    for c in cnn:
        c["muestras_equivalentes"] = {}
        for n, *_ in M.SERIES:
            v, e = M.n_para(*curvas[n], c["acc_val_pct"])
            c["muestras_equivalentes"][n] = {"N": None if v is None else round(v, 1), "texto": M.texto_n(v, e, curvas[n][0]),
                                             "veces_menos_que_180": None if v is None else round(180 / v, 2)}
    out = {"protocolo": f"{DATO_8PX}: 180 train / 1617 val (windep), el mismo en todos los puntos de N = 180",
           "cnn": cnn, "puntos_N180": puntos,
           "nota": "las curvas (muestras equivalentes) son de los datasets mezclados de la ganancia; en N = 180 los dos "
                   "protocolos casi coinciden para detectores 8×8 (95,9 % aquí, ~95,8 % en la curva)"}
    (RES / "comparacion-cnn.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("N = 180 / 1617 (el mismo train y la misma val para todos):")
    for p in sorted(puntos, key=lambda p: -p["acc_val_pct"]):
        print(f"  {p['etiqueta']:<38} {p['acc_val_pct']:5.1f} ± {p['sd_pct']:.1f} %   aprende de los 180: {p['aprende_de_los_180']}")
    print("muestras que necesita cada curva para el acierto de cada CNN (la CNN usó 180):")
    for c in cnn:
        print(f"  {c['etiqueta']:<38} {c['acc_val_pct']:.1f} % → " + " · ".join(
            f"{n.split(' (')[0]} {v['texto']}" for n, v in c["muestras_equivalentes"].items()))
    dibujar(curvas, cnn, puntos)
    return 0


def dibujar(curvas: dict, cnn: list, puntos: list) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=SUP, gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (a1, a2):
        M._ejes(ax)
    # 1. las curvas, con las CNN en N = 180
    M._eje_n(a1, "x"); a1.set_xlim(8, 2600)
    for nombre, col, ls in M.SERIES:
        ns, acc = curvas[nombre]
        a1.plot(ns, acc, color=col, lw=2.2, ls=ls, label=nombre, zorder=3)
    for c, mk, dx in zip(cnn, MARCAS, (0.88, 1.0, 1.13)):     # las tres son N = 180: se separan un poco para que se lean
        a1.errorbar(180 * dx, c["acc_val_pct"], yerr=c["sd_pct"], fmt=mk, color=T1, ms=9, mec=SUP, mew=1.2, capsize=3, lw=1,
                    zorder=5, label=f"{c['etiqueta']}: {c['acc_val_pct']:.1f} %")
    base = cnn[1]                                    # la CNN de 3 capas sin aumento: la flecha de las muestras equivalentes
    eq = base["muestras_equivalentes"]["detectores 8×8 (13)"]["N"]
    a1.annotate("", xy=(eq, base["acc_val_pct"]), xytext=(180, base["acc_val_pct"]),
                arrowprops=dict(arrowstyle="->", color=T1, lw=1.2, shrinkA=6, shrinkB=0), zorder=4)
    a1.annotate(f"la CNN de 3 capas saca {base['acc_val_pct']:.1f} % con 180 muestras;\nlos detectores 8×8 lo sacan con {eq:.0f} "
                f"({180 / eq:.1f}× menos)", xy=(np.sqrt(eq * 180), base["acc_val_pct"]), xytext=(160, 77.5), ha="center", va="center",
                fontsize=9.5, color=T1, arrowprops=dict(arrowstyle="-", color=T2, lw=0.8, shrinkA=4, shrinkB=2))
    a1.set_ylim(a1.get_ylim()[0], 100.5); a1.set_yticks([60, 70, 80, 90, 100])
    a1.set_xlabel("muestras de train N (escala log)"); a1.set_ylabel("acierto sobre los dígitos nuevos (%)")
    a1.legend(frameon=False, loc="lower right", fontsize=9)
    a1.set_title("1 · Nuestras curvas y las CNN del repo (sólo medidas en N = 180)", loc="left", fontsize=12.5, color=T1)
    # 2. N = 180, el mismo train y la misma val para todos
    orden = sorted(puntos, key=lambda p: p["acc_val_pct"])
    colores = {n: c for n, c, _ in M.SERIES}
    for i, p in enumerate(orden):
        col = colores.get(p["etiqueta"], T1)
        mk = "o" if p["etiqueta"] in colores else MARCAS[[c["etiqueta"] for c in cnn].index(p["etiqueta"])]
        a2.errorbar(p["acc_val_pct"], i, xerr=p["sd_pct"], fmt=mk, color=col, ms=9, mec=SUP, mew=1.2, capsize=3, lw=1.2, zorder=3)
        a2.annotate(f"{p['acc_val_pct']:.1f} %", (p["acc_val_pct"] + p["sd_pct"], i), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=T1)
    a2.set_yticks(range(len(orden)))
    a2.set_yticklabels([f"{p['etiqueta']}\naprende de los 180: {p['aprende_de_los_180']}" for p in orden], fontsize=9)
    a2.set_xlim(80, 100); a2.set_xlabel("acierto en val (%) · media ± sd entre semillas")
    a2.grid(True, axis="x", color=GRID, lw=0.8); a2.grid(False, axis="y")
    a2.set_title("2 · En N = 180: mismos 180 de train y 1617 de val para todos", loc="left", fontsize=12.5, color=T1)
    fig.suptitle("Comparación con CNN entrenadas de punta a punta (las del repo: dim-nist, ruido-comb) · dígitos UCI/NIST · "
                 "las CNN NO se reentrenaron: sólo hay su punto de N = 180", x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "comparacion-cnn.png", dpi=125, facecolor=SUP)
    print(f"→ {RES / 'comparacion-cnn.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
