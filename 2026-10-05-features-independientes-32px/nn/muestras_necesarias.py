#!/usr/bin/env python3
"""Muestras necesarias para un nivel de acierto dado: N(ε) = cuántos dígitos de train hacen falta para acertar ε de cada 1000
dígitos nuevos (la «idea D» del análisis del 2026-10-05, pedida por el dueño). Por cada 1000 y no en número absoluto: un
recuento de aciertos depende de cuántos se evalúan; una tasa, no. Sale de los MISMOS datos que la ganancia
(resultados/ganancia.json aquí y en `feat-ind`, por su id, mismas particiones): no se entrena nada.

Por representación, la curva acierto(N) junta los cuatro tamaños de dataset (N de 10 a 2000, 3 semillas cada punto):
media por N, monótona por regresión isotónica (un acierto que baja al añadir muestras es ruido) e invertida interpolando
en log N. SÓLO dentro de lo medido: un nivel que no se alcanza con N ≤ 2000 se da como «> 2000», sin extrapolar.

Y la comprobación que G no pasaba: N(ε) calculado con CADA tamaño de dataset por separado tiene que coincidir.

⚠ Cerca del techo la curva es casi plana, así que un punto de acierto de diferencia son el DOBLE de muestras: N(ε) amplifica
el ruido. Por eso cada N(ε) lleva su rango entre semillas (la curva de cada semilla, con los cuatro tamaños juntos), y la
figura lo dibuja como banda. Medido al escribirlo: con T = 1000, N(950) da 93, 129 y 303 según la semilla.

    python nn/muestras_necesarias.py   → resultados/muestras-necesarias.json y resultados/muestras-necesarias.png
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
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn.registro import por_id              # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# mismos colores y marcadores que las otras dos figuras (validados, light: ALL PASS)
SERIES = [("detectores 8×8 (13)", "#2a78d6", "-"), ("detectores 32×32 (13)", "#eb6834", "-"),
          ("píxeles 8×8", "#1baf7a", "--"), ("píxeles 32×32", "#eda100", "--")]
RAMPA = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # tamaño del dataset: ordinal, validada con --ordinal
REFERENCIA = "píxeles 32×32"                              # el dato crudo a su resolución nativa
OBJETIVOS = (800, 900, 950, 970, 980)                     # aciertos por cada 1000 dígitos nuevos
REJILLA = np.round(np.arange(0.700, 0.9851, 0.0025), 4)


def isotonica(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Regresión isotónica (no decreciente) por «pool adjacent violators», con pesos."""
    v, ww, n = [], [], []
    for yi, wi in zip(y, w):
        v.append(float(yi)); ww.append(float(wi)); n.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tot = ww[-2] + ww[-1]
            v[-2] = (v[-2] * ww[-2] + v[-1] * ww[-1]) / tot; ww[-2] = tot; n[-2] += n[-1]
            v.pop(); ww.pop(); n.pop()
    return np.repeat(v, n)


def curva(filas: list, T: int | None = None, sem: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N, acierto monótono) con la media por N de las filas (de un T y/o una semilla, o de todas)."""
    sel = [f for f in filas if (T is None or f["T"] == T) and (sem is None or f["sem"] == sem)]
    ns = np.array(sorted({f["N"] for f in sel}))
    acc = np.array([np.mean([f["acc"] for f in sel if f["N"] == n]) for n in ns])
    peso = np.array([sum(1 for f in sel if f["N"] == n) for n in ns])
    return ns, isotonica(acc, peso)


def n_para(ns: np.ndarray, acc: np.ndarray, objetivo: float) -> tuple[float | None, str]:
    """N para llegar a `objetivo` (fracción), interpolando en log N. (None, 'debajo'/'encima') si cae fuera de lo medido."""
    if objetivo <= acc[0]:
        return None, "debajo"
    i = int(np.argmax(acc >= objetivo)) if (acc >= objetivo).any() else -1
    if i < 0:
        return None, "encima"
    t = (objetivo - acc[i - 1]) / (acc[i] - acc[i - 1])
    return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1])))), "ok"


def texto_n(v: float | None, estado: str, ns: np.ndarray) -> str:
    return f"{v:.0f}" if estado == "ok" else (f"≤ {ns[0]}" if estado == "debajo" else f"> {ns[-1]}")


def main() -> int:
    aqui = json.loads((RES / "ganancia.json").read_text())
    alli = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    for k in ("huella_y", "huella_particiones"):
        if aqui[k] != alli[k]:
            raise SystemExit(f"✗ '{k}' no coincide entre feat-ind y feat-ind32: no evaluaron los mismos dígitos. Me niego.")
    casos = {**alli["casos"], **aqui["casos"]}
    Ts = aqui["T"]
    out = {"definicion": "N(ε) = muestras de train para acertar ε de cada 1000 dígitos nuevos; curva acierto(N) con los 4 "
                         "tamaños de dataset juntos, isotónica, interpolada en log N; sin extrapolar",
           "objetivos_por_1000": OBJETIVOS, "referencia": REFERENCIA, "representaciones": {}, "por_tamano": {}}
    for nombre, *_ in SERIES:
        ns, acc = curva(casos[nombre])
        fila = {}
        for o in OBJETIVOS:
            v, e = n_para(ns, acc, o / 1000)
            fila[str(o)] = {"N": None if v is None else round(v, 1), "estado": e, "texto": texto_n(v, e, ns)}
        for o in OBJETIVOS:
            por_sem = []
            for sem in (1, 2, 3):
                nss, accs = curva(casos[nombre], sem=sem)
                por_sem.append(texto_n(*n_para(nss, accs, o / 1000), nss))
            fila[str(o)]["por_semilla"] = por_sem
        out["representaciones"][nombre] = {"N_medidos": ns.tolist(), "acierto_isotonico": np.round(acc, 4).tolist(),
                                           "acierto_max": round(float(acc[-1]), 4), "N_para": fila}
    for nombre, *_ in SERIES:                                 # cuántas veces MENOS muestras que la referencia
        for o in OBJETIVOS:
            a, r = out["representaciones"][nombre]["N_para"][str(o)], out["representaciones"][REFERENCIA]["N_para"][str(o)]
            a["veces_menos_que_referencia"] = round(r["N"] / a["N"], 2) if a["N"] and r["N"] else None
    ref = "detectores 8×8 (13)"
    for T in Ts:                                              # la comprobación: cada tamaño de dataset por separado
        ns, acc = curva(casos[ref], T)
        out["por_tamano"][str(T)] = {str(o): texto_n(*n_para(ns, acc, o / 1000), ns) for o in (850, 900, 925, 950, 970)}
    (RES / "muestras-necesarias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'aciertos por 1000 →':<24}" + "".join(f"{o:>9}" for o in OBJETIVOS))
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"{nombre:<24}" + "".join(f"{f[str(o)]['texto']:>9}" for o in OBJETIVOS)
              + f"   (máx {out['representaciones'][nombre]['acierto_max'] * 100:.1f} %)")
    print("rango entre semillas (900 · 950):")
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"  {nombre:<22}  900: {' / '.join(f['900']['por_semilla'])}   950: {' / '.join(f['950']['por_semilla'])}")
    print(f"veces menos muestras que {REFERENCIA}:")
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"  {nombre:<22}" + "".join(f"{(str(f[str(o)]['veces_menos_que_referencia']) + '×') if f[str(o)]['veces_menos_que_referencia'] else '—':>9}" for o in OBJETIVOS))
    print(f"{ref}, cada tamaño por separado (850 / 900 / 925 / 950 / 970 por 1000):")
    for T, f in out["por_tamano"].items():
        print(f"  T = {T:>4}: " + " · ".join(f"{v}" for v in f.values()))

    dibujar(casos, Ts, out)
    return 0


def dibujar(casos: dict, Ts: list, out: dict) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=SUP)
    for ax in (a1, a2):
        ax.set_facecolor(SUP); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.set_yscale("log"); ax.set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        ax.set_yticklabels(["10", "20", "50", "100", "200", "500", "1000", "2000"]); ax.minorticks_off()
        ax.set_xlabel("aciertos por cada 1000 dígitos nuevos (objetivo)")
        ax.set_xlim(700, 990)
    a1.set_ylabel("muestras de train necesarias N (escala log)")
    for nombre, col, ls in SERIES:
        ns, acc = curva(casos[nombre])
        pts = [(o * 1000, n_para(ns, acc, o)[0]) for o in REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        a1.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2.2, ls=ls, label=nombre, zorder=3)
        bandas = []                                   # el rango entre semillas, donde las tres alcanzan el nivel
        for o in REJILLA:
            vs = [n_para(*curva(casos[nombre], sem=sem), o)[0] for sem in (1, 2, 3)]
            if all(v is not None for v in vs):
                bandas.append((o * 1000, min(vs), max(vs)))
        if bandas:
            a1.fill_between([b[0] for b in bandas], [b[1] for b in bandas], [b[2] for b in bandas], color=col, alpha=0.13,
                            lw=0, zorder=2)
        a1.plot(pts[-1][0], pts[-1][1], "o", color=col, ms=7, mec=SUP, mew=1.5, zorder=4)
    a1.legend(frameon=False, loc="upper left", fontsize=9.5)
    a1.set_title("1 · ¿Cuántas muestras cuesta cada nivel de acierto?", loc="left", fontsize=12.5, color=T1)
    r = out["representaciones"]
    a1.text(0.36, 0.97, f"{'':<22}{'N(900)':>7}{'N(950)':>8}{'máx':>6}\n" + "\n".join(
        f"{n:<22}{r[n]['N_para']['900']['texto']:>7}{r[n]['N_para']['950']['texto']:>8}{r[n]['acierto_max'] * 1000:>6.0f}"
        for n, *_ in SERIES)
        + "\n● = lo máximo alcanzado con N ≤ 2000\nbanda = rango entre las 3 semillas", transform=a1.transAxes, ha="left", va="top",
        fontsize=9, color=T2, family="monospace")
    ref = "detectores 8×8 (13)"
    for T, col in zip(Ts, RAMPA):
        ns, acc = curva(casos[ref], T)
        pts = [(o * 1000, n_para(ns, acc, o)[0]) for o in REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        if pts:
            a2.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2.2, label=f"T = {T}  (N de {ns[0]} a {ns[-1]})", zorder=3)
    a2.legend(frameon=False, loc="upper left", fontsize=9.5, title="detectores 8×8 (13), cada dataset por separado",
              title_fontsize=9.5)
    a2.set_title("2 · ¿Independiente del tamaño del dataset? Sí", loc="left", fontsize=12.5, color=T1)
    pt = out["por_tamano"]
    a2.text(0.02, 0.62, "N para 900 / 950 por 1000:\n" + "\n".join(f"T = {T:<5}{pt[str(T)]['900']:>8} / {pt[str(T)]['950']:>6}" for T in Ts)
            + "\n(≤ / >: fuera del rango de N de ese dataset)\nT = 4000 a 950: dentro del ruido; con T = 1000\nlas 3 semillas dan 93 · 129 · 303",
            transform=a2.transAxes, ha="left", va="top", fontsize=9,
            color=T2, family="monospace")
    fig.suptitle("Muestras necesarias N(ε) · los mismos 240 entrenamientos de la ganancia (datasets balanceados de 500–4000 del "
                 "pool de 5620, 43 escritores) · compositor lineal posicional · sin extrapolar", x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "muestras-necesarias.png", dpi=130, facecolor=SUP)
    print(f"→ {RES / 'muestras-necesarias.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
