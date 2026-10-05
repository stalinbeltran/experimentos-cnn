#!/usr/bin/env python3
"""Muestras necesarias para un nivel de acierto dado: N(ε) = cuántos dígitos de train hacen falta para llegar a un acierto ε
sobre los dígitos nuevos (la «idea D» del análisis del 2026-10-05, pedida por el dueño). ε es una TASA (95 % = 950 aciertos
de cada 1000): un recuento de aciertos dependería de cuántos se evalúan. Sale de los MISMOS datos que la ganancia
(resultados/ganancia.json aquí y en `feat-ind`, por su id, mismas particiones): no se entrena nada.

Por representación, la curva acierto(N) junta los cuatro tamaños de dataset (N de 10 a 2000, 3 semillas cada punto):
media por N, monótona por regresión isotónica (un acierto que baja al añadir muestras es ruido) e invertida interpolando
en log N. SÓLO dentro de lo medido: un nivel que no se alcanza con N ≤ 2000 se da como «> 2000», sin extrapolar.

N(ε) es la curva de aprendizaje con los ejes intercambiados: no añade información, cambia la pregunta («¿cuántas muestras
para llegar a ε?» en vez de «¿qué acierto con N?») y la dirección en que se compara (en horizontal, a igual acierto). La
figura pone las dos lado a lado, con el mismo ejemplo leído de las dos formas.

Y la comprobación que G no pasaba: N(ε) calculado con CADA tamaño de dataset por separado tiene que coincidir.

⚠ Cerca del techo la curva es casi plana, así que un punto de acierto de diferencia son el DOBLE de muestras: N(ε) amplifica
el ruido. Por eso cada N(ε) lleva su rango entre semillas (la curva de cada semilla, con los cuatro tamaños juntos), y la
figura lo dibuja como banda. Medido al escribirlo: con T = 1000, N(95 %) da 93, 129 y 303 según la semilla.

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
# mismos colores que las otras dos figuras (validados, light: ALL PASS)
SERIES = [("detectores 8×8 (13)", "#2a78d6", "-"), ("detectores 32×32 (13)", "#eb6834", "-"),
          ("píxeles 8×8", "#1baf7a", "--"), ("píxeles 32×32", "#eda100", "--")]
RAMPA = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # tamaño del dataset: ordinal, validada con --ordinal
REFERENCIA = "píxeles 32×32"                              # el dato crudo a su resolución nativa
EJEMPLO = ("detectores 8×8 (13)", REFERENCIA, 95.0)      # el ejemplo que se lee de las dos formas en el panel 2
OBJETIVOS = (80, 90, 95, 97, 98)                          # acierto objetivo, %
OBJETIVOS_T = (85, 90, 92.5, 95, 97)                      # los de la comprobación por tamaño de dataset
SEMILLAS = (1, 2, 3)
REJILLA = np.round(np.arange(70.0, 98.51, 0.25), 2)       # %
YTICKS_N = [10, 20, 50, 100, 200, 500, 1000, 2000]


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
    """(N, acierto monótono en %) con la media por N de las filas (de un T y/o una semilla, o de todas)."""
    sel = [f for f in filas if (T is None or f["T"] == T) and (sem is None or f["sem"] == sem)]
    ns = np.array(sorted({f["N"] for f in sel}))
    acc = np.array([np.mean([f["acc"] for f in sel if f["N"] == n]) for n in ns]) * 100
    peso = np.array([sum(1 for f in sel if f["N"] == n) for n in ns])
    return ns, isotonica(acc, peso)


def n_para(ns: np.ndarray, acc: np.ndarray, objetivo: float) -> tuple[float | None, str]:
    """N para llegar a `objetivo` (%), interpolando en log N. (None, 'debajo'/'encima') si cae fuera de lo medido."""
    if objetivo <= acc[0]:
        return None, "debajo"
    if not (acc >= objetivo).any():
        return None, "encima"
    i = int(np.argmax(acc >= objetivo))
    t = (objetivo - acc[i - 1]) / (acc[i] - acc[i - 1])
    return float(np.exp(np.log(ns[i - 1]) + t * (np.log(ns[i]) - np.log(ns[i - 1])))), "ok"


def acierto_en(ns: np.ndarray, acc: np.ndarray, n: float) -> float:
    """El acierto (%) de la curva en un N cualquiera dentro de lo medido, interpolando en log N (lo inverso de n_para)."""
    return float(np.interp(np.log(n), np.log(ns), acc))


def texto_n(v: float | None, estado: str, ns: np.ndarray) -> str:
    return f"{v:.0f}" if estado == "ok" else (f"≤ {ns[0]}" if estado == "debajo" else f"> {ns[-1]}")


def clave(o: float) -> str:
    return f"{o:g}"


def main() -> int:
    aqui = json.loads((RES / "ganancia.json").read_text())
    alli = json.loads((por_id("feat-ind").carpeta / "resultados" / "ganancia.json").read_text())
    for k in ("huella_y", "huella_particiones"):
        if aqui[k] != alli[k]:
            raise SystemExit(f"✗ '{k}' no coincide entre feat-ind y feat-ind32: no evaluaron los mismos dígitos. Me niego.")
    casos = {**alli["casos"], **aqui["casos"]}
    Ts = aqui["T"]
    out = {"definicion": "N(ε) = muestras de train para llegar a un acierto ε (%) sobre los dígitos nuevos; curva acierto(N) "
                         "con los 4 tamaños de dataset juntos, isotónica, interpolada en log N; sin extrapolar",
           "objetivos_pct": OBJETIVOS, "referencia": REFERENCIA, "representaciones": {}, "por_tamano": {}}
    for nombre, *_ in SERIES:
        ns, acc = curva(casos[nombre])
        fila = {}
        for o in OBJETIVOS:
            v, e = n_para(ns, acc, o)
            por_sem = []
            for sem in SEMILLAS:
                nss, accs = curva(casos[nombre], sem=sem)
                por_sem.append(texto_n(*n_para(nss, accs, o), nss))
            fila[clave(o)] = {"N": None if v is None else round(v, 1), "estado": e, "texto": texto_n(v, e, ns),
                              "por_semilla": por_sem}
        out["representaciones"][nombre] = {"N_medidos": ns.tolist(), "acierto_isotonico_pct": np.round(acc, 2).tolist(),
                                           "acierto_max_pct": round(float(acc[-1]), 2), "N_para": fila}
    for nombre, *_ in SERIES:                                 # cuántas veces MENOS muestras que la referencia
        for o in OBJETIVOS:
            a, r = out["representaciones"][nombre]["N_para"][clave(o)], out["representaciones"][REFERENCIA]["N_para"][clave(o)]
            a["veces_menos_que_referencia"] = round(r["N"] / a["N"], 2) if a["N"] and r["N"] else None
    ref = "detectores 8×8 (13)"
    for T in Ts:                                              # la comprobación: cada tamaño de dataset por separado
        ns, acc = curva(casos[ref], T)
        out["por_tamano"][str(T)] = {clave(o): texto_n(*n_para(ns, acc, o), ns) for o in OBJETIVOS_T}
    # el ejemplo de las dos lecturas: a igual acierto (horizontal) y a igual N (vertical)
    a_nom, b_nom, eps = EJEMPLO
    na, nb = (n_para(*curva(casos[n]), eps)[0] for n in (a_nom, b_nom))
    out["ejemplo"] = {"a": a_nom, "b": b_nom, "acierto_pct": eps, "N_a": round(na, 1), "N_b": round(nb, 1),
                      "veces_mas_muestras": round(nb / na, 2), "acierto_b_con_N_a_pct": round(acierto_en(*curva(casos[b_nom]), na), 2)}
    out["ejemplo"]["puntos_de_diferencia_con_N_a"] = round(eps - out["ejemplo"]["acierto_b_con_N_a_pct"], 2)
    (RES / "muestras-necesarias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'acierto objetivo →':<24}" + "".join(f"{f'{o:g} %':>9}" for o in OBJETIVOS))
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"{nombre:<24}" + "".join(f"{f[clave(o)]['texto']:>9}" for o in OBJETIVOS)
              + f"   (máx {curva(casos[nombre])[1][-1]:.1f} %)")
    print("rango entre semillas (90 % · 95 %):")
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"  {nombre:<22}  90 %: {' / '.join(f['90']['por_semilla'])}   95 %: {' / '.join(f['95']['por_semilla'])}")
    print(f"veces menos muestras que {REFERENCIA}:")
    for nombre, *_ in SERIES:
        f = out["representaciones"][nombre]["N_para"]
        print(f"  {nombre:<22}" + "".join(f"{(str(f[clave(o)]['veces_menos_que_referencia']) + '×') if f[clave(o)]['veces_menos_que_referencia'] else '—':>9}" for o in OBJETIVOS))
    print(f"{ref}, cada tamaño por separado ({' / '.join(f'{o:g} %' for o in OBJETIVOS_T)}):")
    for T, f in out["por_tamano"].items():
        print(f"  T = {T:>4}: " + " · ".join(f"{v}" for v in f.values()))
    e = out["ejemplo"]
    print(f"ejemplo: para {eps:g} %, {e['a']} {e['N_a']:.0f} y {e['b']} {e['N_b']:.0f} muestras ({e['veces_mas_muestras']}×); "
          f"con {e['N_a']:.0f} muestras, {e['b']} saca {e['acierto_b_con_N_a_pct']:.1f} % ({e['puntos_de_diferencia_con_N_a']:.1f} puntos menos)")

    dibujar(casos, Ts, out)
    return 0


def _ejes(ax) -> None:
    ax.set_facecolor(SUP); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _eje_n(ax, cual: str) -> None:
    (ax.set_yscale if cual == "y" else ax.set_xscale)("log")
    (ax.set_yticks if cual == "y" else ax.set_xticks)(YTICKS_N)
    (ax.set_yticklabels if cual == "y" else ax.set_xticklabels)([str(t) for t in YTICKS_N])
    ax.minorticks_off()


def dibujar(casos: dict, Ts: list, out: dict) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "text.color": T1})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 6.4), facecolor=SUP)
    for ax in (a1, a2, a3):
        _ejes(ax)
    r = out["representaciones"]

    # 1. N(ε): para un acierto dado, cuántas muestras
    _eje_n(a1, "y"); a1.set_xlim(70, 99)
    a1.set_xlabel("acierto objetivo sobre los dígitos nuevos (%)"); a1.set_ylabel("muestras de train necesarias N (escala log)")
    for nombre, col, ls in SERIES:
        ns, acc = curva(casos[nombre])
        pts = [(o, n_para(ns, acc, o)[0]) for o in REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        a1.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2.2, ls=ls, label=nombre, zorder=3)
        a1.plot(pts[-1][0], pts[-1][1], "o", color=col, ms=7, mec=SUP, mew=1.5, zorder=4)
        bandas = []                                   # el rango entre semillas, donde las tres alcanzan el nivel
        for o in REJILLA:
            vs = [n_para(*curva(casos[nombre], sem=sem), o)[0] for sem in SEMILLAS]
            if all(v is not None for v in vs):
                bandas.append((o, min(vs), max(vs)))
        if bandas:
            a1.fill_between([b[0] for b in bandas], [b[1] for b in bandas], [b[2] for b in bandas], color=col, alpha=0.13,
                            lw=0, zorder=2)
    a1.legend(frameon=False, loc="upper left", fontsize=9.5)
    a1.set_title("1 · ¿Cuántas muestras cuesta cada nivel de acierto?", loc="left", fontsize=12.5, color=T1)
    a1.text(0.02, 0.72, f"{'':<22}{'N(90 %)':>8}{'N(95 %)':>9}{'máx':>7}\n" + "\n".join(
        f"{n:<22}{r[n]['N_para']['90']['texto']:>8}{r[n]['N_para']['95']['texto']:>9}{curva(casos[n])[1][-1]:>6.1f}%"
        for n, *_ in SERIES)
        + "\n● = lo máximo alcanzado con N ≤ 2000\nbanda = rango entre las 3 semillas", transform=a1.transAxes, ha="left", va="top",
        fontsize=9, color=T2, family="monospace")

    # 2. el mismo dato invertido: la curva de aprendizaje, y el ejemplo leído de las dos formas
    _eje_n(a2, "x"); a2.set_xlim(8, 2600)
    a2.set_xlabel("muestras de train N (escala log)"); a2.set_ylabel("acierto sobre los dígitos nuevos (%)")
    for nombre, col, ls in SERIES:
        ns, acc = curva(casos[nombre])
        a2.plot(ns, acc, color=col, lw=2.2, ls=ls, label=nombre, zorder=3)
        por_sem = np.array([curva(casos[nombre], sem=sem)[1] for sem in SEMILLAS])
        a2.fill_between(ns, por_sem.min(0), por_sem.max(0), color=col, alpha=0.13, lw=0, zorder=2)
    a2.set_ylim(a2.get_ylim()[0], 103.5); a2.set_yticks([60, 70, 80, 90, 100])   # el techo DESPUÉS de dibujar: deja sitio a la etiqueta
    e = out["ejemplo"]
    eps, na, nb, ab = e["acierto_pct"], e["N_a"], e["N_b"], e["acierto_b_con_N_a_pct"]
    a2.annotate("", xy=(nb, eps), xytext=(na, eps), arrowprops=dict(arrowstyle="<->", color=T1, lw=1.3, shrinkA=0, shrinkB=0), zorder=5)
    a2.annotate(f"en horizontal: mismo {eps:g} % → {na:.0f} contra {nb:.0f} muestras = {e['veces_mas_muestras']:.1f}×",
                xy=(np.sqrt(na * nb), eps), xytext=(np.sqrt(na * nb), 101.2), ha="center", va="center", fontsize=9.5, color=T1,
                arrowprops=dict(arrowstyle="-", color=T2, lw=0.8, shrinkA=2, shrinkB=0))
    a2.annotate("", xy=(na, ab), xytext=(na, eps), arrowprops=dict(arrowstyle="<->", color=T1, lw=1.3, shrinkA=0, shrinkB=0), zorder=5)
    a2.annotate(f"en vertical: mismas {na:.0f} muestras\n{eps:.1f} % contra {ab:.1f} % = {e['puntos_de_diferencia_con_N_a']:.1f} puntos",
                xy=(na, ab), xytext=(6, -8), textcoords="offset points", ha="left", va="top", fontsize=9.5, color=T1)
    a2.legend(frameon=False, loc="lower right", fontsize=9.5)
    a2.set_title("2 · Invertido: la curva de aprendizaje (¿qué acierto da cada N?)", loc="left", fontsize=12.5, color=T1)
    a2.text(0.97, 0.50, f"el panel 1 es este con los ejes cambiados:\nlo que aquí es una distancia HORIZONTAL\n(× muestras a igual acierto) allí "
            f"es vertical.\nCerca del techo, {e['puntos_de_diferencia_con_N_a']:.1f} puntos = {e['veces_mas_muestras']:.1f}× muestras",
            transform=a2.transAxes, ha="right", va="top", fontsize=9, color=T2, family="monospace")

    # 3. la comprobación: cada tamaño de dataset por separado
    _eje_n(a3, "y"); a3.set_xlim(70, 99)
    a3.set_xlabel("acierto objetivo sobre los dígitos nuevos (%)")
    ref = "detectores 8×8 (13)"
    for T, col in zip(Ts, RAMPA):
        ns, acc = curva(casos[ref], T)
        pts = [(o, n_para(ns, acc, o)[0]) for o in REJILLA]
        pts = [(x, v) for x, v in pts if v is not None]
        if pts:
            a3.plot([x for x, _ in pts], [v for _, v in pts], color=col, lw=2.2, label=f"T = {T}  (N de {ns[0]} a {ns[-1]})", zorder=3)
    a3.legend(frameon=False, loc="upper left", fontsize=9.5, title="detectores 8×8 (13), cada dataset por separado",
              title_fontsize=9.5)
    a3.set_title("3 · ¿Independiente del tamaño del dataset? Sí", loc="left", fontsize=12.5, color=T1)
    pt = out["por_tamano"]
    a3.text(0.02, 0.62, "N para 90 % / 95 %:\n" + "\n".join(f"T = {T:<5}{pt[str(T)]['90']:>8} / {pt[str(T)]['95']:>6}" for T in Ts)
            + "\n(≤ / >: fuera del rango de N de ese dataset)\nT = 4000 a 95 %: dentro del ruido; con T = 1000\nlas 3 semillas dan 93 · 129 · 303",
            transform=a3.transAxes, ha="left", va="top", fontsize=9, color=T2, family="monospace")
    fig.suptitle("Muestras necesarias N(ε) y su inversa, la curva de aprendizaje · los mismos 240 entrenamientos de la ganancia "
                 "(datasets balanceados de 500–4000 del pool de 5620, 43 escritores) · compositor lineal posicional · sin extrapolar",
                 x=0.01, ha="left", fontsize=10, color=T2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "muestras-necesarias.png", dpi=125, facecolor=SUP)
    print(f"→ {RES / 'muestras-necesarias.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
