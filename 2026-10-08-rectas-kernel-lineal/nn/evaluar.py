#!/usr/bin/env python3
"""El banco de prueba de `rect-lin` y las métricas del criterio (instrucciones/02-criterio.md).

    python nn/evaluar.py --referencias    Gabor (k × escalas, umbral calibrado en negativos de entreno) y la CNN
    python nn/evaluar.py --tablas         tablas del README desde resultados/rejilla.jsonl y referencias.json
    python nn/evaluar.py --figuras        resultados/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos as D  # noqa: E402

RES = AQUI.parent / "resultados"
GRUPOS = {"fino": (2, 3, 4), "medio": (6, 8), "grueso": (10, 12, 14)}


def r4(v) -> float:
    return round(float(v), 4)


def _desvio(ang: np.ndarray) -> np.ndarray:
    """Distancia del ángulo al centro de orientación más cercano (0–22,5°)."""
    d = np.mod(ang, 45.0)
    return np.minimum(d, 45.0 - d)


@torch.no_grad()
def logits(modelo, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    return np.concatenate([modelo(torch.from_numpy(x[i:i + lote])).numpy() for i in range(0, len(x), lote)])


def metricas(L: np.ndarray, b: dict) -> dict:
    """L (N, 4) logits sobre el banco b → todas las métricas del criterio."""
    det = L.max(1) > 0
    ori_ok = L.argmax(1) == D.orientacion(np.nan_to_num(b["angulo"]))
    acierto = det & ori_ok
    t, g, lg = b["tipo"], b["grosor"], b["largo"]
    cont = t == D.CONTINUA
    m = {}
    for nom, gs in GRUPOS.items():
        m[f"recall_{nom}"] = r4(acierto[cont & np.isin(g, gs)].mean())
    m["recall_total"] = r4(acierto[cont].mean())
    m["recall_por_grosor"] = {int(v): r4(acierto[cont & (g == v)].mean()) for v in np.unique(g[cont])}
    fino = cont & np.isin(g, GRUPOS["fino"])
    m["recall_fino_por_largo"] = {int(v): r4(acierto[fino & (lg == v)].mean()) for v in np.unique(lg[cont])}
    dv = _desvio(b["angulo"])
    m["recall_fino_por_desvio"] = {f"{a}-{z}": r4(acierto[fino & (dv >= a) & (dv < z + 1e-6)].mean())
                                   for a, z in ((0, 7.5), (7.5, 15), (15, 22.5))}
    m["deteccion_fino_sin_orientacion"] = r4(det[fino].mean())
    pun = t == D.PUNTEADA
    m["punteada_por_separacion"] = {int(s): r4(acierto[pun & (b["separacion"] == s)].mean()) for s in np.unique(b["separacion"][pun])}
    cur = t == D.CURVA
    m["curva_deteccion_por_radio"] = {int(r): r4(det[cur & (b["radio"] == r)].mean()) for r in np.unique(b["radio"][cur])}
    m["curva_deteccion_por_radio_y_grosor"] = {f"{int(r)}/{int(gg)}": r4(det[cur & (b["radio"] == r) & (g == gg)].mean())
                                               for r in np.unique(b["radio"][cur]) for gg in (2, 4, 8)}
    # fuerza: mediana del mayor logit de las curvas MENOS la de las rectas de largo 22 y el mismo grosor (en logits)
    fz = {}
    for r in np.unique(b["radio"][cur]):
        dif = [np.median(L.max(1)[cur & (b["radio"] == r) & (g == gg)]) - np.median(L.max(1)[cont & (g == gg) & (lg == 22)])
               for gg in (2, 4, 8)]
        fz[int(r)] = r4(np.mean(dif))
    m["curva_fuerza_menos_recta"] = fz
    for tt in (D.NEG_RUIDO, D.NEG_PUNTOS, D.NEG_MANCHA):
        m[f"fp_{D.NOMBRE_TIPO[tt]}"] = r4(det[t == tt].mean())
    m["fp_total"] = r4(det[t >= D.NEG_RUIDO].mean())
    return m


def banco() -> tuple[np.ndarray, dict]:
    b = D.cargar(D.BANCO)
    return b["imagenes"].astype(np.float32)[:, None], b


def referencias() -> dict:
    import modelo as M  # noqa: PLC0415
    from referencia_cnn import CNN  # noqa: PLC0415
    xb, b = banco()
    e = D.cargar(D.ENTRENO)
    xneg = torch.from_numpy(e["imagenes"][e["tipo"] >= D.NEG_RUIDO].astype(np.float32)[:, None])
    out = {"gabor": []}
    for k in (5, 7, 9):
        for nv in (1, 2, 3):
            g = M.Gabor(k, nv); u = g.calibrar(xneg, 0.05)
            out["gabor"].append({"k": k, "escalas": nv, "umbral": r4(u), **metricas(logits(g, xb), b)})
            print(f"  gabor k={k} esc={nv}: recall fino {out['gabor'][-1]['recall_fino']}  grueso {out['gabor'][-1]['recall_grueso']}")
    cnn = CNN()
    out["cnn"] = {"id": "feat-ind32", "huellas": cnn.huellas, "parametros": cnn.n_parametros(), **metricas(logits(cnn, xb), b)}
    print(f"  cnn: recall fino {out['cnn']['recall_fino']}  grueso {out['cnn']['recall_grueso']}  fp {out['cnn']['fp_total']}")
    RES.mkdir(exist_ok=True)
    (RES / "referencias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- análisis (lee; no entrena nada)
NS = (4, 8, 16, 32, 64, 128, 256, 1000)
SUP, T1, T2, MUT, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIE = {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}          # paleta de referencia, slots 1–3 (validada)


def clave(r) -> tuple:
    return (r["k"], r["escalas"], r["entreno"], r["n"], r["semilla"])


def rejilla() -> tuple[dict, dict]:
    """La rejilla COMPLETA es la de Vast (resultados/trozos/). La del dev (resultados/rejilla.jsonl, parada en 381) sólo
    se usa para comprobar que los comunes son idénticos."""
    v = {clave(r): r for f in sorted((RES / "trozos").glob("rejilla-*.jsonl")) for r in map(json.loads, f.open())}
    d = {clave(r): r for r in map(json.loads, (RES / "rejilla.jsonl").open())} if (RES / "rejilla.jsonl").exists() else {}
    return v, d


def media(v: dict, k, e, modo, n, campo, sub=None) -> float:
    xs = [v[(k, e, modo, n, s)][campo] for s in (0, 1, 2)]
    xs = [x[sub] for x in xs] if sub is not None else xs
    return float(np.mean(xs))


def rango(v, k, e, modo, n, campo) -> tuple[float, float]:
    xs = [v[(k, e, modo, n, s)][campo] for s in (0, 1, 2)]
    return min(xs), max(xs)


def n90(v, k, e, modo) -> int | None:
    techo = media(v, k, e, modo, 1000, "recall_fino")
    for n in NS:
        if media(v, k, e, modo, n, "recall_fino") >= 0.9 * techo and media(v, k, e, modo, n, "fp_total") <= 0.10:
            return n
    return None


def grueso_de(d: dict) -> float:
    return float(np.mean([d[str(g)] for g in (10, 12, 14)]))


def analisis() -> dict:
    v, dev = rejilla()
    if len(v) != 432:
        raise SystemExit(f"✗ la rejilla de Vast tiene {len(v)} brazos, no 432. Me niego a analizar a medias.")
    com = set(v) & set(dev)
    campos = ("recall_fino", "recall_grueso", "fp_total", "K0", "K45")
    reprod = {"comunes": len(com), "identicos": sum(all(v[c][f] == dev[c][f] for f in campos) for c in com)}
    ref = json.loads((RES / "referencias.json").read_text(encoding="utf-8"))
    gab = {(g["k"], g["escalas"]): g for g in ref["gabor"]}
    cnn = ref["cnn"]
    m = lambda k, e, modo, n, c, sub=None: r4(media(v, k, e, modo, n, c, sub))  # noqa: E731
    configs = [(k, e) for k in (5, 7, 9) for e in (1, 2, 3)]
    tabla = []
    for k, e in configs:
        tabla.append({"k": k, "escalas": e, "parametros": v[(k, e, "continua", 4, 0)]["parametros"],
                      "fino_por_n": [m(k, e, "continua", n, "recall_fino") for n in NS],
                      "fp_por_n": [m(k, e, "continua", n, "fp_total") for n in NS],
                      "n90": n90(v, k, e, "continua"),
                      "fino": m(k, e, "continua", 1000, "recall_fino"), "medio": m(k, e, "continua", 1000, "recall_medio"),
                      "grueso": m(k, e, "continua", 1000, "recall_grueso"), "fp": m(k, e, "continua", 1000, "fp_total"),
                      "total": m(k, e, "continua", 1000, "recall_total"),
                      "gabor_fino": gab[(k, e)]["recall_fino"], "gabor_medio": gab[(k, e)]["recall_medio"],
                      "gabor_grueso": gab[(k, e)]["recall_grueso"], "gabor_fp": gab[(k, e)]["fp_total"]})
    T = {(t["k"], t["escalas"]): t for t in tabla}
    # la mejor de H1: mayor recall fino con FP ≤ 0,05 a N=1000
    ok1 = [t for t in tabla if t["fino"] >= 0.90 and t["fp"] <= 0.05]
    mejor = max(ok1 or tabla, key=lambda t: (t["fino"], -t["fp"]))
    bk, be = mejor["k"], mejor["escalas"]
    punt = lambda k, e, modo, seps: r4(np.mean([media(v, k, e, modo, 1000, "punteada_por_separacion", str(s)) for s in seps]))  # noqa: E731
    h = {}
    h["H1"] = {"cumple": bool(ok1), "mejor": f"k={bk}, {be} escala(s)", "fino": mejor["fino"], "fp": mejor["fp"]}
    h["H2"] = {"cumple": mejor["n90"] is not None and mejor["n90"] <= 32, "n90": mejor["n90"]}
    g = gab[(bk, be)]
    h["H3"] = {"cumple": mejor["fino"] - g["recall_fino"] >= 0.05 and mejor["fp"] <= g["fp_total"],
               "aprendido": mejor["fino"], "gabor": g["recall_fino"], "dif": r4(mejor["fino"] - g["recall_fino"]),
               "fp_aprendido": mejor["fp"], "fp_gabor": g["fp_total"]}
    d4 = T[(7, 3)]["grueso"] - T[(7, 1)]["grueso"]
    h["H4"] = {"cumple": d4 >= 0.20, "grueso_1": T[(7, 1)]["grueso"], "grueso_3": T[(7, 3)]["grueso"], "dif": r4(d4)}
    t5, t7, t9 = T[(5, 3)]["total"], T[(7, 3)]["total"], T[(9, 3)]["total"]
    h["H5"] = {"cumple": abs(t7 - t9) <= 0.03 and t5 <= t9 - 0.03, "k5": t5, "k7": t7, "k9": t9}
    h6 = {k: punt(k, 1, "continua", (2, 3, 4)) for k in (7, 9)}
    h["H6"] = {"cumple": all(x >= 0.70 for x in h6.values()), "por_k": h6}
    h6b = {k: r4(punt(k, 3, "continua", (10, 12)) - punt(k, 1, "continua", (10, 12))) for k in (7, 9)}
    h["H6b"] = {"cumple": all(x >= 0.20 for x in h6b.values()), "dif_por_k": h6b}
    h7 = {(k, e): (m(k, e, "punteada", 1000, "recall_fino"), m(k, e, "punteada", 1000, "fp_puntos-sueltos")) for k, e in configs}
    h["H7"] = {"cumple": any(f >= 0.85 and p <= 0.15 for f, p in h7.values()),
               "mejor_fino": max(f for f, _ in h7.values()), "fp_puntos_sueltos": sorted({p for _, p in h7.values()})[::4]}
    cur = {int(r): m(bk, be, "continua", 1000, "curva_deteccion_por_radio", r) for r in ("6", "9", "12", "18", "27", "40")}
    vals = list(cur.values())
    monot = all(b >= a - 0.05 for a, b in zip(vals, vals[1:]))
    cerr = max(cur[6], cur[9])
    h["H8"] = {"cumple": monot and cerr <= 0.5 * mejor["fino"], "monotona": monot, "deteccion_por_radio": cur,
               "radio_le_9": cerr, "limite": r4(0.5 * mejor["fino"]),
               "por_radio_y_grosor": {kk: m(bk, be, "continua", 1000, "curva_deteccion_por_radio_y_grosor", kk)
                                      for kk in v[(bk, be, "continua", 1000, 0)]["curva_deteccion_por_radio_y_grosor"]},
               "fuerza_menos_recta": {int(r): m(bk, be, "continua", 1000, "curva_fuerza_menos_recta", r)
                                      for r in ("6", "9", "12", "18", "27", "40")}}
    h["H9"] = {"cumple": mejor["fino"] >= cnn["recall_fino"] - 0.05 and mejor["grueso"] > cnn["recall_grueso"],
               "fino": (mejor["fino"], cnn["recall_fino"]), "grueso": (mejor["grueso"], cnn["recall_grueso"]),
               "cnn_por_desvio": cnn["recall_fino_por_desvio"], "cnn_por_largo": cnn["recall_fino_por_largo"],
               "mejor_por_desvio": {kk: m(bk, be, "continua", 1000, "recall_fino_por_desvio", kk) for kk in cnn["recall_fino_por_desvio"]}}
    punteadas = {f"{modo} k={k} e={e}": {s: r4(media(v, k, e, modo, 1000, "punteada_por_separacion", str(s))) for s in (2, 3, 4, 6, 8, 10, 12)}
                 for modo in ("continua", "punteada") for k, e in ((7, 1), (9, 1), (9, 3))}
    vast = json.loads((RES / "vast" / "rejilla" / "rejilla.json").read_text(encoding="utf-8"))
    seg = [v[c]["segundos"] / dev[c]["segundos"] for c in com if dev[c]["segundos"] > 5]
    velocidad = {"vast_maquina": vast["maquina"], "vast_minutos_total": vast["minutos"], "vast_coste_usd": vast["coste_usd"],
                 "vast_reloj_rejilla_s": 1474, "s_por_brazo_vast_sobre_dev_mediana": r4(np.median(seg)),
                 "dev_horas_proceso_381": r4(sum(dev[c]["segundos"] for c in dev) / 3600),
                 "vast_horas_proceso_432": r4(sum(r["segundos"] for r in v.values()) / 3600)}
    out = {"reproducibilidad": reprod, "tabla": tabla, "hipotesis": h, "punteadas": punteadas, "velocidad": velocidad,
           "cnn": {k: cnn[k] for k in ("recall_fino", "recall_medio", "recall_grueso", "fp_total", "parametros")}}
    (RES / "analisis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def figuras() -> None:
    import matplotlib  # noqa: PLC0415
    import matplotlib.ticker  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415
    import modelo as M  # noqa: PLC0415
    v, _ = rejilla()
    ref = json.loads((RES / "referencias.json").read_text(encoding="utf-8"))
    gab = {(g["k"], g["escalas"]): g for g in ref["gabor"]}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "axes.titlecolor": T1, "figure.facecolor": SUP, "axes.facecolor": SUP})

    def ejes(ax):
        ax.grid(True, color=REJ, lw=0.6); ax.spines[["top", "right"]].set_visible(False)

    # 1 — curva de aprendizaje: recall fino contra N, un panel por nº de escalas, una línea por k; Gabor (N=0) punteado
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, e in zip(axs, (1, 2, 3)):
        for k in (5, 7, 9):
            y = [media(v, k, e, "continua", n, "recall_fino") for n in NS]
            lo = [min(v[(k, e, "continua", n, s)]["recall_fino"] for s in (0, 1, 2)) for n in NS]
            hi = [max(v[(k, e, "continua", n, s)]["recall_fino"] for s in (0, 1, 2)) for n in NS]
            ax.fill_between(NS, lo, hi, color=SERIE[k], alpha=0.15, lw=0)
            ax.plot(NS, y, color=SERIE[k], lw=2, marker="o", ms=4, label=f"{k}×{k}")
            ax.axhline(gab[(k, e)]["recall_fino"], color=SERIE[k], lw=1, ls=(0, (3, 3)))
        ax.axhline(ref["cnn"]["recall_fino"], color=T2, lw=1, ls=(0, (1, 2)))
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_ylim(0, 1.02)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"{e} escala(s)"); ax.set_xlabel("N (rectas de entrenamiento)"); ejes(ax)
    axs[0].set_ylabel("recall, rectas finas (2–4 px)")
    axs[0].text(4.3, ref["cnn"]["recall_fino"] + 0.02, "CNN feat-ind32", color=T2, fontsize=8)
    axs[2].legend(title="kernel (— Gabor sin entrenar)", frameon=False, loc="lower right", fontsize=8, title_fontsize=8)
    fig.suptitle("Curva de aprendizaje del kernel lineal (entrenado con trazo continuo; media y rango de 3 semillas)", color=T1)
    fig.tight_layout(); fig.savefig(RES / "curva-aprendizaje.png", dpi=130); plt.close(fig)

    # 2 — los kernels aprendidos (k=9, 1 escala, semilla 0) a varios N, junto al Gabor
    cols = [(n, np.array(v[(9, 1, "continua", n, 0)]["K0"]), np.array(v[(9, 1, "continua", n, 0)]["K45"])) for n in (4, 16, 64, 1000)]
    cols.append(("Gabor", M.kernel_gabor(9, 0), M.kernel_gabor(9, 45)))
    fig, axs = plt.subplots(2, len(cols), figsize=(1.9 * len(cols), 4))
    for j, (n, k0, k45) in enumerate(cols):
        for i, K in enumerate((k0, k45)):
            a = max(abs(K).max(), 1e-6)
            axs[i, j].imshow(K, cmap="RdBu_r", norm=TwoSlopeNorm(0, -a, a)); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        axs[0, j].set_title(f"N = {n}" if n != "Gabor" else "Gabor a mano")
    axs[0, 0].set_ylabel("K0 (—)"); axs[1, 0].set_ylabel("K45 (\\)")
    fig.suptitle("Kernels 9×9 aprendidos (1 escala, semilla 0). Rojo +, azul −; escala propia por kernel", color=T1)
    fig.tight_layout(); fig.savefig(RES / "kernels.png", dpi=130); plt.close(fig)

    # 3 — punteadas: recall contra la separación entre puntos, entrenado con continuas o con punteadas
    seps = (2, 3, 4, 6, 8, 10, 12)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, modo in zip(axs, ("continua", "punteada")):
        for k in (5, 7, 9):
            ax.plot(seps, [media(v, k, 1, modo, 1000, "punteada_por_separacion", str(s)) for s in seps],
                    color=SERIE[k], lw=2, marker="o", ms=4, label=f"{k}×{k}")
        ax.axvspan(3, 6, color=REJ, alpha=0.5, lw=0)
        ax.set_title("entrenado con trazo " + {"continua": "continuo", "punteada": "punteado"}[modo]); ax.set_xlabel("separación entre puntos (px)"); ax.set_xticks(seps); ejes(ax)
    axs[1].text(3.1, 0.03, "separaciones vistas\nal entrenar", color=T2, fontsize=8)
    axs[0].set_ylabel("recall, rectas punteadas"); axs[0].set_ylim(0, 1.02); axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Rectas punteadas (N = 1000, 1 escala)", color=T1)
    fig.tight_layout(); fig.savefig(RES / "punteadas.png", dpi=130); plt.close(fig)

    # 4 — curvas: detección por radio y grosor (la mejor config) y fuerza curva − recta
    a = analisis()["hipotesis"]["H8"]
    radios = (6, 9, 12, 18, 27, 40)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for g, c in zip((2, 4, 8), ("#2a78d6", "#eb6834", "#1baf7a")):
        axs[0].plot(radios, [a["por_radio_y_grosor"][f"{r}/{g}"] for r in radios], color=c, lw=2, marker="o", ms=4, label=f"{g} px")
    axs[0].set_xscale("log"); axs[0].set_xticks(radios); axs[0].set_xticklabels(radios); axs[0].set_ylim(0, 1.02)
    axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_xlabel("radio de la curva (px)"); axs[0].set_ylabel("fracción detectada como recta")
    axs[0].legend(title="grosor", frameon=False, fontsize=8, title_fontsize=8); ejes(axs[0])
    axs[1].bar(range(len(radios)), [a["fuerza_menos_recta"][r] for r in radios], color="#2a78d6", width=0.6)
    axs[1].axhline(0, color=T2, lw=1); axs[1].set_xticks(range(len(radios))); axs[1].set_xticklabels(radios)
    axs[1].set_xlabel("radio de la curva (px)"); axs[1].set_ylabel("logit curva − logit recta (mediana)"); ejes(axs[1])
    fig.suptitle("Curvas de 20 px de largo vistas por el mejor kernel (9×9, 1 escala, N = 1000)", color=T1)
    fig.tight_layout(); fig.savefig(RES / "curvas.png", dpi=130); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--referencias", action="store_true")
    p.add_argument("--tablas", action="store_true")
    p.add_argument("--figuras", action="store_true")
    a = p.parse_args()
    if a.referencias:
        referencias(); return 0
    if a.tablas:
        print(json.dumps(analisis(), indent=1, ensure_ascii=False)); return 0
    if a.figuras:
        figuras(); print("figuras en", RES); return 0
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
