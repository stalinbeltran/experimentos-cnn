#!/usr/bin/env python3
"""El banco de prueba (el de `rect-lin`, pasado por el filtro del brazo) y las métricas del criterio. COPIA de
`rect-lin/nn/evaluar.py` con `recall_grueso_largo` añadida.

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
    m["recall_grueso_largo"] = r4(acierto[cont & np.isin(g, GRUPOS["grueso"]) & (lg >= 16)].mean())
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


def banco(pre: str) -> tuple[np.ndarray, dict]:
    import prepro  # noqa: PLC0415
    b = D.cargar(D.BANCO)
    return prepro.aplicar(b["imagenes"].astype(np.float32)[:, None], pre), b


def referencias() -> dict:
    """El Gabor SIN entrenar, con cada filtro (también 'ninguno', como control) y cada k, 1 escala; umbral a FP 5 % sobre
    los negativos de entrenamiento PASADOS POR EL MISMO FILTRO."""
    import modelo as M  # noqa: PLC0415
    import prepro  # noqa: PLC0415
    e = D.cargar(D.ENTRENO)
    neg = e["imagenes"][e["tipo"] >= D.NEG_RUIDO].astype(np.float32)[:, None]
    out = {"gabor": []}
    for pre in prepro.NOMBRES:
        xb, b = banco(pre)
        xneg = torch.from_numpy(prepro.aplicar(neg, pre))
        for k in (5, 7, 9):
            g = M.Gabor(k, 1); u = g.calibrar(xneg, 0.05)
            fila = {"pre": pre, "k": k, "umbral": r4(u), **metricas(logits(g, xb), b)}
            out["gabor"].append(fila)
            print(f"  gabor {pre:8s} k={k}: fino {fila['recall_fino']}  medio {fila['recall_medio']}  grueso-largo "
                  f"{fila['recall_grueso_largo']}  fp {fila['fp_total']}")
    RES.mkdir(exist_ok=True)
    (RES / "referencias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- análisis (lee; no entrena nada)
NS = (4, 8, 16, 32, 64, 128, 256, 1000)
SUP, T1, T2, MUT, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIE = {"ninguno": "#2a78d6", "contorno": "#eb6834", "sobel": "#1baf7a"}   # paleta de referencia, slots 1–3
PRES = ("ninguno", "contorno", "sobel")


def control() -> dict:
    """El control SIN filtro: los brazos «continua, 1 escala» de rect-lin, leídos por id. Sus kernels están guardados y
    son reproducibles bit a bit (medido en rect-lin), así que se RE-EVALÚAN aquí con la métrica nueva sin re-entrenar."""
    import modelo as M  # noqa: PLC0415
    sys.path.insert(0, str(AQUI.parent.parent))
    from expcnn import por_id  # noqa: PLC0415
    org = por_id("rect-lin").carpeta / "resultados" / "trozos"
    xb, b = banco("ninguno")
    out = {}
    for f in sorted(org.glob("rejilla-*.jsonl")):
        for r in map(json.loads, f.open()):
            if r["escalas"] != 1 or r["entreno"] != "continua":
                continue
            m = M.Lineal(r["k"], 1)
            with torch.no_grad():
                m.K[0, 0] = torch.tensor(r["K0"]); m.K[1, 0] = torch.tensor(r["K45"])
                m.a.fill_(r["a"]); m.c.fill_(r["c"])
            met = metricas(logits(m, xb), b)
            # los kernels se guardaron redondeados a 4 decimales: se comprueba que dan lo mismo que lo medido allí
            met["_dif_con_rect_lin"] = r4(max(abs(met[c] - r[c]) for c in ("recall_fino", "recall_grueso", "fp_total")))
            out[("ninguno", r["k"], r["n"], r["semilla"])] = {"pre": "ninguno", "k": r["k"], "n": r["n"], "semilla": r["semilla"], **met}
    return out


def rejilla() -> dict:
    v = {}
    for f in sorted((RES / "trozos").glob("rejilla-*.jsonl")):
        for r in map(json.loads, f.open()):
            v[(r["pre"], r["k"], r["n"], r["semilla"])] = r
    if len(v) != 144:
        raise SystemExit(f"✗ la rejilla tiene {len(v)} brazos, no 144. Me niego a analizar a medias.")
    cache = RES / "control-re-evaluado.json"
    if cache.exists():
        c = {tuple(json.loads(k)): x for k, x in json.loads(cache.read_text()).items()}
    else:
        c = control()
        cache.write_text(json.dumps({json.dumps(list(k)): x for k, x in c.items()}, ensure_ascii=False) + "\n")
    v.update(c)
    return v


def media(v, pre, k, n, campo, sub=None) -> float:
    xs = [v[(pre, k, n, s)][campo] for s in (0, 1, 2)]
    return float(np.mean([x[sub] for x in xs] if sub is not None else xs))


def n90(v, pre, k) -> int | None:
    techo = media(v, pre, k, 1000, "recall_fino")
    for n in NS:
        if media(v, pre, k, n, "recall_fino") >= 0.9 * techo and media(v, pre, k, n, "fp_total") <= 0.10:
            return n
    return None


def analisis() -> dict:
    v = rejilla()
    ref = {(g["pre"], g["k"]): g for g in json.loads((RES / "referencias.json").read_text())["gabor"]}
    m = lambda pre, k, c, n=1000, sub=None: r4(media(v, pre, k, n, c, sub))  # noqa: E731
    tabla = []
    for pre in PRES:
        for k in (5, 7, 9):
            tabla.append({"pre": pre, "k": k, "n90": n90(v, pre, k),
                          **{c: m(pre, k, c) for c in ("recall_fino", "recall_medio", "recall_grueso", "recall_grueso_largo",
                                                       "fp_total", "fp_ruido", "fp_puntos-sueltos", "fp_mancha")},
                          "por_grosor": {g: m(pre, k, "recall_por_grosor", sub=g) for g in v[(pre, k, 1000, 0)]["recall_por_grosor"]},
                          "fino_por_n": [m(pre, k, "recall_fino", n) for n in NS],
                          "grueso_largo_por_n": [m(pre, k, "recall_grueso_largo", n) for n in NS],
                          "gabor": {c: ref[(pre, k)][c] for c in ("recall_fino", "recall_medio", "recall_grueso_largo", "fp_total")}})
    T = {(t["pre"], t["k"]): t for t in tabla}
    filtradas = [t for t in tabla if t["pre"] != "ninguno"]
    g2 = [t for t in filtradas if t["recall_fino"] >= T[("ninguno", t["k"])]["recall_fino"] - 0.03]
    mejor = max(g2 or filtradas, key=lambda t: t["recall_grueso_largo"])
    c = T[("ninguno", mejor["k"])]
    h = {"mejor": f"{mejor['pre']}, k={mejor['k']}" + ("" if g2 else " (NINGUNA cumple G2: la de más grueso-largo)")}
    h["G1"] = {"cumple": mejor["recall_grueso_largo"] >= 0.80, "valor": mejor["recall_grueso_largo"], "control": c["recall_grueso_largo"]}
    h["G2"] = {"cumple": mejor["recall_fino"] >= c["recall_fino"] - 0.03, "valor": mejor["recall_fino"], "control": c["recall_fino"]}
    h["G3"] = {"cumple": mejor["fp_total"] <= 0.05, "valor": mejor["fp_total"], "mancha": mejor["fp_mancha"],
               "puntos": mejor["fp_puntos-sueltos"], "ruido": mejor["fp_ruido"]}
    h["G4"] = {"cumple": mejor["n90"] is not None and c["n90"] is not None and mejor["n90"] <= c["n90"], "valor": mejor["n90"], "control": c["n90"]}
    h["G5"] = {"cumple": mejor["recall_medio"] >= 0.85, "valor": mejor["recall_medio"]}
    h["G6"] = {"cumple": all(T[("contorno", k)]["recall_grueso_largo"] >= T[("sobel", k)]["recall_grueso_largo"] for k in (5, 7, 9)),
               "por_k": {k: (T[("contorno", k)]["recall_grueso_largo"], T[("sobel", k)]["recall_grueso_largo"]) for k in (5, 7, 9)}}
    gg = ref[(mejor["pre"], mejor["k"])]
    h["G7"] = {"cumple": gg["recall_grueso_largo"] >= 0.70, "valor": gg["recall_grueso_largo"]}
    curvas = {pre: {kk: m(pre, mejor["k"], "curva_deteccion_por_radio_y_grosor", sub=kk)
                    for kk in v[(pre, mejor["k"], 1000, 0)]["curva_deteccion_por_radio_y_grosor"]} for pre in PRES}
    dif = max(x["_dif_con_rect_lin"] for kk, x in v.items() if kk[0] == "ninguno")
    out = {"hipotesis": h, "tabla": tabla, "curvas_mejor_k": curvas, "control_max_dif_con_rect_lin": dif}
    (RES / "analisis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def figuras() -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.ticker  # noqa: PLC0415
    from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415
    a = analisis(); v = rejilla()
    T = {(t["pre"], t["k"]): t for t in a["tabla"]}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "axes.labelcolor": T2, "xtick.color": T2,
                         "ytick.color": T2, "axes.titlecolor": T1, "figure.facecolor": SUP, "axes.facecolor": SUP})

    def ejes(ax):
        ax.grid(True, color=REJ, lw=0.6); ax.spines[["top", "right"]].set_visible(False)

    # 1 — recall por grosor (N = 1000), un panel por k, una línea por filtro
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, k in zip(axs, (5, 7, 9)):
        for pre in PRES:
            d = T[(pre, k)]["por_grosor"]; gs = sorted(d, key=int)
            ax.plot([int(g) for g in gs], [d[g] for g in gs], color=SERIE[pre], lw=2, marker="o", ms=4, label=pre)
        ax.axvspan(2, 4, color=REJ, alpha=0.5, lw=0); ax.set_ylim(0, 1.02); ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_title(f"kernel {k}×{k}"); ax.set_xlabel("grosor de la recta (px)"); ejes(ax)
    axs[0].set_ylabel("recall (N = 1000)"); axs[0].text(2.1, 0.03, "grosor\nentrenado", color=T2, fontsize=8)
    axs[2].legend(title="filtro", frameon=False, fontsize=8, title_fontsize=8, loc="lower left")
    fig.suptitle("Recall según el grosor, con y sin filtro de bordes (media de 3 semillas)", color=T1)
    fig.tight_layout(); fig.savefig(RES / "grosor.png", dpi=130); plt.close(fig)

    # 2 — curvas de aprendizaje del mejor k: fino y grueso-largo contra N
    k = int(a["hipotesis"]["mejor"].split("k=")[1][0])
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for ax, campo, tit in zip(axs, ("fino_por_n", "grueso_largo_por_n"), ("rectas finas (2–4 px)", "rectas gruesas (10–14 px, largo ≥ 16)")):
        for pre in PRES:
            ax.plot(NS, T[(pre, k)][campo], color=SERIE[pre], lw=2, marker="o", ms=4, label=pre)
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(0, 1.02); ax.set_title(tit); ax.set_xlabel("N (rectas de entrenamiento, finas)"); ejes(ax)
    axs[0].set_ylabel("recall"); axs[1].legend(title="filtro", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle(f"Curva de aprendizaje, kernel {k}×{k}", color=T1)
    fig.tight_layout(); fig.savefig(RES / "curva-aprendizaje.png", dpi=130); plt.close(fig)

    # 3 — los kernels aprendidos (mejor k, N = 1000, semilla 0) con cada filtro
    import modelo as M  # noqa: PLC0415, F401
    fig, axs = plt.subplots(2, 3, figsize=(6.2, 4.2))
    for j, pre in enumerate(PRES):
        r = v[(pre, k, 1000, 0)] if pre != "ninguno" else None
        if r is None:   # el control: sus kernels están en rect-lin
            sys.path.insert(0, str(AQUI.parent.parent))
            from expcnn import por_id  # noqa: PLC0415
            for f in (por_id("rect-lin").carpeta / "resultados" / "trozos").glob("rejilla-*.jsonl"):
                for x in map(json.loads, f.open()):
                    if (x["k"], x["escalas"], x["entreno"], x["n"], x["semilla"]) == (k, 1, "continua", 1000, 0):
                        r = x
        for i, K in enumerate((np.array(r["K0"]), np.array(r["K45"]))):
            mx = max(abs(K).max(), 1e-6)
            axs[i, j].imshow(K, cmap="RdBu_r", norm=TwoSlopeNorm(0, -mx, mx)); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        axs[0, j].set_title(pre)
    axs[0, 0].set_ylabel("K0 (—)"); axs[1, 0].set_ylabel("K45 (\\)")
    fig.suptitle(f"Kernels {k}×{k} aprendidos (N = 1000, semilla 0). Rojo +, azul −", color=T1)
    fig.tight_layout(); fig.savefig(RES / "kernels.png", dpi=130); plt.close(fig)


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
