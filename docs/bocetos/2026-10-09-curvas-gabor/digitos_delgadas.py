#!/usr/bin/env python3
"""El detector para trazos DELGADOS (`delgadas.py`, calibración elegida allí) sobre TODOS los dígitos, con el compositor
de siempre (pedido del dueño, 2026-10-10).

*«Prueba ahora con todos los dígitos. Dame las estadísticas, gráficamente, y muéstrame los casos de error. Si el dígito
es visiblemente muy grueso no lo incluyas en los errores (los consideraremos un problema separado).»*

Cadena: dígito binario (sin esqueleto, sin borde) → detector delgado (12 Gabor par λ 6 + giro κ, calibración de
`resultados-delgadas.json`) → mapas recto · vector de curvatura (→ ← ↓ ↑) · golpe → max en celdas 8×8 → compositor
lineal (`digitos_bordes.py`: 180 / 1617 / 3823 a ciegas, 3 semillas).

«MUY GRUESO», objetivo y no a ojo: grosor típico = 2 × mediana de la distancia de la tinta al borde. En el dataset sale
en escalones (2,8 · 4 · 4,5 · 5,7–6 px) y se llama muy grueso a **≥ 5,5 px** (los que redondean a 6: 155 de 5620, 2,8 %). Esos fallos no van en la
figura de errores sino en una aparte, para que el dueño juzgue el corte. ⚠ El detector se calibró para ≤ 3 px, y sólo un
tercio de los dígitos lo son (2,8 px); el 64 % tiene 4 px. Por eso todo se da también por grosor.

PREDICCIÓN (escrita antes de correrlo): 0,92–0,95 en val, por debajo del borde gris (0,960): los trazos de 4 px están
fuera de la calibración. Por grosor: mejor en los de 2,8 px que en los de 4.

    python digitos_delgadas.py   → imagenes/23-digitos-delgadas-*.png · resultados-digitos-delgadas.json  (~4 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
import delgadas as DL                                               # noqa: E402
from gemelos_5 import mover                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

MUY_GRUESO = 5.5          # el mismo corte que GRUPOS: incluye los de 5,66 px (2·2√2), que redondean a 6
GRUPOS = (("≈ 2,8 px", 0, 3.5), ("4 px", 3.5, 4.2), ("4,5 px", 4.2, 5.5), ("≥ 6 px (muy grueso)", 5.5, 99))
NAR, AZU, VERDE, GRIS = "#c2410c", "#2a78d6", "#1b7f3b", "#8a8986"


def tal_cual(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


def grosor(x: np.ndarray) -> float:
    m = x > 0
    return float(2 * np.median(distance_transform_edt(m)[m])) if m.any() else 0.0


def grupo(a: float) -> str:
    return next(n for n, lo, hi in GRUPOS if lo <= a < hi)


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    cal = json.loads((AQUI / "resultados-delgadas.json").read_text())["elegida"]
    DL.configurar(**cal)
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    anc = np.array([grosor(x) for x in img])
    X = DB.caract(img, borde=tal_cual)["bordes"]
    print(f"características en {time.time() - t0:.0f} s · calibración {cal}", flush=True)
    v, c = [], []
    for sem in D.SEMILLAS:
        pr = DB.entrenar(X[tr], y[tr], sem)
        pv, pe = pr(X[va]), pr(X[ex])
        v.append(float((pv == y[va]).mean())); c.append(float((pe == y[ex]).mean()))
        if sem == 0:
            pred0, pv0, pe0 = pr, pv, pe
    out = {"calibracion": cal, "muy_grueso_px": MUY_GRUESO,
           "val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
           "ciega": round(float(np.mean(c)), 4), "ciega_rango": [round(min(c), 4), round(max(c), 4)]}
    # por grosor y por dígito (semilla 0)
    por_grosor = {}
    for n, lo, hi in GRUPOS:
        mv, me = (anc[va] >= lo) & (anc[va] < hi), (anc[ex] >= lo) & (anc[ex] < hi)
        por_grosor[n] = {"val_n": int(mv.sum()), "val": round(float((pv0[mv] == y[va][mv]).mean()), 4) if mv.any() else None,
                         "ciega_n": int(me.sum()), "ciega": round(float((pe0[me] == y[ex][me]).mean()), 4) if me.any() else None}
    out["por_grosor_semilla_0"] = por_grosor
    out["por_digito_val_semilla_0"] = [round(float((pv0[y[va] == k] == k).mean()), 3) for k in range(10)]
    conf = np.zeros((10, 10), int)
    for a, b in zip(y[va], pv0):
        conf[a, b] += 1
    out["confusion_val_semilla_0"] = conf.tolist()
    f = np.flatnonzero(pv0 != y[va])
    gruesos = f[anc[va][f] >= MUY_GRUESO]; normales = f[anc[va][f] < MUY_GRUESO]
    out["fallos_val_semilla_0"] = {"total": int(len(f)), "muy_gruesos": int(len(gruesos)), "resto": int(len(normales))}
    cf = {}
    for j in normales:
        k = f"{y[va][j]}→{pv0[j]}"; cf[k] = cf.get(k, 0) + 1
    out["confusiones_sin_gruesos"] = dict(sorted(cf.items(), key=lambda z: -z[1])[:10])
    out["indices_errores_figura"] = [int(va[j]) for j in normales[np.lexsort((pv0[normales], y[va][normales]))]]
    out["indices_muy_gruesos"] = [int(va[j]) for j in gruesos[np.lexsort((pv0[gruesos], y[va][gruesos]))]]
    out["acierto_val_sin_gruesos_semilla_0"] = round(float((pv0[anc[va] < MUY_GRUESO] == y[va][anc[va] < MUY_GRUESO]).mean()), 4)
    print(f"val {out['val']} {out['val_rango']} · a ciegas {out['ciega']} · por grosor {por_grosor} · fallos "
          f"{out['fallos_val_semilla_0']}  ({time.time() - t0:.0f} s)", flush=True)
    # desplazamientos (horizontal; la vertical está recortada en UCI)
    DS = list(range(-4, 5)); acc, cam = [], []
    for k in DS:
        xs = np.stack([mover(img[i], 0, k) for i in va])
        p = pred0(DB.caract(xs, borde=tal_cual)["bordes"])
        acc.append(round(float((p == y[va]).mean()), 4)); cam.append(round(float((p != pv0).mean()), 4))
    out["desplazamiento_horizontal"] = {"d": DS, "acierto": acc, "cambian": cam}
    print(f"desplazamiento horizontal: {acc}", flush=True)
    (AQUI / "resultados-digitos-delgadas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")

    # ── figura 1: estadísticas ──
    C._estilo()
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.4))
    a = axs[0, 0]
    filas = [("detector DELGADO (esto)", out["val"], out["ciega"], NAR),
             ("borde gris re-calibrado λ_b 10 · K 9", 0.9596, 0.9569, AZU),
             ("rectas (referencia)", 0.9546, 0.9466, "#555"),
             ("borde Gabor binarizado λ_b 6", 0.938, 0.9248, "#7fb0e8"),
             ("borde morfológico (retirado)", 0.8893, 0.8678, GRIS)]
    for k, (n, vv, cc, col) in enumerate(filas):
        a.barh(k, vv, color=col, height=0.6); a.text(vv + 0.002, k, f"{vv:.3f}  (a ciegas {cc:.3f})", va="center", fontsize=8.5)
    a.set_yticks(range(len(filas))); a.set_yticklabels([r[0] for r in filas], fontsize=8.5); a.invert_yaxis()
    a.set_xlim(0.85, 1.0); a.set_title("acierto en val (1617), 3 semillas; las otras filas, de sus JSON", loc="left", fontsize=9.5)
    a = axs[0, 1]
    nombres = [n for n, *_ in GRUPOS]; x_ = np.arange(len(nombres)); w = 0.38
    va_ = [por_grosor[n]["val"] or 0 for n in nombres]; ce_ = [por_grosor[n]["ciega"] or 0 for n in nombres]
    a.bar(x_ - w / 2, va_, w, color=NAR, label="val"); a.bar(x_ + w / 2, ce_, w, color="#f2a77f", label="a ciegas")
    for i, n in enumerate(nombres):
        a.text(i - w / 2, va_[i] + 0.01, f"{va_[i]:.3f}\nn {por_grosor[n]['val_n']}", ha="center", fontsize=7.5)
        a.text(i + w / 2, ce_[i] + 0.01, f"{ce_[i]:.3f}\nn {por_grosor[n]['ciega_n']}", ha="center", fontsize=7.5)
    a.set_xticks(x_); a.set_xticklabels(nombres, fontsize=8.5); a.set_ylim(0.5, 1.08); a.legend(frameon=False, fontsize=8)
    a.set_title("acierto por GROSOR del trazo (semilla 0) · el detector se calibró para ≤ 3 px", loc="left", fontsize=9.5)
    a = axs[0, 2]
    a.bar(range(10), out["por_digito_val_semilla_0"], color=NAR)
    for k, vv in enumerate(out["por_digito_val_semilla_0"]):
        a.text(k, vv + 0.005, f"{vv:.2f}", ha="center", fontsize=7.5)
    a.set_xticks(range(10)); a.set_ylim(0.6, 1.04); a.set_title("acierto por dígito (val, semilla 0)", loc="left", fontsize=9.5)
    a = axs[1, 0]
    cn = conf / conf.sum(1, keepdims=True)
    a.imshow(np.where(np.eye(10, dtype=bool), np.nan, cn), cmap="Oranges", vmin=0, vmax=max(0.05, np.nanmax(np.where(np.eye(10, dtype=bool), 0, cn))))
    for i in range(10):
        for j in range(10):
            if conf[i, j] and i != j:
                a.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=7.5)
    a.set_xticks(range(10)); a.set_yticks(range(10)); a.set_xlabel("leído"); a.set_ylabel("real")
    a.set_title("errores (val, semilla 0): real × leído, sin la diagonal", loc="left", fontsize=9.5)
    a = axs[1, 1]
    a.plot(DS, acc, "o-", color=NAR, label="acierto"); a.plot(DS, cam, "o:", color=GRIS, label="fracción que cambia de lectura")
    for k, vv in zip(DS, acc):
        a.text(k, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=7.5)
    a.set_xticks(DS); a.set_ylim(0, 1.05); a.legend(frameon=False, fontsize=8); a.set_xlabel("desplazamiento horizontal (px)")
    a.set_title("desplazamientos (val, semilla 0, sin re-entrenar)", loc="left", fontsize=9.5)
    a = axs[1, 2]; a.axis("off")
    fv = out["fallos_val_semilla_0"]
    lin = [f"val: {out['val']:.3f} (3 semillas, {out['val_rango'][0]:.3f}–{out['val_rango'][1]:.3f})",
           f"a ciegas: {out['ciega']:.3f} ({out['ciega_rango'][0]:.3f}–{out['ciega_rango'][1]:.3f})", "",
           f"fallos en val (semilla 0): {fv['total']}", f"  de ellos muy gruesos (≥ {MUY_GRUESO:g} px): {fv['muy_gruesos']}",
           f"  el resto (figura de errores): {fv['resto']}",
           f"acierto en val sin los muy gruesos: {out['acierto_val_sin_gruesos_semilla_0']:.3f}", "",
           "confusiones más frecuentes (sin gruesos):"] + [f"  {k}: {vv}" for k, vv in out["confusiones_sin_gruesos"].items()]
    a.text(0, 1, "\n".join(lin), va="top", fontsize=9, family="monospace", color=C.T1)
    for a in list(axs.flat):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"23 · El detector para trazos DELGADOS (λ par {cal['lam']:g}, sin esqueleto ni borde) sobre TODOS los "
                 "dígitos, con el compositor lineal de siempre", fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "23-digitos-delgadas-1-estadisticas.png", dpi=100); plt.close(fig)

    # ── figura 2: los errores (sin los muy gruesos); figura 3: los muy gruesos aparte ──
    for nombre, lista, titulo in (("2-errores", normales, f"los {len(normales)} errores de val (semilla 0) SIN los muy gruesos"),
                                  ("3-muy-gruesos", gruesos, f"los {len(gruesos)} errores de val de dígitos MUY GRUESOS "
                                                             f"(grosor típico ≥ {MUY_GRUESO:g} px), aparte")):
        if not len(lista):
            continue
        orden = lista[np.lexsort((pv0[lista], y[va][lista]))]
        cols = 12; filas = int(np.ceil(len(orden) / cols))
        fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.25, filas * 1.45 + 0.9))
        axs = np.atleast_2d(axs)
        for k, ax in enumerate(axs.flat):
            if k >= len(orden):
                ax.axis("off"); continue
            j = orden[k]; i = va[j]
            BG.dibujar_detector(ax, img[i].astype(np.float32), img[i].astype(float),
                                f"{y[i]}→{pv0[j]} · {anc[i]:.1f}px", NAR)
        fig.suptitle(f"23 · {titulo}\n«real→leído · grosor típico» · azul pálido = dígito · verde recto · naranja curvo · "
                     "flecha al centro de curvatura · gris: sin giro medible", fontsize=10, color=C.T1)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(DB.IMG / f"23-digitos-delgadas-{nombre}.png", dpi=100)
        plt.close(fig)
    # ── figura 4: los MISMOS errores de la figura 2, el dígito original sin ningún proceso (pedido del dueño) ──
    orden = normales[np.lexsort((pv0[normales], y[va][normales]))]
    cols = 12; filas = int(np.ceil(len(orden) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.25, filas * 1.45 + 0.9))
    for k, ax in enumerate(np.atleast_2d(axs).flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(orden):
            ax.axis("off"); continue
        j = orden[k]; i = va[j]
        ax.imshow(d["imagenes"][i], cmap="gray_r", vmin=0, vmax=d["imagenes"].max())
        ax.set_title(f"{y[i]}→{pv0[j]} · {anc[i]:.1f}px", fontsize=8.5, pad=2, color=NAR)
    fig.suptitle(f"23 · los mismos {len(orden)} errores de la figura 2, ORIGINALES sin ningún proceso (el dato del dataset tal "
                 "cual)\n«real→leído · grosor típico»", fontsize=10.5, color=C.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(DB.IMG / "23-digitos-delgadas-4-originales.png", dpi=100)
    plt.close(fig)
    print(f"→ 23-digitos-delgadas-*.png · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
