#!/usr/bin/env python3
"""El detector de curvas con la entrada SIEMPRE DELGADA: todo pasa antes por el esqueleto morfológico (pedido del dueño,
2026-10-10).

*«El esqueleto morfológico se ve bien. Ahora en el experimento detector de curvas, entrena el detector de curvas con esa
entrada siempre delgada (para las curvas definidas de ese detector, previamente adelgazadas por el morfológico).»*

La cadena pasa a ser:   imagen binaria ─► ESQUELETO (skimage.skeletonize) ─► detector de curvas (Gabor par + giro κ)
Ya no hay paso de borde: el esqueleto ES la línea que el detector lee. ⚠ La regla del dueño del 2026-10-10 prohíbe la
morfología para sacar el BORDE; aquí la pide él mismo para ADELGAZAR (la vio en `dig-delg`, figura 2-D).

«Entrenar» este detector es CALIBRARLO: no tiene pesos, tiene cuatro números que dependen del formato de la entrada
(el λ del Gabor par que lee las líneas, el umbral de trazo TAU, el suavizado de la energía σE y la coherencia mínima).
Se eligen con «las curvas definidas de ese detector»: el banco de `rect-lin` (rectas de todos los grosores, arcos de
R ≤ 27 y negativos), ESQUELETIZADO, puntuando por igual tres tasas (rectas → sólo trozos rectos · arcos → un trozo curvo
con su radio a ±25 % · negativos → ningún trozo). Los dígitos no se usan para calibrar. El giro mínimo de un trozo
curvo (2 °/px) y la distancia ±4 px no se tocan.

Luego, los dígitos: esqueleto → detector calibrado → el compositor de siempre (`digitos_bordes.py`: lineal, 180 / 1617,
3 semillas, 3823 a ciegas), y su curva de desplazamientos (regla del 2026-10-09).

PREDICCIÓN (escrita antes de correrlo, 2026-10-10): la calibración elige un λ par mayor que 3 (una línea de 1 px a 3 px
de franja se ve mal en oblicuo: lo midió franja1px.py) y un TAU alto; en el banco esqueletizado acierta ≥ 90 % de arcos y
de rectas. En dígitos, 0,94–0,96: el esqueleto quita el grosor —lo que confundía al detector—, pero mete ramitas en los
cruces y en los extremos; no supera con claridad al borde gris (0,960).

    python esqueleto.py   → imagenes/20-esqueleto-*.png · resultados-esqueleto.json  (~10 min)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import bordes as B                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import galeria                                                      # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
from gemelos_5 import mover                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

REJILLA = {"lam": (3.0, 4.0, 6.0), "tau": (0.3, 0.5, 0.8, 1.2), "sigma_e": (0.5, 1.0), "coh_min": (0.25, 0.35)}
DS = list(range(-4, 5))
NAR, AZU, VERDE, GRIS = "#c2410c", "#2a78d6", "#1b7f3b", "#8a8986"


def esqueleto(x: np.ndarray) -> np.ndarray:
    return skeletonize(np.asarray(x) > 0).astype(np.float32)


def puntuar_banco(b: dict, ids: np.ndarray) -> dict:
    """Las tres tasas sobre figuras del banco, esqueletizadas (con la configuración ACTUAL del detector)."""
    t = b["tipo"]; ok = {"rectas": [], "arcos": [], "negativos": []}
    for i in ids:
        _, kappa, _, vs = C.detectar(esqueleto(b["imagenes"][i])); cur, nr = B.trozos(kappa, vs)
        if t[i] == 0:
            ok["rectas"].append(bool(nr) and not cur)
        elif t[i] == 2:
            ok["arcos"].append(B.acierta("curva", int(b["radio"][i]), 1, cur, vs))   # esqueleto: el radio es R
        else:
            ok["negativos"].append(not cur and not nr)
    r = {k: round(100 * float(np.mean(v)), 1) for k, v in ok.items()}
    r["media"] = round(float(np.mean(list(r.values()))), 1)
    return r


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    b = dict(np.load(exigir_dataset("rect-lin-banco-r20261008") / "datos.npz"))
    t, rad = b["tipo"], b["radio"]
    defin = np.flatnonzero((t == 0) | ((t == 2) & (rad <= 27)) | (t >= 3))          # sin punteadas: no son curvas ni rectas
    calib = defin[::2]                                                             # la mitad para calibrar
    prueba = defin[1::2]                                                           # la otra mitad, para comprobar
    out: dict = {"rejilla": {k: list(v) for k, v in REJILLA.items()}, "calibracion": []}

    # ── 1. calibrar con el banco esqueletizado ──
    mejor, mejor_p = None, -1.0
    for lam, tau, se, coh in itertools.product(*REJILLA.values()):
        GR.configurar(tau, se, coh, lam)
        r = puntuar_banco(b, calib)
        out["calibracion"].append({"lam": lam, "tau": tau, "sigma_e": se, "coh_min": coh, **r})
        if r["media"] > mejor_p:
            mejor_p, mejor = r["media"], {"lam": lam, "tau": tau, "sigma_e": se, "coh_min": coh}
        print(f"λ {lam:g} TAU {tau} σE {se} coh {coh}: {r}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"ELEGIDA: {mejor}")
    GR.configurar(mejor["tau"], mejor["sigma_e"], mejor["coh_min"], mejor["lam"])
    out["elegida"] = mejor
    out["banco_mitad_de_prueba"] = puntuar_banco(b, prueba)
    a, na, r_, nr = BG.galeria_puntos(lambda x: esqueleto(x))
    out["galeria"] = {"arcos": f"{a}/{na}", "rectas": f"{r_}/{nr}"}
    print(f"banco (mitad de prueba): {out['banco_mitad_de_prueba']} · galería: {out['galeria']}", flush=True)

    # ── 2. los dígitos ──
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    X = DB.caract(img, borde=esqueleto)["bordes"]
    v, c, preds = [], [], {}
    for sem in D.SEMILLAS:
        pr = DB.entrenar(X[tr], y[tr], sem); pv = pr(X[va])
        v.append(float((pv == y[va]).mean())); c.append(float((pr(X[ex]) == y[ex]).mean()))
        if sem == 0:
            pred0, pv0 = pr, pv
    out["digitos"] = {"val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
                      "ciega": round(float(np.mean(c)), 4),
                      "por_digito_val": [round(float((pv0[y[va] == k] == k).mean()), 3) for k in range(10)]}
    conf = {}
    for j in np.flatnonzero(pv0 != y[va]):
        kk = f"{y[va][j]}→{pv0[j]}"; conf[kk] = conf.get(kk, 0) + 1
    out["digitos"]["confusiones_semilla_0"] = dict(sorted(conf.items(), key=lambda z: -z[1])[:8])
    out["digitos"]["fallos_semilla_0"] = int((pv0 != y[va]).sum())
    print(f"dígitos: {out['digitos']}  ({time.time() - t0:.0f} s)", flush=True)

    # ── 3. desplazamientos (semilla 0, sin re-entrenar; ⚠ la vertical está recortada: los dígitos ocupan los 32 px) ──
    des = {}
    for dire, f in {"horizontal": lambda a_, k: mover(a_, 0, k), "vertical": lambda a_, k: mover(a_, k, 0)}.items():
        acc, cam, rec = [], [], []
        for k in DS:
            xs = np.stack([f(img[i], k) for i in va])
            p = pred0(DB.caract(xs, borde=esqueleto)["bordes"])
            acc.append(round(float((p == y[va]).mean()), 4)); cam.append(round(float((p != pv0).mean()), 4))
            rec.append(round(float(np.mean([xs[n].sum() < img[i].sum() for n, i in enumerate(va)])), 4))
        des[dire] = {"acierto": acc, "cambian": cam, "recortados": rec}
        print(f"desplazamiento {dire}: {acc}", flush=True)
    out["desplazamientos"] = {"d": DS, **des}
    (AQUI / "resultados-esqueleto.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── figuras ──
    C._estilo()
    # 1 · lo que ve el detector sobre el esqueleto: galería (8) + un dígito de cada clase (10)
    gal = [it for g in galeria.trazos().values() for it in g[2]]
    gsel = [gal[k] for k in (2, 5, 7, 10, 13, 18, 21, 28)]
    dsel = [va[np.flatnonzero(y[va] == k)[0]] for k in range(10)]
    fig, axs = plt.subplots(2, 18, figsize=(26, 3.9))
    for j, (etq, x, *_r) in enumerate(gsel):
        BG.dibujar_detector(axs[1, j], esqueleto(x), x.astype(float), etq)
        axs[0, j].imshow(x, cmap="gray_r", vmin=0, vmax=1); axs[0, j].set_title("galería", fontsize=8)
    for j, i in enumerate(dsel):
        BG.dibujar_detector(axs[1, 8 + j], esqueleto(img[i]), img[i].astype(float), f"dígito {y[i]} · lee {pv0[np.flatnonzero(va == i)[0]]}")
        axs[0, 8 + j].imshow(img[i], cmap="gray_r", vmin=0, vmax=1)
    for a_ in axs.flat:
        a_.set_xticks([]); a_.set_yticks([])
    axs[0, 0].set_ylabel("entrada", fontsize=9); axs[1, 0].set_ylabel("esqueleto +\ndetector", fontsize=9)
    fig.suptitle(f"20 · El detector sobre el ESQUELETO (calibrado con el banco esqueletizado: λ par {mejor['lam']:g} · TAU "
                 f"{mejor['tau']} · σE {mejor['sigma_e']} · coh {mejor['coh_min']}) · verde recto · naranja curvo · flecha "
                 "al centro · gris: sin giro medible", fontsize=10.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "20-esqueleto-1-detector.png", dpi=95); plt.close(fig)
    # 2 · resultados
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    a_ = axs[0]
    filas = [("esqueleto (esto)", out["digitos"]["val"], out["digitos"]["ciega"], NAR),
             ("borde gris re-calibrado λ_b 10 · K 9", 0.9596, 0.9569, AZU),
             ("borde Gabor binarizado λ_b 6", 0.938, 0.9248, "#7fb0e8"),
             ("borde morfológico (retirado)", 0.8893, 0.8678, GRIS), ("rectas (referencia)", 0.9546, 0.9466, "#555")]
    for k, (n, vv, cc, col) in enumerate(filas):
        a_.barh(k, vv, color=col, height=0.6); a_.text(vv + 0.002, k, f"{vv:.3f}  (a ciegas {cc:.3f})", va="center", fontsize=8.5)
    a_.set_yticks(range(len(filas))); a_.set_yticklabels([f[0] for f in filas], fontsize=8.5); a_.invert_yaxis()
    a_.set_xlim(0.85, 1.0); a_.set_title("acierto en val (1617, 3 semillas); las otras filas, de sus JSON", loc="left", fontsize=9.5)
    a_ = axs[1]
    cal = sorted(out["calibracion"], key=lambda z: -z["media"])[:12]
    a_.axis("off")
    lin = ["calibración (banco esqueletizado, mitad de calibrar)", "λ   TAU  σE   coh   rectas arcos neg  media"]
    lin += [f"{z['lam']:<3g} {z['tau']:<4} {z['sigma_e']:<4} {z['coh_min']:<5} {z['rectas']:6.1f} {z['arcos']:5.1f} "
            f"{z['negativos']:5.1f} {z['media']:5.1f}" for z in cal]
    bp = out["banco_mitad_de_prueba"]
    lin += ["", f"elegida, en la mitad de PRUEBA: rectas {bp['rectas']} · arcos {bp['arcos']} · neg {bp['negativos']}",
            f"galería: arcos {out['galeria']['arcos']} · rectas {out['galeria']['rectas']}"]
    a_.text(0, 1, "\n".join(lin), va="top", fontsize=8.2, family="monospace", color=C.T1)
    a_ = axs[2]
    for dire, col in (("horizontal", NAR), ("vertical", GRIS)):
        a_.plot(DS, des[dire]["acierto"], "o-", color=col, label=f"{dire}" + (" (⚠ recortada)" if dire == "vertical" else ""))
    a_.plot(DS, np.array(des["horizontal"]["cambian"]), ":", color=NAR, label="% que cambia (horizontal), en fracción")
    a_.set_xticks(DS); a_.set_ylim(0, 1); a_.legend(frameon=False, fontsize=8); a_.set_xlabel("desplazamiento d (px)")
    a_.set_title("desplazamientos (semilla 0, sin re-entrenar)", loc="left", fontsize=9.5)
    for a_ in axs[[0, 2]]:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("20 · El detector de curvas con la entrada SIEMPRE DELGADA (esqueleto morfológico): calibración y dígitos",
                 fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "20-esqueleto-2-resultados.png", dpi=105); plt.close(fig)
    print(f"→ 20-esqueleto-*.png · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
