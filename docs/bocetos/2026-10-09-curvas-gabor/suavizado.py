#!/usr/bin/env python3
"""¿Suavizar el dígito antes de sacarle el borde arregla los fallos? (pedido del dueño, 2026-10-10)

Hoy la cadena es: dígito binario → BORDE (contorno de 1 px) → Gabor λ = 3 → orientación y giro. El borde hereda la
ESCALERA del dígito (los de UCI vienen escalados desde 8×8), y en la figura 15 se midió que sólo el 18–25 % del borde
deja medir el giro. Aquí, ANTES del borde, el dígito se suaviza con una gaussiana de σ px y se vuelve a binarizar en
0,5: la forma es la misma, el contorno sale menos escalonado. Todo lo demás, igual (`digitos_bordes.py`).

Pedido: «prueba sólo los dígitos de error». Se enseñan los 181 que falla σ = 0 (semilla 0), pero el compositor se
re-entrena para cada σ con el train suavizado igual, y ⚠ se da TAMBIÉN el acierto en val entero (3 semillas) y cuántos
aciertos de antes se ROMPEN: mirar sólo los fallos sesga a favor de cualquier cambio (un re-entreno con otra entrada
arregla algunos por azar y rompe otros que esa vista no enseña).

PREDICCIÓN (antes de correrlo): σ 0,5–1 sube el borde medible y arregla una parte de los fallos; σ ≥ 2 empieza a
borrar huecos (el lazo pequeño del 8, el ojo del 9) y empeora.

    python suavizado.py   → imagenes/16-suavizado-*.png · resultados-suavizado.json  (~3 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import bordes as B                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0)
VERDE, NAR = "#1b7f3b", "#c2410c"


def suavizar(img: np.ndarray, s: float) -> np.ndarray:
    """(n,32,32) o (32,32) binario → suavizado con gaussiana σ = s y re-binarizado en 0,5 (s = 0: tal cual)."""
    if s == 0:
        return img
    ax = (0, s, s) if img.ndim == 3 else s
    return (gaussian_filter(img.astype(np.float32), ax) > 0.5).astype(np.uint8)


def medible(x: np.ndarray) -> float:
    b = B.bordes(x); c = C.campo(b); k, m, _ = C.giro(c)
    return float((m & np.isfinite(k) & (b > 0)).sum() / max(b.sum(), 1))


def dibujar(ax, x: np.ndarray, titulo: str, color: str, negrita=False):
    """El dígito (azul pálido) y lo que ve el detector en su borde: gris sin giro · verde recto · naranja curvo."""
    b = B.bordes(x); c = C.campo(b); kappa, mask, emin = C.giro(c)
    vs = C.veredicto(c, kappa, mask, emin); curvos, _ = B.trozos(kappa, vs, c["theta"])
    med = mask & np.isfinite(kappa)
    ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=ListedColormap(["#e3edf8"]), vmin=0, vmax=1)
    ax.imshow(np.where(b & ~med, 1.0, np.nan), cmap=ListedColormap(["#7d7c78"]), vmin=0, vmax=1)
    ax.imshow(np.where(med & (np.abs(kappa) < C.KAPPA_MIN), 1.0, np.nan), cmap="Greens", vmin=0, vmax=1.4)
    ax.imshow(np.where(med & (np.abs(kappa) >= C.KAPPA_MIN), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=10)
    for t in curvos:
        (cx, cy), dd = t["centroide"], np.deg2rad(t["direccion"])
        ax.annotate("", xy=(cx - 0.5 + 4 * np.cos(dd), cy - 0.5 + 4 * np.sin(dd)), xytext=(cx - 0.5, cy - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=DB.AZU, lw=1.1, mutation_scale=7))
    ax.set_title(titulo, fontsize=9, pad=2, color=color, fontweight="bold" if negrita else "normal")
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    B.configurar()
    pred, res = {}, {}
    for s in SIGMAS:
        xs = suavizar(img, s)
        X = DB.caract(xs)["bordes"]
        accs = []
        for sem in D.SEMILLAS:
            p = DB.entrenar(X[tr], y[tr], sem)(X[va])
            accs.append(float((p == y[va]).mean()))
            if sem == 0:
                pred[s] = p
        res[s] = {"val_3_semillas": round(float(np.mean(accs)), 4), "val_rango": [round(min(accs), 4), round(max(accs), 4)],
                  "val_semilla_0": round(accs[0], 4)}
        print(f"σ {s:<4} val {res[s]['val_3_semillas']:.4f} {res[s]['val_rango']}  ({time.time() - t0:.0f} s)", flush=True)
    base = pred[0.0]
    fallos = np.flatnonzero(base != y[va])                                 # posiciones dentro de va
    aciertos = np.flatnonzero(base == y[va])
    for s in SIGMAS:
        p = pred[s]
        res[s].update({"de_los_181_arreglados": int((p[fallos] == y[va][fallos]).sum()),
                       "de_los_181_siguen_mal": int((p[fallos] != y[va][fallos]).sum()),
                       "aciertos_que_se_rompen": int((p[aciertos] != y[va][aciertos]).sum()),
                       "borde_medible_en_los_181": round(float(np.mean([medible(suavizar(img[va[j]], s)) for j in fallos])), 3)})
        r = res[s]
        print(f"σ {s:<4} de 181: arreglados {r['de_los_181_arreglados']:3d} · rotos nuevos {r['aciertos_que_se_rompen']:3d} · "
              f"borde medible {r['borde_medible_en_los_181']:.3f}", flush=True)
    mejor = max(SIGMAS, key=lambda s: res[s]["val_3_semillas"])
    print(f"mejor σ por val entero (3 semillas): {mejor}")

    C._estilo()
    # ── figura 1: el resumen por σ y ejemplos de fallos a lo largo de σ ──
    ejemplos = []
    orden_conf = sorted({(int(y[va][j]), int(base[j])) for j in fallos},
                        key=lambda k: -int(((y[va][fallos] == k[0]) & (base[fallos] == k[1])).sum()))
    for real, leido in orden_conf[:10]:
        j = fallos[(y[va][fallos] == real) & (base[fallos] == leido)][0]; ejemplos.append(j)
    fig = plt.figure(figsize=(14, 3.6 + 1.25 * len(ejemplos)))
    gs = fig.add_gridspec(2 + len(ejemplos), len(SIGMAS), height_ratios=[3.2, 0.25] + [1] * len(ejemplos), hspace=0.5,
                          wspace=0.12, top=0.95, bottom=0.01)
    a = fig.add_subplot(gs[0, :3])
    xs_ = np.arange(len(SIGMAS))
    a.bar(xs_ - 0.2, [res[s]["de_los_181_arreglados"] for s in SIGMAS], 0.4, color=VERDE, label="de los 181, arreglados")
    a.bar(xs_ + 0.2, [res[s]["aciertos_que_se_rompen"] for s in SIGMAS], 0.4, color=NAR, label="aciertos de antes que se rompen")
    for k, s in enumerate(SIGMAS):
        a.text(k - 0.2, res[s]["de_los_181_arreglados"] + 1, str(res[s]["de_los_181_arreglados"]), ha="center", fontsize=8)
        a.text(k + 0.2, res[s]["aciertos_que_se_rompen"] + 1, str(res[s]["aciertos_que_se_rompen"]), ha="center", fontsize=8)
    a.set_xticks(xs_); a.set_xticklabels([f"σ {s:g}" for s in SIGMAS]); a.legend(frameon=False, fontsize=8)
    a.set_title("semilla 0: cuántos de los 181 fallos se arreglan, y cuántos aciertos se pierden", loc="left", fontsize=9.5)
    a.spines[["top", "right"]].set_visible(False)
    a2 = fig.add_subplot(gs[0, 3:])
    a2.plot(xs_, [res[s]["val_3_semillas"] for s in SIGMAS], "o-", color=DB.AZU, label="acierto val entero (3 semillas)")
    for k, s in enumerate(SIGMAS):
        a2.text(k, res[s]["val_3_semillas"] + 0.004, f"{res[s]['val_3_semillas']:.3f}", ha="center", fontsize=8)
    a3 = a2.twinx()
    a3.plot(xs_, [res[s]["borde_medible_en_los_181"] for s in SIGMAS], "s--", color=C.T2, label="borde medible (los 181)")
    a3.set_ylim(0, 0.6); a3.set_ylabel("fracción del borde medible", fontsize=8)
    a2.set_xticks(xs_); a2.set_xticklabels([f"σ {s:g}" for s in SIGMAS]); a2.set_ylim(0.8, 0.95)
    a2.set_title("acierto en val entero (azul) y borde medible en los 181 (gris)", loc="left", fontsize=9.5)
    for k, j in enumerate(ejemplos):
        for c_, s in enumerate(SIGMAS):
            ax = fig.add_subplot(gs[2 + k, c_])
            ok = pred[s][j] == y[va][j]
            dibujar(ax, suavizar(img[va[j]], s), f"σ {s:g}: lee {pred[s][j]}" + (" ✓" if ok else ""), VERDE if ok else NAR, ok)
            if c_ == 0:
                ax.set_ylabel(f"real {y[va][j]}", fontsize=9)
    fig.suptitle("16 · Suavizar el dígito ANTES de sacarle el borde (gaussiana σ, re-binarizada en 0,5): resumen y un fallo "
                 "de cada confusión frecuente a lo largo de σ", fontsize=11, color=C.T1, y=0.99)
    fig.savefig(DB.IMG / "16-suavizado-1-niveles.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→ 16-suavizado-1-niveles.png")

    # ── figura 2: los 181 fallos con el mejor σ ──
    cols = 16; filas = int(np.ceil(len(fallos) / cols))
    orden = fallos[np.lexsort((base[fallos], y[va][fallos]))]
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 0.95, filas * 1.12 + 0.6))
    for k, ax in enumerate(axs.flat):
        if k >= len(orden):
            ax.axis("off"); continue
        j = orden[k]; ok = pred[mejor][j] == y[va][j]
        dibujar(ax, suavizar(img[va[j]], mejor), f"{y[va][j]}→{base[j]}→{pred[mejor][j]}" + (" ✓" if ok else ""),
                VERDE if ok else NAR, ok)
    fig.suptitle(f"16 · Los 181 fallos de σ = 0, con el dígito suavizado a σ = {mejor:g} (el mejor en val entero) · "
                 f"«real→leído con σ 0→leído con σ {mejor:g}» · verde ✓ = arreglado\n"
                 f"arreglados {res[mejor]['de_los_181_arreglados']} de 181 · aciertos de antes que se rompen "
                 f"{res[mejor]['aciertos_que_se_rompen']} · val entero {res[0.0]['val_3_semillas']:.3f} → "
                 f"{res[mejor]['val_3_semillas']:.3f} (3 semillas)", fontsize=10, color=C.T1, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.3)
    fig.savefig(DB.IMG / "16-suavizado-2-fallos.png", dpi=100); plt.close(fig)
    print("→ 16-suavizado-2-fallos.png")
    (AQUI / "resultados-suavizado.json").write_text(json.dumps(
        {"sigmas": list(SIGMAS), "mejor_por_val": mejor, "por_sigma": {f"{s:g}": r for s, r in res.items()},
         "ejemplos_val": [int(va[j]) for j in ejemplos]}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
