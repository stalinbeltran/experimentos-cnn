#!/usr/bin/env python3
"""El borde de Gabor impar SIN binarizar, con el detector RE-CALIBRADO para él (pedido del dueño, 2026-10-10).

En `bordes_gabor.py` el borde gris se hundía (0,33–0,62 en dígitos salvo λ 4 sin suavizar, 0,909) porque el detector se
había calibrado sobre un contorno binario de 1 px: una banda gris de 2–3 px dejaba 3–17 píxeles medibles por dígito.
Aquí se re-calibran, PARA CADA Gabor de borde, los números del detector que dependen del formato de la entrada:

    TAU        umbral de energía para «aquí hay trazo» (era 0,5)
    SIGMA_E    suavizado de la energía antes del umbral (era 1)
    COH_MIN    coherencia mínima para creer la orientación (era 0,35)
    LAM        el λ del Gabor PAR que lee las líneas (era 3, la franja de 1,5 px; una banda gris de 2–3 px puede
               pedir más)

⚠ La calibración se elige SÓLO con la galería de 32 trazos (arcos con su radio a ±25 % + rectas sin curva), nunca con
los dígitos: así los dígitos de val y los de a ciegas siguen siendo una prueba. El giro mínimo de un trozo curvo
(2 °/px) y la distancia ±4 px no se tocan: no dependen del formato del borde.

PREDICCIÓN (escrita antes de correrlo, 2026-10-10): con la calibración propia el borde gris se acerca al binarizado
(0,92–0,94 en val) pero no lo supera con claridad; el mejor, λ_b 4–6.

    python gris_recalibrado.py   → imagenes/18-gris-*.png · resultados-gris-recalibrado.json  (~15 min)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import bordes as B                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

BORDES = [(lb, kb) for lb in (3.0, 4.0, 6.0, 8.0, 10.0, 12.0) for kb in (5, 7, 9)]   # 10–12 y K 9: añadidos al salir el mejor en el borde del rango (λ_b 8)
REJILLA = {"tau": (0.1, 0.2, 0.3, 0.4, 0.5), "sigma_e": (0.5, 1.0), "coh_min": (0.25, 0.35), "lam": (3.0, 4.0)}
VERDE, NAR, AZU = "#1b7f3b", "#c2410c", "#2a78d6"


def configurar(tau: float, sigma_e: float, coh_min: float, lam: float) -> None:
    """Como `bordes.configurar`, con los números a elegir (el detector lee estos globales en cada llamada)."""
    B.configurar()                                       # deja campo suavizado, KAPPA_MIN 2, etc.
    C.TAU, C.COH_MIN, C.LAM = tau, coh_min, lam
    C.GABOR = torch.from_numpy(np.stack([C.kernel_gabor(C.K, t, lam) for t in C.THETAS])[:, None])
    from scipy.ndimage import gaussian_filter

    def campo(x):
        c = B._campo_original(x)
        c["Emax"] = gaussian_filter(c["Emax"], sigma_e)
        return c
    C.campo = campo


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    out: dict = {"rejilla": {k: list(v) for k, v in REJILLA.items()}, "por_borde": {}}

    # ── 1. calibración con la galería, para cada Gabor de borde ──
    elegidas = {}
    for lb, kb in BORDES:
        bg = BG.BordeGabor(lb, k=kb, binarizar=False)
        mejor, mejor_p = None, (-1, -1)
        for tau, se, coh, lam in itertools.product(*REJILLA.values()):
            configurar(tau, se, coh, lam)
            a, na, r, nr = BG.galeria_puntos(bg)
            p = (a + r, -abs(lam - 3.0))                 # empate: el λ par de siempre (3)
            if p > mejor_p:
                mejor_p, mejor = p, {"tau": tau, "sigma_e": se, "coh_min": coh, "lam": lam, "galeria_arcos": f"{a}/{na}",
                                     "galeria_rectas": f"{r}/{nr}"}
        elegidas[(lb, kb)] = mejor
        print(f"borde λ_b {lb:g} K {kb}: calibración {mejor}  ({time.time() - t0:.0f} s)", flush=True)

    # ── 2. los dígitos, con la calibración de cada borde ──
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")

    def evaluar(X):
        v, c = [], []
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem)
            v.append(float((pr(X[va]) == y[va]).mean())); c.append(float((pr(X[ex]) == y[ex]).mean()))
        return {"val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
                "ciega": round(float(np.mean(c)), 4)}

    def medibles(borde, n=300):
        return float(np.mean([np.isfinite(C.giro(C.campo(borde(x.astype(np.float32))))[0]).sum() for x in img[:n]]))

    # referencia: el binarizado de bordes_gabor.py (λ_b 6, K 5) con la calibración de antes
    B.configurar()
    ref = BG.BordeGabor(6.0, binarizar=True)
    out["referencia_binarizado"] = {**evaluar(DB.caract(img, borde=ref)["bordes"]), "medibles": medibles(ref)}
    print(f"referencia binarizado λ_b 6: {out['referencia_binarizado']}", flush=True)
    for (lb, kb), cal in elegidas.items():
        bg = BG.BordeGabor(lb, k=kb, binarizar=False)
        sin = {}
        B.configurar()                                   # sin re-calibrar, para ver cuánto aporta re-calibrar
        sin = {**evaluar(DB.caract(img, borde=bg)["bordes"]), "medibles": medibles(bg)}
        configurar(cal["tau"], cal["sigma_e"], cal["coh_min"], cal["lam"])
        con = {**evaluar(DB.caract(img, borde=bg)["bordes"]), "medibles": medibles(bg)}
        out["por_borde"][f"λ_b {lb:g} · K {kb}"] = {"calibracion": cal, "sin_recalibrar": sin, "recalibrado": con}
        print(f"dígitos · gris λ_b {lb:g} K {kb}: sin re-calibrar val {sin['val']:.4f} (medibles {sin['medibles']:.0f}) · "
              f"re-calibrado val {con['val']:.4f} {con['val_rango']} a ciegas {con['ciega']:.4f} (medibles {con['medibles']:.0f})"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados-gris-recalibrado.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")

    # ── figura ──
    C._estilo()
    claves = list(out["por_borde"])
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.3, 1]})
    a = axs[0]; x_ = np.arange(len(claves)); w = 0.38
    s_ = [out["por_borde"][k]["sin_recalibrar"]["val"] for k in claves]
    c_ = [out["por_borde"][k]["recalibrado"]["val"] for k in claves]
    a.bar(x_ - w / 2, s_, w, color="#c9c8c4", label="gris, calibración de antes")
    a.bar(x_ + w / 2, c_, w, color=AZU, label="gris, RE-CALIBRADO con la galería")
    for i in range(len(claves)):
        a.text(i - w / 2, s_[i] + 0.008, f"{s_[i]:.3f}", ha="center", fontsize=7.5, rotation=90)
        a.text(i + w / 2, c_[i] + 0.008, f"{c_[i]:.3f}", ha="center", fontsize=7.5, rotation=90)
    rb = out["referencia_binarizado"]["val"]
    a.axhline(rb, color=NAR, ls="--", lw=1.2); a.text(len(claves) - 0.5, rb + 0.006, f"binarizado λ_b 6: {rb:.3f}",
                                                      ha="right", fontsize=8.5, color=NAR)
    a.set_xticks(x_); a.set_xticklabels(claves, fontsize=8.5, rotation=20); a.set_ylim(0.2, 1.02)
    a.set_title("acierto en val (1617, 3 semillas) con el borde de Gabor impar GRIS (sin binarizar)", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8.5, loc="lower left"); a.spines[["top", "right"]].set_visible(False)
    a = axs[1]; a.axis("off")
    lin = [f"{'borde':14s} {'TAU':>4s} {'σE':>4s} {'coh':>5s} {'λ':>3s}  galería   val    ciega  medibles"]
    for k in claves:
        r = out["por_borde"][k]; cal, co = r["calibracion"], r["recalibrado"]
        lin.append(f"{k:14s} {cal['tau']:4.1f} {cal['sigma_e']:4.1f} {cal['coh_min']:5.2f} {cal['lam']:3.0f}  "
                   f"{cal['galeria_arcos']:>5s}+{cal['galeria_rectas']:<4s} {co['val']:.3f}  {co['ciega']:.3f}  {co['medibles']:5.0f}")
    rbm = out["referencia_binarizado"]
    lin += ["", f"referencia: binarizado λ_b 6 (calibración de antes)", f"   val {rbm['val']:.3f} · a ciegas {rbm['ciega']:.3f}"
            f" · medibles {rbm['medibles']:.0f}", "", "calibración = la mejor en la galería de 32 trazos",
            "(los dígitos NO se usaron para elegirla); medibles =",
            "píxeles con giro medible por dígito (300 dígitos)"]
    a.text(0, 1, "\n".join(lin), va="top", fontsize=8.3, family="monospace", color=C.T1)
    fig.suptitle("18 · El borde de Gabor impar SIN binarizar, con el detector re-calibrado para él", fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "18-gris-recalibrado.png", dpi=105); plt.close(fig)
    print("→ 18-gris-recalibrado.png"); print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
