#!/usr/bin/env python3
"""El detector de curvas ajustado SÓLO para trazos DELGADOS (≤ 3 px), sobre la imagen tal cual (pedido del dueño,
2026-10-10).

*«El esqueletizado no funciona con estas curvas gruesas. Vamos a cambiar la estrategia. Vamos a ajustar Gabor para
rectas delgadas solamente. Delgadas ≤ 3 px. Debe reconocer rectas, curvas y posición del centro aproximada. No apliques
esqueletización a nada. Dime cómo queda el detector.»*

Cadena:   imagen binaria (trazo de 1–3 px) ─► banco de Gabor PAR (12 orientaciones, 9×9, λ a elegir) ─► orientación local
          θ y coherencia ─► giro κ a lo largo de la tangente ─► TROZOS rectos y curvos, cada uno con:
              · tipo (recto / curvo)
              · posición de su centro (centroide del trozo, px)
              · si es curvo: radio (57,3 / |κ|), dirección hacia su centro de curvatura y ese centro
Sin esqueleto y sin paso de borde. Es el detector de `curvas.py`/`bordes.py` con su calibración rehecha para lo delgado.

LA CALIBRACIÓN (los números que dependen del formato de la entrada: λ del Gabor par, umbral de trazo TAU, suavizado de
la energía σE, coherencia mínima y giro mínimo de un trozo curvo) se elige con figuras SINTÉTICAS DELGADAS de verdad
conocida —rectas y arcos de 1–3 px, en posición, ángulo, radio y largo al azar, más negativos—, porque el banco de
`rect-lin` no guarda posiciones y no dejaría medir el error de posición. Mitad para calibrar, mitad para comprobar
(semillas distintas); y además la galería (sus trazos de ≤ 3 px) y el banco (sus figuras de grosor ≤ 3).

PREDICCIÓN (escrita antes de correrlo, 2026-10-10): la calibración elige λ 4–6 (franja de 2–3 px: la de λ 3 se queda
corta para 3 px y la escalera de 1 px oblicuo la rompe, franja1px.py); rectas y arcos de R 5–25 bien clasificados en
≥ 90 %; radio a ±25 % en ~80 % de los arcos; centro del trozo a ≤ 1,5 px de la verdad en la mediana.

    python delgadas.py   → imagenes/22-delgadas-*.png · resultados-delgadas.json  (~3 min)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import galeria                                                      # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

REJILLA = {"lam": (3.0, 4.0, 5.0, 6.0), "tau": (0.3, 0.6, 0.9, 1.2), "sigma_e": (0.5, 1.0), "coh_min": (0.25, 0.35),
           "kappa_min": (1.5, 2.0)}
N_POR_TIPO = {"calibrar": 150, "probar": 300}
TOL_R = 0.25
NAR, AZU, VERDE = "#c2410c", "#2a78d6", "#1b7f3b"


# ─── figuras sintéticas delgadas, con su verdad ───────────────────────────────────────────────────────────────────
def sinteticas(n: int, semilla: int) -> list[dict]:
    rng = np.random.default_rng(semilla); out = []
    while sum(f["tipo"] == "recta" for f in out) < n:
        L, a, g = rng.uniform(12, 24), rng.uniform(0, 180), int(rng.integers(1, 4))
        cx, cy = rng.uniform(8, 24, 2)
        x = C.recta(cx, cy, L, a, g)
        if x.sum() >= 0.9 * L * g and _dentro(x):
            out.append({"tipo": "recta", "x": x, "centro": (cx, cy), "grosor": g, "largo": L})
    while sum(f["tipo"] == "arco" for f in out) < n:
        R, L, a, g = float(np.exp(rng.uniform(np.log(5), np.log(25)))), rng.uniform(14, 24), rng.uniform(0, 360), int(rng.integers(1, 4))
        mx, my = rng.uniform(8, 24, 2)
        x = C.arco(mx, my, R, a, L, g)
        if x.sum() >= 0.8 * min(L, 2 * np.pi * R) * g and _dentro(x):
            out.append({"tipo": "arco", "x": x, "centro": (mx, my), "radio": R, "grosor": g,
                        "centro_curv": galeria._centro(mx, my, R, a)})
    for k in range(n):
        x = (rng.random((32, 32)) < 0.01).astype(np.uint8) if k % 2 else C.discos(rng.uniform(4, 28, size=(6, 2)))
        out.append({"tipo": "negativo", "x": x})
    return out


def _dentro(x) -> bool:
    return x[:2].sum() + x[-2:].sum() + x[:, :2].sum() + x[:, -2:].sum() == 0      # 2 px de margen: nada recortado


def configurar(lam, tau, sigma_e, coh_min, kappa_min) -> None:
    GR.configurar(tau, sigma_e, coh_min, lam)
    C.KAPPA_MIN = kappa_min


# ─── el detector: trozos con su posición ──────────────────────────────────────────────────────────────────────────
def detectar(x: np.ndarray) -> dict:
    c = C.campo(x.astype(np.float32)); kappa, mask, emin = C.giro(c)
    vs = C.veredicto(c, kappa, mask, emin)
    rectos, curvos = [], []
    for v in vs:
        med = v["nucleo"] & np.isfinite(kappa)
        for es_curvo in (False, True):
            sel = med & ((np.abs(kappa) >= C.KAPPA_MIN) if es_curvo else (np.abs(kappa) < C.KAPPA_MIN))
            lab, n = label(sel, structure=np.ones((3, 3)))
            for z in range(1, n + 1):
                m = lab == z
                if m.sum() < C.MIN_TROZO:
                    continue
                ys, xs = np.nonzero(m)
                t = {"centro": (float(xs.mean() + 0.5), float(ys.mean() + 0.5)), "px": int(m.sum()), "m": m}
                if es_curvo:
                    th, k = np.deg2rad(c["theta"][ys, xs]), kappa[ys, xs]
                    kv = np.array([(-k * np.sin(th)).mean(), (k * np.cos(th)).mean()]); d = np.arctan2(kv[1], kv[0])
                    R = float(57.3 / np.median(np.abs(k)))
                    t.update({"radio": R, "direccion": float(np.degrees(d) % 360),
                              "centro_curv": (t["centro"][0] + R * np.cos(d), t["centro"][1] + R * np.sin(d))})
                    curvos.append(t)
                else:
                    rectos.append(t)
    return {"rectos": sorted(rectos, key=lambda t: -t["px"]), "curvos": sorted(curvos, key=lambda t: -t["px"]),
            "c": c, "kappa": kappa, "mask": mask}


def evaluar(figs: list[dict]) -> dict:
    ok = {"recta": [], "arco": [], "negativo": []}; radio, pos_r, pos_a, pos_cc = [], [], [], []
    for f in figs:
        d = detectar(f["x"])
        if f["tipo"] == "recta":
            bien = bool(d["rectos"]) and not d["curvos"]; ok["recta"].append(bien)
            if d["rectos"]:
                pos_r.append(float(np.hypot(*np.subtract(d["rectos"][0]["centro"], f["centro"]))))
        elif f["tipo"] == "arco":
            ok["arco"].append(bool(d["curvos"]))
            if d["curvos"]:
                t = d["curvos"][0]
                radio.append(abs(np.log(t["radio"] / f["radio"])) < np.log(1 + TOL_R))
                pos_a.append(float(np.hypot(*np.subtract(t["centro"], f["centro"]))))
                pos_cc.append(float(np.hypot(*np.subtract(t["centro_curv"], f["centro_curv"]))))
        else:
            ok["negativo"].append(not d["rectos"] and not d["curvos"])
    r = {f"{k}_bien": round(100 * float(np.mean(v)), 1) for k, v in ok.items()}
    r["radio_a_25pc"] = round(100 * float(np.sum(radio)) / max(len(ok["arco"]), 1), 1)    # sobre TODOS los arcos
    r["pos_recta_mediana_px"] = round(float(np.median(pos_r)), 2) if pos_r else None
    r["pos_arco_mediana_px"] = round(float(np.median(pos_a)), 2) if pos_a else None
    r["pos_arco_p90_px"] = round(float(np.percentile(pos_a, 90)), 2) if pos_a else None
    r["centro_curv_mediana_px"] = round(float(np.median(pos_cc)), 2) if pos_cc else None
    r["puntos"] = round((r["recta_bien"] + r["radio_a_25pc"] + r["negativo_bien"]) / 3, 1)
    return r


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    cal, pru = sinteticas(N_POR_TIPO["calibrar"], 101), sinteticas(N_POR_TIPO["probar"], 202)
    out: dict = {"rejilla": {k: list(v) for k, v in REJILLA.items()}, "n": N_POR_TIPO, "calibracion": []}
    mejor, mejor_p = None, -1.0
    for combo in itertools.product(*REJILLA.values()):
        p = dict(zip(REJILLA, combo)); configurar(**p)
        r = evaluar(cal); out["calibracion"].append({**p, **r})
        if r["puntos"] > mejor_p:
            mejor_p, mejor = r["puntos"], p
    print(f"calibración: {len(out['calibracion'])} combinaciones en {time.time() - t0:.0f} s · ELEGIDA {mejor} ({mejor_p})",
          flush=True)
    configurar(**mejor); out["elegida"] = mejor
    out["prueba_sinteticas"] = evaluar(pru)
    print(f"prueba (sintéticas, otra semilla): {out['prueba_sinteticas']}", flush=True)
    # galería: sólo sus trazos de ≤ 3 px
    gal = [it for g in galeria.trazos().values() for it in g[2] if it[4] <= 3]
    gf = [{"tipo": "arco" if esp == "curva" else "recta", "x": x, "centro": (16, 16) if cr is None else None,
           "radio": R, "centro_curv": cr, "etq": etq} for etq, x, esp, R, g, cr in gal]
    gr = {"rectas": 0, "n_rectas": 0, "arcos": 0, "arcos_radio": 0, "n_arcos": 0}
    for f in gf:
        d = detectar(f["x"])
        if f["tipo"] == "recta":
            gr["n_rectas"] += 1; gr["rectas"] += bool(d["rectos"]) and not d["curvos"]
        else:
            gr["n_arcos"] += 1; gr["arcos"] += bool(d["curvos"])
            gr["arcos_radio"] += bool(bool(d["curvos"]) and abs(np.log(d["curvos"][0]["radio"] / f["radio"])) < np.log(1 + TOL_R))
    out["galeria_delgados"] = gr
    # banco de rect-lin: sus figuras de grosor ≤ 3 (sin posición: sólo clasificación)
    b = dict(np.load(exigir_dataset("rect-lin-banco-r20261008") / "datos.npz"))
    t, g, lg, rad = b["tipo"], b["grosor"], b["largo"], b["radio"]
    bk = {}
    for nombre, ids, que in (("rectas g ≤ 3, largo 16–22", np.flatnonzero((t == 0) & (g <= 3) & (lg >= 16)), "recta"),
                             ("rectas g ≤ 3, largo 10", np.flatnonzero((t == 0) & (g <= 3) & (lg < 16)), "recta"),
                             ("arcos g 2, R ≤ 27", np.flatnonzero((t == 2) & (g <= 3) & (rad <= 27)), "arco"),
                             ("arcos g 2, R 40", np.flatnonzero((t == 2) & (g <= 3) & (rad > 27)), "arco"),
                             ("negativos", np.flatnonzero(t >= 3)[::2], "negativo")):
        s = []
        for i in ids:
            d = detectar(b["imagenes"][i])
            s.append(bool(d["rectos"]) and not d["curvos"] if que == "recta" else bool(d["curvos"]) if que == "arco"
                     else not d["rectos"] and not d["curvos"])
        bk[nombre] = {"bien_pc": round(100 * float(np.mean(s)), 1), "n": int(len(ids))}
    out["banco_delgados"] = bk
    print(f"galería: {gr} · banco: {bk}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados-delgadas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── figura: ejemplos de prueba con lo que devuelve el detector ──
    C._estilo()
    rng = np.random.default_rng(7)
    ej = [pru[i] for i in rng.choice([k for k, f in enumerate(pru) if f["tipo"] == "recta"], 8, replace=False)] + \
         [pru[i] for i in rng.choice([k for k, f in enumerate(pru) if f["tipo"] == "arco"], 16, replace=False)]
    fig, axs = plt.subplots(3, 8, figsize=(17, 8.8))
    for ax, f in zip(axs.flat, ej):
        d = detectar(f["x"])
        ax.imshow(np.where(f["x"] > 0, 1.0, np.nan), cmap=BG.ListedColormap(["#d6e4f5"]), vmin=0, vmax=1)
        for tz in d["rectos"]:
            ax.imshow(np.where(tz["m"], 1.0, np.nan), cmap=BG.ListedColormap([VERDE]), vmin=0, vmax=1)
            ax.plot(tz["centro"][0] - 0.5, tz["centro"][1] - 0.5, "+", color=C.T1, ms=9, mew=1.8)
        for tz in d["curvos"]:
            ax.imshow(np.where(tz["m"], 1.0, np.nan), cmap=BG.ListedColormap([NAR]), vmin=0, vmax=1)
            (cx, cy), dd = tz["centro"], np.deg2rad(tz["direccion"])
            ax.plot(cx - 0.5, cy - 0.5, "+", color=C.T1, ms=9, mew=1.8)
            ax.annotate("", xy=(cx - 0.5 + 5 * np.cos(dd), cy - 0.5 + 5 * np.sin(dd)), xytext=(cx - 0.5, cy - 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=AZU, lw=1.3, mutation_scale=8))
        ax.plot(f["centro"][0] - 0.5, f["centro"][1] - 0.5, "x", color=AZU, ms=7, mew=1.5)
        if f["tipo"] == "recta":
            tit = "RECTA ✓" if d["rectos"] and not d["curvos"] else ("CURVA ✗" if d["curvos"] else "NADA ✗")
            col = VERDE if "✓" in tit else NAR
            tit += f" · g {f['grosor']}"
        else:
            if d["curvos"]:
                R = d["curvos"][0]["radio"]; okr = abs(np.log(R / f["radio"])) < np.log(1 + TOL_R)
                tit = f"CURVA R≈{R:.0f} (real {f['radio']:.0f}) {'✓' if okr else '✗'}"; col = VERDE if okr else NAR
            else:
                tit = "RECTA ✗" if d["rectos"] else "NADA ✗"; col = NAR
            tit += f" · g {f['grosor']}"
        ax.set_title(tit, fontsize=8, color=col); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
    p = out["prueba_sinteticas"]
    fig.suptitle(f"22 · El detector para trazos DELGADOS (1–3 px), sobre la imagen, sin esqueleto · λ par {mejor['lam']:g} · "
                 f"TAU {mejor['tau']} · σE {mejor['sigma_e']} · coh {mejor['coh_min']} · giro mín. {mejor['kappa_min']} °/px\n"
                 f"verde = trozo recto · naranja = trozo curvo · + = centro que da el detector · x = centro real · flecha = "
                 f"hacia el centro de curvatura\nprueba ({N_POR_TIPO['probar']} por tipo): rectas {p['recta_bien']} % · "
                 f"arcos {p['arco_bien']} % (radio ±25 %: {p['radio_a_25pc']} %) · negativos {p['negativo_bien']} % · "
                 f"centro: mediana {p['pos_recta_mediana_px']} px (rectas), {p['pos_arco_mediana_px']} px (arcos)",
                 fontsize=10, color=C.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.6); fig.savefig(C.IMG / "22-delgadas-detector.png", dpi=100); plt.close(fig)
    print(f"→ 22-delgadas-detector.png · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
