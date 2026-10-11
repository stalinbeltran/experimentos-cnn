#!/usr/bin/env python3
"""rizo-2 — el detector COMPUESTO de los tres primeros (R rayos, P plantillas, V votos de curvatura), pedido por el dueño
el 2026-10-11 tras ver que P y R encuentran el rizo (91 / 83 %) pero ponen 10–17 candidatos de más por dígito a lo largo
de todo el contorno (figura 8), y que V da un candidato y acierta poco (36 %).

Cómo se compone (reglas escritas a mano, sin entrenar nada con dígitos):
  1. CANDIDATOS = la unión de los de P (S ≥ 0,8), los de R (F ≥ 0,5, sin exigir anillo) y los lazos de V; dos candidatos
     a ≤ 2 px son el mismo (se queda el de más puntuación).
  2. En cada candidato se miden las TRES evidencias, en su propio píxel, aunque ese detector no lo hubiera propuesto:
       p = la mejor correlación de plantilla ahí (0–1)                       ← ¿se parece a un arco con hueco?
       r = la fracción de 36 rayos que chocan antes de 8 px (0–1)            ← ¿está encerrado?
       v = los votos de curvatura suavizados ahí, relativos al máximo de la imagen (0–1)  ← ¿los trazos de alrededor GIRAN hacia aquí?
  3. PUNTUACIÓN = r (el encierro), y ACUERDO = cuántas de las tres pasan su umbral (p ≥ 0,8 · r ≥ 0,5 · v ≥ 0,25).
     ⚠ Primero fue p + r + v y el top-1 acertaba el 40 %; probadas nueve combinaciones sobre los 47 doses etiquetados
     (2026-10-11), el encierro SOLO da el 70 % (top-2 81 %), y sumarle p o v lo empeora: p vale casi lo mismo en el
     rizo (0,90) que en el resto del contorno (0,86), y v es ~0 en los dos. O sea: P y R sirven para PROPONER (entre
     sus candidatos está el rizo el 100 % de las veces) y el encierro para ELEGIR; V no aporta. Elegido mirando las
     etiquetas, así que es una regla ajustada a 47 doses, no una verdad medida aparte.
  4. El rizo compuesto es el candidato de MÁS puntuación de la mitad superior (y < 16); también se reportan todos los de
     acuerdo ≥ 2. Radio, cierre y abertura: los de la plantilla que mejor casa ahí.

    python nn/compuesto.py   → resultados/9-compuesto.png · 9-compuesto.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import detector as DT                                               # noqa: E402
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
import rayos as RY                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

S_P, F_R, V_MIN_REL, SIGMA_V = 0.8, 0.5, 0.25, 1.0
UMBRAL = {"p": S_P, "r": F_R, "v": V_MIN_REL}


def evidencias(x: np.ndarray) -> dict:
    """Los tres mapas 32×32 (p, r, v) y las listas de candidatos de cada detector."""
    x = np.asarray(x) > 0
    best, idx = PL.mapa(x)
    ys, xs = np.nonzero(~x); ch, _, _ = RY._rayos(x, ys, xs, 8.0)
    r = np.zeros(x.shape); r[ys, xs] = ch.mean(1)
    acc = gaussian_filter(DT.votos(x.astype(np.uint8)), SIGMA_V); v = acc / acc.max() if acc.max() > 0 else acc
    return {"p": best, "idx": idx, "r": r, "v": v,
            "cand_P": PL.rizos(x, S_P), "cand_R": RY.rizos(x, F_R, 8.0, 99.0), "cand_V": DT.lazos(x.astype(np.uint8), 1.0, 3.0)}


def compuesto(x: np.ndarray) -> list[dict]:
    e = evidencias(x); x = np.asarray(x) > 0
    cands = []
    for origen in ("P", "R", "V"):
        for c in e["cand_" + origen]:
            cx, cy = c["centro"]; ix, iy = min(31, max(0, int(cx))), min(31, max(0, int(cy)))
            if x[iy, ix]:
                continue
            p, r, v = float(e["p"][iy, ix]), float(e["r"][iy, ix]), float(e["v"][iy, ix])
            m = PL.META[e["idx"][iy, ix]]
            cands.append({"centro": (ix + 0.5, iy + 0.5), "p": p, "r": r, "v": v, "puntuacion": r,
                          "acuerdo": int(p >= S_P) + int(r >= F_R) + int(v >= V_MIN_REL), "origen": origen,
                          "radio": float(m["radio"]), "cierre": float(m["cierre"]),
                          "orientacion": None if m["abertura"] is None else float(m["abertura"])})
    out = []
    for c in sorted(cands, key=lambda z: -z["puntuacion"]):
        if any(EV._dist(c["centro"], o["centro"]) <= 2.0 for o in out):
            continue
        out.append(c)
    return out


def main() -> int:
    t0 = time.time(); d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8)
    E = EV.etiquetas(); E_ok = [e for e in E if not e["dudoso"]]
    R = {e["i"]: compuesto(IM[e["i"]]) for e in E}
    arriba = {i: [c for c in r if c["centro"][1] < 16] for i, r in R.items()}
    res = {"umbrales": UMBRAL, "doses": {}}
    acierta = lambda c, e: EV._dist(c["centro"], e["centro"]) <= EV.TOL                      # noqa: E731
    for nombre, filtro in (("todos", lambda cs: cs), ("acuerdo>=2", lambda cs: [c for c in cs if c["acuerdo"] >= 2]),
                           ("acuerdo=3", lambda cs: [c for c in cs if c["acuerdo"] == 3]),
                           ("top-1", lambda cs: cs[:1]), ("top-2", lambda cs: cs[:2])):
        enc = [any(acierta(c, e) for c in filtro(arriba[e["i"]])) for e in E_ok]
        n = [len(filtro(arriba[e["i"]])) for e in E_ok]
        res["doses"][nombre] = {"encontrados_pc": round(100 * float(np.mean(enc)), 1), "candidatos_arriba_por_dos": round(float(np.mean(n)), 2)}
    # cada detector solo, con el mismo "top-1 de la mitad superior", para comparar en igualdad
    for origen in ("P", "R", "V"):
        enc = []
        for e in E_ok:
            cs = sorted([c for c in R[e["i"]] if c["origen"] == origen and c["centro"][1] < 16],
                        key=lambda c: -{"P": c["p"], "R": c["r"], "V": c["v"]}[origen])
            enc.append(bool(cs) and acierta(cs[0], e))
        res["doses"][f"top-1 solo {origen}"] = {"encontrados_pc": round(100 * float(np.mean(enc)), 1)}
    # por clase: % con algún candidato de acuerdo >= 2 arriba (informativo)
    va = np.flatnonzero(d["particion"] == "val"); Y = d["etiquetas"][va].astype(int)
    sub = va[::3]; Ys = Y[::3]
    tiene = np.array([any(c["acuerdo"] >= 2 and c["centro"][1] < 16 for c in compuesto(IM[i])) for i in sub])
    res["clases_acuerdo2_arriba_pc"] = {str(k): round(100 * float(tiene[Ys == k].mean()), 1) for k in range(10)}
    res["ms_por_digito"] = round(1000 * (time.time() - t0) / (len(E) + len(sub)), 1)
    (AQUI / "resultados" / "9-compuesto.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))

    por_fila = 8; filas = int(np.ceil(len(E) / por_fila))
    fig, axs = plt.subplots(filas, por_fila, figsize=(1.6 * por_fila, 1.75 * filas + 0.9))
    for ax, e in zip(axs.flat, E):
        x = IM[e["i"]]; cs = arriba[e["i"]]; top = cs[:1]
        ok = bool(top) and acierta(top[0], e)
        EV.dibujar(ax, x, top, e, f"{e['i']} · {len([c for c in cs if c['acuerdo'] >= 2])} con acuerdo≥2" + (" ✓" if ok else " ✗"), EV.VERDE if ok else EV.NAR)
        for c in cs[1:]:
            if c["acuerdo"] >= 2:
                ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color="#e0a080", ms=6, mew=1)
    for ax in axs.flat[len(E):]:
        ax.axis("off")
    r1 = res["doses"]["top-1"]["encontrados_pc"]
    fig.suptitle(f"rizo-2 · 9 · el COMPUESTO (P + R + V): el candidato de más puntuación de la mitad superior (cruz azul y arco), "
                 f"verde = etiqueta, cruces claras = otros con acuerdo ≥ 2 · top-1 acierta {r1} %", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(AQUI / "resultados" / "9-compuesto.png", dpi=90)
    print(f"→ resultados/9-compuesto.png · 9-compuesto.json · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
