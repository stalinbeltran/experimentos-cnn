#!/usr/bin/env python3
"""rizo-2 — los PEORES casos del detector de huecos suaves (pedido por el dueño el 2026-10-13): todos los objetos
«cerrado» que marca en val, y de ellos los que NO son un hueco. Referencia, sin etiquetas a mano: un hueco es una región
de FONDO cerrada (no toca el borde, área ≥ 3 px) en M > u para algún u ∈ {0,3 · 0,5 · 0,7}; un objeto es falso si su
centro no cae en ninguna. Se ordenan por fuerza, de peor a mejor.

    python nn/huecos_suaves_falsos.py --entrada /ruta/delgados-k3x3.npz [--min 0.3] [--peores 48]
        → resultados/43-huecos-falsos.json · 43-huecos-falsos.png (los peores) · 43-huecos-falsos-hist.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import evaluar as EV                                                # noqa: E402
import huecos_suaves as HS                                          # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

UMBRALES = (0.3, 0.5, 0.7); AREA_MIN = 3


def huecos_reales(m: np.ndarray, umbrales=UMBRALES) -> np.ndarray:
    """(32,32) bool: píxeles que están dentro de un hueco cerrado de M > u para algún u (o de la tinta si se pasa binaria)."""
    dentro = np.zeros((32, 32), bool)
    for u in umbrales:
        lab, n = ndimage.label(m <= u)
        borde = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
        for k in range(1, n + 1):
            if k not in borde and (lab == k).sum() >= AREA_MIN:
                dentro |= lab == k
    return dentro


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--entrada", required=True); ap.add_argument("--clave", default="M")
    ap.add_argument("--min", type=float, default=0.3, help="fuerza mínima para contar un objeto cerrado"); ap.add_argument("--peores", type=int, default=48)
    a = ap.parse_args()
    d = np.load(exigir_dataset(EV.DATASET) / "datos.npz"); y, part = d["etiquetas"].astype(int), d["particion"]
    va = np.flatnonzero(part == "val"); M = np.load(a.entrada)[a.clave].astype(np.float32)[va]; tinta = (d["imagenes"][va] > 0).astype(np.float32)
    cer, _ = HS.mapas(M)
    # dos referencias: la topología del MAPA (M > 0,3/0,5/0,7) y la de la TINTA original (dilatada 1 px, para que un
    # hueco que el adelgazado ensancha siga casando). Un objeto es: «hueco» si cae en alguna; «anillo roto» si cae
    # sólo en la tinta (el mapa abrió un hueco que la tinta tenía cerrado: fallo de la referencia, no del detector);
    # «falso» si no cae en ninguna: ahí NO hay hueco, y ésos son los peores casos.
    objetos, falsos, rotos, por_clase = [], [], [], {c: {"objetos": 0, "huecos": 0, "anillo_roto": 0, "falsos": 0, "digitos": 0, "digitos_con_falso": 0, "duplicados": 0} for c in range(10)}
    for j, i in enumerate(va):
        ref_m = huecos_reales(M[j]); ref_t = ndimage.binary_dilation(huecos_reales(tinta[j], (0.5,)), iterations=1); c = int(y[i]); por_clase[c]["digitos"] += 1; algun = False
        lab_m, _ = ndimage.label(ref_m); vistos = set()
        for o in HS.objetos(cer[j]):
            if o["fuerza"] < a.min:
                continue
            cy, cx = int(o["cy"]), int(o["cx"]); en_m, en_t = bool(ref_m[cy, cx]), bool(ref_t[cy, cx])
            tipo = "hueco" if en_m else ("anillo_roto" if en_t else "falso"); objetos.append((i, o, tipo)); por_clase[c]["objetos"] += 1; por_clase[c][{"hueco": "huecos", "anillo_roto": "anillo_roto", "falso": "falsos"}[tipo]] += 1
            if en_m:
                k = int(lab_m[cy, cx]); por_clase[c]["duplicados"] += k in vistos; vistos.add(k)
            reg = {"idx": int(i), "clase": c, "cx": o["cx"], "cy": o["cy"], "fuerza": round(o["fuerza"], 3)}
            if tipo == "falso":
                falsos.append(reg); algun = True
            elif tipo == "anillo_roto":
                rotos.append(reg)
        por_clase[c]["digitos_con_falso"] += algun
    falsos.sort(key=lambda f: -f["fuerza"]); rotos.sort(key=lambda f: -f["fuerza"])
    fuerzas_ok = np.array([o["fuerza"] for _, o, t in objetos if t == "hueco"]); fuerzas_no = np.array([o["fuerza"] for _, o, t in objetos if t == "falso"]); fuerzas_rot = np.array([o["fuerza"] for _, o, t in objetos if t == "anillo_roto"])
    out = {"entrada": Path(a.entrada).name, "fuerza_min": a.min, "umbrales_referencia": UMBRALES, "objetos": len(objetos), "huecos": int(len(fuerzas_ok)), "anillo_roto": int(len(fuerzas_rot)), "falsos": len(falsos),
           "duplicados_en_el_mismo_hueco": int(sum(v["duplicados"] for v in por_clase.values())),
           "falsos_fuerza_ge_0.5": int((fuerzas_no >= 0.5).sum()), "falsos_fuerza_ge_0.7": int((fuerzas_no >= 0.7).sum()),
           "fuerza_mediana_verdaderos": round(float(np.median(fuerzas_ok)), 3), "fuerza_mediana_falsos": round(float(np.median(fuerzas_no)), 3) if len(fuerzas_no) else None,
           "fuerza_mediana_anillo_roto": round(float(np.median(fuerzas_rot)), 3) if len(fuerzas_rot) else None,
           "por_clase": por_clase, "peores": falsos[:a.peores], "anillos_rotos_peores": rotos[:24], "todos_los_falsos": falsos}
    (AQUI / "resultados" / "43-huecos-falsos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("peores", "todos_los_falsos", "anillos_rotos_peores")}, ensure_ascii=False, indent=1))
    # figura: los peores
    peores = falsos[:a.peores]; cols = 12; filas = int(np.ceil(len(peores) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.55, filas * 1.8 + 0.8)); axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, f in zip(axs.flat, peores):
        j = int(np.flatnonzero(va == f["idx"])[0]); m = M[j]
        ax.imshow(np.where(m > 0.05, m, np.nan), cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        reales = huecos_reales(m); ref_t = ndimage.binary_dilation(huecos_reales(tinta[j], (0.5,)), iterations=1)
        yy, xx = np.nonzero(reales); ax.plot(xx, yy, "s", color="#9fd3ff", ms=2.2, mew=0, alpha=0.8)
        yy, xx = np.nonzero(ref_t & ~reales); ax.plot(xx, yy, "s", color="#ffe08a", ms=2.2, mew=0, alpha=0.8)
        for o in HS.objetos(cer[j]):
            if o["fuerza"] >= a.min:
                cy, cx = int(o["cy"]), int(o["cx"]); col = "#2a78d6" if reales[cy, cx] else "#d19a00" if ref_t[cy, cx] else "#e8432b"
                ax.plot(o["cx"] - 0.5, o["cy"] - 0.5, "+", color=col, ms=5 + 9 * o["fuerza"], mew=2)
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"{f['clase']} · idx {f['idx']} · {f['fuerza']:.2f}", fontsize=7.5, color="#b45309")
    fig.suptitle(f"rizo-2 · 43 · los {len(peores)} PEORES falsos huecos del detector suave en val: {len(falsos)} falsos de verdad, {len(rotos)} de anillo roto y {len(fuerzas_ok)} huecos, entre {len(objetos)} objetos con fuerza ≥ {a.min}\n"
                 "cruz roja = «cerrado» sin hueco ni en el mapa ni en la tinta (tamaño = fuerza) · ámbar = hueco sólo en la tinta (el mapa rompió el anillo) · azul = hueco · celeste/ámbar = las dos referencias", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(AQUI / "resultados" / "43-huecos-falsos.png", dpi=100)
    # histograma de fuerzas
    fig, ax = plt.subplots(figsize=(7, 3.4)); b = np.linspace(a.min, 1, 15)
    ax.hist(fuerzas_ok, b, alpha=0.7, color="#2a78d6", label=f"hueco ({len(fuerzas_ok)})"); ax.hist(fuerzas_rot, b, alpha=0.7, color="#d19a00", label=f"anillo roto ({len(fuerzas_rot)})"); ax.hist(fuerzas_no, b, alpha=0.7, color="#e8432b", label=f"falso ({len(fuerzas_no)})")
    ax.set_xlabel("fuerza del objeto «cerrado»"); ax.set_ylabel("objetos en val"); ax.legend(); ax.set_title("rizo-2 · 43 · fuerza de los objetos cerrados: ¿separa los falsos?", fontsize=9)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "43-huecos-falsos-hist.png", dpi=110)
    print("→ resultados/43-huecos-falsos.json · 43-huecos-falsos.png · 43-huecos-falsos-hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
