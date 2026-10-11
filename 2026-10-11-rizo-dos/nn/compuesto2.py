#!/usr/bin/env python3
"""rizo-2 — el compuesto con las plantillas RODEADAS (cob ≥ 0,85) como proponente (pedido por el dueño, 2026-10-11).

Igual que `compuesto.py` pero con el banco rodeado (`plantillas_rodeadas.banco((0.85, 1.0), 0.5)`) en vez del actual,
y se comparan tres formas de elegir entre los candidatos: por encierro (r, como en compuesto.py), por la puntuación de
la plantilla rodeada (p'), y por la suma. También se dibujan los FALLOS del mejor.

    python nn/compuesto2.py   → resultados/12-compuesto2.json · 12-compuesto2-fallos.png · 12-compuesto2.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import compuesto as CP                                              # noqa: E402
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
import plantillas_rodeadas as PR                                    # noqa: E402
import rayos as RY                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

B_ROD, META_ROD = PR.banco((0.85, 1.0), 0.5)
RMAX = 8.0      # alcance de los rayos del encierro; `digitos_cerradas.py --rmax` lo sube para los lazos grandes (0, 6, 9)
S_ROD = 0.6     # el umbral con el que el banco rodeado encuentra el 87 % (figura 11); elegir lo hace el compuesto


@torch.no_grad()
def mapa_rodeado(x):
    r = F.conv2d(torch.from_numpy((np.asarray(x) > 0).astype(np.float32))[None, None], B_ROD, padding=PL.K // 2)[0]
    best, idx = r.max(0); return best.numpy(), idx.numpy()


def compuesto2(x: np.ndarray) -> list[dict]:
    x = np.asarray(x) > 0
    best, idx = mapa_rodeado(x)
    ys, xs = np.nonzero(~x); ch, _, _ = RY._rayos(x, ys, xs, RMAX); r = np.zeros(x.shape); r[ys, xs] = ch.mean(1)
    cands = []
    for origen, lista in (("P'", PR.rizos(x, B_ROD, META_ROD, S_ROD)), ("R", RY.rizos(x, CP.F_R, 8.0, 99.0))):
        for c in lista:
            ix, iy = min(31, max(0, int(c["centro"][0]))), min(31, max(0, int(c["centro"][1])))
            if x[iy, ix]:
                continue
            m = META_ROD[idx[iy, ix]]
            cands.append({"centro": (ix + 0.5, iy + 0.5), "p": float(best[iy, ix]), "r": float(r[iy, ix]), "origen": origen,
                          "radio": float(m["radio"]), "cierre": float(m["cierre"]),
                          "orientacion": None if m["abertura"] is None else float(m["abertura"])})
    out = []
    for c in sorted(cands, key=lambda z: -z["r"]):
        if any(EV._dist(c["centro"], o["centro"]) <= 2.0 for o in out):
            continue
        out.append(c)
    return out


def main() -> int:
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8)
    E = EV.etiquetas(); E_ok = [e for e in E if not e["dudoso"]]
    R = {e["i"]: [c for c in compuesto2(IM[e["i"]]) if c["centro"][1] < 16] for e in E}
    acierta = lambda c, e: EV._dist(c["centro"], e["centro"]) <= EV.TOL                      # noqa: E731
    res = {"banco": "cob ≥ 0,85, hueco ≥ 0,5", "s_rodeado": S_ROD, "f_rayos": CP.F_R, "doses": {}}
    res["doses"]["todos"] = {"encontrados_pc": round(100 * float(np.mean([any(acierta(c, e) for c in R[e["i"]]) for e in E_ok])), 1),
                             "candidatos_arriba_por_dos": round(float(np.mean([len(R[e["i"]]) for e in E_ok])), 2)}
    claves = {"encierro r": lambda c: c["r"], "plantilla rodeada p'": lambda c: c["p"], "r + p'": lambda c: c["r"] + c["p"],
              "solo candidatos de P', por p'": lambda c: c["p"] if c["origen"] == "P'" else -1}
    for n, k in claves.items():
        t1 = [bool(R[e["i"]]) and acierta(max(R[e["i"]], key=k), e) for e in E_ok]
        t2 = [any(acierta(c, e) for c in sorted(R[e["i"]], key=lambda c: -k(c))[:2]) for e in E_ok]
        res["doses"][f"top-1 por {n}"] = {"top1_pc": round(100 * float(np.mean(t1)), 1), "top2_pc": round(100 * float(np.mean(t2)), 1)}
    mejor = max(claves, key=lambda n: res["doses"][f"top-1 por {n}"]["top1_pc"]); res["mejor_eleccion"] = mejor
    (AQUI / "resultados" / "12-compuesto2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))

    k = claves[mejor]
    # los fallos del mejor, de cerca: el elegido (azul/naranja), la etiqueta (verde), los demás candidatos (cruces claras con su r)
    fallos = [e for e in E_ok if not (R[e["i"]] and acierta(max(R[e["i"]], key=k), e))]
    nc = min(8, max(1, len(fallos))); nf = int(np.ceil(len(fallos) / nc))
    fig, axs = plt.subplots(nf, nc, figsize=(2.1 * nc + 0.5, 2.3 * nf + 0.8), squeeze=False)
    for ax, e in zip(axs.flat, fallos):
        x = IM[e["i"]]; cs = sorted(R[e["i"]], key=lambda c: -k(c)); top = cs[:1]
        EV.dibujar(ax, x, top, e, f"idx {e['i']} · elegido r {top[0]['r']:.2f} p' {top[0]['p']:.2f}" if top else f"idx {e['i']} · sin candidatos", EV.NAR)
        for c in cs[1:]:
            col = EV.VERDE if acierta(c, e) else "#e0a080"
            ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color=col, ms=7, mew=1.3)
            ax.annotate(f"{c['r']:.2f}", (c["centro"][0] - 0.5, c["centro"][1] - 0.5), fontsize=5.5, color=col, xytext=(3, -3), textcoords="offset points")
        ax.set_xlim(2, 26); ax.set_ylim(18, -0.5)
    for ax in axs.flat[len(fallos):]:
        ax.axis("off")
    fig.suptitle(f"rizo-2 · 12 · los {len(fallos)} FALLOS del compuesto con plantillas rodeadas (elige por {mejor}): naranja = el elegido, verde = la etiqueta,\n"
                 "cruz verde = un candidato que sí acertaba (con su encierro r), cruces claras = los demás · recortado a la mitad superior", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(AQUI / "resultados" / "12-compuesto2-fallos.png", dpi=100); plt.close(fig)

    por_fila = 8; filas = int(np.ceil(len(E) / por_fila))
    fig, axs = plt.subplots(filas, por_fila, figsize=(1.6 * por_fila, 1.75 * filas + 0.9))
    for ax, e in zip(axs.flat, E):
        cs = sorted(R[e["i"]], key=lambda c: -k(c)); top = cs[:1]; ok = bool(top) and acierta(top[0], e)
        EV.dibujar(ax, IM[e["i"]], top, e, f"{e['i']} · {len(cs)} cand." + (" ✓" if ok else " ✗"), EV.VERDE if ok else EV.NAR)
    for ax in axs.flat[len(E):]:
        ax.axis("off")
    fig.suptitle(f"rizo-2 · 12 · el compuesto con plantillas RODEADAS (P' cob ≥ 0,85 + R), elige por {mejor}: "
                 f"top-1 acierta {res['doses'][f'top-1 por {mejor}']['top1_pc']} %", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(AQUI / "resultados" / "12-compuesto2.png", dpi=90)
    print("→ resultados/12-compuesto2.json · 12-compuesto2-fallos.png · 12-compuesto2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
