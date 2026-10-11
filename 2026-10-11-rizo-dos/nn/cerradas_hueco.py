#!/usr/bin/env python3
"""rizo-2 — el detector de curvas cerradas pidiendo MÁS FONDO EN EL CENTRO (pedido del dueño, 2026-10-11).

Sobre los candidatos del compuesto (plantillas rodeadas ∪ rayos) se añade la distancia del candidato a la tinta (h,
transformada de distancia): se exige h ≥ H_MIN y se elige por encierro r más α·h (h recortado a 4 px). Varias
combinaciones, medidas en (a) los 47 doses etiquetados (top-1 ≤ 2,5 px) y (b) dibujadas sobre los 119 dígitos de val
que falla el compositor (figura 13), para que el dueño vea dónde caen.

    python nn/cerradas_hueco.py   → resultados/14-hueco.json · 14-hueco-<combinación>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt as edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import compuesto2 as C2                                             # noqa: E402
import rayos as RY                                                  # noqa: E402
import evaluar as EV                                                # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

COMBOS = {"h0-a0 (actual)": (0.0, 0.0, False), "h1.5-a0": (1.5, 0.0, False), "h2.5-a0": (2.5, 0.0, False), "h0-a1": (0.0, 1.0, False),
          "h0-a0-r14": (0.0, 0.0, True), "h1.5-a0-r14": (1.5, 0.0, True), "h2.5-a0-r14": (2.5, 0.0, True), "h1.5-a1-r14": (1.5, 1.0, True)}
R_MIN = 0.85


_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def candidatos(x):
    """h = el radio inscrito del hueco: la mayor distancia a la tinta a ≤ 2 px del candidato, y el candidato se MUEVE a ese
    píxel (el centro del hueco). ⚠ Primera versión: h en el píxel propuesto, que casi siempre está pegado a la tinta
    (h = 1 en un 2, 0,85–1 en todos): exigir h ≥ 1,5 tiraba el 75 % de los rizos. El hueco hay que medirlo en su centro."""
    dt = edt(np.asarray(x) == 0); cs = C2.compuesto2(x)
    for c in cs:
        cerca = np.hypot(_XX - c["centro"][0], _YY - c["centro"][1]) <= 2.0
        k = int(np.argmax(np.where(cerca, dt, -1))); c["h"] = float(dt.flat[k]); c["centro_hueco"] = (k % 32 + 0.5, k // 32 + 0.5)
    # el encierro medido EN EL CENTRO DEL HUECO y con rayos más largos (14 px): los lazos grandes del 0, 6 y 9 tienen el
    # centro a más de 8 px de la tinta, y con rayos de 8 su encierro sale bajo y se pierden (visto en la figura 14 con h ≥ 2,5)
    if cs:
        ys = np.array([int(c["centro_hueco"][1]) for c in cs]); xs = np.array([int(c["centro_hueco"][0]) for c in cs])
        ch, _, _ = RY._rayos(np.asarray(x) > 0, ys, xs, 14.0)
        for c, f in zip(cs, ch.mean(1)):
            c["r14"] = float(f)
    return cs


def elegir(cs, hmin, alfa, largo=False):
    r = (lambda c: c["r14"]) if largo else (lambda c: c["r"])
    ok = [dict(c, centro=c["centro_hueco"]) for c in cs if c["h"] >= hmin and r(c) >= R_MIN]
    out = []
    for c in sorted(ok, key=lambda c: -(r(c) + alfa * min(c["h"], 4) / 4)):
        if not any(EV._dist(c["centro"], o["centro"]) <= 2.0 for o in out):        # dos propuestas con el mismo centro de hueco son una
            out.append(c)
    return out


def main() -> int:
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8); Y = d["etiquetas"].astype(int)
    E = [e for e in EV.etiquetas() if not e["dudoso"]]
    fallos = json.loads((AQUI / "resultados" / "13-digitos.json").read_text())["indices_fallos_val_s0"]
    C_dos = {e["i"]: candidatos(IM[e["i"]]) for e in E}; C_fal = {i: candidatos(IM[i]) for i in fallos}
    res = {}
    for nombre, (hmin, alfa, largo) in COMBOS.items():
        arriba = {e["i"]: [c for c in elegir(C_dos[e["i"]], hmin, alfa, largo) if c["centro"][1] < 16] for e in E}
        t1 = np.mean([bool(arriba[e["i"]]) and EV._dist(arriba[e["i"]][0]["centro"], e["centro"]) <= EV.TOL for e in E])
        enc = np.mean([any(EV._dist(c["centro"], e["centro"]) <= EV.TOL for c in arriba[e["i"]]) for e in E])
        nf = np.mean([len(elegir(C_fal[i], hmin, alfa, largo)) for i in fallos])
        res[nombre] = {"h_min": hmin, "alfa": alfa, "rayos_px": 14 if largo else 8, "doses_top1_pc": round(100 * float(t1), 1), "doses_encontrados_pc": round(100 * float(enc), 1),
                       "candidatos_por_digito_en_los_119": round(float(nf), 2)}
        print(nombre, res[nombre], flush=True)
        # figura sobre los 119 fallos
        cols = 12; filas = int(np.ceil(len(fallos) / cols))
        fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.3, filas * 1.5 + 0.8))
        for ax in axs.flat:
            ax.axis("off")
        for ax, i in zip(axs.flat, fallos):
            cs = elegir(C_fal[i], hmin, alfa, largo)
            ax.imshow(np.where(IM[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            for q, c in enumerate(cs):
                ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=9 if q == 0 else 6, mew=1.8 if q == 0 else 1)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_title(f"{Y[i]} · idx {i} · {len(cs)}", fontsize=6.5, pad=2)
        fig.suptitle(f"rizo-2 · 14 · curvas cerradas con fondo en el centro: h ≥ {hmin:g} px, encierro con rayos de {14 if largo else 8} px, elige por encierro + {alfa:g}·h — sobre los 119 dígitos "
                     f"que falla el compositor · cruz azul = la mejor, claras = las demás (encierro ≥ {R_MIN}) · {nf:.1f} por dígito", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(AQUI / "resultados" / f"14-hueco-{nombre.split(' ')[0]}.png", dpi=90); plt.close(fig)
    (AQUI / "resultados" / "14-hueco.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("→ resultados/14-hueco.json · 14-hueco-*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
