#!/usr/bin/env python3
"""rizo-2 — VISTA PREVIA del vértice en 4 direcciones N, S, E, O (el dueño, 2026-10-17: «necesito detectar este feature en las
4 direcciones… antes de proceder muéstrame una imagen de cómo quedarían en la matriz 3×3»). Nada se entrena ni se cambia:
`cresta_curvas.vertices` sigue buscando sólo la punta de ARRIBA (N). Aquí las otras tres salen GIRANDO la tinta 90/180/270°,
pasando el MISMO detector y devolviendo la posición al dígito original; la rejilla es la del bloque `vertice` (mancha
gaussiana σ 2 px, máximo por celda de `adaptive_max_pool2d`, § 95).

Convención de nombres que propone la figura: el vértice se nombra por hacia dónde APUNTA su punta (N = la cima del 4l); un
entrante por hacia dónde ABRE (los conos de ±60° del § 62–66: der → E, izq → O, abajo → S, arriba → N); y las celdas de la
rejilla 3×3 por su posición (NO N NE / O C E / SO S SE).

    ../.venv/bin/python nn/vertice_4dir_vista.py   → resultados/101-vertice-4-direcciones.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DATASET = "uci-optdigits-orig-32px-r20261005"                            # el de rizo-2 (web/servidor.py)
N = 32
# dirección → (giros CCW de np.rot90 que llevan esa punta ARRIBA, cómo vuelve (cx, cy) del giro al dígito original)
DIRS = {
    "N": (0, lambda x, y: (x, y)),
    "S": (2, lambda x, y: (N - x, N - y)),
    "E": (1, lambda x, y: (N - y, x)),        # rot90 k=1: la columna derecha pasa a ser la fila de arriba
    "O": (3, lambda x, y: (y, N - x)),
}
CELDAS = ["NO", "N", "NE", "O", "C", "E", "SO", "S", "SE"]


def vertices_dir(tinta, d: str) -> list[dict]:
    k, vuelta = DIRS[d]
    out = []
    for v in CC.vertices(np.rot90(tinta, k)):
        cx, cy = vuelta(v["cx"], v["cy"]); out.append(dict(v, cx=cx, cy=cy, dir=d))
    return out


def rejilla(vs) -> np.ndarray:
    m = CC._mapa(vs, lambda q: 1.0, 2.0)
    return F.adaptive_max_pool2d(torch.from_numpy(m)[None], 3)[0].numpy()


def main() -> int:
    d = dict(np.load(exigir_dataset(DATASET) / "datos.npz"))
    tinta = (d["imagenes"] > 0).astype(np.uint8); y = d["etiquetas"].astype(int); va = np.flatnonzero(d["particion"] == "val")
    # dónde dispara cada dirección en val, por clase, y en qué celda cae
    cuenta = {k: [0] * 10 for k in DIRS}; celda = {k: np.zeros(9, int) for k in DIRS}; ejemplos = {k: [] for k in DIRS}
    for i in va:
        for k in DIRS:
            vs = vertices_dir(tinta[i], k)
            if vs:
                cuenta[k][y[i]] += 1
                c = min(int(vs[0]["cy"] * 3 / N), 2) * 3 + min(int(vs[0]["cx"] * 3 / N), 2); celda[k][c] += 1
                ejemplos[k].append(int(i))
    # un ejemplo por dirección, de la clase donde más dispara SIN contar el 1: en el 1 grueso dispara casi siempre (es un
    # vértice de verdad, § 95) y taparía el caso que interesa, que es el resto de los dígitos
    elegido = {}
    for k in DIRS:
        sin1 = [c if j != 1 else -1 for j, c in enumerate(cuenta[k])]
        top = int(np.argmax(sin1)) if max(sin1) > 0 else 1; cand = [i for i in ejemplos[k] if y[i] == top]
        elegido[k] = cand[len(cand) // 2] if cand else None

    fig, ax = plt.subplots(4, 3, figsize=(11, 15.5), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    nombre = {"N": "N — punta ARRIBA (la de hoy)", "S": "S — punta ABAJO", "E": "E — punta a la DERECHA", "O": "O — punta a la IZQUIERDA"}
    for r, k in enumerate(DIRS):
        i = elegido[k]
        a = ax[r, 0]
        if i is None:
            a.text(0.5, 0.5, "no dispara\nen val", ha="center", va="center"); a.axis("off")
        else:
            vs = vertices_dir(tinta[i], k)
            a.imshow(tinta[i], cmap="Greys", extent=(0, N, N, 0))
            for t in (32 / 3, 64 / 3):
                a.axhline(t, color="tab:blue", lw=0.8, ls="--"); a.axvline(t, color="tab:blue", lw=0.8, ls="--")
            for v in vs:
                a.plot(v["cx"], v["cy"], "o", ms=11, mfc="none", mec="red", mew=2.5)
            a.set_title(f"{nombre[k]}\nval #{i}, un {y[i]}", fontsize=10); a.set_xticks([]); a.set_yticks([])
        a = ax[r, 1]
        g = rejilla(vertices_dir(tinta[i], k)) if i is not None else np.zeros((3, 3))
        a.imshow(g, cmap="Reds", vmin=0, vmax=1)
        for rr in range(3):
            for cc in range(3):
                a.text(cc, rr, f"{CELDAS[rr * 3 + cc]}\n{g[rr, cc]:.2f}", ha="center", va="center", fontsize=10,
                       color="white" if g[rr, cc] > 0.5 else "black")
        a.set_title(f"vértice {k}: su matriz 3×3\n(+ cuenta = {len(vertices_dir(tinta[i], k)) if i is not None else 0})", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        a = ax[r, 2]; a.axis("off")
        tot = sum(cuenta[k]); filas = [f"en val ({len(va)} dígitos) dispara en {tot}:"]
        filas += ["  " + " · ".join(f"{c}:{cuenta[k][c]}" for c in range(5)), "  " + " · ".join(f"{c}:{cuenta[k][c]}" for c in range(5, 10)), "",
                  "celda donde cae (de esos):"]
        for rr in range(3):
            filas.append("  " + "  ".join(f"{CELDAS[rr * 3 + cc]:>2} {celda[k][rr * 3 + cc]:>4}" for cc in range(3)))
        a.text(0, 0.95, "\n".join(filas), va="top", family="monospace", fontsize=10)
    fig.suptitle("rizo-2 · 101 · VISTA PREVIA: el vértice en 4 direcciones (detector de hoy, GIRADO)\n"
                 "nombre = hacia dónde APUNTA la punta · celdas de la rejilla: NO N NE / O C E / SO S SE · "
                 "entrantes (conos 60°): der→E, izq→O, abajo→S, arriba→N", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0)
    out = AQUI / "resultados" / "101-vertice-4-direcciones"
    fig.savefig(out.with_suffix(".png"), dpi=110)
    json.dump({"val": int(len(va)), "por_clase": cuenta, "por_celda": {k: dict(zip(CELDAS, map(int, v))) for k, v in celda.items()},
               "ejemplos": elegido}, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps({k: sum(v) for k, v in cuenta.items()}), {k: dict(zip(CELDAS, map(int, v))) for k, v in celda.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
