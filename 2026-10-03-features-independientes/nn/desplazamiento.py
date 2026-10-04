#!/usr/bin/env python3
"""Corrida 13: compositores tolerantes a ±1 celda (A posicional · B máx 3×3 · C máx 2×2→4×4 · D aumento con
desplazamientos), evaluados con las 5 transformaciones de la corrida 12. Criterio en 02-criterio.md § «Corrida 13».

    python nn/desplazamiento.py      → resultados/desplazamiento.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import curva                                    # noqa: E402
import datos                                    # noqa: E402
import generalizacion as G                      # noqa: E402

RES = AQUI.parent / "resultados"
VARIANTES = ("A_posicional", "B_max3", "C_max2_4x4", "D_aumento")
CASOS = {"fino+grueso": ["fino", "grueso"], "cae3": ["cae3"], "todos": ["fino", "grueso", "dig", "cae5", "cae3"],
         "píxeles crudos": None}
DIRS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def desplazar(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(x)
    src = x[..., max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
    out[..., max(0, dy):max(0, dy) + src.shape[-2], max(0, dx):max(0, dx) + src.shape[-1]] = src
    return out


def preparar(s: np.ndarray, variante: str) -> np.ndarray:
    """s (N, J, 8, 8) → entradas del lineal según la variante."""
    t = torch.from_numpy(s)
    if variante == "B_max3":
        t = Fn.max_pool2d(t, 3, 1, 1)
    elif variante == "C_max2_4x4":
        t = Fn.max_pool2d(t, 2, 2)
    return t.flatten(1).numpy()


def ajustar(x, y, sem):
    return G.acc(x, y, np.ones(len(y), bool), None, sem)


def main() -> int:
    dig = datos.digitos(); x0, y, tr = dig["x"], dig["y"], dig["train"]
    _, test = curva.reparto(y, tr)
    yt, xtr, ytr = y[test], x0[tr], y[tr]
    xt = {"limpio": x0[test]} | {t: G.transformar(x0[test], t) for t in G.TRANSFORMACIONES}
    xaug = np.concatenate([xtr] + [desplazar(xtr, dy, dx) for dy, dx in DIRS]); yaug = np.tile(ytr, 9)
    t0 = time.time()
    grupos = ["fino", "grueso", "dig", "cae5", "cae3"]
    m_tr = {g: G.mapas_grupo(g, xtr).astype(np.float32) for g in grupos}
    m_aug = {g: G.mapas_grupo(g, xaug).astype(np.float32) for g in grupos}
    m_te = {t: {g: G.mapas_grupo(g, xt[t]).astype(np.float32) for g in grupos} for t in xt}
    print(f"mapas listos ({time.time() - t0:.0f} s)", flush=True)
    salida = {"test": int(test.sum()), "variantes": VARIANTES, "transformaciones": ["limpio", *G.TRANSFORMACIONES], "casos": {}}
    for caso, gs in CASOS.items():
        junta = (lambda d: d) if gs is None else (lambda d: np.concatenate([d[g] for g in gs], 1))
        s_tr = xtr if gs is None else junta(m_tr); s_aug = xaug if gs is None else junta(m_aug)
        s_te = {t: (xt[t] if gs is None else junta(m_te[t])) for t in xt}
        for v in VARIANTES:
            base = "A_posicional" if v == "D_aumento" else v
            Xtr, Ytr = (preparar(s_aug, base), yaug) if v == "D_aumento" else (preparar(s_tr, base), ytr)
            Xte = {t: preparar(s_te[t], base) for t in xt}
            accs = {t: [] for t in xt}
            for sem in (1, 2, 3):
                W = ajustar(Xtr, Ytr, sem)
                for t in xt:
                    accs[t].append(G.evaluar(W, Xte[t], yt))
            fila = {t: round(float(np.mean(a)), 4) for t, a in accs.items()}
            fila["G3_medio"] = round(float(np.mean([fila[t] / fila["limpio"] for t in G.TRANSFORMACIONES])), 4)
            salida["casos"].setdefault(caso, {})[v] = fila
            print(f"{caso:<15} {v:<12} limpio {fila['limpio']:.3f} · desp {fila['desplazar']:.3f} · ruido {fila['ruido']:.3f} · "
                  f"engr {fila['engrosar']:.3f} · adel {fila['adelgazar']:.3f} · ocl {fila['ocluir']:.3f} · G3 medio {fila['G3_medio']:.3f}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
    (RES / "desplazamiento.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
