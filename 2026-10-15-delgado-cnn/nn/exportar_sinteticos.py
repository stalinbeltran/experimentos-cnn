#!/usr/bin/env python3
"""delg-cnn — exporta trazos SINTÉTICOS de prueba (semilla 202) con su VERDAD (la polilínea de la línea media, sub-píxel)
y el mapa M de la red, como DATO para que otros experimentos midan sus detectores de curvas contra una verdad conocida.

    python nn/exportar_sinteticos.py [--n 600] [--pesos k3x3] [--semilla 202]   → resultados/sinteticos-verdad-<pesos>.npz
Claves: X (n,32,32) tinta · M (n,32,32) mapa · puntos (P,2) todas las polilíneas concatenadas · inicio (n+1,) offsets ·
        grosor (P,) · tipo (n,) recta/curva/lazo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinteticos import curva_natural, engrosar  # noqa: E402
from entrenar_local import SoloKernels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--pesos", default="k3x3"); ap.add_argument("--semilla", type=int, default=202)
    a = ap.parse_args(); rng = np.random.default_rng(a.semilla)
    X, pts, ini, gro, tipo = [], [], [0], [], []
    while len(X) < a.n:
        c = curva_natural(rng)
        if len(c["puntos"]) * 0.25 < 10:
            continue
        X.append(engrosar(c["puntos"], c["grosor"])); pts.append(c["puntos"]); gro.append(c["grosor"]); ini.append(ini[-1] + len(c["puntos"])); tipo.append(c["tipo"])
    X = np.stack(X).astype(np.float32)
    pesos = AQUI / "nn" / "pesos" / a.pesos; cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    with torch.no_grad():
        M = torch.cat([red(torch.from_numpy(X[i:i + 256])[:, None])[:, 0] for i in range(0, len(X), 256)]).numpy().astype(np.float16)
    out = AQUI / "resultados" / f"sinteticos-verdad-{a.pesos}.npz"
    np.savez_compressed(out, X=X.astype(np.uint8), M=M, puntos=np.concatenate(pts).astype(np.float32), inicio=np.array(ini), grosor=np.concatenate(gro).astype(np.float32), tipo=np.array(tipo))
    print(f"→ {out.relative_to(AQUI)}  {len(X)} trazos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
