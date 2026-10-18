#!/usr/bin/env python3
"""delg-cnn — pasa TODOS los dígitos del dataset por la red de sólo kernels y guarda el mapa de línea media M (float16,
5620×32×32, mismo orden que datos.npz) en resultados/delgados-<pesos>.npz. Es la ENTRADA que otros experimentos pueden
leer como dato (nunca importando código de aquí) para medir sus detectores sobre dígitos adelgazados.

    python nn/exportar_delgados.py [--pesos k3x3]   → resultados/delgados-k3x3.npz

SUAVIZADO previo (el dueño, 2026-10-17: «tenemos un preproceso que adelgaza; vamos a crear otro que los suavice… lo ruidoso viene
de que los dígitos originales están binarizados: su suavidad original se perdió»). Antes de la red, la tinta binaria pasa por una
gaussiana de `--suavizar σ` px y se vuelve a binarizar a 0,5: el contorno escalonado se redondea y la red sigue viendo lo que vio al
entrenar (tinta BINARIA de curvas suaves, nn/sinteticos.py). Con `--gris` la gaussiana entra tal cual, sin rebinarizar
(⚠ FUERA de lo que vio la red al entrenar).

    python nn/exportar_delgados.py --suavizar 1.0 [--gris]   → resultados/delgados-k3x3-s1.0[g].npz

OTRO DATASET (2026-10-18): `--dataset <nombre publicado>`. Si sus imágenes son GRIS (float, p. ej. el de mismo trazo de rizo-2
§ 118) entran tal cual, sin binarizar; si son 0/1, como siempre.

    python nn/exportar_delgados.py --pesos k3x3-g0.7 --dataset uci-optdigits-orig-32px-trazo4-r20261018
        → resultados/delgados-k3x3-g0.7-uci-optdigits-orig-32px-trazo4-r20261018.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parents[0])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from expcnn import exigir_dataset  # noqa: E402
from entrenar_local import SoloKernels  # noqa: E402
from aplicar_digitos import DATASET  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--pesos", default="k3x3")
    ap.add_argument("--suavizar", type=float, default=0.0, help="σ (px) de la gaussiana previa; 0 = sin suavizar (lo de siempre)")
    ap.add_argument("--gris", action="store_true", help="con --suavizar: NO rebinarizar (fuera de lo que vio la red)")
    ap.add_argument("--dataset", default=DATASET, help="dataset publicado; si es gris (float) entra tal cual"); a = ap.parse_args()
    d = np.load(exigir_dataset(a.dataset) / "datos.npz")
    gris = np.issubdtype(d["imagenes"].dtype, np.floating)
    T = d["imagenes"].astype(np.float32) if gris else (d["imagenes"] > 0).astype(np.float32)
    if a.suavizar > 0:
        T = np.stack([ndimage.gaussian_filter(t, a.suavizar, mode="constant") for t in T])
        if not a.gris:
            T = (T > 0.5).astype(np.float32)
    X = torch.from_numpy(T)[:, None]
    pesos = AQUI / "nn" / "pesos" / a.pesos; cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    with torch.no_grad():
        M = torch.cat([red(X[i:i + 256])[:, 0] for i in range(0, len(X), 256)]).numpy().astype(np.float16)
    suf = (f"-s{a.suavizar:g}{'g' if a.gris else ''}" if a.suavizar > 0 else "") + ("" if a.dataset == DATASET else f"-{a.dataset}")
    out = AQUI / "resultados" / f"delgados-{a.pesos}{suf}.npz"
    np.savez_compressed(out, M=M, etiquetas=d["etiquetas"], particion=d["particion"], dataset=a.dataset, pesos=a.pesos,
                        suavizar=a.suavizar, gris=a.gris, tinta=T.astype(np.float16))
    print(f"→ {out.relative_to(AQUI)}  {M.shape}  {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
