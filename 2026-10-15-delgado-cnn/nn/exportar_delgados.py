#!/usr/bin/env python3
"""delg-cnn — pasa TODOS los dígitos del dataset por la red de sólo kernels y guarda el mapa de línea media M (float16,
5620×32×32, mismo orden que datos.npz) en resultados/delgados-<pesos>.npz. Es la ENTRADA que otros experimentos pueden
leer como dato (nunca importando código de aquí) para medir sus detectores sobre dígitos adelgazados.

    python nn/exportar_delgados.py [--pesos k3x3]   → resultados/delgados-k3x3.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parents[0])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from expcnn import exigir_dataset  # noqa: E402
from entrenar_local import SoloKernels  # noqa: E402
from aplicar_digitos import DATASET  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--pesos", default="k3x3"); a = ap.parse_args()
    d = np.load(exigir_dataset(DATASET) / "datos.npz")
    X = torch.from_numpy((d["imagenes"] > 0).astype(np.float32))[:, None]
    pesos = AQUI / "nn" / "pesos" / a.pesos; cfg = json.loads((pesos / "config.json").read_text())
    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(pesos / "best.pt")); red.eval()
    with torch.no_grad():
        M = torch.cat([red(X[i:i + 256])[:, 0] for i in range(0, len(X), 256)]).numpy().astype(np.float16)
    out = AQUI / "resultados" / f"delgados-{a.pesos}.npz"
    np.savez_compressed(out, M=M, etiquetas=d["etiquetas"], particion=d["particion"], dataset=DATASET, pesos=a.pesos)
    print(f"→ {out.relative_to(AQUI)}  {M.shape}  {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
