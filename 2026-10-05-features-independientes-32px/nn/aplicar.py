#!/usr/bin/env python3
"""Pasa los 13 detectores (best.pt) sobre los dígitos de 32×32 y deja los mapas 8×8 en resultados/mapas-digitos.npz
(los 5620: windep con su reparto 180/1617 y los 3823 `extra`), la firma por clase y una rejilla para MIRAR.

    python nn/aplicar.py [--n 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import features as F                           # noqa: E402
import modelo                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
RES = AQUI.parent / "resultados"
MAPAS = RES / "mapas-digitos.npz"


def detectores(pesos: Path = PESOS) -> tuple[list, np.ndarray, dict]:
    faltan = [f for f in F.CON_TRAZO if not (pesos / f / "best.pt").is_file()]
    if faltan:
        raise SystemExit(f"✗ faltan detectores entrenados en {pesos}: {faltan}")
    reds, umbrales, huellas = [], np.zeros(len(F.CON_TRAZO), np.float32), {}
    for j, f in enumerate(F.CON_TRAZO):
        red, est = modelo.cargar(pesos / f / "best.pt")
        reds.append(red); umbrales[j] = est["umbral"]; huellas[f] = modelo.huella_pesos(red)
    return reds, umbrales, huellas


@torch.no_grad()
def mapas(reds: list, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    """x (N,1,32,32) -> σ (N, 13, 8, 8)."""
    out = np.zeros((len(x), len(reds), modelo.LADO, modelo.LADO), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(x[i:i + lote])
        for j, red in enumerate(reds):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out


def firma(s: np.ndarray, y: np.ndarray, umbrales: np.ndarray) -> dict:
    pres = s.reshape(len(y), len(F.CON_TRAZO), -1).max(2) >= umbrales[None]
    return {str(c): {f: round(float(pres[y == c, j].mean()), 3) for j, f in enumerate(F.CON_TRAZO)} for c in range(10)}


def rejilla(x, s, y, val, umbrales, destino: Path, n: int = 2, semilla: int = 3) -> Path:
    from PIL import Image, ImageDraw                     # noqa: PLC0415
    rng = np.random.default_rng(semilla)
    idx = np.concatenate([rng.choice(np.flatnonzero((y == c) & val), n, replace=False) for c in range(10)])
    lado, sep = 64, 2
    cols = 1 + len(F.CON_TRAZO)
    im = Image.new("L", (cols * (lado + sep) + sep, len(idx) * (lado + sep) + sep), 128)
    for r, i in enumerate(idx):
        tiles = [1 - x[i, 0]] + [1 - s[i, j] for j in range(len(F.CON_TRAZO))]
        for k, t in enumerate(tiles):
            tile = Image.fromarray((t * 255).astype(np.uint8)).resize((lado, lado), Image.NEAREST)
            if k > 0 and s[i, k - 1].max() >= umbrales[k - 1]:
                f, c = np.unravel_index(s[i, k - 1].argmax(), (8, 8)); e = lado // 8
                ImageDraw.Draw(tile).rectangle([c * e, f * e, c * e + e - 1, f * e + e - 1], outline=60)
            im.paste(tile, (sep + k * (lado + sep), sep + r * (lado + sep)))
    destino.parent.mkdir(parents=True, exist_ok=True); im.save(destino)
    return destino


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2)
    a = p.parse_args()
    reds, umbrales, huellas = detectores()
    dig = datos.digitos(con_extra=True); t0 = time.time()
    s = mapas(reds, dig["x"])
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAPAS, sigma=s, y=dig["y"], train=dig["train"], extra=dig["extra"], umbrales=umbrales)
    w = ~dig["extra"]
    fi = firma(s[w], dig["y"][w], umbrales)
    (RES / "firma-por-clase.json").write_text(json.dumps({"umbrales": {f: float(u) for f, u in zip(F.CON_TRAZO, umbrales)},
                                                          "huellas_best": huellas, "columnas": list(F.CON_TRAZO),
                                                          "firma_windep": fi}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    png = rejilla(dig["x"], s, dig["y"], w & ~dig["train"], umbrales, RES / "mapas-digitos.png", a.n)
    print(f"mapas de {len(s)} dígitos × 13 detectores en {time.time() - t0:.1f} s → {MAPAS.name}, {png.name}, firma-por-clase.json")
    print(f"{'clase':<6}" + "".join(f"{f[:6]:>7}" for f in F.CON_TRAZO))
    for c, fila in fi.items():
        print(f"{c:<6}" + "".join(f"{fila[f]:>7.2f}" for f in F.CON_TRAZO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
