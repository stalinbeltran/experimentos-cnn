#!/usr/bin/env python3
"""Pasa los 13 detectores (best.pt) sobre los 1797 dígitos de `uci-optdigits-8px-r20261002` y deja los
mapas en resultados/mapas-digitos.npz (lo que leen los compositores) y una rejilla PNG para MIRAR.

    python nn/aplicar.py                 mapas + resultados/mapas-digitos.png (2 dígitos por clase, val)
    python nn/aplicar.py --n 4           más dígitos por clase en la rejilla

Rejilla: por fila un dígito; columnas = el dígito y los 13 mapas σ(logits) (negro = 1). Una celda
con marco = el máximo del mapa cuando supera el umbral (hallada).
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


@torch.no_grad()
def mapas(dig: dict, pesos: Path = PESOS) -> dict:
    x = torch.from_numpy(dig["x"])
    out, huellas, umbrales = np.zeros((len(x), len(F.CON_TRAZO), modelo.LADO, modelo.LADO), np.float32), {}, np.zeros(len(F.CON_TRAZO), np.float32)
    for j, f in enumerate(F.CON_TRAZO):
        red, est = modelo.cargar(pesos / f / "best.pt")
        out[:, j] = torch.sigmoid(red(x))[:, 0].numpy()
        huellas[f] = modelo.huella_pesos(red); umbrales[j] = est.get("umbral", modelo.UMBRAL)
    return {"sigma": out, "y": dig["y"], "train": dig["train"], "huellas": huellas, "umbrales": umbrales}


def resumen(m: dict) -> dict:
    """Por clase, la fracción de dígitos en que cada detector dispara (max σ ≥ umbral): la «firma» de la clase."""
    pres = m["sigma"].reshape(len(m["y"]), len(F.CON_TRAZO), -1).max(2) >= m["umbrales"][None]
    return {str(c): {f: round(float(pres[m["y"] == c, j].mean()), 3) for j, f in enumerate(F.CON_TRAZO)} for c in range(10)}


def rejilla(dig: dict, m: dict, destino: Path, n: int = 2, semilla: int = 3) -> Path:
    from PIL import Image, ImageDraw                     # noqa: PLC0415
    rng = np.random.default_rng(semilla)
    idx = np.concatenate([rng.choice(np.flatnonzero((m["y"] == c) & ~m["train"]), n, replace=False) for c in range(10)])
    esc, sep, cols = 5, 2, 1 + len(F.CON_TRAZO)
    W, H = cols * (8 * esc + sep) + sep, len(idx) * (8 * esc + sep) + sep
    im = Image.new("L", (W, H), 128)
    for r, i in enumerate(idx):
        tiles = [1 - dig["x"][i, 0]] + [1 - m["sigma"][i, j] for j in range(len(F.CON_TRAZO))]
        for k, t in enumerate(tiles):
            tile = Image.fromarray((t * 255).astype(np.uint8)).resize((8 * esc, 8 * esc), Image.NEAREST)
            if k > 0:
                s = m["sigma"][i, k - 1]
                if s.max() >= m["umbrales"][k - 1]:
                    f, c = np.unravel_index(s.argmax(), s.shape)
                    ImageDraw.Draw(tile).rectangle([c * esc, f * esc, c * esc + esc - 1, f * esc + esc - 1], outline=60)
            im.paste(tile, (sep + k * (8 * esc + sep), sep + r * (8 * esc + sep)))
    destino.parent.mkdir(parents=True, exist_ok=True); im.save(destino)
    return destino


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2)
    a = p.parse_args()
    faltan = [f for f in F.CON_TRAZO if not (PESOS / f / "best.pt").is_file()]
    if faltan:
        raise SystemExit(f"✗ faltan detectores entrenados: {faltan}. Primero nn/lanzar.sh todas.")
    dig = datos.digitos(); t0 = time.time()
    m = mapas(dig)
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAPAS, sigma=m["sigma"], y=m["y"], train=m["train"], umbrales=m["umbrales"])
    firma = resumen(m)
    (RES / "firma-por-clase.json").write_text(json.dumps({"umbrales": {f: float(u) for f, u in zip(F.CON_TRAZO, m["umbrales"])}, "huellas_best": m["huellas"],
                                                          "columnas": list(F.CON_TRAZO), "firma": firma}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    png = rejilla(dig, m, RES / "mapas-digitos.png", a.n)
    print(f"mapas de {len(m['y'])} dígitos × {len(F.CON_TRAZO)} detectores en {time.time() - t0:.1f} s → {MAPAS.name}, {png.name}, firma-por-clase.json")
    print(f"{'clase':<6}" + "".join(f"{f[:6]:>7}" for f in F.CON_TRAZO))
    for c, fila in firma.items():
        print(f"{c:<6}" + "".join(f"{fila[f]:>7.2f}" for f in F.CON_TRAZO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
