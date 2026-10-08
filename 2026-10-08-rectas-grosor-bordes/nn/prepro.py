#!/usr/bin/env python3
"""El ÚNICO pre-proceso de `rect-bor`: un filtro de bordes, aplicado IGUAL a todo lo que ve el detector (entrenamiento,
banco de prueba, negativos de calibración del Gabor). Regla del repo del 2026-10-07: un pre-proceso, el mismo para todos.

    ninguno   la imagen tal cual (el control; es lo que midió rect-lin)
    contorno  x − erosión(x) con cruz 3×3: el borde INTERIOR de la tinta, 1 px. Un trazo de 2 px queda igual (la erosión
              lo borra entero); uno grueso queda como DOS rectas finas paralelas, más los extremos.
    sobel     magnitud del gradiente de Sobel, normalizada a [0, 1] (÷ 4·√2, su máximo en una imagen binaria). Bordes de
              ~2 px a los dos lados del salto, dentro y fuera de la tinta.

    python nn/prepro.py      dibuja en ASCII una recta gruesa, una fina y una mancha con cada filtro
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NOMBRES = ("ninguno", "contorno", "sobel")
_CRUZ = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)[None, None]
_SX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None]


@torch.no_grad()
def aplicar(x: np.ndarray, nombre: str) -> np.ndarray:
    """(N,1,32,32) float32 en {0,1} → (N,1,32,32) float32."""
    if nombre == "ninguno":
        return x
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    if nombre == "contorno":
        erosion = (F.conv2d(t, _CRUZ, padding=1) >= 5).float()          # los 5 de la cruz son tinta (fuera = fondo)
        return (t - erosion).numpy()
    if nombre == "sobel":
        gx = F.conv2d(t, _SX, padding=1); gy = F.conv2d(t, _SX.transpose(-1, -2), padding=1)
        return (torch.sqrt(gx ** 2 + gy ** 2) / (4 * 2 ** 0.5)).clamp(0, 1).numpy()
    raise SystemExit(f"✗ pre-proceso '{nombre}' desconocido: {NOMBRES}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import datos as D
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    mancha = (((xx - 16) ** 2 + (yy - 16) ** 2) <= 25).astype(np.uint8)
    for titulo, img in (("recta 10° · 10 px", D.recta(16, 16, 22, 10, 10)), ("recta 45° · 2 px", D.recta(16, 16, 20, 45, 2)),
                        ("mancha r=5", mancha)):
        x = img.astype(np.float32)[None, None]
        for nom in NOMBRES:
            y = aplicar(x, nom)[0, 0]
            print(f"— {titulo} · {nom}")
            print("\n".join("".join("#" if v > 0.5 else ("+" if v > 0.1 else ".") for v in r[4:28]) for r in y[8:24]))
