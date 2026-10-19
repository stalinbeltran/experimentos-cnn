#!/usr/bin/env python3
"""rizo-2 § 128 — el LAZO con la posición de su centro y su ALTURA RELATIVA como FEATURE del revisor (el dueño, 2026-10-19:
«crear un nuevo feature, que sea como el lazo con posición de su centro, y que indique cuál es su altura… haz que la altura sea
relativa a 1, así es independiente del tamaño en px… procede»). La vista previa es § 127 (`nn/lazo_altura_vista.py`).

Sobre el DELGADO, con los lazos de `cresta_curvas.objetos()`:
  · altura relativa = alto del lazo / alto del dígito, los DOS sobre la línea media (el lazo: máx − mín de la y de su polígono; el
    dígito: de la primera a la última fila del esqueleto de M > 0,4). Un 0 ≈ 1 (val: mediana 0,97); 6, 9, 4, 8 ≈ 0,4.
  · rejilla: gaussiana σ 2 px en el CENTRO de cada lazo con pico = su altura relativa, máximo por celda (como `lazos_area` con el
    área) + la cuenta de lazos. Las mismas columnas que el bloque `lazos`: celdas² + 1.
⚠ 5 de los 156 ochos de val dan un lazo EXTERIOR que rodea a los dos (≈ 0,99, como un 0): es del detector de lazos y se deja así
(§ 127); pendiente enseñárselo al dueño.

Fuera de `cresta_curvas.py` a propósito, como la cúspide: tocar ese fichero cambia su hash y con él la huella de todo compositor
guardado y todas las cachés de features.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

import cresta_curvas as CC

AQUI = Path(__file__).resolve().parents[1]
SIGMA = 2.0                                                              # px: el de `lazos` y `lazos_area`


def alto_digito(M) -> float:
    """De la primera a la última fila de la LÍNEA MEDIA (esqueleto de M > 0,4). NaN si no hay cresta."""
    filas = np.flatnonzero(skeletonize(np.asarray(M, np.float32) > 0.4).any(1))
    return float(filas[-1] - filas[0]) if len(filas) > 1 else float("nan")


def altura_relativa(z, alto_dig: float) -> float:
    p = np.asarray(z["pts"], float)
    return float((p[:, 1].max() - p[:, 1].min()) / alto_dig) if alto_dig and np.isfinite(alto_dig) else 0.0


def bloque(M, celdas: int = 3, lazos=None) -> list[float]:
    """celdas² + 1 números: la rejilla (pico = altura relativa) y la cuenta de lazos. `lazos` evita recalcular objetos()."""
    M = np.asarray(M, np.float32)
    lz = CC.objetos(M)["lazos"] if lazos is None else lazos
    a = alto_digito(M)
    m = CC._mapa(lz, lambda z: altura_relativa(z, a), SIGMA)
    return F.adaptive_max_pool2d(torch.from_numpy(m)[None], celdas)[0].numpy().ravel().tolist() + [float(len(lz))]


def huella_modulo() -> str:
    """El hash de lo que calcula este bloque: este fichero y `cresta_curvas.py` (de donde salen la cresta y los lazos)."""
    return hashlib.sha1((AQUI / "nn" / "lazo_altura.py").read_bytes() + (AQUI / "nn" / "cresta_curvas.py").read_bytes()).hexdigest()[:10]
