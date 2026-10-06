#!/usr/bin/env python3
"""El DETECTOR de `feat-bor`: COPIA del de `feat-ind32` (nn/modelo.py, 2026-10-06) con UN cambio, el número de canales
de ENTRADA (`entrada`): 1 para `lineas` y `contorno`, 4 para `signo` (nn/bordes.py). Todo lo demás es idéntico: mismas
capas, strides, objetivo, lectura y huella. AUTOCONTENIDO: entra (N, entrada, 32, 32) y sale UN mapa 8×8 de logits.

    Conv3×3 16 (32×32) → Conv3×3/2 32 (16×16) → Conv3×3/2 32 (8×8) → 3 × Conv3×3 32 (8×8) → Conv1×1 → 1 canal 8×8

Todas con padding y ReLU. Campo receptivo 33 px (calculado: 3, 5, 9, 17, 25, 33): cubre lo mismo que los 7×7 celdas
(28 px) de `feat-ind`, que es el criterio con el que se eligió la profundidad (sustituye al ensayo C2 del plan, ver
REGLAS.md de feat-ind32). 41.825 parámetros con entrada 1; con 4, 432 más (la primera convolución).

    python nn/modelo.py     parámetros y un forward
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

ENTRADA = 32
LADO = 8                     # el mapa de salida
CANALES = (16, 32, 32, 32, 32, 32)
STRIDES = (1, 2, 2, 1, 1, 1)
K = 3
UMBRAL = 0.5
SIGMA = 0.6                  # en celdas del mapa 8×8, como en feat-ind
PESO_OBJETIVO = 8.0


class Detector(nn.Module):
    def __init__(self, canales=CANALES, strides=STRIDES, entrada: int = 1):
        super().__init__()
        self.canales, self.strides, self.entrada = tuple(canales), tuple(strides), int(entrada)
        capas, cin = [], self.entrada
        for c, s in zip(self.canales, self.strides):
            capas += [nn.Conv2d(cin, c, K, stride=s, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))              # (N, 1, 8, 8) logits

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def objetivo(anclas: torch.Tensor) -> torch.Tensor:
    """(N, 2) (fila, col) de la celda 8×8 -> (N, 1, 8, 8): gaussiana en el ancla; una fila con -1 da todo cero."""
    g = torch.arange(LADO, dtype=torch.float32)
    f = anclas[:, 0].float()[:, None, None]; c = anclas[:, 1].float()[:, None, None]
    d2 = (g[None, :, None] - f) ** 2 + (g[None, None, :] - c) ** 2
    t = torch.exp(-d2 / (2 * SIGMA ** 2))
    t[anclas[:, 0] < 0] = 0.0
    return t[:, None]


def leer(logits: torch.Tensor, umbral: float = UMBRAL):
    """(N,1,8,8) -> (hallada (N,), posición (N,2), confianza (N,) = max σ). El umbral es POR DETECTOR."""
    p = torch.sigmoid(logits).flatten(1)
    conf, idx = p.max(1)
    pos = torch.stack([idx // LADO, idx % LADO], 1)
    return conf >= umbral, pos, conf


def huella_pesos(red: nn.Module) -> str:
    h = hashlib.sha256()
    for p in red.parameters():
        h.update(p.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


def cargar(ruta) -> tuple[Detector, dict]:
    est = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = est.get("config", {})
    red = Detector(cfg.get("canales", CANALES), cfg.get("strides", STRIDES), cfg.get("entrada", 1))
    red.load_state_dict(est["modelo"]); red.eval()
    return red, est


def main() -> int:
    for e in (1, 4):
        red = Detector(entrada=e)
        print(f"entrada {e} canal(es), canales {CANALES}, strides {STRIDES}: {red.n_parametros()} parámetros; "
              f"{ENTRADA}×{ENTRADA} → {tuple(red(torch.rand(2, e, ENTRADA, ENTRADA)).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
