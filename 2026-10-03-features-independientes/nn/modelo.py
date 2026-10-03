#!/usr/bin/env python3
"""El DETECTOR de `feat-ind`, AUTOCONTENIDO: totalmente convolucional con padding, 3 × Conv(3×3) + ReLU
(16, 32, 32 canales desde la corrida 2; 8, 16, 16 en la 1) y una Conv(1×1) → UN mapa 8×8 de logits. Campo receptivo 7×7. 14.081 parámetros
(calculado: 160 + 4.640 + 9.248 + 33; la corrida 1 tenía 3.585).

Lectura del mapa (ESPECIFICACION.md §5): hallada si max σ ≥ UMBRAL; posición = celda del máximo.
Objetivo de entrenamiento: gaussiana de σ = SIGMA celdas en el ancla; todo cero si no está.

    python nn/modelo.py     parámetros y un forward
"""

from __future__ import annotations

import hashlib
import sys

import torch
import torch.nn as nn

LADO = 8
CANALES = (16, 32, 32)      # corrida 1: (8, 16, 16); ENMIENDA corrida 2: el doble, +0,07 F1 en arco-E medido
K = 3
UMBRAL = 0.5
SIGMA = 0.6
PESO_OBJETIVO = 8.0


class Detector(nn.Module):
    def __init__(self, canales=CANALES):
        super().__init__()
        self.canales = tuple(canales)
        capas, cin = [], 1
        for c in self.canales:
            capas += [nn.Conv2d(cin, c, K, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))              # (N, 1, 8, 8) logits

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def objetivo(anclas: torch.Tensor) -> torch.Tensor:
    """(N, 2) (fila, col) -> (N, 1, 8, 8): gaussiana en el ancla; una fila con -1 da todo cero."""
    g = torch.arange(LADO, dtype=torch.float32)
    f = anclas[:, 0].float()[:, None, None]; c = anclas[:, 1].float()[:, None, None]
    d2 = (g[None, :, None] - f) ** 2 + (g[None, None, :] - c) ** 2
    t = torch.exp(-d2 / (2 * SIGMA ** 2))
    t[anclas[:, 0] < 0] = 0.0
    return t[:, None]


def leer(logits: torch.Tensor, umbral: float = UMBRAL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(N,1,8,8) -> (hallada bool (N,), posición (N,2) fila/col, confianza (N,) = max σ).
    ENMIENDA 2026-10-03: el umbral es POR DETECTOR, elegido sobre train y guardado en su checkpoint
    (`est["umbral"]`); UMBRAL es sólo el defecto de la corrida 1."""
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
    red = Detector(est.get("config", {}).get("canales", CANALES)); red.load_state_dict(est["modelo"]); red.eval()
    return red, est


def main() -> int:
    red = Detector()
    print(f"canales {CANALES}, k={K}, padding: {red.n_parametros()} parámetros; salida {tuple(red(torch.rand(2, 1, LADO, LADO)).shape)}")
    t = objetivo(torch.tensor([[3, 4], [-1, -1]]))
    print(f"objetivo: pico {t[0, 0, 3, 4]:.2f}, vecino {t[0, 0, 3, 5]:.2f}, diagonal {t[0, 0, 4, 5]:.2f}, negativo suma {t[1].sum():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
