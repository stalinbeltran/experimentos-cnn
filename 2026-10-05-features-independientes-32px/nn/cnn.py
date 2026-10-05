#!/usr/bin/env python3
"""Las dos CNN «tradicionales» (entrenadas de punta a punta sobre los dígitos) contra las que se comparan las curvas de los
detectores (2026-10-05, pedido del dueño). AUTOCONTENIDAS: no importan nada de nadie.

  CNN3   la de `ruido-nist`, COPIADA (Regla 0): 3 × Conv(3×3, C = 8) + ReLU sin padding (8 → 6 → 4 → 2), promedio global y
         Linear(8 → 10). 1.338 parámetros. Entra el 8×8 (cuentas/16). Es la «CNN normal» que ya había en el repo.
  LeNet5 la clásica de LeCun et al. (1998) para dígitos de 32×32 (el formato para el que se diseñó), en su versión moderna
         (ReLU y max-pool): Conv 5×5 6 → pool 2 → Conv 5×5 16 → pool 2 → 400 → 120 → 84 → 10. 61.706 parámetros.
         Entra el bitmap 32×32 binario.

    python nn/cnn.py      parámetros y un forward de cada una
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN3(nn.Module):
    entrada = 8

    def __init__(self, C: int = 8):
        super().__init__()
        capas, cin = [], 1
        for _ in range(3):
            capas += [nn.Conv2d(cin, C, 3), nn.ReLU()]
            cin = C
        self.tronco = nn.Sequential(*capas)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cabeza = nn.Linear(C, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(torch.flatten(self.gap(self.tronco(x)), 1))


class LeNet5(nn.Module):
    entrada = 32

    def __init__(self):
        super().__init__()
        self.tronco = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),      # 32 → 28 → 14
                                    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))     # 14 → 10 → 5
        self.cabeza = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
                                    nn.Linear(84, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.tronco(x))


# nombre → (clase, lr). Mismos pasos y lote para las dos (los de ruido-nist: 3996 pasos de 20).
MODELOS = {"cnn3": (CNN3, 3e-3), "lenet5": (LeNet5, 1e-3)}
ETIQUETAS = {"cnn3": "CNN 3 capas 8×8 (la del repo)", "lenet5": "LeNet-5 32×32"}


def n_parametros(red: nn.Module) -> int:
    return sum(p.numel() for p in red.parameters())


def main() -> int:
    for nombre, (clase, lr) in MODELOS.items():
        red = clase()
        print(f"{nombre:<7} {n_parametros(red):>6} parámetros · lr {lr:g} · {clase.entrada}×{clase.entrada} → "
              f"{tuple(red(torch.rand(2, 1, clase.entrada, clase.entrada)).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
