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


# --- las tres variantes «CNN con el compositor de los detectores» (pedidas el 2026-10-05) -------------------------------
# El compositor es el de `feat-ind`: σ de 13 mapas 8×8 → aplanar (832) → Linear → 10. Todo derivable: el umbral y el argmax
# de la lectura de un detector NO intervienen (el compositor lee los mapas continuos).

FAMILIAS = ("arco-E", "arco-W", "arco-N", "arco-S", "recta-V", "recta-H", "recta-S", "recta-B", "lazo",
            "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW")   # el orden de features.CON_TRAZO de feat-ind


class CNN3Pos(nn.Module):
    """A: la CNN de 3 capas del repo con el compositor posicional en vez de promedio global. Cambia lo mínimo: padding (los
    mapas siguen en 8×8), la última capa da 13 mapas de logits (sin ReLU) y la cabeza es σ → 832 → 10. 9.943 parámetros."""
    entrada = 8

    def __init__(self):
        super().__init__()
        self.tronco = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(8, 13, 3, padding=1))
        self.compositor = nn.Linear(13 * 64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compositor(torch.sigmoid(self.tronco(x)).flatten(1))


class Detector8(nn.Module):
    """COPIA del detector de `feat-ind` (corrida 2): 3 × Conv 3×3 con padding (16, 32, 32) + ReLU y Conv 1×1 → un mapa 8×8 de
    logits. Sólo para COMPROBAR que el banco agrupado de abajo es exactamente 13 de estos (nn/curvas_cnn.py --exportar-init)."""

    def __init__(self, canales=(16, 32, 32)):
        super().__init__()
        capas, cin = [], 1
        for c in canales:
            capas += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))


class Banco13(nn.Module):
    """B y C: los 13 detectores de `feat-ind` EN PARALELO como una sola red —convoluciones agrupadas, o sea exactamente 13
    redes independientes, sólo que en una pasada— y el compositor posicional. 183.053 + 8.330 = 191.383 parámetros.
    B la entrena de punta a punta desde cero; C arranca de los detectores sintéticos (nn/init-detectores-8px.pt)."""
    entrada = 8

    def __init__(self, n: int = 13, canales=(16, 32, 32)):
        super().__init__()
        c1, c2, c3 = canales
        self.detectores = nn.Sequential(nn.Conv2d(1, n * c1, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(n * c1, n * c2, 3, padding=1, groups=n), nn.ReLU(),
                                        nn.Conv2d(n * c2, n * c3, 3, padding=1, groups=n), nn.ReLU(),
                                        nn.Conv2d(n * c3, n, 1, groups=n))
        self.compositor = nn.Linear(n * 64, 10)

    def mapas(self, x: torch.Tensor) -> torch.Tensor:
        return self.detectores(x)                      # (N, 13, 8, 8) logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compositor(torch.sigmoid(self.mapas(x)).flatten(1))

    @staticmethod
    def estado_desde_detectores(estados: list[dict]) -> dict:
        """El state_dict de `self.detectores` a partir de los 13 state_dict de Detector8, en ese orden."""
        cat = lambda k: torch.cat([e[k] for e in estados], 0)          # noqa: E731
        return {"0.weight": cat("tronco.0.weight"), "0.bias": cat("tronco.0.bias"),
                "2.weight": cat("tronco.2.weight"), "2.bias": cat("tronco.2.bias"),
                "4.weight": cat("tronco.4.weight"), "4.bias": cat("tronco.4.bias"),
                "6.weight": cat("salida.weight"), "6.bias": cat("salida.bias")}


# nombre → (clase, lr). Mismos pasos y lote para todas (los de ruido-nist: 3996 pasos de 20). `ajuste13` no tiene un lr:
# entrena en dos fases (AJUSTE).
MODELOS = {"cnn3": (CNN3, 3e-3), "lenet5": (LeNet5, 1e-3),
           "cnn3pos": (CNN3Pos, 3e-3), "aprendidos13": (Banco13, 2e-3), "ajuste13": (Banco13, None)}
ETIQUETAS = {"cnn3": "CNN 3 capas 8×8 (la del repo)", "lenet5": "LeNet-5 32×32",
             "cnn3pos": "A · CNN 3 capas + compositor", "aprendidos13": "B · detectores aprendidos",
             "ajuste13": "C · detectores sintéticos + ajuste fino"}
# C: primero sólo el compositor con los detectores congelados (la mitad de los pasos), después todo, con los detectores a un
# lr 10× menor que el compositor. El compositor recién inicializado es aleatorio: sin la fase congelada, sus primeros
# gradientes deshacen lo que los detectores traen.
AJUSTE = {"pasos_congelado": 1998, "lr_compositor_congelado": 3e-3, "lr_compositor": 1e-3, "lr_detectores": 1e-4}
INIT_AJUSTE = "init-detectores-8px.pt"


def n_parametros(red: nn.Module) -> int:
    return sum(p.numel() for p in red.parameters())


def main() -> int:
    for nombre, (clase, lr) in MODELOS.items():
        red = clase()
        print(f"{nombre:<13} {n_parametros(red):>7} parámetros · lr {lr if lr else 'en dos fases'} · {clase.entrada}×{clase.entrada} → "
              f"{tuple(red(torch.rand(2, 1, clase.entrada, clase.entrada)).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
