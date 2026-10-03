#!/usr/bin/env python3
"""La red de `ruido-nist`, AUTOCONTENIDA: 3 × Conv2d(3×3, C = 8) + ReLU sin padding (8→6→4→2),
promedio global y `Linear(8 → 10)`. 1.338 parámetros (calculado el 2026-10-02: 80 + 584 + 584 + 90).

    python nn/modelo.py                 mapas, parámetros y un forward
    python nn/modelo.py --inicializar   crea nn/init/init-s{1,2,3}.pt (si no están) y casa sus huellas

Los PESOS INICIALES son un FICHERO, no una llamada a `manual_seed`: `init-s<s>.pt` se crea una
vez (con `torch.manual_seed(s)`), se guarda con su huella en `init/huellas.json`, y TODOS los
escenarios de la semilla s lo cargan. Así «pesos idénticos en todos los escenarios» no depende de
que torch inicialice igual dentro de un año: depende de un fichero commiteado y de una huella.
Entrenar se NIEGA si el fichero no está (R2), en vez de crearlo al vuelo en silencio.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
INIT = AQUI / "init"
HUELLAS_INIT = INIT / "huellas.json"

L = 3
K = 3               # kernel 3×3, sin padding: cada capa quita 2 px
C = 8
CLASES = 10
LADO = 8
SEMILLAS = (1, 2, 3)


def mapas(lado: int = LADO) -> list[int]:
    out, w = [], lado
    for _ in range(L):
        w = w - K + 1
        if w < 1:
            raise ValueError(f"lado {lado}: el mapa se agota con {L} capas de {K}×{K}")
        out.append(w)
    return out


class Red(nn.Module):
    def __init__(self, C_: int = C):
        super().__init__()
        capas, cin = [], 1
        for _ in range(L):
            capas += [nn.Conv2d(cin, C_, K), nn.ReLU()]      # `valid`: sin padding, sin stride
            cin = C_
        self.tronco = nn.Sequential(*capas)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cabeza = nn.Linear(C_, CLASES)
        self.lados = mapas()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(torch.flatten(self.gap(self.tronco(x)), 1))

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())


def huella_pesos(red: nn.Module) -> str:
    """sha256 (16 hex) de todos los parámetros en orden, en float32 little-endian."""
    h = hashlib.sha256()
    for p in red.parameters():
        h.update(p.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


def nueva(semilla: int) -> Red:
    torch.manual_seed(semilla)
    return Red()


def ruta_init(semilla: int) -> Path:
    return INIT / f"init-s{semilla}.pt"


def _huellas_guardadas() -> dict:
    return json.loads(HUELLAS_INIT.read_text(encoding="utf-8")) if HUELLAS_INIT.is_file() else {}


def inicializar(semillas=SEMILLAS) -> int:
    """Crea los que falten y CASA los que estén contra `huellas.json`. Idempotente."""
    INIT.mkdir(parents=True, exist_ok=True)
    reg = _huellas_guardadas()
    ok = True
    for s in semillas:
        f = ruta_init(s)
        if f.is_file():
            red = cargar_inicial(s)
            print(f"  init-s{s}.pt ya está: huella {huella_pesos(red)} casa con huellas.json")
            continue
        red = nueva(s)
        torch.save({"modelo": red.state_dict(), "semilla": s, "torch": torch.__version__}, f)
        reg[str(s)] = huella_pesos(red)
        print(f"  init-s{s}.pt creado con torch.manual_seed({s}): huella {reg[str(s)]}")
    HUELLAS_INIT.write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


def cargar_inicial(semilla: int) -> Red:
    """Los pesos iniciales de la semilla, desde el FICHERO, con su huella casada. Se niega si falta."""
    f = ruta_init(semilla)
    if not f.is_file():
        raise RuntimeError(f"no está {f.relative_to(AQUI.parent)}: los pesos iniciales son un fichero compartido. "
                           f"Créalo con `python nn/modelo.py --inicializar` (y commitéalo) antes de entrenar.")
    est = torch.load(f, map_location="cpu", weights_only=False)
    red = Red()
    red.load_state_dict(est["modelo"])
    esperada = _huellas_guardadas().get(str(semilla))
    h = huella_pesos(red)
    if esperada is None or h != esperada:
        raise RuntimeError(f"init-s{semilla}.pt da la huella {h} y huellas.json dice {esperada}: no son los pesos "
                           f"iniciales con los que se entrenó el resto. Me niego.")
    return red


def main() -> int:
    if "--inicializar" in sys.argv:
        return inicializar()
    red = Red()
    print(f"L={L} k={K} C={C}: mapas {LADO}->{'->'.join(map(str, red.lados))}, {red.n_parametros()} parametros "
          f"(cabeza {sum(p.numel() for p in red.cabeza.parameters())})")
    print(f"forward: salida {tuple(red(torch.rand(2, 1, LADO, LADO)).shape)}")
    for s in SEMILLAS:
        print(f"  init-s{s}.pt: {'está' if ruta_init(s).is_file() else 'FALTA (--inicializar)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
