#!/usr/bin/env python3
"""La referencia CNN: los 4 detectores de recta de `feat-ind32`, leídos por su ID (registro) y comprobados por la HUELLA de
su firma. Si una huella no casa, se niega antes de evaluar nada (R2).

La clase `Detector` es una COPIA de `nn/modelo.py` de feat-ind32 (no se importa: Regla 0). Sólo hace falta para cargar.

    recta-H 0° (—) · recta-B 45° (\\) · recta-V 90° (|) · recta-S 135° (/)  — la convención de este experimento.

Salida como la del detector lineal: (N, 4) «logits» = logit(max σ del mapa 8×8) − logit(umbral de su firma), así que
«detecta» es > 0 igual que aquí.

    python nn/referencia_cnn.py --comprobar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn import por_id  # noqa: E402

ID = "feat-ind32"
ORDEN = ("recta-H", "recta-B", "recta-V", "recta-S")       # 0, 45, 90, 135


class Detector(nn.Module):
    """COPIA de feat-ind32 nn/modelo.py: Conv3×3 16 → 3×3/2 32 → 3×3/2 32 → 3 × 3×3 32 → 1×1, ReLU. 41.825 parámetros."""

    def __init__(self, canales=(16, 32, 32, 32, 32, 32), strides=(1, 2, 2, 1, 1, 1)):
        super().__init__()
        capas, cin = [], 1
        for c, s in zip(canales, strides):
            capas += [nn.Conv2d(cin, c, 3, stride=s, padding=1), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        return self.salida(self.tronco(x))


def huella_pesos(red: nn.Module) -> str:
    """COPIA de feat-ind32: la huella de su firma se calcula así."""
    h = hashlib.sha256()
    for p in red.parameters():
        h.update(p.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        org = por_id(ID).carpeta
        firma = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))
        self.reds = nn.ModuleList()
        umbrales, self.huellas = [], {}
        for f in ORDEN:
            est = torch.load(org / "nn" / "pesos" / f / "best.pt", map_location="cpu", weights_only=False)
            cfg = est.get("config", {})
            red = Detector(cfg.get("canales", (16, 32, 32, 32, 32, 32)), cfg.get("strides", (1, 2, 2, 1, 1, 1)))
            red.load_state_dict(est["modelo"]); red.eval()
            h = huella_pesos(red)
            if h != firma["huellas_best"][f]:
                raise SystemExit(f"✗ {ID}/{f}: huella {h} ≠ {firma['huellas_best'][f]} de su firma: sus pesos cambiaron. Me niego.")
            self.reds.append(red); umbrales.append(firma["umbrales"][f]); self.huellas[f] = h
        self.register_buffer("u", torch.logit(torch.tensor(umbrales)))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.stack([r(x).flatten(1).amax(1) for r in self.reds], 1)    # max logit del mapa = logit(max σ)
        return m - self.u

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.reds.parameters())


def comprobar() -> int:
    import numpy as np  # noqa: PLC0415
    sys.path.insert(0, str(AQUI))
    import datos as D  # noqa: PLC0415
    cnn = CNN()
    x = torch.from_numpy(np.stack([D.recta(16, 16, 20, a, 3) for a in (0, 45, 90, 135)]).astype(np.float32)[:, None])
    o = cnn(x)
    bien = o.argmax(1).tolist() == [0, 1, 2, 3] and bool((o.amax(1) > 0).all())
    print(f"  [   ok] {ID}: 4 huellas casan con su firma · {cnn.n_parametros()} parámetros en total")
    print(f"  [{'ok' if bien else 'FALLA':>5}] — \\ | / finas caen en su orientación: {o.argmax(1).tolist()}  logits {o.amax(1).numpy().round(2).tolist()}")
    return 0 if bien else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    raise SystemExit(comprobar() if p.parse_args().comprobar else (print(__doc__) or 0))
