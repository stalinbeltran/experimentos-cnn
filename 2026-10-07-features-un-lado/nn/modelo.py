#!/usr/bin/env python3
"""El DETECTOR de `feat-1lado`. Entra la TINTA tal cual (N, 1, 32, 32) y sale UN mapa 8×8 de logits.

    tinta → BORDES (capa FIJA: 8 kernels 3×3 de borde de un solo lado + ReLU) → 8 canales 32×32 → TRONCO → 8×8

La capa de bordes es el ÚNICO pre-proceso (regla del dueño del 2026-10-07) y va DENTRO del modelo como buffer, no como
script delante: así es imposible aplicar a los dígitos uno distinto que al entrenamiento. Sus kernels son los del boceto
(`docs/bocetos/2026-10-07-borde-de-un-lado/ver_proceso.py:33`): k_θ = cos θ·Sx + sin θ·Sy (Sobel /4), θ = 0, 45, …, 315,
y ReLU deja pasar sólo «la tinta AUMENTA yendo hacia θ». `conv2d` de torch es correlación, igual que
`scipy.ndimage.correlate` con relleno de ceros (lo comprueba `--comprobar`). Su huella va en `config.json`.

El TRONCO es la red de `feat-bor` / `feat-ind32` sin cambios (Conv3×3 16 → /2 32 → /2 32 → 3 × 32 → Conv1×1, ReLU,
campo receptivo 33 px). Dos BRAZOS:

    control      el tronco mira los 8 canales A LA VEZ (primera convolución de 8 canales de entrada)
    compartido   el MISMO tronco (1 canal de entrada) se aplica a CADA canal por separado, y se queda el MÁXIMO de los
                 8 mapas de logits: el detector ve un lado del trazo cada vez, y el grosor (la distancia entre lados)
                 no le llega. Cuesta 8 pasadas por imagen.

    python nn/modelo.py               parámetros y un forward
    python nn/modelo.py --comprobar   los kernels, la correlación, la equivariancia del máximo y la huella fija
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
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
ANGULOS = (0, 45, 90, 135, 180, 225, 270, 315)          # → ↘ ↓ ↙ ← ↖ ↑ ↗ (y crece hacia ABAJO)
FLECHAS = ("→", "↘", "↓", "↙", "←", "↖", "↑", "↗")
BRAZOS = ("control", "compartido")
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4     # tinta crece hacia la DERECHA
SY = SX.T                                                           # tinta crece hacia ABAJO


def kernels() -> torch.Tensor:
    """(8, 1, 3, 3) float32: k_θ = cos θ·Sx + sin θ·Sy. COPIA de `kernel()` de ver_proceso.py del boceto."""
    ks = [np.cos(np.deg2rad(g)) * SX + np.sin(np.deg2rad(g)) * SY for g in ANGULOS]
    return torch.tensor(np.stack(ks)[:, None], dtype=torch.float32)


def huella_kernels(k: torch.Tensor) -> str:
    return hashlib.sha256(k.detach().contiguous().numpy().astype("<f4").tobytes()).hexdigest()[:16]


HUELLA_KERNELS = huella_kernels(kernels())


class Detector(nn.Module):
    def __init__(self, brazo: str = "control", canales=CANALES, strides=STRIDES):
        super().__init__()
        if brazo not in BRAZOS:
            raise ValueError(f"brazo '{brazo}' desconocido: {BRAZOS}")
        self.brazo, self.canales, self.strides = brazo, tuple(canales), tuple(strides)
        self.register_buffer("bordes", kernels())            # FIJO: no es parámetro, el optimizador no lo ve
        cin = len(ANGULOS) if brazo == "control" else 1
        capas = []
        for c, s in zip(self.canales, self.strides):
            capas += [nn.Conv2d(cin, c, K, stride=s, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def canales_borde(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 32, 32) tinta → (N, 8, 32, 32) bordes de un lado (≥ 0)."""
        return torch.relu(nn.functional.conv2d(x, self.bordes, padding=1))

    def por_canal(self, x: torch.Tensor) -> torch.Tensor:
        """Sólo `compartido`: (N, 8, 8, 8), el mapa de logits de cada canal antes del máximo."""
        b = self.canales_borde(x); n = len(b)
        return self.salida(self.tronco(b.reshape(n * len(ANGULOS), 1, ENTRADA, ENTRADA))).reshape(n, len(ANGULOS), LADO, LADO)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.brazo == "control":
            return self.salida(self.tronco(self.canales_borde(x)))            # (N, 1, 8, 8)
        return self.por_canal(x).max(1, keepdim=True).values                # (N, 1, 8, 8)

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


@torch.no_grad()
def logits_por_lotes(red: nn.Module, x: np.ndarray, lote: int = 512) -> torch.Tensor:
    """El forward de muchas imágenes a trozos: con `compartido` un lote de 6000 serían 48.000 pasadas a la vez."""
    out = [red(torch.from_numpy(np.ascontiguousarray(x[i:i + lote])).float()) for i in range(0, len(x), lote)]
    return torch.cat(out) if out else torch.zeros(0, 1, LADO, LADO)


def huella_pesos(red: nn.Module) -> str:
    h = hashlib.sha256()
    for p in red.parameters():
        h.update(p.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


def cargar(ruta) -> tuple[Detector, dict]:
    est = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = est.get("config", {})
    if cfg.get("huella_kernels") != HUELLA_KERNELS:
        raise SystemExit(f"✗ {ruta}: sus kernels de borde ({cfg.get('huella_kernels')}) no son los de este código "
                         f"({HUELLA_KERNELS}). Me niego: sería otro pre-proceso.")
    red = Detector(cfg["brazo"], cfg.get("canales", CANALES), cfg.get("strides", STRIDES))
    red.load_state_dict(est["modelo"]); red.eval()
    if huella_kernels(red.bordes) != HUELLA_KERNELS:
        raise SystemExit(f"✗ {ruta}: el buffer de bordes guardado no es el de este código. Me niego.")
    return red, est


class Tinta(nn.Module):
    """La REFERENCIA, sólo para leerla: el detector de `feat-ind32` (tinta, sin bordes), con el mismo tronco. Se copia la
    forma aquí (Regla 0) para cargar sus `best.pt` por su id; nunca se entrena desde este experimento."""

    def __init__(self, canales=CANALES, strides=STRIDES):
        super().__init__()
        capas, cin = [], 1
        for c, s in zip(canales, strides):
            capas += [nn.Conv2d(cin, c, K, stride=s, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))


def cargar_tinta(ruta) -> tuple[Tinta, dict]:
    est = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = est.get("config", {})
    red = Tinta(cfg.get("canales", CANALES), cfg.get("strides", STRIDES))
    red.load_state_dict(est["modelo"]); red.eval()
    return red, est


def comprobar() -> int:
    from scipy.ndimage import correlate                                # noqa: PLC0415
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien); print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    rng = np.random.default_rng(0)
    img = (rng.random((32, 32)) < 0.3).astype(np.float32)
    red = Detector("control")
    b = red.canales_borde(torch.from_numpy(img)[None, None])[0].numpy()
    ref = np.stack([np.maximum(0, correlate(img, kernels()[i, 0].numpy().astype(float), mode="constant")) for i in range(8)])
    prueba("conv2d de torch == correlate de scipy (relleno 0), los 8 canales", np.abs(b - ref).max() < 1e-5,
           f"error máx {np.abs(b - ref).max():.2e}")
    barra = np.zeros((32, 32), np.float32); barra[4:28, 10:20] = 1
    bb = red.canales_borde(torch.from_numpy(barra)[None, None])[0].numpy()
    cols0 = np.flatnonzero(bb[0].sum(0) > 0); cols4 = np.flatnonzero(bb[4].sum(0) > 0)
    prueba("barra vertical: el canal → (0°) sólo responde en su borde IZQUIERDO", cols0.max() <= 10, f"columnas {cols0.tolist()}")
    prueba("barra vertical: el canal ← (180°) sólo responde en su borde DERECHO", cols4.min() >= 19, f"columnas {cols4.tolist()}")
    for brazo in BRAZOS:
        torch.manual_seed(1); r = Detector(brazo)
        prueba(f"{brazo}: los kernels NO son parámetros (el optimizador no los ve)",
               all(p.data_ptr() != r.bordes.data_ptr() for p in r.parameters()) and "bordes" in dict(r.named_buffers()))
        opt = torch.optim.Adam(r.parameters(), lr=0.1)
        x = torch.rand(4, 1, 32, 32)
        opt.zero_grad(); r(x).sum().backward(); opt.step()
        prueba(f"{brazo}: la huella de los kernels no cambia tras un paso", huella_kernels(r.bordes) == HUELLA_KERNELS)
        prueba(f"{brazo}: (N,1,32,32) → (N,1,8,8)", tuple(r(x).shape) == (4, 1, 8, 8), f"{r.n_parametros()} parámetros")
    torch.manual_seed(2); r = Detector("compartido"); x = torch.rand(3, 1, 32, 32)
    with torch.no_grad():
        sueltos = torch.stack([r.salida(r.tronco(r.canales_borde(x)[:, i:i + 1]))[:, 0] for i in range(8)], 1)
        prueba("compartido == máximo de 8 pasadas sueltas del mismo tronco",
               torch.allclose(r(x)[:, 0], sueltos.max(1).values, atol=1e-6))
        prueba("logits_por_lotes == forward de golpe", torch.allclose(logits_por_lotes(r, x.numpy(), lote=2), r(x), atol=1e-6))
    print("el modelo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    if "--comprobar" in sys.argv:
        return comprobar()
    for brazo in BRAZOS:
        red = Detector(brazo)
        print(f"{brazo:<11} canales {CANALES}, strides {STRIDES}: {red.n_parametros()} parámetros; kernels {HUELLA_KERNELS}; "
              f"{ENTRADA}×{ENTRADA} → {tuple(red(torch.rand(2, 1, ENTRADA, ENTRADA)).shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
