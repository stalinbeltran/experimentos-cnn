#!/usr/bin/env python3
"""delg-cnn — SÓLO KERNELS: una pila de convoluciones 3×3 (sin ninguna cabeza densa) que recibe la tinta gruesa
y devuelve un mapa continuo M ∈ [0,1] de la línea media.

Primer tanteo (2026-10-13, pedido por el dueño): 20 trazos de entrenamiento, ¿se puede aprender de ellos?

    python nn/entrenar_local.py                 # 20 trazos, 5 capas de 8 canales, 400 épocas (segundos en CPU)
    python nn/entrenar_local.py --n 20 --capas 5 --canales 8 --epocas 400 --sigma 0.8

Salidas: nn/pesos/<nombre>/{best.pt,config.json,metrics.jsonl} (nombre = k3x3, k5x5… o --nombre) y
         resultados/1-kernels-<n>-<nombre>.png  (entrada, verdad, salida: los 20 vistos y 10 NUEVOS, semilla aparte)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinteticos import N, _XX, _YY, curva_natural, engrosar  # noqa: E402


def verdad(puntos: np.ndarray, sigma: float) -> np.ndarray:
    """Campana alrededor de la línea media: T = exp(−d²/2σ²), d = distancia al punto más cercano de la polilínea."""
    d2 = np.full((N, N), np.inf)
    for px, py in puntos:
        d2 = np.minimum(d2, (_XX - px) ** 2 + (_YY - py) ** 2)
    return np.exp(-d2 / (2 * sigma**2))


def conjunto(n: int, semilla: int, sigma: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(semilla)
    X, T, cs = [], [], []
    while len(X) < n:
        c = curva_natural(rng)
        if len(c["puntos"]) * 0.25 < 10:
            continue
        X.append(engrosar(c["puntos"], c["grosor"])); T.append(verdad(c["puntos"], sigma)); cs.append(c)
    return np.stack(X).astype(np.float32), np.stack(T).astype(np.float32), cs


class SoloKernels(nn.Module):
    """Convoluciones 3×3 con ReLU entre medias y sigmoide al final. Sin pooling, sin densa: la salida es el mismo plano."""

    def __init__(self, capas: int, canales: int, kernel: int = 3):
        super().__init__()
        L, c_in, pad = [], 1, kernel // 2
        for i in range(capas - 1):
            L += [nn.Conv2d(c_in, canales, kernel, padding=pad), nn.ReLU()]; c_in = canales
        L += [nn.Conv2d(c_in, 1, kernel, padding=pad)]
        self.red = nn.Sequential(*L)

    def forward(self, x):
        return torch.sigmoid(self.red(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--capas", type=int, default=5)
    ap.add_argument("--canales", type=int, default=8)
    ap.add_argument("--epocas", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--semilla", type=int, default=101)
    ap.add_argument("--kernel", type=int, default=3, help="lado del kernel (impar): 3 o 5")
    ap.add_argument("--nombre", default=None, help="carpeta de pesos nn/pesos/<nombre>; por defecto k<kernel>x<kernel>")
    a = ap.parse_args()
    a.nombre = a.nombre or f"k{a.kernel}x{a.kernel}"
    torch.manual_seed(a.semilla)

    X, T, cs = conjunto(a.n, a.semilla, a.sigma)
    Xn, Tn, csn = conjunto(10, 202, a.sigma)                 # NUNCA vistos: ¿generaliza o memoriza?
    x, t = torch.from_numpy(X)[:, None], torch.from_numpy(T)[:, None]
    xn, tn = torch.from_numpy(Xn)[:, None], torch.from_numpy(Tn)[:, None]
    red = SoloKernels(a.capas, a.canales, a.kernel)
    n_par = sum(p.numel() for p in red.parameters())
    opt = torch.optim.Adam(red.parameters(), lr=1e-2)
    peso = 1 + 4 * (t > 0.5).float()                         # la línea media es el 3–5 % de los píxeles
    bce = nn.BCELoss(reduction="none")
    out = AQUI / "nn" / "pesos" / a.nombre; out.mkdir(parents=True, exist_ok=True)
    t0, mejor = time.time(), np.inf
    with (out / "metrics.jsonl").open("w") as f:
        for ep in range(1, a.epocas + 1):
            red.train(); opt.zero_grad()
            perdida = (bce(red(x), t) * peso).mean(); perdida.backward(); opt.step()
            red.eval()
            with torch.no_grad():
                pn = (bce(red(xn), tn) * (1 + 4 * (tn > 0.5).float())).mean().item()
            f.write(json.dumps({"epoca": ep, "train": perdida.item(), "nuevos": pn}) + "\n")
            if pn < mejor:
                mejor = pn; torch.save(red.state_dict(), out / "best.pt")
            if ep % 50 == 0 or ep == 1:
                print(f"época {ep:4d}  train {perdida.item():.4f}  nuevos {pn:.4f}")
    dur = time.time() - t0
    (out / "config.json").write_text(json.dumps({**vars(a), "parametros": n_par, "segundos": round(dur, 1),
                                                  "mejor_nuevos": mejor}, indent=1))
    print(f"{n_par} parámetros · {dur:.1f} s · mejor pérdida en nuevos {mejor:.4f}")

    # figura: los 20 vistos + los 10 nuevos, tres filas por bloque (tinta, verdad, salida)
    red.load_state_dict(torch.load(out / "best.pt")); red.eval()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        M, Mn = red(x)[:, 0].numpy(), red(xn)[:, 0].numpy()
    bloques = [("VISTOS en el entrenamiento", X, T, M, cs), ("NUEVOS (nunca vistos)", Xn, Tn, Mn, csn)]
    ncol = max(len(X), len(Xn))
    fig, axs = plt.subplots(6, ncol, figsize=(0.95 * ncol, 6.4))
    for b, (nombre, Xb, Tb, Mb, cb) in enumerate(bloques):
        for j in range(ncol):
            for k in range(3):
                ax = axs[3 * b + k, j]; ax.set_xticks([]); ax.set_yticks([])
                if j >= len(Xb):
                    ax.axis("off"); continue
                img = (Xb, Tb, Mb)[k][j]
                ax.imshow(img, cmap=("Greys", "magma", "magma")[k], vmin=0, vmax=(1.6, 1, 1)[k], interpolation="nearest")
                if k == 0 and j == 0:
                    ax.set_ylabel(f"{nombre}\ntinta", fontsize=6)
                elif j == 0:
                    ax.set_ylabel(("verdad σ=%.1f" % a.sigma, "salida M")[k - 1], fontsize=6)
                if k == 0:
                    ax.set_title(cb[j]["tipo"], fontsize=6)
    fig.suptitle(f"delg-cnn · sólo kernels [{a.nombre}]: {a.capas} conv {a.kernel}×{a.kernel} × {a.canales} canales ({n_par} parámetros), "
                 f"{a.n} trazos, {a.epocas} épocas, {dur:.0f} s · mejor pérdida en nuevos {mejor:.3f}", fontsize=8)
    fig.tight_layout(); (AQUI / "resultados").mkdir(exist_ok=True)
    fig.savefig(AQUI / "resultados" / f"1-kernels-{a.n}-{a.nombre}.png", dpi=130)
    print(f"→ resultados/1-kernels-{a.n}-{a.nombre}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
