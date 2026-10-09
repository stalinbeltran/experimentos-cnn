#!/usr/bin/env python3
"""TANTEO RÁPIDO (pedido por el dueño, 2026-10-09: «es solo prueba, para ver qué tal le va a un compositor de números con
este detector»). No es un estudio de este experimento: no tiene criterio propio. Lo escrito antes de correrlo es sólo la
predicción de abajo.

Detector: el Gabor 9×9 FIJO de `rect-lin` (4 orientaciones), SIN tomar el max de la imagen: se conserva el MAPA.
    A  ReLU(Gabor ⋆ x)                                  → max en celdas 4×4 → 4 × 8×8 = 256 características
    B  A, y después INTEGRADO a lo largo de su orientación (una recta de 15 px en esa dirección) → idem, 256
    C  B en dos escalas (la imagen y su mitad, para el trazo grueso)                              → 512
Referencias: píxeles 8×8 (media 4×4, 64) y píxeles 32×32 (1024), con el mismo compositor.
Compositor: el de los `feat-*` (Linear, Adam, 300 épocas a lote completo, lr 1e-2, L2 1e-3), 3 semillas.
Dígitos: `uci-optdigits-orig-32px-r20261005`; C1 = 180 train / 1617 val (windep); a ciegas: los 3823 de otros escritores.
Comparación: `feat-ind32` (13 detectores CNN, 832 características) dio 0,949 en C1.

PREDICCIÓN (escrita antes de correrlo): B ≥ A, porque la integración separa recta de mancha; C ≥ B por el grosor; y
ninguno llega a 0,949, porque sólo hay RECTAS (faltan arcos, lazo y esquinas, que en `feat-cortas` resultaron ser lo
que más pesaba).

    python nn/prueba_digitos.py   → resultados/prueba-digitos.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset                                   # noqa: E402
import modelo as M                                                  # noqa: E402

RES = AQUI.parent / "resultados"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
SEMILLAS = (0, 1, 2)
EPOCAS, LR, L2 = 300, 1e-2, 1e-3
L_INT = 15
GABOR = torch.from_numpy(np.stack([M.kernel_gabor(9, a) for a in (0, 45, 90, 135)])[:, None])


def linea(ang: float, largo: int = L_INT) -> np.ndarray:
    """Kernel de integración: una recta de 1 px y `largo` px en la dirección `ang` (y hacia abajo), suma 1."""
    r = largo // 2; yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    t = np.deg2rad(ang); u = xx * np.cos(t) + yy * np.sin(t); v = -xx * np.sin(t) + yy * np.cos(t)
    k = ((np.abs(v) <= 0.5) & (np.abs(u) <= r)).astype(np.float32)
    return k / k.sum()


INTEG = torch.from_numpy(np.stack([linea(a) for a in (0, 45, 90, 135)])[:, None])


@torch.no_grad()
def caracteristicas(x: np.ndarray, integrar: bool, escalas: int) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32)[:, None])
    salidas = []
    for e in range(escalas):
        xe = xt if e == 0 else F.avg_pool2d(xt, 2 ** e)
        m = F.relu(F.conv2d(xe, GABOR, padding=4))
        if integrar:
            m = F.conv2d(m, INTEG, padding=L_INT // 2, groups=4)        # cada orientación, a lo largo de SÍ MISMA
        salidas.append(F.adaptive_max_pool2d(m, 8).flatten(1))
    return torch.cat(salidas, 1).numpy()


def compositor(xtr, ytr, xte_s, sem):
    torch.manual_seed(sem)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    W = torch.nn.Linear(xtr.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(EPOCAS):
        opt.zero_grad(); F.cross_entropy(W(xt), yt).backward(); opt.step()
    with torch.no_grad():
        return [float((W(torch.from_numpy((x - mu) / sd).float()).argmax(1).numpy() == y).mean()) for x, y in xte_s]


def main() -> int:
    torch.set_num_threads(2)
    d = dict(np.load(exigir_dataset(DIGITOS) / "datos.npz"))
    img, y, part = d["imagenes"], d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = part == "train", part == "val", part == "extra"
    variantes = {
        "pixeles 8×8": lambda x: x.reshape(len(x), 8, 4, 8, 4).mean((2, 4)).reshape(len(x), -1),
        "pixeles 32×32": lambda x: x.reshape(len(x), -1).astype(np.float32),
        "A Gabor (mapa)": lambda x: caracteristicas(x, False, 1),
        "B Gabor + integrado": lambda x: caracteristicas(x, True, 1),
        "C B en 2 escalas": lambda x: caracteristicas(x, True, 2),
    }
    out = {}
    for nombre, f in variantes.items():
        X = f(img)
        acc = np.array([compositor(X[tr], y[tr], [(X[va], y[va]), (X[ex], y[ex])], s) for s in SEMILLAS])
        out[nombre] = {"caracteristicas": int(X.shape[1]), "val_1617": round(float(acc[:, 0].mean()), 4),
                       "val_rango": [round(float(acc[:, 0].min()), 4), round(float(acc[:, 0].max()), 4)],
                       "ciega_3823": round(float(acc[:, 1].mean()), 4)}
        print(f"  {nombre:22s} {X.shape[1]:5d} caract. · val {out[nombre]['val_1617']:.4f} {out[nombre]['val_rango']} · "
              f"a ciegas {out[nombre]['ciega_3823']:.4f}", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "prueba-digitos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
