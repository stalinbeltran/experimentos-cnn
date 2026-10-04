#!/usr/bin/env python3
"""Corrida 10: compositor que COMBINA detectores — Conv 3×3 (J → 16) + ReLU sobre los mapas apilados, y una
lineal 16·64 → 10. Criterio en instrucciones/02-criterio.md § «Corrida 10».

    python nn/compositor_comb.py        los bancos → resultados/compositor-comb.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compositor as C                          # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
CANALES, LR, L2, EPOCAS = 16, 3e-3, 1e-3, 300
FICHEROS = {"cae3": "-cae3", "fino": "", "grueso": "-c4", "dig": "-dig", "cae5": "-cae5"}
BANCOS = {
    "cae3": ["cae3"],
    "fino": ["fino"],
    "grueso": ["grueso"],
    "fino+grueso": ["fino", "grueso"],
    "cae3+fino+grueso": ["cae3", "fino", "grueso"],
    "todos": ["fino", "grueso", "dig", "cae5", "cae3"],
}


class Combinante(nn.Module):
    def __init__(self, j: int):
        super().__init__()
        self.comb = nn.Conv2d(j, CANALES, 3, padding=1)
        self.cabeza = nn.Linear(CANALES * 64, 10)

    def forward(self, x):
        return self.cabeza(Fn.relu(self.comb(x)).flatten(1))


def mapas(nombre):
    m = dict(np.load(RES / f"mapas-digitos{FICHEROS[nombre]}.npz"))
    return m["sigma"].astype(np.float32), m["y"], m["train"]


def correr(s, y, tr, semilla):
    torch.manual_seed(semilla)
    red = Combinante(s.shape[1]); opt = torch.optim.Adam(red.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(s[tr]), torch.from_numpy(y[tr])
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(red(xt), yt).backward(); opt.step()
    with torch.no_grad():
        pv = red(torch.from_numpy(s[~tr])).argmax(1).numpy(); pt = red(xt).argmax(1).numpy()
    return float((pv == y[~tr]).mean()), float((pt == y[tr]).mean()), sum(p.numel() for p in red.parameters())


def main() -> int:
    salida = {"canales": CANALES, "lr": LR, "l2": L2, "epocas": EPOCAS, "semillas": C.SEMILLAS, "bancos": {}}
    t0 = time.time()
    for banco, grupos in BANCOS.items():
        partes = [mapas(g) for g in grupos]
        y, tr = partes[0][1], partes[0][2]
        assert all(np.array_equal(p[1], y) and np.array_equal(p[2], tr) for p in partes)
        s = np.concatenate([p[0] for p in partes], 1)
        rs = [correr(s, y, tr, sem) for sem in C.SEMILLAS]
        x = s.reshape(len(s), -1)
        lin = [C.logistica(x[tr], y[tr], x[~tr], y[~tr], sem)["acc_val"] for sem in C.SEMILLAS]
        v = [r[0] for r in rs]
        b = {"detectores": int(s.shape[1]), "parametros": rs[0][2],
             "comb_val": round(float(np.mean(v)), 4), "comb_val_sd": round(float(np.std(v, ddof=1)), 4),
             "comb_train": round(float(np.mean([r[1] for r in rs])), 4), "lineal_val": round(float(np.mean(lin)), 4)}
        b["delta"] = round(b["comb_val"] - b["lineal_val"], 4)
        salida["bancos"][banco] = b
        print(f"{banco:<17} {b['detectores']:>2} det · lineal {b['lineal_val']:.4f} → combinante {b['comb_val']:.4f} ± {b['comb_val_sd']:.4f} "
              f"(train {b['comb_train']:.3f}, {b['parametros']} par.) · Δ {b['delta']:+.4f}  [{time.time() - t0:.0f} s]", flush=True)
    (RES / "compositor-comb.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
