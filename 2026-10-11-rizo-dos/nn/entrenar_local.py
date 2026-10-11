#!/usr/bin/env python3
"""rizo-2 — entrena un brazo de la estrategia C con SINTÉTICOS (nunca con dígitos).

    python nn/entrenar_local.py c-base [--epocas 12] [--n 24000]   → nn/pesos/c-base/{best.pt, config.json, metrics.jsonl}

El nombre del script es el contrato con el freno (`cerrable.mjs` casa `entrenar_local.py`). Datos: `sinteticos.conjunto`
con semilla 1 (entrenar) y 101 (validar: la misma semilla con la que se calibran R y P); 70 % positivos. Objetivo: mapa de
calor gaussiano (σ 1) en el centro, BCE con los positivos pesados ×10; radio, cierre y abertura con L1 sólo donde el
mapa real pasa de 0,5. Se guarda el mejor por pérdida de validación.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import sinteticos as SN                                             # noqa: E402
from modelo import BRAZOS, Rizo                                     # noqa: E402

_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def tensores(conj: list[dict]):
    X = np.stack([f["x"] for f in conj]).astype(np.float32)[:, None]
    Y = np.zeros((len(conj), 5, 32, 32), np.float32)
    for k, f in enumerate(conj):
        if f["tipo"] != "rizo":
            continue
        cx, cy = f["centro"]
        Y[k, 0] = np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / 2.0)
        Y[k, 1] = f["radio"] / 7; Y[k, 2] = f["cierre"]
        Y[k, 3], Y[k, 4] = np.cos(np.deg2rad(f["abertura"])), np.sin(np.deg2rad(f["abertura"]))
    return torch.from_numpy(X), torch.from_numpy(Y)


def perdida(out, Y):
    peso = 1 + 9 * (Y[:, 0] > 0.1).float()
    l_heat = (F.binary_cross_entropy_with_logits(out[:, 0], Y[:, 0], reduction="none") * peso).mean()
    m = (Y[:, 0] > 0.5).float()[:, None]
    l_reg = ((out[:, 1:] - Y[:, 1:]).abs() * m).sum() / (m.sum() * 4 + 1e-6)
    return l_heat + l_reg, l_heat, l_reg


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("brazo", choices=list(BRAZOS)); ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--semilla", type=int, default=1); a = ap.parse_args()
    torch.manual_seed(a.semilla); torch.set_num_threads(2); t0 = time.time()
    destino = AQUI / "nn" / "pesos" / a.brazo; destino.mkdir(parents=True, exist_ok=True)
    n_pos = int(a.n * 0.7)
    Xt, Yt = tensores(SN.conjunto(n_pos, a.n - n_pos, a.semilla)); Xv, Yv = tensores(SN.conjunto(700, 300, 101))
    print(f"datos: {len(Xt)} entrenar · {len(Xv)} validar ({time.time() - t0:.0f} s)", flush=True)
    red = Rizo(**BRAZOS[a.brazo]); opt = torch.optim.Adam(red.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epocas)
    (destino / "config.json").write_text(json.dumps({"brazo": a.brazo, **BRAZOS[a.brazo], "parametros": red.parametros(), "epocas": a.epocas,
                                                      "n": a.n, "semilla": a.semilla, "lr": 2e-3}, indent=1) + "\n")
    print(f"{a.brazo}: {red.parametros()} parámetros", flush=True)
    mejor = float("inf"); log = open(destino / "metrics.jsonl", "w")
    for ep in range(1, a.epocas + 1):
        red.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]; opt.zero_grad()
            l, _, _ = perdida(red(Xt[idx]), Yt[idx]); l.backward(); opt.step(); tot += float(l) * len(idx)
        sched.step(); red.eval()
        with torch.no_grad():
            lv, lh, lr = perdida(red(Xv), Yv)
        fila = {"epoca": ep, "train": round(tot / len(Xt), 4), "val": round(float(lv), 4), "val_heat": round(float(lh), 4),
                "val_reg": round(float(lr), 4), "s": round(time.time() - t0)}
        log.write(json.dumps(fila) + "\n"); log.flush(); print(fila, flush=True)
        if float(lv) < mejor:
            mejor = float(lv); torch.save(red.state_dict(), destino / "best.pt")
    torch.save(red.state_dict(), destino / "last.pt")
    print(f"→ {destino}/best.pt (val {mejor:.4f}) · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
