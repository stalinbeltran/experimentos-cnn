#!/usr/bin/env python3
"""Corrida 9: compositor POSICIONAL con L2 y resolución del mapa elegidos por validación cruzada de 5 pliegues
DENTRO de los 180 de train (val no interviene). Criterio en instrucciones/02-criterio.md § «Corrida 9».

    python nn/compositor_reg.py         los 6 bancos → resultados/compositor-reg.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compositor as C                          # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
L2S = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
RESOLUCIONES = (8, 4)
PLIEGUES = 5
SEMILLA_CV = 0
# banco → [(fichero de mapas, índices de detectores que se toman)]
BANCOS = {
    "fino": [("", None)],
    "fino+grueso": [("-c24", None)],
    "fino+grueso+dig": [("-c24", None), ("-dig", None)],
    "fino+grueso+cae5": [("-c24", None), ("-cae5", None)],
    "fino+grueso+cae3": [("-c24", None), ("-cae3", None)],
    "todos": [("-c24", None), ("-dig", None), ("-cae5", None), ("-cae3", None)],
}


def cargar(partes):
    sig, y, tr = [], None, None
    for suf, _ in partes:
        m = dict(np.load(RES / f"mapas-digitos{suf}.npz"))
        if y is not None and not (np.array_equal(m["y"], y) and np.array_equal(m["train"], tr)):
            raise SystemExit(f"✗ {suf}: no son los mismos dígitos")
        y, tr = m["y"], m["train"]; sig.append(m["sigma"].astype(np.float32))
    return np.concatenate(sig, 1), y, tr


def reducir(s: np.ndarray, res: int) -> np.ndarray:
    if res == 8:
        return s.reshape(len(s), -1)
    return s.reshape(len(s), s.shape[1], 4, 2, 4, 2).mean((3, 5)).reshape(len(s), -1)


def ajustar(x, y, l2: float, semilla: int) -> torch.nn.Linear:
    torch.manual_seed(semilla)
    W = torch.nn.Linear(x.shape[1], 10)
    opt = torch.optim.Adam(W.parameters(), lr=C.LR, weight_decay=l2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(C.EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def acierto(W, x, y) -> float:
    with torch.no_grad():
        return float((W(torch.from_numpy(x)).argmax(1).numpy() == y).mean())


def pliegues(y: np.ndarray) -> list[np.ndarray]:
    """Estratificados por clase: cada pliegue lleva la misma proporción de cada dígito."""
    rng = np.random.default_rng(SEMILLA_CV); asig = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c)); asig[idx] = np.arange(len(idx)) % PLIEGUES
    return [asig == k for k in range(PLIEGUES)]


def main() -> int:
    salida = {"l2s": L2S, "resoluciones": RESOLUCIONES, "pliegues": PLIEGUES, "epocas": C.EPOCAS, "lr": C.LR, "bancos": {}}
    t0 = time.time()
    for banco, partes in BANCOS.items():
        s, y, tr = cargar(partes)
        ytr = y[tr]; fs = pliegues(ytr); rejilla = []
        for res in RESOLUCIONES:
            x = reducir(s, res); xtr = x[tr]
            for l2 in L2S:
                accs = [acierto(ajustar(xtr[~f], ytr[~f], l2, 1), xtr[f], ytr[f]) for f in fs]
                rejilla.append({"res": res, "l2": l2, "cv": round(float(np.mean(accs)), 4)})
        mejor = max(rejilla, key=lambda r: (r["cv"], r["l2"], -r["res"]))      # empate → más regularizado
        x = reducir(s, mejor["res"])
        vals, trains = [], []
        for sem in C.SEMILLAS:
            W = ajustar(x[tr], y[tr], mejor["l2"], sem)
            vals.append(acierto(W, x[~tr], y[~tr])); trains.append(acierto(W, x[tr], y[tr]))
        # referencia: el compositor de siempre (8×8, L2 0,001) sobre el mismo banco
        x8 = reducir(s, 8); ref = [acierto(ajustar(x8[tr], y[tr], 0.001, sem), x8[~tr], y[~tr]) for sem in C.SEMILLAS]
        salida["bancos"][banco] = {"detectores": int(s.shape[1]), "entradas": int(x.shape[1]), "elegido": mejor,
                                   "acc_val": round(float(np.mean(vals)), 4), "acc_val_sd": round(float(np.std(vals, ddof=1)), 4),
                                   "acc_train": round(float(np.mean(trains)), 4),
                                   "acc_val_sin_regularizar": round(float(np.mean(ref)), 4), "rejilla_cv": rejilla}
        b = salida["bancos"][banco]
        print(f"{banco:<18} {b['detectores']:>2} det · CV elige {mejor['res']}×{mejor['res']} L2 {mejor['l2']:<5} (cv {mejor['cv']:.4f}) · "
              f"val {b['acc_val']:.4f} ± {b['acc_val_sd']:.4f} (train {b['acc_train']:.3f}) · sin regularizar {b['acc_val_sin_regularizar']:.4f}"
              f"  [{time.time() - t0:.0f} s]", flush=True)
    salida["segundos"] = round(time.time() - t0, 1)
    (RES / "compositor-reg.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
