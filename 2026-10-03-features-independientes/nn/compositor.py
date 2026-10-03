#!/usr/bin/env python3
"""Los compositores de `feat-ind` sobre los mapas de los dígitos (resultados/mapas-digitos.npz, de
nn/aplicar.py). Dos, y los dos LINEALES (regresión logística, Adam, L2), entrenados con las 180 de
train y medidos sobre las 1617 de val:

    presencia   13 entradas: el máximo de cada mapa          («¿QUÉ features hay?», ciego a la posición)
    posicional  13 × 64 = 832 entradas: los mapas enteros    («...y DÓNDE»)

    python nn/compositor.py              entrena los dos y escribe resultados/compositores.json

Sin selección de modelo sobre val: épocas y L2 fijos aquí (ESCRITOS antes de correr). Se reporta la
exactitud, la matriz de confusión y el par 6↔9 de cada uno (02-criterio.md §B).
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
import features as F                           # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
EPOCAS, LR, L2, SEMILLAS = 300, 1e-2, 1e-3, (1, 2, 3)


def entradas(m: dict) -> dict:
    s = m["sigma"]
    return {"presencia": s.reshape(len(s), len(F.CON_TRAZO), -1).max(2), "posicional": s.reshape(len(s), -1)}


def logistica(xtr, ytr, xva, yva, semilla: int) -> dict:
    torch.manual_seed(semilla)
    W = torch.nn.Linear(xtr.shape[1], 10)
    opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xtr_t, ytr_t, xva_t = torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(xva)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xtr_t), ytr_t).backward(); opt.step()
    with torch.no_grad():
        pred = W(xva_t).argmax(1).numpy(); pred_tr = W(xtr_t).argmax(1).numpy()
    conf = np.zeros((10, 10), int)
    for a, b in zip(yva, pred):
        conf[a, b] += 1
    return {"acc_val": round(float((pred == yva).mean()), 4), "acc_train": round(float((pred_tr == ytr).mean()), 4),
            "confusion": conf.tolist(), "acc_por_clase": {str(c): round(float((pred[yva == c] == c).mean()), 3) for c in range(10)},
            "6_como_9": int(conf[6, 9]), "9_como_6": int(conf[9, 6]), "2_como_5": int(conf[2, 5]), "5_como_2": int(conf[5, 2])}


def main() -> int:
    sufijo = sys.argv[sys.argv.index("--sufijo") + 1] if "--sufijo" in sys.argv else ""
    f = RES / f"mapas-digitos{sufijo}.npz"
    if not f.is_file():
        raise SystemExit("✗ no está resultados/mapas-digitos.npz: primero nn/aplicar.py")
    m = dict(np.load(f)); e = entradas(m); tr = m["train"]; y = m["y"]
    salida = {"epocas": EPOCAS, "lr": LR, "l2": L2, "semillas": SEMILLAS, "n_train": int(tr.sum()), "n_val": int((~tr).sum()), "azar": 0.1}
    t0 = time.time()
    for nombre, x in e.items():
        x = x.astype(np.float32)
        corridas = [logistica(x[tr], y[tr], x[~tr], y[~tr], s) for s in SEMILLAS]
        accs = [c["acc_val"] for c in corridas]
        salida[nombre] = {"entradas": int(x.shape[1]), "acc_val_media": round(float(np.mean(accs)), 4), "acc_val_sd": round(float(np.std(accs, ddof=1)), 4),
                          "acc_val_por_semilla": accs, "acc_train_media": round(float(np.mean([c["acc_train"] for c in corridas])), 4),
                          "semilla_1": corridas[0]}
        c1 = corridas[0]
        print(f"{nombre:<11} ({x.shape[1]:>3} entradas): acc_val {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} (train {salida[nombre]['acc_train_media']:.3f}) · "
              f"6→9 {c1['6_como_9']} 9→6 {c1['9_como_6']} · 2→5 {c1['2_como_5']} 5→2 {c1['5_como_2']}")
    salida["delta_posicional_menos_presencia"] = round(salida["posicional"]["acc_val_media"] - salida["presencia"]["acc_val_media"], 4)
    salida["segundos"] = round(time.time() - t0, 1)
    (RES / f"compositores{sufijo}.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Δ posicional − presencia = {salida['delta_posicional_menos_presencia']:+.4f} → resultados/compositores{sufijo}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
