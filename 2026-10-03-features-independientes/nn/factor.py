#!/usr/bin/env python3
"""Factor de generalización (2026-10-05, para la figura de `feat-ind32`): F(N) = dígitos NO vistos bien reconocidos / N
dígitos de train. «No vistos» = el test de 717 (corrida 11) + los 3823 de otros 30 escritores (de
`uci-optdigits-orig-32px-r20261005`, reducidos a 8×8 por bloques 4×4) = 4540. Banco `fino` (13 detectores) con el
compositor posicional, y los píxeles crudos de 8×8 como referencia. 3 semillas.

    python nn/factor.py      → resultados/factor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset              # noqa: E402
import aplicar                                  # noqa: E402
import compositor as C                          # noqa: E402
import curva                                    # noqa: E402
import datos                                    # noqa: E402

RES = AQUI.parent / "resultados"


def main() -> int:
    m = dict(np.load(RES / "mapas-digitos.npz")); s, y, tr = m["sigma"], m["y"], m["train"]
    orden, test = curva.reparto(y, tr)
    x8 = datos.digitos()["x"]
    z = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz"); e = z["particion"] == "extra"
    xe8 = (z["imagenes"][e].reshape(-1, 8, 4, 8, 4).sum((2, 4)).astype(np.float32) / 16)[:, None]
    ye = z["etiquetas"][e].astype(np.int64)
    se = aplicar.mapas({"x": xe8, "y": ye, "train": np.zeros(len(ye), bool)})["sigma"]
    casos = {"detectores 8×8 (13)": (s.reshape(len(s), -1), se.reshape(len(se), -1)),
             "píxeles 8×8": (x8.reshape(len(x8), -1), xe8.reshape(len(xe8), -1))}
    out = {"no_vistos": int(test.sum()) + len(ye), "test": int(test.sum()), "extra": len(ye), "casos": {}}
    for caso, (xa, xe) in casos.items():
        xa, xe = xa.astype(np.float32), xe.astype(np.float32)
        xp, yp = np.concatenate([xa[test], xe]), np.concatenate([y[test], ye])
        filas = {}
        for n in curva.TAMANOS:
            a = [C.logistica(xa[orden[:n]], y[orden[:n]], xp, yp, sem)["acc_val"] for sem in C.SEMILLAS]
            filas[n] = {"acc": round(float(np.mean(a)), 4), "acc_sd": round(float(np.std(a, ddof=1)), 4),
                        "aciertos": round(float(np.mean(a)) * len(yp), 1), "F": round(float(np.mean(a)) * len(yp) / n, 2)}
            print(f"{caso:<20} N={n:>4}: acc {filas[n]['acc']:.4f} · F {filas[n]['F']:.1f}", flush=True)
        out["casos"][caso] = filas
    (RES / "factor.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
