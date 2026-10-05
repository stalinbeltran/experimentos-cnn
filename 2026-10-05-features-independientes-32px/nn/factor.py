#!/usr/bin/env python3
"""Factor de generalización: F(N) = dígitos NO vistos bien reconocidos / N dígitos de train. «No vistos» = el test de 717
+ los 3823 de otros 30 escritores = 4540. Los 13 detectores de 32×32 con el compositor posicional, y los píxeles
crudos de 32×32 como referencia. 3 semillas. `feat-ind` calcula lo mismo a 8×8 (su `nn/factor.py`).

    python nn/factor.py      → resultados/factor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import componer as K                            # noqa: E402
import datos                                    # noqa: E402

RES = AQUI.parent / "resultados"


def main() -> int:
    m = dict(np.load(RES / "mapas-digitos.npz")); s, y, tr, ex = m["sigma"], m["y"], m["train"], m["extra"]
    x32 = datos.digitos(con_extra=True)["x"]
    w = ~ex
    orden, test = K.reparto(y[w], tr[w])
    casos = {"detectores 32×32 (13)": s.reshape(len(s), -1), "píxeles 32×32": x32.reshape(len(x32), -1)}
    out = {"no_vistos": int(test.sum()) + int(ex.sum()), "test": int(test.sum()), "extra": int(ex.sum()), "casos": {}}
    for caso, x in casos.items():
        x = x.astype(np.float32); xw, yw = x[w], y[w]
        xp, yp = np.concatenate([xw[test], x[ex]]), np.concatenate([yw[test], y[ex]])
        filas = {}
        for n in K.TAMANOS:
            a = [float((K.predecir(K.ajustar(xw[orden[:n]], yw[orden[:n]], sem), xp) == yp).mean()) for sem in K.SEMILLAS]
            filas[n] = {"acc": round(float(np.mean(a)), 4), "acc_sd": round(float(np.std(a, ddof=1)), 4),
                        "aciertos": round(float(np.mean(a)) * len(yp), 1), "F": round(float(np.mean(a)) * len(yp) / n, 2)}
            print(f"{caso:<22} N={n:>4}: acc {filas[n]['acc']:.4f} · F {filas[n]['F']:.1f}", flush=True)
        out["casos"][caso] = filas
    (RES / "factor.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
