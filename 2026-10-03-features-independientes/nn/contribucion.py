#!/usr/bin/env python3
"""Contribución de cada detector (dato curioso, no decide nada): quitar su mapa y re-entrenar el compositor
posicional (semilla 1). contribución = acierto con todos − acierto sin él. Positivo = ayudaba.

    python nn/contribucion.py --sufijo -c24dig     → resultados/contribucion-c24dig.json
    python nn/contribucion.py --sufijo -dig
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compositor as C                          # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"


def main() -> int:
    suf = sys.argv[sys.argv.index("--sufijo") + 1]
    m = dict(np.load(RES / f"mapas-digitos{suf}.npz"))
    s, y, tr = m["sigma"].astype(np.float32), m["y"], m["train"]
    nombres = [str(n) for n in m["nombres"]]
    J = s.shape[1]; t0 = time.time()

    def acc(cols):
        x = s[:, cols].reshape(len(y), -1)
        return C.logistica(x[tr], y[tr], x[~tr], y[~tr], 1)["acc_val"]

    base = acc(list(range(J)))
    filas = []
    for j in range(J):
        a = acc([k for k in range(J) if k != j])
        filas.append({"detector": nombres[j], "acc_sin": a, "contribucion": round(base - a, 4),
                      "digitos": int(round((base - a) * int((~tr).sum())))})
    filas.sort(key=lambda f: -f["contribucion"])
    out = {"sufijo": suf, "acc_todos": base, "n_val": int((~tr).sum()), "segundos": round(time.time() - t0, 1), "detectores": filas}
    (RES / f"contribucion{suf}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{suf}: acierto con los {J} = {base:.4f}  ({out['segundos']} s)")
    for f in filas:
        print(f"  {f['detector']:<20} {f['contribucion']:+.4f}  ({f['digitos']:+d} dígitos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
