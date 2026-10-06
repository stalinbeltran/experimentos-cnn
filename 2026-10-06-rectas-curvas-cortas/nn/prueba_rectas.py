#!/usr/bin/env python3
"""La prueba «a mano» del README (§3): rectas dibujadas sin ruido, de largo L y grosor g, centradas, horizontales y
verticales, y qué detectores cortos se encienden (máximo del mapa ≥ su umbral). Sirve para dos preguntas: ¿se enciende la
recta corta sobre una recta LARGA? y ¿una recta GRUESA enciende curvas cortas?

    python nn/prueba_rectas.py      → la tabla por pantalla y resultados/prueba-rectas.json
"""
from __future__ import annotations

import json

import numpy as np

import evaluar as V


def main() -> int:
    fams = list(V.F.CON_TRAZO)
    reds = V.cargar_banco(V.PESOS, fams)
    um = {f: json.loads((V.PESOS / f / "summary.json").read_text(encoding="utf-8"))["best"]["umbral"] for f in fams}
    casos = {}
    for g in (2, 3, 4, 5, 6, 8):
        for L in (10, 20):
            h = np.zeros((32, 32), np.float32); h[16 - g // 2:16 - g // 2 + g, 16 - L // 2:16 - L // 2 + L] = 1
            v = np.zeros((32, 32), np.float32); v[16 - L // 2:16 - L // 2 + L, 16 - g // 2:16 - g // 2 + g] = 1
            casos[f"H g{g} L{L}"], casos[f"V g{g} L{L}"] = h, v
    x = np.stack(list(casos.values()))[:, None]
    s = V.mapas(reds, x).reshape(len(x), len(fams), -1).max(2)
    out = {"umbrales": um, "casos": {}}
    print(" " * 10 + " ".join(f"{f:>8}" for f in fams))
    for i, k in enumerate(casos):
        out["casos"][k] = {f: {"max": round(float(s[i, j]), 3), "enciende": bool(s[i, j] >= um[f])} for j, f in enumerate(fams)}
        print(f"{k:<10}" + " ".join((("ON " if s[i, j] >= um[f] else " - ") + f"{s[i, j]:.2f}").rjust(8) for j, f in enumerate(fams)))
    (V.RES / "prueba-rectas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
