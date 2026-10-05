#!/usr/bin/env python3
"""Corrida 14 (2026-10-05, pedida para comparar con `feat-ind32`): SÓLO el banco `fino` (los 13 detectores de la corrida
2), con exactamente lo que midió `feat-ind32` a 32×32 — curva por N sobre el test de 717, compositor B (máx 3×3) con
dígitos desplazados, y los 3823 dígitos de otros escritores (de `uci-optdigits-orig-32px-r20261005`, reducidos aquí
a 8×8 contando bloques 4×4, que es como se hicieron los de 8 px).

    python nn/curva_fino.py      → resultados/curva-fino.json
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
import desplazamiento as D                      # noqa: E402
import generalizacion as G                      # noqa: E402

RES = AQUI.parent / "resultados"
ORIG32 = "uci-optdigits-orig-32px-r20261005"


def acc(x, y, idx, test, sem):
    return C.logistica(x[idx], y[idx], x[test], y[test], sem)["acc_val"]


def main() -> int:
    m = dict(np.load(RES / "mapas-digitos.npz")); s, y, tr = m["sigma"], m["y"], m["train"]
    orden, test = curva.reparto(y, tr)
    x = s.reshape(len(s), -1).astype(np.float32)
    out = {"banco": "fino (13 detectores, corrida 2)", "test": int(test.sum()), "curva": {}, "con_extra": {}, "desplazado": {}}
    for n in curva.TAMANOS:
        a = [acc(x, y, orden[:n], test, sem) for sem in C.SEMILLAS]
        out["curva"][n] = round(float(np.mean(a)), 4); print(f"N={n:>4}: {out['curva'][n]:.4f}", flush=True)
    # otros escritores, reducidos a 8×8
    z = np.load(exigir_dataset(ORIG32) / "datos.npz")
    e = z["particion"] == "extra"
    xe8 = (z["imagenes"][e].reshape(-1, 8, 4, 8, 4).sum((2, 4)).astype(np.float32) / 16)[:, None]
    se = aplicar.mapas({"x": xe8, "y": z["etiquetas"][e], "train": np.zeros(e.sum(), bool)})["sigma"]
    xe, ye = se.reshape(len(se), -1).astype(np.float32), z["etiquetas"][e].astype(np.int64)
    for nombre, (xa, ya) in {"extra_solo": (xe, ye), "1080+extra": (np.concatenate([x[orden[:1080]], xe]), np.concatenate([y[orden[:1080]], ye]))}.items():
        a = [C.logistica(xa, ya, x[test], y[test], sem)["acc_val"] for sem in C.SEMILLAS]
        out["con_extra"][nombre] = round(float(np.mean(a)), 4); print(f"{nombre}: {out['con_extra'][nombre]:.4f}", flush=True)
    # A y B, limpio y desplazado (la transformación de la corrida 12)
    dig_x = aplicar.datos.digitos()["x"]
    sd = G.mapas_grupo("fino", G.transformar(dig_x[test], "desplazar")).astype(np.float32)
    for v in ("A_posicional", "B_max3"):
        Xtr, Xte, Xd = D.preparar(s[tr], v), D.preparar(s[test], v), D.preparar(sd, v)
        r = {"limpio": [], "desplazar": []}
        for sem in C.SEMILLAS:
            W = G.acc(Xtr, y[tr], np.ones(int(tr.sum()), bool), None, sem)
            r["limpio"].append(G.evaluar(W, Xte, y[test])); r["desplazar"].append(G.evaluar(W, Xd, y[test]))
        out["desplazado"][v] = {k: round(float(np.mean(a)), 4) for k, a in r.items()}
        print(v, out["desplazado"][v], flush=True)
    (RES / "curva-fino.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
