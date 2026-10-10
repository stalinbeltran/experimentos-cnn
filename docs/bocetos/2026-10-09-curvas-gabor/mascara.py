#!/usr/bin/env python3
"""¿Por qué el detector DELGADO marca en gris píxeles FUERA del dígito? (observado por el dueño en la figura 23-2,
2026-10-10). Y qué cuesta cada arreglo.

El gris de las figuras es la MÁSCARA DE TRAZO del detector: los píxeles donde la energía de algún Gabor par supera TAU,
pero donde el giro no se pudo medir. Con TAU 0,3 (la calibración de `delgadas.py`) se sale de la tinta por dos vías:
los EXTREMOS del trazo (el Gabor es alargado y sigue respondiendo 3–4 px más allá de donde acaba la línea) y el LADO de
trazos oblicuos, curvas y cruces (un Gabor de otra orientación cruza el trazo y responde algo desde fuera).

Se comparan, con TODO lo demás igual: TAU 0,3 (el de ahora) · TAU 0,6 (empataba exacto en la calibración) · TAU 0,9 ·
TAU 0,3 con la máscara recortada a la tinta. Para cada uno: % de la máscara fuera de la tinta (300 dígitos de val), las
figuras sintéticas delgadas de prueba de `delgadas.py` y los dígitos con el compositor de siempre.

    python mascara.py   → resultados-mascara.json  (~5 min)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import delgadas as DL                                               # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

VARIANTES = (("TAU 0,3 (la de delgadas.py)", None, False), ("TAU 0,6 (empataba)", 0.6, False),
             ("TAU 0,9", 0.9, False), ("TAU 0,3 + máscara ∩ tinta", None, True))


def main() -> int:
    torch.set_num_threads(2)
    cal = json.loads((AQUI / "resultados-delgadas.json").read_text())["elegida"]
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, p = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = (np.flatnonzero(p == k) for k in ("train", "val", "extra"))
    pru = DL.sinteticas(300, 202)
    out = {}
    for nombre, tau, tinta in VARIANTES:
        DL.configurar(**{**cal, **({"tau": tau} if tau else {})})
        if tinta:
            base = C.campo

            def campo(x, base=base):
                c = base(x); c["Emax"] = c["Emax"] * (np.asarray(x) > 0.5); return c
            C.campo = campo
        fuera = []
        for i in va[:300]:
            x = img[i] > 0; k, m, _ = C.giro(C.campo(x.astype(np.float32)))
            fuera.append((m & ~x).sum() / max(m.sum(), 1))
        s = DL.evaluar(pru)
        X = DB.caract(img, borde=lambda x: x.astype(np.float32))["bordes"]
        v, c = [], []
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem)
            v.append(float((pr(X[va]) == y[va]).mean())); c.append(float((pr(X[ex]) == y[ex]).mean()))
        out[nombre] = {"mascara_fuera_de_la_tinta_pc": round(100 * float(np.mean(fuera)), 1), "sinteticas": s,
                       "digitos_val": round(float(np.mean(v)), 4), "digitos_ciega": round(float(np.mean(c)), 4)}
        print(nombre, out[nombre], flush=True)
    (AQUI / "resultados-mascara.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
