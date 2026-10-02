#!/usr/bin/env python3
"""El informe de `bor-ae`: aplica la seccion B de `instrucciones/02-criterio.md` TAL CUAL.

    python nn/informe.py     lee nn/pesos/*/best.pt, escribe resultados/informe.json,
                             resultados/INFORME.md y resultados/kernels.png, e imprime los
                             brazos que VAN al banco

QUE DECIDE: un brazo va al banco si su codificador final NO es una delta (`delta <= 0,5`).
Una delta es la condicion `identidad` del banco con otro nombre: importarla no mide nada.
Se reporta, por brazo y sobre `eval`, R^2 de la reconstruccion, `delta` y codigo activo.
No se declara ganador: la utilidad la dice el banco.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import datos
from entrenar_local import LAMBDA, medir, varianza_r
from modelo import BRAZOS, construir, delta

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RESULTADOS = EXP / "resultados"
MAX_DELTA = 0.5


def main() -> int:
    X = torch.from_numpy(datos.ventanas("eval")[0])
    var_r = varianza_r(X)
    brazos = sorted(b for b in BRAZOS if (PESOS / b / "best.pt").is_file())
    if not brazos:
        print("✗ no hay ningun brazo entrenado en nn/pesos/")
        return 1
    filas, kernels = [], {}
    for b in brazos:
        k, s = BRAZOS[b]
        ck = torch.load(PESOS / b / "best.pt", map_location="cpu", weights_only=False)
        red = construir(k, s)
        red.load_state_dict(ck["modelo"])
        red.eval()
        m = medir(red, X, float(LAMBDA or 0.0), var_r)
        kernels[b] = red.kernel().detach().numpy().copy()
        filas.append({"brazo": b, "k": k, "semilla": s, "epoca_best": int(ck.get("epoca") or 0),
                      "r2": round(m["r2"], 4), "delta": round(m["delta"], 3),
                      "activa": round(m["activa"], 4), "al_banco": bool(m["delta"] <= MAX_DELTA)})
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "informe.json").write_text(json.dumps(
        {"criterio": "instrucciones/02-criterio.md, seccion B", "particion": "eval",
         "lambda": LAMBDA, "max_delta": MAX_DELTA, "brazos": filas}, indent=1) + "\n",
        encoding="utf-8")
    L = ["# `bor-ae` — informe (generado por `nn/informe.py`; no se transcribe nada a mano)", "",
         f"`λ` = {LAMBDA} (del tanteo, `resultados/tanteo-lambda.json`). Sobre **`eval`**. Va al "
         f"banco el brazo cuyo codificador **no** es una delta (`delta ≤ {MAX_DELTA}`).", "",
         "| brazo | época | R² | delta | código activo | ¿al banco? |", "|---|---:|---:|---:|---:|---|"]
    for f in filas:
        L.append(f"| {f['brazo']} | {f['epoca_best']} | {f['r2']:.3f} | {f['delta']:.2f} | "
                 f"{100 * f['activa']:.1f} % | {'✅ sí' if f['al_banco'] else '✗ no (delta)'} |")
    n = sum(f["al_banco"] for f in filas)
    L += ["", f"**{n} de {len(filas)}** brazos van al banco.", "", "![los kernels](kernels.png)", ""]
    (RESULTADOS / "INFORME.md").write_text("\n".join(L), encoding="utf-8")
    _figura(kernels)
    print("\n".join(L))
    print("al banco:", " ".join(f["brazo"] for f in filas if f["al_banco"]) or "(ninguno)")
    return 0


def _figura(kernels: dict) -> None:
    from PIL import Image                                   # noqa: PLC0415
    ks = sorted({BRAZOS[b][0] for b in kernels})
    ss = sorted({BRAZOS[b][1] for b in kernels})
    celda, pad = 19 * 6, 6
    hoja = Image.new("L", (len(ks) * (celda + pad) + pad, len(ss) * (celda + pad) + pad), 255)
    for b, w in kernels.items():
        k, s = BRAZOS[b]
        g = 127.5 + 127.5 * w / max(1e-9, float(np.abs(w).max()))
        im = Image.fromarray(g.round().astype(np.uint8)).resize((celda * k // 19,) * 2, Image.NEAREST)
        hoja.paste(im, (pad + ks.index(k) * (celda + pad) + (celda - im.width) // 2,
                        pad + ss.index(s) * (celda + pad) + (celda - im.height) // 2))
    hoja.save(RESULTADOS / "kernels.png")


if __name__ == "__main__":
    raise SystemExit(main())
