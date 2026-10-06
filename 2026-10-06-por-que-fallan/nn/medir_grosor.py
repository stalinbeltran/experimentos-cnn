#!/usr/bin/env python3
"""¿Cuánto más gruesos son los dígitos que las features con que se entrenaron los detectores? Grosor medio de un trazo =
píxeles de tinta ÷ píxeles de su esqueleto (área ÷ longitud). En los dígitos de NIST (5620, 32×32) y en las features
sintéticas de entrenamiento (`feat-ind32-sinteticas-32px-r20261005`, dibujadas a 2, 3 y 4 px).

    python nn/medir_grosor.py     → resultados/grosor-digitos.json y resultados/normalizacion.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset          # noqa: E402
import normalizar as N                       # noqa: E402

RES = AQUI.parent / "resultados"


def grosor(x: np.ndarray) -> np.ndarray:
    x = (x > 0).astype(np.uint8); e = N.esqueleto(x)
    return x.reshape(len(x), -1).sum(1) / np.maximum(1, e.reshape(len(e), -1).sum(1))


def main() -> int:
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    xd, y = d["imagenes"], d["etiquetas"]
    s = np.load(exigir_dataset("feat-ind32-sinteticas-32px-r20261005") / "datos.npz")
    vacio = 13
    m = (s["principal"] != vacio) & (s["secundaria"] < 0)                   # una sola feature, con trazo
    gd, gs = grosor(xd), grosor(s["imagenes"][m])
    pct = lambda v: {str(q): round(float(np.percentile(v, q)), 2) for q in (5, 25, 50, 75, 95)}
    out = {"digitos": {"n": int(len(gd)), "percentiles": pct(gd),
                       "por_clase_mediana": {str(c): round(float(np.median(gd[y == c])), 2) for c in range(10)}},
           "sinteticas_entrenamiento": {"n": int(len(gs)), "percentiles": pct(gs),
                                        "por_grosor_dibujado_mediana": {str(w): round(float(np.median(gs[s["grosor"][m] == w])), 2)
                                                                        for w in (2, 3, 4)}}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "grosor-digitos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    # la figura: 12 dígitos, su versión normalizada, y la normalizada de su versión engrosada y adelgazada
    import torch, torch.nn.functional as Fn                                     # noqa: E401
    from PIL import Image, ImageDraw
    idx = np.random.default_rng(3).choice(len(xd), 12, replace=False)
    x = xd[idx].astype(np.float32)[:, None]
    t = torch.from_numpy(x)
    filas = {"original": x[:, 0], "normalizado (3 px)": N.normalizar(x)[:, 0],
             "engrosado 2 px y normalizado": N.normalizar(Fn.max_pool2d(t, 5, 1, 2).numpy())[:, 0],
             "adelgazado 1 px y normalizado": N.normalizar((-Fn.max_pool2d(-t, 3, 1, 1)).numpy())[:, 0]}
    lado, sep, txt = 64, 4, 230
    im = Image.new("L", (txt + 12 * (lado + sep), len(filas) * (lado + sep) + sep), 255); dr = ImageDraw.Draw(im)
    for r, (nombre, a) in enumerate(filas.items()):
        dr.text((4, sep + r * (lado + sep) + lado // 2 - 6), nombre, fill=0)
        for j in range(12):
            im.paste(Image.fromarray(((1 - a[j]) * 255).astype(np.uint8)).resize((lado, lado), Image.NEAREST),
                     (txt + j * (lado + sep), sep + r * (lado + sep)))
    im.save(RES / "normalizacion.png", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
