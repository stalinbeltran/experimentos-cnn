#!/usr/bin/env python3
"""S5 de la iteración 4: el compositor TOLERANTE a la posición. Los mismos mapas de una combinación, con el mismo compositor
lineal, en tres variantes:

  exacto   los mapas 8×8 tal cual (lo de siempre)
  max3     cada mapa pasa por un máximo 3×3 de paso 1 (sigue siendo 8×8): una feature corrida una celda cae en el mismo peso
  media3   lo mismo con la media (se mira, no decide: el criterio se escribió para max3)

Compositor de 180 (3 semillas) y la curva 36 / 180 / 1080 sobre el test fijo de 717, como en nn/evaluar.py.

    python nn/tolerante.py lineas-nada lineas-nada+norm3     → resultados/tolerante.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
import evaluar as V                               # noqa: E402

RES = AQUI.parent / "resultados"


def variante(s: np.ndarray, que: str) -> np.ndarray:
    t = torch.from_numpy(s)
    if que == "max3":
        t = Fn.max_pool2d(t, 3, 1, 1)
    elif que == "media3":
        t = Fn.avg_pool2d(t, 3, 1, 1, count_include_pad=False)
    elif que != "exacto":
        raise ValueError(que)
    return t.numpy().reshape(len(s), -1)


def main(combos: list[str]) -> int:
    torch.set_num_threads(2)
    dg = V._digitos(); w = dg["w"]; x = dg["x"][w]; y = dg["y"][w]; tr = dg["train"][w]
    orden, test = V.reparto(y, tr)
    ruta = RES / "tolerante.json"
    out = json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else {}
    for combo in combos:
        nombre, prepro = combo.rsplit("-", 1)
        bancos = [V.banco(n) for n in nombre.split("+")]
        s = np.concatenate([V.mapas(b, V.preparar(x, b["rep"], vi)) for b in bancos for vi in prepro.split("+")], 1)
        fila = {}
        for que in ("exacto", "max3", "media3"):
            xm = variante(s, que).astype(np.float32)
            c = V.acierto(xm[tr], y[tr], xm[~tr], y[~tr])
            fila[que] = {"compositor_180": V.r4(np.mean(c)), "por_semilla": c,
                         "curva_717": {str(n): V.r4(np.mean(V.acierto(xm[orden[:n]], y[orden[:n]], xm[test], y[test])))
                                       for n in V.TAMANOS}}
            print(f"  {combo:<22} {que:<7} compositor {fila[que]['compositor_180']:.4f} · curva "
                  + "/".join(f"{fila[que]['curva_717'][str(n)]:.3f}" for n in V.TAMANOS), flush=True)
        e, m = fila["exacto"], fila["max3"]
        fila["S5a"] = {"subida_36": V.r4(m["curva_717"]["36"] - e["curva_717"]["36"]),
                       "veredicto": "confirmada" if m["curva_717"]["36"] - e["curva_717"]["36"] >= 0.02 else "refutada"}
        fila["S5b"] = {"cambio_180": V.r4(m["compositor_180"] - e["compositor_180"]),
                       "veredicto": "confirmada" if m["compositor_180"] >= e["compositor_180"] - 0.002 else "refutada"}
        print(f"  {combo:<22} S5a {fila['S5a']} · S5b {fila['S5b']}", flush=True)
        out[combo] = fila
        RES.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
