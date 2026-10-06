#!/usr/bin/env python3
"""¿QUÉ TIENEN los dígitos que fallan? Para la referencia (`lineas-nada`) y las combinaciones que se pidan, el compositor
(180/1617, semilla 1) y, por cada dígito de val, tres propiedades medidas en sus píxeles, SIN detectores:

  grosor    tinta ÷ esqueleto (px)
  relleno   fracción de la tinta que es interior (sus 8 vecinos también son tinta): zonas macizas, no trazo
  huecos    cuántas regiones de fondo encerradas tiene (0 en un 1 o un 7; 1 en un 0, 6 o 9; 2 en un 8) — y si coincide con lo
            que su clase suele tener (la moda de su clase): un 8 sin sus dos huecos es un 8 que el grosor cerró

y la tasa de error por tercil de cada propiedad, y por «huecos como su clase / distintos».

    python nn/errores.py lineas-nada lineas-nada+norm3 ...    → resultados/errores.json, resultados/errores.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
import evaluar as V                               # noqa: E402
import normalizar as N                            # noqa: E402

RES = AQUI.parent / "resultados"


def huecos(im: np.ndarray) -> int:
    """Regiones de fondo (vecindad-4) que no tocan el borde."""
    fondo = im == 0; vis = np.zeros_like(fondo); n = 0
    h, w = fondo.shape
    for i in range(h):
        for j in range(w):
            if fondo[i, j] and not vis[i, j]:
                pila, toca = [(i, j)], False; vis[i, j] = True
                while pila:
                    a, b = pila.pop()
                    if a in (0, h - 1) or b in (0, w - 1):
                        toca = True
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, b + db
                        if 0 <= c < h and 0 <= d < w and fondo[c, d] and not vis[c, d]:
                            vis[c, d] = True; pila.append((c, d))
                n += not toca
    return n


def propiedades(x: np.ndarray) -> dict:
    x = (x[:, 0] > 0.5).astype(np.uint8)
    e = N.esqueleto(x)
    grosor = x.reshape(len(x), -1).sum(1) / np.maximum(1, e.reshape(len(e), -1).sum(1))
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    interior = np.ones_like(x)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            interior &= p[:, di:di + 32, dj:dj + 32]
    relleno = interior.reshape(len(x), -1).sum(1) / np.maximum(1, x.reshape(len(x), -1).sum(1))
    return {"grosor": grosor, "relleno": relleno, "huecos": np.array([huecos(im) for im in x])}


def predicciones(combo: str, dg: dict) -> np.ndarray:
    nombre, prepro = combo.rsplit("-", 1)
    bancos = [V.banco(n) for n in nombre.split("+")]; w = dg["w"]
    s = np.concatenate([V.mapas(b, V.preparar(dg["x"][w], b["rep"], vi)) for b in bancos for vi in prepro.split("+")], 1).reshape(w.sum(), -1)
    tr = dg["train"][w]; y = dg["y"][w]
    W = V.ajustar(s[tr], y[tr], 1)
    return W(torch.from_numpy(s[~tr])).argmax(1).numpy()


def main(combos: list[str]) -> int:
    dg = V._digitos(); w = dg["w"]; y = dg["y"][w]; tr = dg["train"][w]; yv = y[~tr]
    pr = propiedades(dg["x"][w]); pv = {k: v[~tr] for k, v in pr.items()}
    moda = {c: int(np.bincount(pr["huecos"][y == c]).argmax()) for c in range(10)}
    como_su_clase = pv["huecos"] == np.array([moda[c] for c in yv])
    out = {"moda_de_huecos_por_clase": moda, "propiedades": {}, "combos": {}}
    for k in ("grosor", "relleno"):
        q = np.percentile(pv[k], [33.3, 66.7])
        out["propiedades"][k] = {"terciles": [round(float(v), 3) for v in q]}
    out["propiedades"]["huecos"] = {"frac_val_distintos_de_su_clase": round(float((~como_su_clase).mean()), 4)}
    for combo in combos:
        err = predicciones(combo, dg) != yv
        fila = {"error_total": round(float(err.mean()), 4), "n_errores": int(err.sum())}
        for k in ("grosor", "relleno"):
            q = out["propiedades"][k]["terciles"]
            t = np.digitize(pv[k], q)
            fila[f"error_por_tercil_de_{k}"] = [round(float(err[t == i].mean()), 4) for i in range(3)]
        fila["error_huecos_como_su_clase"] = round(float(err[como_su_clase].mean()), 4)
        fila["error_huecos_distintos"] = round(float(err[~como_su_clase].mean()), 4)
        fila["frac_de_errores_con_huecos_distintos"] = round(float((~como_su_clase)[err].mean()), 4)
        out["combos"][combo] = fila
        print(f"  {combo:<22} error {fila['error_total']:.3f} · por tercil de grosor {fila['error_por_tercil_de_grosor']} · de relleno "
              f"{fila['error_por_tercil_de_relleno']} · huecos como su clase {fila['error_huecos_como_su_clase']:.3f} / distintos "
              f"{fila['error_huecos_distintos']:.3f} ({fila['frac_de_errores_con_huecos_distintos']:.0%} de los errores)", flush=True)
    print(f"  moda de huecos por clase {moda} · {out['propiedades']['huecos']['frac_val_distintos_de_su_clase']:.1%} de val con otros huecos")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "errores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
