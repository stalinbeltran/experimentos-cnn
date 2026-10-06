#!/usr/bin/env python3
"""¿DÓNDE falla el compositor, más allá del grosor? Diagnóstico SIN entrenar nada (la iteración 4 de 02-criterio.md):

  D1  confusiones: los pares de clases que más se confunden (val 1617; compositor 180, las 3 semillas sumadas) y el error
      por clase
  D2  referencias SIN detectores, con el MISMO compositor: los píxeles crudos (32×32 → 1024) y promediados a 8×8 (64). Es
      lo que dice cuánto aportan los detectores, y si el techo es de los detectores o del compositor
  D3  el error por tercil de tres propiedades geométricas del dígito, medidas en sus píxeles:
        inclinación   |α| = |cov(fila, columna) / var(fila)|: columnas que se desplaza la tinta por cada fila (0 = derecho)
        descentrado   distancia del centro de masas al centro del lienzo (px)
        alto, ancho   de la caja que encierra la tinta (px)
        grosor, relleno   los de nn/errores.py, aquí con las 3 semillas (allí, la 1)

    python nn/diagnostico.py lineas-nada lineas-nada+norm3     → resultados/diagnostico.json
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
import errores as E                               # noqa: E402
import evaluar as V                               # noqa: E402

RES = AQUI.parent / "resultados"


def geometria(x: np.ndarray) -> dict:
    """x (N,1,32,32) 0/1 → inclinación, descentrado, alto y ancho por dígito."""
    t = (x[:, 0] > 0.5).astype(np.float64)
    fil, col = np.mgrid[:32, :32]
    m = t.sum((1, 2)).clip(1)
    f0 = (t * fil).sum((1, 2)) / m; c0 = (t * col).sum((1, 2)) / m
    vf = (t * (fil - f0[:, None, None]) ** 2).sum((1, 2)) / m
    cv = (t * (fil - f0[:, None, None]) * (col - c0[:, None, None])).sum((1, 2)) / m
    filas, cols = t.any(2), t.any(1)
    alto = np.array([np.ptp(np.flatnonzero(r)) + 1 if r.any() else 0 for r in filas])
    ancho = np.array([np.ptp(np.flatnonzero(c)) + 1 if c.any() else 0 for c in cols])
    return {"inclinacion": np.abs(cv / vf.clip(1e-9)), "alfa": cv / vf.clip(1e-9),
            "descentrado": np.hypot(f0 - 15.5, c0 - 15.5), "alto": alto.astype(float), "ancho": ancho.astype(float)}


def vectores(combo: str, x: np.ndarray) -> np.ndarray:
    if combo == "pixeles":
        return x.reshape(len(x), -1)
    if combo == "pixeles8":
        return Fn.avg_pool2d(torch.from_numpy(x), 4).numpy().reshape(len(x), -1)
    nombre, prepro = combo.rsplit("-", 1)
    bancos = [V.banco(n) for n in nombre.split("+")]
    return np.concatenate([V.mapas(b, V.preparar(x, b["rep"], vi)) for b in bancos for vi in prepro.split("+")], 1).reshape(len(x), -1)


def main(combos: list[str]) -> int:
    torch.set_num_threads(2)
    dg = V._digitos(); w = dg["w"]; x = dg["x"][w]; y = dg["y"][w]; tr = dg["train"][w]; yv = y[~tr]
    g = {k: v[~tr] for k, v in (geometria(x) | {k: v for k, v in E.propiedades(x).items() if k != "huecos"}).items()}
    out = {"propiedades": {k: {"terciles": [round(float(q), 3) for q in np.percentile(v, [33.3, 66.7])],
                               "mediana": round(float(np.median(v)), 3)} for k, v in g.items() if k != "alfa"},
           "combos": {}}
    for combo in ["pixeles", "pixeles8"] + combos:
        s = vectores(combo, x).astype(np.float32)
        preds = [V.ajustar(s[tr], y[tr], sem)(torch.from_numpy(s[~tr])).argmax(1).numpy() for sem in (1, 2, 3)]
        conf = np.zeros((10, 10), int)
        for p in preds:
            np.add.at(conf, (yv, p), 1)
        err = np.stack([p != yv for p in preds]).mean(0)                 # fracción de semillas que fallan cada dígito
        pares = {}
        for a in range(10):
            for b in range(a + 1, 10):
                pares[f"{a}↔{b}"] = int(conf[a, b] + conf[b, a])
        tot = int(conf.sum() - np.trace(conf))
        top = sorted(pares.items(), key=lambda kv: -kv[1])[:8]
        fila = {"acierto": round(float(1 - err.mean()), 4), "errores_3_semillas": tot,
                "pares": [{"par": k, "n": n, "frac": round(n / max(1, tot), 3)} for k, n in top],
                "error_por_clase": {str(c): round(float(err[yv == c].mean()), 4) for c in range(10)},
                "confusion": conf.tolist()}
        for k in ("inclinacion", "descentrado", "alto", "ancho", "grosor", "relleno"):
            t = np.digitize(g[k], out["propiedades"][k]["terciles"])
            fila[f"error_por_tercil_de_{k}"] = [round(float(err[t == i].mean()), 4) if (t == i).any() else None for i in range(3)]
        out["combos"][combo] = fila
        print(f"  {combo:<20} acierto {fila['acierto']:.4f} · pares " + ", ".join(f"{p['par']} {p['n']}" for p in fila["pares"][:5])
              + " · error por clase " + " ".join(f"{c}:{v:.3f}" for c, v in fila["error_por_clase"].items()), flush=True)
        print("  " + " " * 20 + " por tercil: " + " · ".join(f"{k} {fila[f'error_por_tercil_de_{k}']}" for k in
                                                         ("inclinacion", "descentrado", "grosor", "relleno")), flush=True)
    print("  terciles: " + " · ".join(f"{k} {v['terciles']} (mediana {v['mediana']})" for k, v in out["propiedades"].items()))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "diagnostico.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
