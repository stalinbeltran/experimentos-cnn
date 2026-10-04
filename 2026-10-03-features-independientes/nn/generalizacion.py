#!/usr/bin/env python3
"""Corrida 12: capacidad generalizadora (G1 brecha, G2 eficiencia de datos, G3 robustez a 5 transformaciones no
vistas) de cada banco de detectores con el compositor lineal posicional, y de los píxeles crudos como referencia.
Definición y criterio en instrucciones/02-criterio.md § «Corrida 12».

    python nn/generalizacion.py      → resultados/generalizacion.json + resultados/transformaciones.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import aplicar                                  # noqa: E402
import compositor as C                          # noqa: E402
import curva                                    # noqa: E402
import datos                                    # noqa: E402
import obtenedor                                # noqa: E402
import obtenedor_cnn                            # noqa: E402

RES = AQUI.parent / "resultados"
SEMILLA_T = 2027
TRANSFORMACIONES = ("desplazar", "ruido", "engrosar", "adelgazar", "ocluir")
BANCOS = {"fino": ["fino"], "grueso": ["grueso"], "fino+grueso": ["fino", "grueso"], "dig": ["dig"],
          "cae5": ["cae5"], "cae3": ["cae3"], "todos": ["fino", "grueso", "dig", "cae5", "cae3"]}


def transformar(x: np.ndarray, que: str) -> np.ndarray:
    """x (N,1,8,8) en [0,1] → transformada, determinista (semilla fija por transformación)."""
    rng = np.random.default_rng(SEMILLA_T + TRANSFORMACIONES.index(que))
    t = torch.from_numpy(x)
    if que == "desplazar":
        out = np.zeros_like(x)
        dirs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        for i, k in enumerate(rng.integers(0, 8, len(x))):
            dy, dx = dirs[k]
            src = x[i, 0, max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
            out[i, 0, max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
        return out
    if que == "ruido":
        return np.clip(x + rng.normal(0, 0.15, x.shape).astype(np.float32), 0, 1)
    if que == "engrosar":
        return (0.5 * t + 0.5 * Fn.max_pool2d(t, 3, 1, 1)).numpy()
    if que == "adelgazar":
        return (0.5 * t - 0.5 * Fn.max_pool2d(-t, 3, 1, 1)).numpy()
    if que == "ocluir":
        out = x.copy()
        for i, (r, c) in enumerate(rng.integers(0, 6, (len(x), 2))):
            out[i, 0, r:r + 3, c:c + 3] = 0
        return out
    raise ValueError(que)


@torch.no_grad()
def mapas_grupo(grupo: str, x: np.ndarray) -> np.ndarray:
    """Los mapas (N, 13, 8, 8) de un grupo sobre imágenes x (N,1,8,8) cualesquiera."""
    d = {"x": x, "y": np.zeros(len(x), np.int64), "train": np.zeros(len(x), bool)}   # aplicar.mapas los pide; no se usan
    if grupo == "fino":
        return aplicar.mapas(d, (aplicar.PESOS,))["sigma"]
    if grupo == "grueso":
        return aplicar.mapas(d, (AQUI / "pesos-c4",))["sigma"]
    xt = torch.from_numpy(x)
    if grupo == "dig":
        est = torch.load(obtenedor.PESOS / "kernels.pt", weights_only=False)
        return np.stack([obtenedor.DetectorKernel(k)(xt)[:, 0].numpy() for k in est["kernels"]], 1)
    obtenedor_cnn.usar(grupo)
    est = torch.load(obtenedor_cnn.PESOS / "autocodificador.pt", weights_only=False)
    ae = obtenedor_cnn.Autocodificador(); ae.load_state_dict(est["autocodificador"]); ae.eval()
    return obtenedor_cnn.salida(ae.enc(xt), torch.tensor(est["escalas"]))[:, est["orden"]].numpy()


def acc(x, y, entrena, evalua, sem):
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=C.LR, weight_decay=C.L2)
    xt, yt = torch.from_numpy(x[entrena]), torch.from_numpy(y[entrena])
    for _ in range(C.EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def evaluar(W, x, y):
    with torch.no_grad():
        return float((W(torch.from_numpy(x)).argmax(1).numpy() == y).mean())


def main() -> int:
    dig = datos.digitos(); x0, y, tr = dig["x"], dig["y"], dig["train"]
    orden, test = curva.reparto(y, tr)
    tr180 = np.zeros(len(y), bool); tr180[orden[:180]] = True
    assert np.array_equal(tr180, tr)
    xt = {"limpio": x0[test]} | {t: transformar(x0[test], t) for t in TRANSFORMACIONES}
    yt = y[test]
    t0 = time.time()
    # mapas: limpios para todo el dataset; transformados sólo para el test
    grupos = ["fino", "grueso", "dig", "cae5", "cae3"]
    limpio = {g: mapas_grupo(g, x0).astype(np.float32) for g in grupos}
    trans = {t: {g: mapas_grupo(g, xt[t]).astype(np.float32) for g in grupos} for t in TRANSFORMACIONES}
    # comprobación: los mapas recalculados son los mismos que usaron las corridas anteriores
    for g, suf in (("fino", ""), ("grueso", "-c4"), ("dig", "-dig"), ("cae5", "-cae5"), ("cae3", "-cae3")):
        guardado = np.load(RES / f"mapas-digitos{suf}.npz")["sigma"]
        assert np.allclose(limpio[g], guardado, atol=1e-5), f"{g}: los mapas recalculados no casan con los guardados"
    print(f"mapas listos y casados con los guardados ({time.time() - t0:.0f} s)", flush=True)
    casos = {b: g for b, g in BANCOS.items()} | {"píxeles crudos": None}
    salida = {"test": int(test.sum()), "transformaciones": TRANSFORMACIONES, "casos": {}}
    for caso, gs in casos.items():
        if gs is None:
            X = x0.reshape(len(y), -1); XT = {t: xt[t].reshape(len(yt), -1) for t in TRANSFORMACIONES}
        else:
            X = np.concatenate([limpio[g] for g in gs], 1).reshape(len(y), -1)
            XT = {t: np.concatenate([trans[t][g] for g in gs], 1).reshape(len(yt), -1) for t in TRANSFORMACIONES}
        r = {"limpio": [], "train": [], "n36": [], "n1080": []} | {t: [] for t in TRANSFORMACIONES}
        for sem in C.SEMILLAS:
            W = acc(X, y, tr, test, sem)
            r["train"].append(evaluar(W, X[tr], y[tr])); r["limpio"].append(evaluar(W, X[test], yt))
            for t in TRANSFORMACIONES:
                r[t].append(evaluar(W, XT[t], yt))
            for n in (36, 1080):
                idx = np.zeros(len(y), bool); idx[orden[:n]] = True
                r[f"n{n}"].append(evaluar(acc(X, y, idx, test, sem), X[test], yt))
        m = {k: float(np.mean(v)) for k, v in r.items()}
        fila = {"acc_train": round(m["train"], 4), "acc_test": round(m["limpio"], 4),
                "G1_brecha": round(m["train"] - m["limpio"], 4),
                "acc_n36": round(m["n36"], 4), "acc_n1080": round(m["n1080"], 4), "G2": round(m["n36"] / m["n1080"], 4),
                "acc_transformado": {t: round(m[t], 4) for t in TRANSFORMACIONES},
                "G3": {t: round(m[t] / m["limpio"], 4) for t in TRANSFORMACIONES}}
        fila["G3_medio"] = round(float(np.mean(list(fila["G3"].values()))), 4)
        salida["casos"][caso] = fila
        print(f"{caso:<15} test {fila['acc_test']:.3f} · G1 {fila['G1_brecha']:.3f} · G2 {fila['G2']:.3f} · G3 "
              + " ".join(f"{t[:4]} {fila['G3'][t]:.3f}" for t in TRANSFORMACIONES) + f" · G3 medio {fila['G3_medio']:.3f}"
              f"  [{time.time() - t0:.0f} s]", flush=True)
    (RES / "generalizacion.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    dibujar(xt)
    return 0


def dibujar(xt):
    """Las 5 transformaciones sobre 8 dígitos del test, para VER qué se pidió generalizar."""
    from PIL import Image, ImageDraw                    # noqa: PLC0415
    esc, sep, etq = 6, 4, 70; t = 8 * esc; filas = ["limpio", *TRANSFORMACIONES]
    im = Image.new("L", (etq + 8 * (t + sep), len(filas) * (t + sep) + sep), 255); d = ImageDraw.Draw(im)
    for r, f in enumerate(filas):
        d.text((4, r * (t + sep) + t // 2), f, fill=0)
        for c in range(8):
            im.paste(Image.fromarray(((1 - xt[f][c * 37, 0]) * 255).astype(np.uint8)).resize((t, t), Image.NEAREST),
                     (etq + c * (t + sep), sep + r * (t + sep)))
    im.save(RES / "transformaciones.png")


if __name__ == "__main__":
    raise SystemExit(main())
