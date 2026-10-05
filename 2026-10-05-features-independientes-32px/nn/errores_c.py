#!/usr/bin/env python3
"""Los dígitos que FALLA C (detectores sintéticos + ajuste fino), pedido del dueño el 2026-10-05. En Vast sólo se guardó el
resumen de cada entrenamiento, así que C se REENTRENA aquí con el mismo código y las mismas semillas (curvas_cnn.entrenar_red)
para el caso del modelo final —T = 4000, p = 50 %: N = 2000 de train y 2000 nuevos— y se comprueba que el acierto coincide
con el de Vast antes de usar sus errores.

Por cada error: el bitmap de 32×32 (lo que vería una persona), el 8×8 que ve la red, real → predicho y la confianza.

    python nn/errores_c.py [--T 4000 --p 0.5]   → resultados/errores-c.json y resultados/errores-c.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import curvas_cnn as K                          # noqa: E402
import datos                                    # noqa: E402

RES = AQUI.parent / "resultados"
SUP, T1, T2 = "#fcfcfb", "#0b0b0b", "#52514e"
MODELO = "ajuste13"


def _errores(args) -> dict:
    T, p, sem = args
    red, tr, te, extra, _ = K.entrenar_red(MODELO, T, p, sem)
    d = K.dato(); x8, y = d[8], d["y"]
    red.eval()
    with torch.no_grad():
        prob = torch.softmax(torch.cat([red(x8[te][i:i + 512]) for i in range(0, len(te), 512)]), 1).numpy()
    pred = prob.argmax(1); real = y[te]
    vast = json.loads(K.ruta(MODELO, T, p, sem).read_text())["acc"] if K.ruta(MODELO, T, p, sem).exists() else None
    malos = np.flatnonzero(pred != real)
    return {"sem": sem, "n_test": int(len(te)), "acc_local": round(float((pred == real).mean()), 4), "acc_vast": vast,
            "errores": [{"indice": int(te[i]), "real": int(real[i]), "predicho": int(pred[i]),
                         "p_predicho": round(float(prob[i, pred[i]]), 3), "p_real": round(float(prob[i, real[i]]), 3)}
                        for i in malos]}


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--T", type=int, default=4000); a.add_argument("--p", type=float, default=0.5)
    x = a.parse_args()
    with Pool(3) as pool:
        por_sem = pool.map(_errores, [(x.T, x.p, s) for s in (1, 2, 3)])
    origen = datos.digitos(con_extra=True)["origen"]
    for r in por_sem:
        for e in r["errores"]:
            e["origen"] = str(origen[e["indice"]])
        igual = r["acc_vast"] is not None and abs(r["acc_local"] - r["acc_vast"]) < 1e-9
        print(f"semilla {r['sem']}: acierto aquí {r['acc_local']:.4f} · en Vast {r['acc_vast']} → "
              f"{'el MISMO modelo' if igual else '⚠ NO coincide: los errores son de un modelo parecido, no del de Vast'} · "
              f"{len(r['errores'])} errores de {r['n_test']}")
    pares = Counter((e["real"], e["predicho"]) for r in por_sem for e in r["errores"])
    out = {"modelo": "C · detectores sintéticos + ajuste fino", "T": x.T, "p": x.p, "semillas": por_sem,
           "pares": {f"{a}→{b}": n for (a, b), n in pares.most_common()}}
    (RES / "errores-c.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("pares real→predicho (las 3 semillas):", ", ".join(f"{a}→{b} ×{n}" for (a, b), n in pares.most_common()))
    dibujar(por_sem, pares, x)
    return 0


def dibujar(por_sem: list, pares: Counter, x) -> None:
    d = K.dato(); x32, x8 = d[32].numpy()[:, 0], d[8].numpy()[:, 0]
    errores = [(r["sem"], e) for r in por_sem for e in r["errores"]]
    orden = {par: i for i, (par, _) in enumerate(pares.most_common())}       # los pares más frecuentes primero
    errores.sort(key=lambda se: (orden[(se[1]["real"], se[1]["predicho"])], -se[1]["p_predicho"]))
    cols = 10; filas = int(np.ceil(len(errores) / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(cols * 1.75, filas * 1.25 + 1.3), facecolor=SUP)
    for ax in np.ravel(ejes):
        ax.axis("off")
    for ax, (sem, e) in zip(np.ravel(ejes), errores):
        i = e["indice"]
        img = np.ones((32, 32 + 4 + 32))                                     # 32×32 | hueco | 8×8 ampliado ×4
        img[:, :32] = 1 - x32[i]
        img[:, 36:] = 1 - np.kron(x8[i], np.ones((4, 4)))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{e['real']}→{e['predicho']}  {e['p_predicho']:.2f}", fontsize=10, color=T1, pad=2)
        ax.text(0.5, -0.06, f"s{sem} · {e['origen']} · #{i}", transform=ax.transAxes, ha="center", va="top", fontsize=7, color=T2)
    total = sum(len(r["errores"]) for r in por_sem); n = sum(r["n_test"] for r in por_sem)
    # cada semilla saca SU dataset del mismo pool: un dígito puede estar en el test de varias y fallar en varias
    veces = Counter(e["indice"] for r in por_sem for e in r["errores"])
    distintos, repetidos = len(veces), sum(1 for v in veces.values() if v > 1)
    fig.suptitle(f"Los {total} fallos de C (detectores sintéticos + ajuste fino) con N = {int(x.T * x.p)} de train · {n} dígitos "
                 f"nuevos en 3 semillas ({100 * (1 - total / n):.1f} % de acierto) · son {distintos} dígitos distintos: {repetidos} fallan "
                 f"en más de una semilla (mismo #)\ncada caso: el 32×32 original | el 8×8 que ve la red · real→predicho y la confianza · "
                 f"los más frecuentes: "
                 + " · ".join(f"{a}→{b} ×{c}" for (a, b), c in pares.most_common(8)),
                 x=0.01, ha="left", fontsize=10.5, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (filas * 1.25 + 1.3)))
    fig.savefig(RES / "errores-c.png", dpi=110, facecolor=SUP)
    print(f"→ {RES / 'errores-c.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
