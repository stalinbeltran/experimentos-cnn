#!/usr/bin/env python3
"""El dataset: EL MISMO fichero publicado, con las etiquetas colapsadas.

    python nn/datos.py --comprobar     ¿esta el publicado y es el del manifiesto?

NO GENERA NADA, Y ESO ES EL PUNTO
    Este experimento NO rinde ni una imagen: lee el dataset publicado
    `esquinas300-32px-r4-r20260907` del repo de datos, el MISMO que uso `esq-2d`.
    Es exactamente para lo que aquel se etiqueto con LAS CUATRO esquinas en vez
    de con la diagonal que medía: para que el siguiente no tuviera que re-rendir.

    Consecuencia que hay que aprovechar al leer los resultados: los dos
    experimentos se comparan sobre las MISMAS ventanas, no sobre dos muestras
    equivalentes. Cualquier diferencia es de la tarea o de la red, nunca del dato.

LA ETIQUETA, COLAPSADA
    Por orden del dueno (2026-09-07): «no queremos diferenciar las esquinas entre
    si. Si detecta una esquina, cualquiera (por ahora limitadas a las mismas tl y
    br) el resultado es valido».

        existe = existe_tl OR existe_br
        (x, y) = la de la que exista

    ⚠ Y no hay ambiguedad posible en el "OR", cosa que NO se supone: el propio
    manifiesto del dataset cuenta que NINGUNA ventana contiene mas de una esquina
    (el parrafo mide >= 64 px reducidos y la ventana 32). Si algun dia eso dejara
    de ser cierto, `cargar()` se niega en vez de quedarse con una de las dos.

    ⚠ `tr` y `bl` siguen siendo NEGATIVOS. El encargo dice "por ahora limitadas a
    las mismas tl y br", asi que una ventana con la esquina superior-derecha es un
    negativo duro -- y el mejor que hay, porque tiene la misma forma local girada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from expcnn import exigir_dataset, ruta_dataset

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent

DATASET = "esquinas300-32px-r4-r20260907"
PARTES = ("train", "val", "muestra")
CONGELADAS = "muestras-congeladas.npz"
BUSCADAS = ("tl", "br")          # las que cuentan como positivo
NEGATIVAS = ("tr", "bl")         # etiquetadas en el dataset, pero negativas aqui


def _colapsar(z) -> dict:
    """(existe, x, y) de CUALQUIERA de las buscadas."""
    n = len(z["clase"])
    existe = np.zeros(n, np.int64)
    x = np.full(n, -1.0, np.float32)
    y = np.full(n, -1.0, np.float32)
    cuantas = sum(z[f"existe_{c}"] for c in BUSCADAS)
    if int(cuantas.max(initial=0)) > 1:
        raise SystemExit(
            f"✗ {(cuantas > 1).sum()} ventana(s) contienen MAS DE UNA de {BUSCADAS}. "
            "El colapso a una sola etiqueta seria ambiguo y aqui se niega en vez de "
            "quedarse con una: el manifiesto dice que no puede pasar, y si pasa, lo "
            "que cambio es el dataset.")
    for c in BUSCADAS:
        hay = z[f"existe_{c}"] == 1
        existe[hay] = 1
        x[hay] = z[f"x_{c}"][hay]
        y[hay] = z[f"y_{c}"][hay]
    return {"ventanas": z["ventanas"], "existe": existe, "x": x, "y": y,
            "clase": z["clase"], "imagen": z["imagen"]}


def cargar(parte: str) -> dict:
    """Una particion del dataset publicado, ya colapsada."""
    return _colapsar(np.load(exigir_dataset(DATASET) / f"{parte}.npz", allow_pickle=True))


def congeladas() -> dict:
    return _colapsar(np.load(exigir_dataset(DATASET) / CONGELADAS, allow_pickle=True))


def comprobar() -> int:
    """¿Esta el publicado, es el del manifiesto, y que sale al colapsar?"""
    pub = ruta_dataset(DATASET)
    if pub is None:
        print(f"✗ '{DATASET}' no esta publicado en el repo de datos.")
        print("  Lo produce el experimento `esq-2d`: python nn/datos.py --imagenes 300 --publicar")
        return 1
    man = json.loads((pub / "manifiesto.json").read_text())
    print(f"publicado en {pub}\n")
    ok = True
    for nombre, esp in man["particiones"].items():
        h = hashlib.sha256((pub / f"{nombre}.npz").read_bytes()).hexdigest()[:16]
        ok &= h == esp["sha256_16"]
        print(f"  {nombre:>8}: {h} contra {esp['sha256_16']} … "
              f"{'igual' if h == esp['sha256_16'] else 'DISTINTO'}")
    print()
    for parte in PARTES:
        d = cargar(parte)
        n, pos = len(d["existe"]), int(d["existe"].sum())
        clases = {}
        for c, e in zip(d["clase"], d["existe"]):
            if e == 1:
                clases[str(c)] = clases.get(str(c), 0) + 1
        print(f"  {parte:>8}: {n:>5} ventanas · {pos} positivas ({100*pos/n:.1f} %) · "
              + " ".join(f"{k}={v}" for k, v in sorted(clases.items())))
    d = congeladas()
    print(f"  {'muestras':>8}: {len(d['existe'])} congeladas · {int(d['existe'].sum())} positivas")
    print("\n✓ el dataset publicado es el del manifiesto y el colapso es univoco."
          if ok else "\n✗ el publicado NO coincide: no entrenes hasta aclararlo.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comprobar", action="store_true")
    p.parse_args()
    return comprobar()


if __name__ == "__main__":
    raise SystemExit(main())
