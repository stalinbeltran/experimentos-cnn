#!/usr/bin/env python3
"""Los datos de `feat-1lado` (COPIA del `nn/datos.py` de `feat-bor`, 2026-10-07, sin el paso a bordes ni la
generación de la prueba gruesa, que ya está publicada). Se leen por su nombre y su huella, no se regeneran:

  ENTRENAR  `feat-ind32-sinteticas-32px-r20261005`: las mismas 32.400 features de feat-ind32 y feat-bor (2–4 px).
  GRUESO    `feat-bor-sinteticas-grueso-32px-r20261006`: la prueba de grosor (2–12 px), publicada por feat-bor.
  DIGITOS   `uci-optdigits-orig-32px-r20261005`: los 5620 dígitos de NIST (windep con su reparto 180/1617).

Aquí NO se transforma nada al cargar: entra la TINTA tal cual. El único pre-proceso (los 8 bordes de un lado) es la
primera capa del modelo (nn/modelo.py), igual para todo lo que ve el detector.

    python nn/datos.py --comprobar       huellas y conjuntos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset  # noqa: E402
import features as F                                               # noqa: E402

DATASET = "feat-ind32-sinteticas-32px-r20261005"
GRUESO = "feat-bor-sinteticas-grueso-32px-r20261006"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
N_TRAIN = 2000


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _cargar(nombre: str) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{nombre}/datos.npz: '{k}' no casa con su manifiesto. Me niego.")
    d["manifiesto"] = man; d["nombre"] = nombre
    return d


def cargar(nombre: str = DATASET) -> dict:
    d = _cargar(nombre)
    if list(d["manifiesto"]["familias"]) != list(F.FAMILIAS):
        raise RuntimeError("las familias del dataset y del código no casan. Me niego.")
    return d


def conjunto(d: dict, familia: str, particion: str | None) -> dict:
    """COPIA de `conjunto` de feat-ind32. Positivos: principal == f. Negativos: principal
    != f y secundaria != f, y ni la principal ni la secundaria una familia que CONTIENE a f. `particion` None = todas (la
    prueba de grosor no tiene reparto). x (N, 1, 32, 32) uint8 0/1: la tinta; se pasa a float en cada lote."""
    i = F.FAMILIAS.index(familia)
    part = np.ones(len(d["principal"]), bool) if particion is None else d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    cont = [F.FAMILIAS.index(c) for c in F.contenedoras(familia)]
    if cont:
        neg &= ~np.isin(d["principal"], cont) & ~np.isin(d["secundaria"], cont)
    return {"x_pos": d["imagenes"][pos][:, None], "ancla_pos": d["ancla"][pos].astype(np.int64),
            "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos], "secundaria_pos": d["secundaria"][pos],
            "x_neg": d["imagenes"][neg][:, None], "familia_neg": d["principal"][neg], "grosor_neg": d["grosor"][neg]}


def digitos() -> dict:
    """Los 5620 dígitos 32×32 (windep con su reparto 180/1617; el resto `extra`). x (N,1,32,32) uint8 0/1."""
    d = _cargar(DIGITOS)
    return {"x": d["imagenes"][:, None].astype(np.uint8), "y": d["etiquetas"].astype(np.int64),
            "train": d["particion"] == "train", "val": d["particion"] == "val", "origen": d["origen"].astype(str)}


def comprobar() -> int:
    ok = True

    def mira(que, cond, det=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FALLA':>5}] {que}" + (f"  {det}" if det else "")); ok &= bool(cond)

    d = cargar()
    mira(f"{DATASET}: huellas del manifiesto", True, f"{len(d['imagenes'])} imágenes")
    c = conjunto(d, "arco-E", "train")
    mira("arco-E train = 2000 positivas de (1, 32, 32) uint8 0/1",
         len(c["x_pos"]) == N_TRAIN and c["x_pos"].shape[1:] == (1, 32, 32) and c["x_pos"].dtype == np.uint8
         and int(c["x_pos"].max()) == 1)
    mira("negativos de arco-E: sin lazo (lo contiene)", F.FAMILIAS.index("lazo") not in set(np.unique(c["familia_neg"])))
    try:
        g = cargar(GRUESO)
        gr = g["grosor"][g["principal"] != F.FAMILIAS.index(F.VACIO)]
        mira(f"{GRUESO}: huellas, {len(g['imagenes'])} imágenes", True,
             "grosores " + ", ".join(f"{w}: {(gr == w).sum()}" for w in sorted(set(gr.tolist()))))
    except (SystemExit, RuntimeError) as e:
        mira(f"{GRUESO} publicado", False, str(e))
    dg = digitos()
    mira(f"{DIGITOS}: 5620 dígitos, 180/1617 en windep", len(dg["y"]) == 5620 and dg["train"].sum() == 180 and dg["val"].sum() == 1617)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
